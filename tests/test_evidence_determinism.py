# -*- coding: utf-8 -*-
"""Evidence selection must not depend on Python's string hashing, on numpy's order for equal
scores or weights, or on a vocabulary of at least 1500 n-grams.

The tie rules, stated here independently of the code:
  n-grams                 score descending, exactly equal scores by feature index ascending
  contributing documents  weight descending, exactly equal weights by doc_id ascending
  representative word     most documents, then shortest, then alphabetical (the per-n-gram
                          representative that decides merges: most documents, then alphabetical)
  word forms              most documents, then alphabetical
  source documents        doc_id ascending
Synthetic runs only (tests/synthetic_run.py); no corpus text.
"""
import importlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synthetic_run import build_run  # noqa: E402


def ref_order(values):
    """The rule: descending value, exact ties by ascending index (a full, stable ordering)."""
    values = np.asarray(values)
    return np.lexsort((np.arange(len(values)), -values))


def make_evidence(run, x, y, outfile, seed="0", topk=15, readout="global"):
    return subprocess.run([sys.executable, os.path.join(CODE, "make_evidence.py"), "--x", str(x), "--y", str(y),
                           "--out", run, "--data", os.path.join(run, "work"), "--code", CODE,
                           "--readout", readout, "--topk", str(topk), "--outfile", str(outfile)],
                          capture_output=True, text=True, env=dict(os.environ, PYTHONHASHSEED=str(seed)))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return build_run(tmp_path_factory.mktemp("evidence"))


@pytest.fixture
def lib():
    return importlib.import_module("evidence_lib")


# ============================================================ make_evidence.py end to end
class TestMakeEvidence:
    def test_output_does_not_depend_on_the_hash_seed(self, run, tmp_path):
        # on this map (0.3, 0.7) and (1.0, -1.0) have representative words tied on df and length
        for x, y in ((0.3, 0.7), (1.0, -1.0), (-0.5, 0.0)):
            blobs = set()
            for seed in range(6):
                out = tmp_path / f"ev_{x}_{y}_{seed}.json"
                p = make_evidence(run, x, y, out, seed)
                assert p.returncode == 0, p.stderr
                blobs.add(out.read_bytes())
            assert len(blobs) == 1, f"({x}, {y}): {len(blobs)} different packages over 6 hash seeds"

    def test_contributing_documents_follow_the_tie_rule(self, run, tmp_path):
        from kmlib import sph_weights
        P = np.load(os.path.join(run, "coords.npy"))
        for x, y in ((-1.0, -0.5), (-1.0, 0.5), (-1.0, 1.0)):
            w = sph_weights(np.array([[x, y]]), P, h_mode="global")[0]
            assert len(set(np.sort(w)[::-1][:4].tolist())) < 4, "the point was chosen for an exact tie"
            out = tmp_path / f"ev_{x}_{y}.json"
            p = make_evidence(run, x, y, out)
            assert p.returncode == 0, p.stderr
            got = [c["doc"] for c in json.load(open(out))["contributing_documents_top3"]]
            assert got == ref_order(w)[:3].tolist(), (x, y)

    @pytest.mark.parametrize("n_features", [40, 800, 1500])
    def test_vocabularies_at_or_below_the_candidate_pool(self, tmp_path, n_features):
        run = build_run(tmp_path, max_features=n_features)
        X = np.load(os.path.join(run, "work", "X_raw.npy"))
        assert X.shape[1] == n_features
        docs = json.load(open(os.path.join(run, "corpus", "docs_clean.json")))
        words = [set(d["text"].split()) for d in docs]
        out = tmp_path / "ev.json"
        p = make_evidence(run, -0.5, 0.0, out, topk=15)
        assert p.returncode == 0, p.stderr
        ev = json.load(open(out))
        N = len(docs)
        for lens in ("theme_lens_L2", "concentration_lens_L1"):
            assert 1 <= len(ev[lens]) <= 15
            for e in ev[lens]:
                assert any(e["term"] in ws for ws in words), e["term"]      # a word of the corpus
                if e["df"] <= 3:
                    ids = [s["doc"] for s in e["source_docs"]]
                    assert ids == sorted(ids) and len(ids) == e["df"] and all(0 <= i < N for i in ids)
        assert all(0 <= c["doc"] < N for c in ev["contributing_documents_top3"])

    def test_rejects_a_non_positive_topk(self, run, tmp_path):
        p = make_evidence(run, -0.5, 0.0, tmp_path / "ev.json", topk=0)
        assert p.returncode != 0 and "topk" in (p.stderr + p.stdout)


# ============================================================ the shared selection rules
class TestTopIndices:
    def test_ties_across_the_boundary_take_the_lowest_indices(self, lib):
        scores = np.array([5.0, 3.0, 3.0, 3.0, 3.0, 1.0, 3.0])
        assert lib.top_indices(scores, 3).tolist() == [0, 1, 2]
        assert lib.top_indices(scores, 5).tolist() == [0, 1, 2, 3, 4]
        assert lib.top_indices(scores, 6).tolist() == [0, 1, 2, 3, 4, 6]

    def test_fixed_expectation_for_every_numpy(self, lib):
        """Literal answer for a tie-heavy vector, so every numpy build (CI: 2.2 and 2.4) must agree."""
        scores = np.array([0.2, 0.9, 0.5, 0.9, 0.5, 0.2, 0.5, 0.9, 0.1, 0.5, 0.2, 0.9])
        assert lib.top_indices(scores, 6).tolist() == [1, 3, 7, 11, 2, 4]
        assert lib.top_indices(scores, 12).tolist() == [1, 3, 7, 11, 2, 4, 6, 9, 0, 5, 10, 8]

    @pytest.mark.parametrize("seed", range(20))
    def test_equals_the_full_ordering_on_tie_heavy_vectors(self, lib, seed):
        rng = np.random.default_rng(seed)
        scores = rng.integers(0, 6, size=int(rng.integers(1, 400))).astype(float) / 7.0
        for n in (1, 5, len(scores) // 2 or 1, len(scores), len(scores) + 3):
            assert lib.top_indices(scores, n).tolist() == ref_order(scores)[:n].tolist()

    @pytest.mark.parametrize("seed", range(5))
    def test_without_ties_the_order_is_the_old_one(self, lib, seed):
        scores = np.random.default_rng(seed).normal(size=5000)
        assert len(np.unique(scores)) == len(scores)
        old = np.argpartition(scores, -1500)[-1500:]
        old = old[np.argsort(scores[old])[::-1]]
        assert lib.top_indices(scores, 1500).tolist() == old.tolist()

    @pytest.mark.parametrize("bad", [np.array([]), np.array([1.0, np.nan]), np.array([1.0, np.inf]),
                                     np.ones((2, 2))])
    def test_rejects_inputs_it_cannot_order(self, lib, bad):
        with pytest.raises(ValueError):
            lib.top_indices(bad, 1)

    def test_integer_and_boolean_scores(self, lib):
        assert lib.top_indices(np.array([1, 2, 2, 0], dtype=np.uint8), 4).tolist() == [1, 2, 0, 3]
        assert lib.top_indices(np.array([False, True, True]), 2).tolist() == [1, 2]

    def test_rejects_a_non_positive_count(self, lib):
        with pytest.raises(ValueError):
            lib.top_indices(np.ones(3), 0)


class TestRankDocuments:
    def test_equal_weights_by_ascending_doc_id(self, lib):
        w = np.array([0.1, 0.3, 0.2, 0.3, 0.1])
        assert lib.rank_documents(w, 3).tolist() == [1, 3, 2]
        assert lib.rank_documents(w, 10).tolist() == [1, 3, 2, 0, 4]


class TestPickNgrams:
    VOCAB = ["alpha", "alphabet", "betamax", "gamma", "delt", "epsilon", "x y", "zeta"]

    def test_filtered_candidates_in_rule_order(self, lib):
        sc = np.array([0.5, 0.9, 0.9, 0.5, 0.5, 0.5, 0.9, 0.5])
        # candidates: 1, 2, 6 (0.9), then 0, 3, 4, 5, 7 (0.5); "x y" fails term_ok, "alpha" is in "alphabet"
        assert lib.pick_ngrams(sc, self.VOCAB, [], 4) == [("alphabet", 1), ("betamax", 2), ("gamma", 3),
                                                            ("delt", 4)]

    def test_pool_boundary_is_part_of_the_rule(self, lib):
        sc = np.array([0.5, 0.9, 0.9, 0.5, 0.5, 0.5, 0.9, 0.5])
        # pool 5 keeps 1, 2, 6 and the two lowest indices among the 0.5 ties (0, 3)
        assert lib.pick_ngrams(sc, self.VOCAB, [], 8, pool=5) == [("alphabet", 1), ("betamax", 2), ("gamma", 3)]

    def test_returns_fewer_terms_when_the_candidates_run_out(self, lib):
        sc = np.linspace(1, 0, len(self.VOCAB))
        got = lib.pick_ngrams(sc, self.VOCAB, [], 50)
        assert [t for t, _ in got] == ["alpha", "betamax", "gamma", "delt", "epsilon", "zeta"]

    def test_vocabulary_and_scores_must_agree(self, lib):
        with pytest.raises(ValueError):
            lib.pick_ngrams(np.ones(3), self.VOCAB, [], 2)


WORD_DOCS = [{"text": "betaxyz alphxyz gammaword lamxyzz"}, {"text": "alphxyz deltaword omegaxyz"},
             {"text": "zetaxyz etaxyz thetaxyz iotaxyz kappaxyz"}]


class TestWordResolver:
    def test_representative_word(self, lib):
        r = lib.WordResolver(["xyz "], WORD_DOCS)
        # words ending in "xyz": alphxyz (2 docs) beats the one-document words
        got = r.resolve([("xyz", 0)])
        assert got[0]["term"] == "alphxyz" and got[0]["docs"] == [0, 1, 2]

    def test_word_forms_tie_alphabetically_and_stop_at_four(self, lib):
        r = lib.WordResolver(["xyz "], WORD_DOCS)
        forms = r.resolve([("xyz", 0)])[0]["word_forms"]
        # alphxyz (df 2) first, then the df-1 forms in alphabetical order
        assert forms == ["alphxyz", "betaxyz", "etaxyz", "iotaxyz"]

    def test_merged_term_prefers_documents_then_length_then_alphabet(self, lib):
        docs = [{"text": "pqrsab pqrsac"}, {"text": "pqrsab pqrsac"}]
        r = lib.WordResolver([" pqrs"], docs)
        assert r.resolve([("pqrs", 0)])[0]["term"] == "pqrsab"

    def test_shorter_word_beats_the_alphabet_for_every_term(self, lib):
        # one n-gram, no merge: bxyz and abcxyz both in one document; the shorter wins over the alphabet
        got = lib.WordResolver(["xyz "], [{"text": "abcxyz bxyz"}]).resolve([("xyz", 0)])[0]
        assert got["term"] == "bxyz" and got["word_forms"] == ["abcxyz", "bxyz"]

    def test_ties_are_broken_alphabetically_not_by_document_order(self, lib):
        # pqrszz comes first in the documents, pqrsaa first in the alphabet; same df, same length
        docs = [{"text": "pqrszz"}, {"text": "pqrsaa"}]
        got = lib.WordResolver([" pqrs"], docs).resolve([("pqrs", 0)])[0]
        assert got["term"] == "pqrsaa" and got["word_forms"] == ["pqrsaa", "pqrszz"] and got["docs"] == [0, 1]

    def test_entry_representative_tie_decides_the_merge(self, lib):
        # "dq " is in zzzdq (doc 0) and abcdq (doc 1), one document each: the entry's representative is
        # abcdq by the alphabet, and abcdq starts with the word "abcd" of the second entry, so the two
        # merge into one term; with zzzdq as representative they would stay two terms
        docs = [{"text": "zzzdq"}, {"text": "abcdq"}, {"text": "abcd"}]
        got = lib.WordResolver(["dq ", " abcd "], docs).resolve([("dq", 0), ("abcd", 1)])
        assert got == [{"term": "abcd", "word_forms": ["abcd", "abcdq", "zzzdq"], "docs": [0, 1, 2]}]

    def test_merge_decision_ignores_length(self, lib):
        # "xyz " is in abcdxyz (doc 0) and bxyz (doc 1): the n-gram's representative for the merge
        # test is abcdxyz (documents, then alphabet — not the shorter bxyz), which starts with the
        # word "abcd" of the second n-gram, so the two merge; the merged term is then the shortest
        docs = [{"text": "abcdxyz"}, {"text": "bxyz"}, {"text": "abcd"}]
        got = lib.WordResolver(["xyz ", " abcd "], docs).resolve([("xyz", 0), ("abcd", 1)])
        assert got == [{"term": "abcd", "word_forms": ["abcd", "abcdxyz", "bxyz"], "docs": [0, 1, 2]}]

    def test_fragment_without_a_containing_word(self, lib):
        r = lib.WordResolver(["qqqq"], WORD_DOCS)
        assert r.resolve([("qqqq", 0)]) == [{"term": "qqqq", "word_forms": [], "docs": []}]


TIE_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
from evidence_lib import WordResolver
docs = [{"text": "betaxyz alphxyz gammaword lamxyzz"}, {"text": "deltaword omegaxyz"},
        {"text": "zetaxyz etaxyz thetaxyz iotaxyz kappaxyz"}]
print(json.dumps(WordResolver(["xyz "], docs).resolve([("xyz", 0)])))
"""


def test_word_resolution_does_not_depend_on_the_hash_seed():
    got = set()
    for seed in range(8):
        p = subprocess.run([sys.executable, "-c", TIE_SCRIPT, CODE], capture_output=True, text=True,
                           env=dict(os.environ, PYTHONHASHSEED=str(seed)))
        assert p.returncode == 0, p.stderr
        got.add(p.stdout)
    assert len(got) == 1
