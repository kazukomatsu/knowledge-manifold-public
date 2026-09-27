# -*- coding: utf-8 -*-
"""証拠の語と寄与文献の選び方 (make_evidence.py と grid_scan.py が共有する).

同じスコア・同じ重みの並べ方を明示し、Python の文字列 hash (プロセスごとに乱数化される) や
numpy のソート実装 (版や CPU で変わる) に結果を依存させない。扱うのは値が完全に一致する同点
だけで、丸めや許容誤差は入れない (浮動小数点の微小な差はそのまま順位の差になる)。

  n-gram    スコアの降順。完全に同じスコアは特徴番号 (vocab の列番号) の昇順。上位
            CANDIDATE_POOL 件の境界にかかる同点も同じ規則で選ぶので、候補集合は一意に決まる。
  寄与文献  重みの降順。完全に同じ重みは doc_id の昇順。
  代表語    含む文献の数が最も多い語、同数なら短い語、最後は辞書順 (コードポイント順)。語の併合を
            判定するときの各 n-gram の代表は、文献数のあと辞書順。
  語形      含む文献の数の降順、同数は辞書順で、最大 MAX_WORD_FORMS 個。
  出典文献  doc_id の昇順。
スコアの定義 (L2・L1 レンズ)、語フィルタ (kmlib.term_ok)、部分文字列の重複除去、語の併合規則は
以前のまま。
"""
import re

import numpy as np

from kmlib import term_ok

CANDIDATE_POOL = 1500   # 語の候補にする上位 n-gram の数。語彙がこれより少なければ語彙数まで
MAX_WORD_FORMS = 4
_WORD_RE = re.compile(r"[a-z][a-z\-]{2,}")


def _scores(values):
    s = np.asarray(values)
    if s.dtype.kind in "biu":               # 符号なし整数や bool は符号反転で壊れるので実数にしてから並べる
        s = s.astype(np.float64)
    if s.ndim != 1 or s.size == 0:
        raise ValueError(f"expected a non-empty 1-D vector of scores, got shape {s.shape}")
    if not np.isfinite(s).all():
        raise ValueError(f"{int((~np.isfinite(s)).sum())} scores are not finite")
    return s


def top_indices(values, n):
    """値の大きい順に n 個の番号 (同じ値は番号の小さい順). 全体を並べた先頭 n 個と一致する."""
    s = _scores(values)
    if n < 1:
        raise ValueError(f"the number of items must be at least 1, got {n}")
    n = min(int(n), s.size)
    if n == s.size:
        return np.lexsort((np.arange(s.size), -s))
    kth = np.partition(s, s.size - n)[s.size - n]       # n 番目に大きい値 (値はソート実装によらない)
    above = np.flatnonzero(s > kth)
    tied = np.flatnonzero(s == kth)[:n - len(above)]     # 境界の同点は番号の小さい方から
    cand = np.concatenate([above, tied])
    return cand[np.lexsort((cand, -s[cand]))]


def rank_documents(weights, k):
    """寄与文献の上位 k 件: 重みの降順、同じ重みは doc_id の昇順."""
    return top_indices(weights, k)


def pick_ngrams(scores, vocab, stoplist, k, pool=CANDIDATE_POOL):
    """上位 pool 件の n-gram から、語フィルタと部分文字列の重複除去を通った最大 k 個を (n-gram, 列番号) で返す.

    候補が尽きれば k 個より少なく返す (0 個もあり得る)。
    """
    s = _scores(scores)
    if len(vocab) != s.size:
        raise ValueError(f"{s.size} scores for a vocabulary of {len(vocab)} n-grams")
    if k < 1:
        raise ValueError(f"the number of terms must be at least 1, got {k}")
    out = []
    for c in top_indices(s, pool):
        t = vocab[c].strip()
        if not term_ok(t, stoplist):
            continue
        if any(t in x or x in t for x, _ in out):
            continue
        out.append((t, int(c)))
        if len(out) >= k:
            break
    return out


def _match(word, frag):
    lead, trail, s = frag.startswith(" "), frag.endswith(" "), frag.strip()
    return (word == s) if (lead and trail) else (word.startswith(s) if lead else (word.endswith(s) if trail else (s in word)))


class WordResolver:
    """選んだ n-gram を出典テキスト中の語へ復元し、語を共有する項目を併合する."""

    def __init__(self, vocab, docs):
        self.vocab = vocab
        self.tok = [sorted(set(_WORD_RE.findall(d["text"].lower()))) for d in docs]
        self._cache = {}

    def words_for(self, frag_raw):
        """n-gram (前後の空白は語境界) を含む語 -> その語を含む文献番号の集合."""
        if frag_raw not in self._cache:
            lead, trail, s = frag_raw.startswith(" "), frag_raw.endswith(" "), frag_raw.strip()
            hits = {}
            for di, ts in enumerate(self.tok):
                if lead and trail:
                    found = [w for w in ts if w == s]
                elif lead:
                    found = [w for w in ts if w.startswith(s)]
                elif trail:
                    found = [w for w in ts if w.endswith(s)]
                else:
                    found = [w for w in ts if s in w]
                for w in found:
                    hits.setdefault(w, set()).add(di)
            self._cache[frag_raw] = hits
        return self._cache[frag_raw]

    def resolve(self, picks):
        """[(n-gram, 列番号)] -> [{"term", "word_forms", "docs"}] (項目の順は最初の n-gram の順)."""
        entries = [{"frags": [s], "raw": [self.vocab[c]], "words": self.words_for(self.vocab[c])}
                   for s, c in picks]
        n = len(entries)
        par = list(range(n))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        def rep_of(e):
            w = e["words"]
            return min(w, key=lambda t: (-len(w[t]), t)) if w else e["frags"][0]

        for i in range(n):
            for j in range(i + 1, n):
                a, b = entries[i], entries[j]
                ra, rb = rep_of(a), rep_of(b)
                morph = len(ra) >= 4 and len(rb) >= 4 and (ra.startswith(rb) or rb.startswith(ra))
                if set(a["words"]) & set(b["words"]) or morph:
                    par[find(i)] = find(j)
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(entries[i])
        out = []
        for g in groups.values():
            allw, frs, rawfr = {}, [], []
            for e in g:
                frs += e["frags"]
                rawfr += e["raw"]
                for w, ds in e["words"].items():
                    allw.setdefault(w, set()).update(ds)
            if not allw:
                out.append({"term": frs[0], "word_forms": [], "docs": []})
                continue
            # 群の n-gram の半数以上に合う語形だけを残す (以前と同じ)
            thr = max(1, (len(rawfr) + 1) // 2)
            keep = {w: d for w, d in allw.items() if sum(_match(w, f) for f in rawfr) >= thr} or allw
            rep = min(keep, key=lambda t: (-len(keep[t]), len(t), t))
            forms = sorted(keep, key=lambda t: (-len(keep[t]), t))[:MAX_WORD_FORMS]
            out.append({"term": rep, "word_forms": forms, "docs": sorted(set().union(*keep.values()))})
        return out
