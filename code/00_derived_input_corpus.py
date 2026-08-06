# -*- coding: utf-8 -*-
"""Step 0: derived input をパイプラインのコーパス形式へ変換.
入力: KM_DERIVED (texts/*.md + normalized_records/*.json + freeze_manifest.json)
出力: KM_OUT/corpus/docs_clean.json, corpus_metadata.csv, extraction_log.json
前処理は v5.0 §1.1 (決定論的アーティファクト除去 → NFKC → 小文字化 → 空白圧縮)。
"""
import json, re, csv, unicodedata, os, time
from pathlib import Path

DI = Path(os.environ["KM_DERIVED"])
OUT = Path(os.environ["KM_OUT"])
(OUT / "corpus").mkdir(parents=True, exist_ok=True)
t0 = time.time()

CLEAN_STEPS = [
    ("markdown_image", r"!\[[^\]]*\]\([^)]*\)", " "),
    ("markdown_link_target", r"\[([^\]]*)\]\([^)]*\)", r"\1"),
    ("bare_url", r"(?:https?://|www\.)\S+", " "),
    ("page_tag", r"\S*_page_\d+\S*", " "),
    ("doi", r"\b10\.\d{4,9}/[^\s\"<>\]]+", " "),
    ("doi_label", r"\b[Dd][Oo][Ii]\s*:\s*\S+", " "),
    ("sbref_anchor", r"#?\bsb(?:ref|fig|tab|eq)\w*\d+\b", " "),
    ("rule_lines", r"^[\s|:\-=_*]{4,}$", " "),
    ("table_pipe_rule", r"\|[\s:\-]+\|", " "),
    ("long_digits", r"\d{6,}", " "),
]

def clean_text(t):
    counts = {}
    for name, pat, rep in CLEAN_STEPS:
        flags = re.M if name == "rule_lines" else 0
        t, n = re.subn(pat, rep, t, flags=flags)
        counts[name] = n
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t, counts

fm = json.load(open(DI / "freeze_manifest.json"))
text_files = sorted((DI / "texts").glob("*.md"))
assert text_files, f"no texts/*.md under {DI}"

docs, rows, log_docs = [], [], []
for idx, tf in enumerate(text_files):
    did = tf.stem
    raw = tf.read_text(encoding="utf-8")
    cleaned, counts = clean_text(raw)
    rec_path = DI / "normalized_records" / f"{did}.json"
    rec = json.load(open(rec_path)) if rec_path.exists() else {}
    meta_in = rec.get("metadata", {})
    docs.append({"doc_id": idx, "filename": did, "text": cleaned})
    rows.append({
        "doc_id": idx, "filename": did,
        "title": rec.get("document_title", did),
        "authors": "", "year": (rec.get("document_date") or "")[:4],
        "venue": meta_in.get("venue", ""), "doi": meta_in.get("doi", ""),
        "abstract": "", "keywords": ",".join(rec.get("technical_themes", [])),
        "citation_count": "", "source_pdf_path": meta_in.get("source_pdf", did),
        "chars_raw": len(raw), "chars_cleaned": len(cleaned)})
    log_docs.append({"doc_id": idx, "document_id": did, "chars_raw": len(raw),
                     "chars_cleaned": len(cleaned), "removal_counts": counts})

json.dump(docs, open(OUT / "corpus/docs_clean.json", "w", encoding="utf-8"))
with open(OUT / "corpus_metadata.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
log = {"source": str(DI), "derivation_id": fm.get("derivation_id"),
       "freeze_manifest_hashes": {k: fm.get(k) for k in
           ["source_raw_file_hash", "text_hash", "llm_used"]},
       "n_docs": len(docs),
       "extraction_method": "git-repo derived input (texts/*.md) preprocessed per v5.0 §1.1",
       "failed_pages": "n/a", "clean_steps_order": [s[0] for s in CLEAN_STEPS],
       "per_doc": log_docs, "elapsed_sec": round(time.time() - t0, 2)}
json.dump(log, open(OUT / "extraction_log.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print(f"docs: {len(docs)}  chars: {sum(len(d['text']) for d in docs)}  {time.time()-t0:.1f}s")
