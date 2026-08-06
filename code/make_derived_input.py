# -*- coding: utf-8 -*-
"""
結合Markdown -> パイプライン入力(frozen derived input)への変換
================================================================
pdf_to_markdown.py が出力した 1 本の Markdown（PAPER [n/N] 区切り）を、
ext_v50 パイプラインが要求するディレクトリ構成に展開する。

    <出力先>/
      freeze_manifest.json
      texts/KM0001.md ...             各論文の本文（frontmatter を除いたもの）
      normalized_records/KM0001.json  各論文の書誌

LLM は使用しない。書誌は結合 Markdown の frontmatter から読み取る。

usage:
  python3 make_derived_input.py --input 100paper_rev.md --output inputs/derived/papers100_local
"""
import argparse, hashlib, json, re
from pathlib import Path

# PAPER [n/N] のブロック区切り
BLOCK_RE = re.compile(r"={60,}\s*\nPAPER \[(\d+)/(\d+)\]\s*\n={60,}\s*\n", re.M)
# ブロック先頭の YAML 風 frontmatter
FM_RE = re.compile(r"\A\s*---\s*\n(.*?)\n---\s*\n", re.S)


def parse_frontmatter(block: str):
    """ブロック先頭の frontmatter を辞書として取り出し、本文と分けて返す"""
    m = FM_RE.match(block)
    if not m:
        return {}, block
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, block[m.end():]


def strip_footer(body: str) -> str:
    """[END OF DOCUMENT: ...] 以降を落とす

    フッターにはファイル名が含まれるため、TF-IDF の特徴語に
    "118.pdf" のような断片が混じり得る。ただし既存の解析結果を
    再現する場合は残す必要があるため、既定では除去しない。
    """
    i = body.find("[END OF DOCUMENT")
    if i > 0:
        body = body[:i]
        body = re.sub(r"\n---\s*\n\s*\Z", "\n", body)
    return body.strip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="結合 Markdown ファイル")
    ap.add_argument("--output", required=True, help="出力する derived input ディレクトリ")
    ap.add_argument("--derivation-id", default=None, help="freeze_manifest に記録する ID")
    ap.add_argument("--strip-footer", action="store_true",
                    help="[END OF DOCUMENT: ...] フッターを本文から除去する。"
                         "既定では残す（既存の解析結果と同一の入力にするため）")
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    (out / "texts").mkdir(parents=True, exist_ok=True)
    (out / "normalized_records").mkdir(parents=True, exist_ok=True)

    raw = src.read_text(encoding="utf-8")
    parts = BLOCK_RE.split(raw)
    # parts = [前文, n, N, block, n, N, block, ...]
    blocks = [(parts[i + 1], parts[i + 3]) for i in range(0, len(parts) - 3, 3)]
    if not blocks:
        raise SystemExit(f"[エラー] '{src}' に PAPER [n/N] 区切りが見つかりません。")

    concat_text = []
    n_title = n_doi = n_year = 0

    for idx, (num, block) in enumerate(blocks, 1):
        did = f"KM{idx:04d}"
        meta, body = parse_frontmatter(block)
        body = strip_footer(body) if args.strip_footer else body.strip("\n")

        # 本文（TF-IDF の入力そのもの）
        (out / "texts" / f"{did}.md").write_text(body, encoding="utf-8")
        concat_text.append(body)

        title = meta.get("title", "").strip()
        doi = meta.get("doi", "").replace("\\_", "_").strip()
        year = meta.get("year", "").strip()
        n_title += bool(title); n_doi += bool(doi); n_year += bool(year)

        rec = {
            "schema_version": "1",
            "source_raw_file_id": did,
            "derivation_id": args.derivation_id or out.name,
            "document_title": title or meta.get("filename", did),
            "document_type": "academic_paper",
            "document_date": f"{year}-01-01" if re.fullmatch(r"\d{4}", year) else "",
            "date_precision": "year" if year else "unknown",
            "technical_themes": [],
            "metadata": {
                "doi": doi,
                "venue": meta.get("journal", ""),
                "source_pdf": meta.get("filename", ""),
            },
        }
        (out / "normalized_records" / f"{did}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")

    sha = lambda s: "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()
    manifest = {
        "freeze_manifest_schema_version": "1.0",
        "derivation_id": args.derivation_id or out.name,
        "source_raw_file_id": f"KM0001..KM{len(blocks):04d}",
        "source_raw_file_path": src.name,
        "source_raw_file_hash": sha(raw),
        "text_hash": sha("".join(concat_text)),
        "llm_used": False,
        "llm_provider": None,
        "llm_model": None,
        "n_documents": len(blocks),
    }
    (out / "freeze_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"documents      : {len(blocks)}")
    print(f"  with title   : {n_title}")
    print(f"  with doi     : {n_doi}")
    print(f"  with year    : {n_year}")
    print(f"chars (bodies) : {sum(len(t) for t in concat_text):,}")
    print(f"output         : {out.resolve()}")
    if n_title < len(blocks):
        print(f"[注意] title 欠落 {len(blocks)-n_title} 件。"
              f" normalized_records/*.json の document_title を確認してください。")


if __name__ == "__main__":
    main()
