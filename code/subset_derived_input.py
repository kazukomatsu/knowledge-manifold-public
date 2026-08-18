# -*- coding: utf-8 -*-
"""derived input から N 編を切り出す。
usage:
  python3 subset_derived_input.py --input derived400 --output derived200 --n 200 [--seed 0]
  python3 subset_derived_input.py --input derived400 --output derived200 --ids ids.txt
--ids: KM番号 (KM0001形式) または整数を1行1件。--n はシャッフル抽出 (seed固定で再現可)。
"""
import argparse, json, os, random, shutil
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--n", type=int)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ids")
a = ap.parse_args()

texts = sorted(os.listdir(os.path.join(a.input, "texts")))
stems = [t[:-3] for t in texts if t.endswith(".md")]
if a.ids:
    want = set()
    for line in open(a.ids):
        s = line.strip()
        if not s: continue
        want.add(s if s.startswith("KM") else f"KM{int(s):04d}")
    sel = [s for s in stems if s in want]
    missing = want - set(sel)
    assert not missing, f"not found: {sorted(missing)}"
else:
    assert a.n, "--n か --ids を指定"
    rng = random.Random(a.seed)
    sel = sorted(rng.sample(stems, a.n))

os.makedirs(os.path.join(a.output, "texts"), exist_ok=True)
os.makedirs(os.path.join(a.output, "normalized_records"), exist_ok=True)
for s in sel:
    shutil.copy(os.path.join(a.input, "texts", s + ".md"),
                os.path.join(a.output, "texts", s + ".md"))
    shutil.copy(os.path.join(a.input, "normalized_records", s + ".json"),
                os.path.join(a.output, "normalized_records", s + ".json"))
m = json.load(open(os.path.join(a.input, "freeze_manifest.json")))
m["derivation_id"] = os.path.basename(a.output.rstrip("/"))
m["subset_of"] = {"source_derivation_id": os.path.basename(a.input.rstrip("/")),
                  "n_selected": len(sel), "selection": ("ids-file" if a.ids else f"random seed={a.seed}"),
                  "selected_ids": sel}
m["source_raw_file_id"] = f"{sel[0]}..{sel[-1]} (subset {len(sel)})"
json.dump(m, open(os.path.join(a.output, "freeze_manifest.json"), "w"), ensure_ascii=False, indent=1)
print(f"selected {len(sel)} docs -> {a.output}")
