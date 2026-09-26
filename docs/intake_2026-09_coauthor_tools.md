# 受領記録: 共著者提供の後処理コード(2026-09)

共著者(川越)から 2026-09 に受け取った 3 つの ZIP を調べ、この repo に足りない機能だけを
取り込んだ記録です。ZIP そのもの、添付のメール本文、投稿バンドル側の成果物(LLM 応答、
補足資料、ジャーナルコーパスの結果)はこの repo に入れていません。使い方は
[`postprocessing_ja.md`](postprocessing_ja.md)。

## 受け取ったもの

| ZIP | バイト数 | SHA-256 | 説明(提供者) |
|---|---|---|---|
| `repository_release_20260915.zip` | 2,279,846 | `5e29e802bcfba07c5e0fb49685a2e1ac9fb02274bdced1c1658d4c4fbad726c6` | 論文のコード一式。クラスターID追加とクラスター数変更のコード・README を含む |
| `scan_code.zip` | 18,974 | `3c7b6bffb7fee2173615dda6279aeef12c407711b7b046b67b06283b59f4a2d7` | 既存の解析結果に対する格子走査の追加コード |
| `grid_scan_ex_polymer.zip` | 299,458 | `5e6ae45918d967ab034289bfdcb0e26ba9d80bb037dcde4c0db9e4a7e877e168` | Polymer 100 編への適用例 |

比較の基準は main の clean commit `08013cdf1188e67718b1ecaa2b6cea8d5fbdd017`(origin/main と一致)。
日付の新しい ZIP を正本とはみなさず、ファイルごとに中身と既存の修正を比べて判断しました。

## ファイルごとの判断

判断: 採択 = そのまま / 部分採択 = 機能を取り込み改修 / 不採択 / 保留 = 著者の判断が要る /
同一 = public と同じ内容で対応不要。

### `repository_release_20260915.zip` の `code/`

| ファイル | public | 差分 | 判断 | 理由・検証 |
|---|---|---|---|---|
| `add_cluster_to_csv.py` | 無 | 新規(6 行) | 部分採択 | 目的: 座標 CSV にクラスタ列。原作はカレントディレクトリ依存で行順に付けるだけ。doc_id 結合・件数/重複/欠損/既存列の検査・上書き防止を足した。依存: numpy。検証: `TestAddCluster`、Polymer run で原作と内容一致(原作は CRLF、新版は LF) |
| `recluster.py` | 無 | 新規 | 部分採択 | 目的: マップ固定で任意の k。定義(svd 10 次元・`kmeans(seed=0)`・文献数順)を保存。同数の順の安定化、k 検査、空クラスタ、語彙 < 600、`--no-terms`、meta を追加。依存: kmlib.kmeans, term_ok, matplotlib。検証: `TestRecluster`(同梱 data/derived で k=5 が公開分割と一致)、Polymer run で k=4〜7 のラベル・CSV・dat がバイト一致、clusters JSON 一致 |
| `km_query.py` | 無(削除済み) | commit 749708b で削除したものとバイト一致 | 不採択 | 必要な `code/data/` の 10 ファイル(`X_sel.npy`, `vocab_sel.json`, `mean_sel.npy`, `gpr.npz`, `paths.npz` など)が同梱されておらず、生成するスクリプトも無い。import した時点で失敗する。`X_sel.npy`+`vocab_sel.json` は本文を復元し得る組合せで公開判断が別途要る。SPH・エントロピーを kmlib と別に再実装し、読み出し層を常に adaptive にしている(Sec. 3.3 の選択則と不一致)。格子走査には不要。749708b の判断を維持 |
| `kmlib.py` | 有 | 15 行 | 不採択 | public の `trustworthiness_continuity`・`knn_preservation` にある `kind="stable"`(CORRECTIONS.md #4)が無い。取り込むと continuity_k10 が CPU により 1.2e-5 ずれる不具合が戻る。後処理に要る関数(sph_weights, L2Field, GPR, kmeans, term_ok)は public と同一 |
| `04_fields.py` | 有 | 3 行 | 不採択 | `gpr_info.json` の implementation 記述が修正前(「custom NumPy GPR (sklearn unavailable)…」)に戻る(CORRECTIONS.md #3) |
| `09_manifest.py` | 有 | 2 行 | 不採択 | 測地線 optimizer の記述が修正前(「custom L-BFGS … scipy L-BFGS-B unavailable」)に戻る(CORRECTIONS.md #3) |
| `11_reports.py.ja.bak` | 無 | 旧版の退避 | 不採択 | public の `11_reports.py` は release 版と同一(書き直し後)。退避ファイルは不要 |
| `__pycache__/*.pyc` | 無 | ビルド生成物 | 不採択 | |
| 上記以外の 19 ファイル(`00_…`〜`15_…`, `make_evidence.py`, `run_v50.sh`, `term_stoplist.txt`, `validate.py`, `verify_reference.py` など) | 有 | なし | 同一 | バイト一致を確認 |

### `repository_release_20260915.zip` のその他

| 対象 | 判断 | 理由 |
|---|---|---|
| `code.zip`(入れ子) | 不採択 | `code/` の古いスナップショット。`add_cluster_to_csv.py`・`recluster.py` を含まない以外は `code/` と同一 |
| `README.md` | 部分採択 | "Post-processing utilities" 節の内容を `postprocessing_ja.md` と README に反映。ファイル自体は投稿バンドル用(環境を Python 3.10 と書くなど public と前提が違う) |
| `LICENSE_NOTE.md` | 保留 | 派生データを CC BY 4.0 とする案。public は repo 全体(派生データ含む)を MIT としており食い違う。著者間で決めること。今回は変更しない |
| `config/`, `results/`, `results_external/`, `evidence/`, `supplementary/` | 不採択 | 投稿バンドル側の成果物(LLM の応答 6 本、補足資料 PDF・TeX、ジャーナルコーパスの結果、図対応表)。public は code-and-derived-data release(README の対応表で "bundle only")。なお `config/gpr_info.json` の implementation も修正前の記述 |

### `scan_code.zip`

| ファイル | public | 判断 | 理由・検証 |
|---|---|---|---|
| `scan/k5_grid_scan.py` | 無 | 部分採択 → `code/grid_scan.py` | 目的: 全格子点の空白度・混合・支持・不確実性・計量。数値定義は保存。直した点: dx=0.05 で座標 CSV と details のキーが小数 1 桁固定のため重複・上書きが起きる → dx に応じた桁数と `ix, iy`。2/dx が整数でない dx は linspace が黙って別の間隔にする → 拒否。既定出力先が run の外の共通 `grid_scan/` → run 内の `grid_scan/k{K}_dx{dx}/`。上書き防止。入力の整合検査(文書順、座標 CSV と manifest の hash、GPR の学習点と scale、SVD・X_raw と Gram、本文と X_raw、文献一覧の文字数と本文、ラベルと k-means 分割)。N < 9 で d_k8 と adaptive h が壊れる → 明示的に失敗。語彙 < 1500 で argpartition が落ちる → 語彙数まで。固有値の丸めの負値の扱い、縮退と h の折れのフラグ。語の set 反復順を整列。丸めない値の npz、meta と runinfo の分離。依存: kmlib, scikit-learn(Matérn 再学習)。検証: `TestGridAxis`, `TestEigenAnalysis`, `TestGridScan`、実 run での比較(下記) |
| `scan/k5_grid_viz.py` | 無 | 部分採択 → `code/grid_viz.py` | 閾値と同時条件は保存。行数の平方根と並び順からの格子推定をやめ、meta と ix/iy から組み立てて欠落・重複・座標の食い違いを検査。座標とラベルの sha256 照合。画素の中心を格子点に合わせた(原作は半セルずれ)。検証: `TestGridViz`、実 run で候補一覧が原作・配布例と一致 |
| `scan/kmlib.py` | 有 | 不採択 | release の `kmlib.py` とバイト一致(安定ソート無し) |
| `scan/term_stoplist.txt` | 有 | 同一 | バイト一致 |
| `scan/README_scangrid.md` | 無 | 部分採択 | `postprocessing_ja.md` に書き直して統合。`verbalization_50w` を「LLM 言語化」と書いていた箇所は、スクリプトは空欄で出すことを明記 |

### `grid_scan_ex_polymer.zip`

8 ファイル(`k5_grid_scan_raw.csv` 441 行、`k5_grid_scan_details.json`、meta 2 つ、候補一覧、
図 2 枚)。**比較用の提供例として扱い、repo には入れない**(Polymer の論文タイトルと、
レンズ語に著者名の断片を含む。golden にもしない)。`verbalization_50w` の 441 行の英文は
LLM による簡易な文章化で、スクリプトの出力ではない。

## 検証

環境: macOS 26.7(Apple M4)、Python 3.11.16、numpy 2.4.6 / scipy 1.17.1 / scikit-learn 1.9.0 /
matplotlib 3.11.1(`requirements.txt` の固定版。README の基準は 3.11.15 で、パッチ版だけ違う)。
入力はこのマシンのローカル run 2 つ(Polymer と J. Informetrics の 100 編、コーパスは repo 外)。

| 項目 | 結果 |
|---|---|
| 既存テスト(変更前 08013cd) | 35 passed |
| `verify_reference.py`(変更前・変更後) | ALL 28 METRICS REPRODUCED |
| 全テスト(変更後) | 既存 35 + 新規 83 = 118 passed(Python 3.11 固定版、Python 3.10 + numpy 2.2.6 の両方) |
| 新 scan と原作 scan(同じ環境, dx=0.1, topk 15/15) | 両 run で CSV の共通 35 列(空欄の verbalization_50w を含む)が 441 点すべて文字列一致、details 441 点一致、Matérn θ・u 相関一致 |
| 新 scan と配布例(Python 3.11/numpy 2.4.6) | 数値の列は全点で文字列一致。L2 語は全点一致、L1 語は 154 点で相違 |
| 同上(Python 3.9.6/numpy 1.26.4) | 原作・新版とも数値・L1・L2・寄与文献の**すべてが 441 点で一致**(配布例を完全再現)。numpy 2.0.2 では L1 が 146 点で相違(環境を作り直して再確認)、2.0.2 と 2.4.6 の間でも 43 点相違 → 相違は同点 n-gram の並びの numpy 実装差 |
| 候補一覧 | 原作・配布例と一致(Polymer: B 76 点/1 領域、A 41 点/3 領域) |
| 再実行(PYTHONHASHSEED 0 と 7、別プロセス) | CSV・details・meta がバイト一致、npz の全配列一致 |
| 数値のみと語彙付き | CSV 全列一致 |
| dx = 0.05 | 1681 点、CSV・details・(ix, iy) が一対一、間隔の誤差 7e-17。dx=0.1 と共通の 441 点は全項目一致 |
| make_evidence.py(`--readout global --topk 15`、PYTHONHASHSEED 4 通り) | 両 run 各 10 点、290 語位置で語・df・出典・上位 3 文献が一致。H・N_eff・u は 3 桁丸めの範囲 |
| 固有値の関係(npz) | tr g = λ1+λ2、det g = λ1λ2、e1ᵀge1 = λ1、grad_norm² = tr g が相対 6e-16 以内。λ2 の最小 0.0038(負値なし)、(λ1−λ2)/λ1 の最小 0.027 |
| δ 感度(dx=0.1 の全格子点) | h が滑らかな 400 点: δ を 1e-2→1e-3→1e-4→1e-5 と変えたときの λ の相対変化の最大 2.5e-4、2.5e-6、2.4e-8(Polymer)/ 2.0e-4、2.1e-6、2.1e-8(JOI)。x=0, y=0 上の 41 点: 5.8e-3〜1.3e-2、5.7e-4〜1.3e-3、5.8e-5〜1.3e-4(h の折れ、`metric_kink`)。有限距離の cos 変化との差は s=±1e-4 で最大 3.4e-4(JOI 4.9e-4。軸上は最大 24%) |
| 再クラスタリング | Polymer run の k=4〜7 で原作とラベル・CSV・dat がバイト一致、JSON 一致。k=5 は標準分割と一致(対応 0→3, 1→4, 2→0, 3→2, 4→1) |

比較に使った run の入力(sha256 の先頭 16 桁): Polymer `coords.npy` 0344e8a328197e5f,
`X_raw.npy` 0a48e9f2fcb4da6e, `gpr_model.pkl` b24b481369182c1c; J. Informetrics `coords.npy`
ea5e5b7577342dce, `X_raw.npy` f632567842f09dbd, `gpr_model.pkl` 1ce55c92fa39c497。

## 独立レビュー

実装とは別の context のレビュー担当(Claude の subagent。Codex などの外部ツールは使っていない)に、
成果物と受け入れ条件だけを渡して反証的に検証させた。1 回目の判定は REVISE で、指摘は次の 4 点。
すべて対応し、2 回目の判定は APPROVED(重大な指摘なし)。2 回目に残った軽微な点(data/ の
下を指す symlink 経由の書き込み、manifest の無い run での文献一覧の差し替え、`--help` の既定
出力先、速度低下の原因の記述、有限距離の符号別の最大値、README の照合範囲の書き方)も
commit 前に直した(3 回目のレビューはしていない)。

1. 別 run の `corpus_metadata.csv`・`docs_clean.json`・ラベルを混ぜても検出できないのに、文書では
   検査すると書いていた → 座標 CSV・manifest の hash・SVD と Gram・X_raw と Gram・本文と X_raw・
   ラベルと k-means 分割の照合を実装し(`postproc_lib.check_run`)、照合できた範囲を meta に記録。
   意図して別のラベルを使う `--custom-labels` を追加。
2. hash seed に依存しないことのテストが、直した不具合を戻しても通っていた → 同点の代表語を
   作った単体テストと、8 通りの PYTHONHASHSEED で回すテストを追加。変異(set に戻す、d_k5 の
   添字、Matérn 5/2、閾値 1e-2、既定ソート、寄与文献の並べ方)がすべて検出されることを確認。
3. 寄与文献 top10 の同値の重みが numpy・CPU 依存であることを書いていなかった → 既知の制限に
   件数とともに記載(並べ方は make_evidence.py と揃えたまま)。
4. lam2 ≤ 0 の点で原作と表記が変わることを書いていなかった → 記載。

あわせて、同梱データ `data/` への書き込み禁止、大文字小文字だけ違う出力パスで入力を上書きする
問題(APFS)、`--force` 後に古い details・図が残る問題、zscale の厳密比較、既定の出力先が語数・δ で
分かれない問題を直し、数値の記述(δ 感度・有限距離・主方向誤差の統計量、ファイル数、列数)を
実測に合わせた。

## 見つかったが今回は直していないこと

- `make_evidence.py`: 代表語が同数で並ぶと Python の文字列 hash で選ばれる(PYTHONHASHSEED で
  `word_forms` の順、合成コーパスでは語そのものが変わる)。語彙 1500 未満で失敗する。
  同点 n-gram の並びと同値の寄与文献の並びが numpy の版・CPU に依存する(`grid_scan.py` と共通)。
  直すと証拠パッケージの語が変わり得るので、別途判断が要る。
- global SPH の h の折れ(x=0, y=0)は `04_fields.py grid` の計量(`grid_fields.npz`、
  補助図 figD)と H_field/N_eff の場にも同じく現れる。本体には手を入れていない。
- `LICENSE_NOTE.md` のライセンス案(上記、保留)。
- 提案にあったシルエット係数による k 推薦とキーワード指定クラスタリングは提供コードに無く、
  未実装。
