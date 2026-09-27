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
| 新 scan と原作 scan(同じ環境, dx=0.1, topk 15/15。PR #1 の時点。同点規則の後の変化は CHANGELOG) | 両 run で CSV の共通 35 列(空欄の verbalization_50w を含む)が 441 点すべて文字列一致、details 441 点一致、Matérn θ・u 相関一致 |
| 新 scan と配布例(Python 3.11/numpy 2.4.6) | 数値の列は全点で文字列一致。L2 語は全点一致、L1 語は 154 点で相違 |
| 同上(Python 3.9.6/numpy 1.26.4) | 原作・新版とも数値・L1・L2・寄与文献の**すべてが 441 点で一致**(配布例を完全再現)。numpy 2.0.2 では L1 が 146 点で相違(環境を作り直して再確認)、2.0.2 と 2.4.6 の間でも 43 点相違 → 相違は同点 n-gram の並びの numpy 実装差 |
| 候補一覧 | 原作・配布例と一致(Polymer: B 76 点/1 領域、A 41 点/3 領域) |
| 再実行(PYTHONHASHSEED 0 と 7、別プロセス) | CSV・details・meta がバイト一致、npz の全配列一致 |
| 数値のみと語彙付き | CSV 全列一致 |
| dx = 0.05 | 1681 点、CSV・details・(ix, iy) が一対一、間隔の誤差 7e-17。dx=0.1 と共通の 441 点は全項目一致 |
| make_evidence.py(`--readout global --topk 15`、PYTHONHASHSEED 4 通り。PR #1 の時点) | 両 run 各 10 点、290 語位置で語・df・出典・上位 3 文献が一致。H・N_eff・u は 3 桁丸めの範囲 |
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
commit 前に直した。これらの修正と、その後の数理説明の修正(折れ点での計量は片側微分の平均の内積で
あって片側計量の平均ではないこと、H_field・N_eff の十字状の谷を平滑化長の定義による影響として
切り分けて読むこと)は、push 前に別 context のレビューにかけ、対象 commit と結果を PR に記録する。

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

## 残件の追跡

この表が残件の正です(2026-09-26 更新)。担当が決まっていない項目は「未定」。

| # | 項目 | 現状 | 判断事項 | 次の行動 | 担当・判断者 | 着手条件 | 完了条件 |
|---|---|---|---|---|---|---|---|
| 0 | `make_evidence.py` の決定論性と小語彙 | **解決**(ブランチ fix/evidence-determinism)。同点規則を `code/evidence_lib.py` に明示し `grid_scan.py` と共有、語彙 1500 未満に対応 | — | PR の merge | 小松 | 独立レビューの APPROVED と Linux CI(3.11・3.10)の通過 | merge |
| A | ライセンスの食い違い | 公開 repo は全体 MIT、投稿バンドルの `LICENSE_NOTE.md` は派生データ CC BY 4.0 案(下記) | 派生データの条件、バンドル側のコードの条件、書誌リストと LLM 応答の扱い | 著者に確認(質問案は作成済み、公開 repo には置かない) | 判断: 著者 4 名(小松・川越・大林・岡部)。質問の送付: 未定 | なし(不具合修正とは独立) | 両配布物の条件が決まり、LICENSE・README・CITATION.cff・バンドル側の記述が一致 |
| B | `recluster.py` の `top_ngrams` の定義 | 原作の定義(平均を引かない生の n-gram)のまま。13_llm_export とは別定義であることを文書化済み | 維持 / 別オプション追加 / 既定値変更(下記) | 使い道を決めて方針を選ぶ | 判断: 川越・小松。実装: 未定 | クラスタ特徴語の使い道(命名か注記か)が決まる | 選んだ方針の実装・テスト・文書 |
| C1 | k の推薦 | 未実装(提供コードに無い) | 目的・指標・出力の形(下記) | 仕様を決める | 判断: 著者。実装: 未定 | 下記の決定事項がそろう | 合成データと実 run での評価結果を添えた実装 |
| C2 | キーワード指定クラスタリング | 未実装(提供コードに無い) | 目的・照合方法・割当規則(下記) | 仕様を決める | 判断: 著者。実装: 未定 | 下記の決定事項がそろう | 同上 |
| D | 研究室コーパスでの格子走査 | 未実施(本文が手元に無い) | — | 下記チェックリストどおりに実施 | 実施: 未定(本文を正規に持つ人) | 研究室コーパスの run がそろう | チェックリストの全項目を記録 |
| E | global SPH の h の折れ(x=0, y=0) | 文書化・`metric_kink` で明示。`04_fields.py grid` の計量(`grid_fields.npz`、補助図 figD)と H_field/N_eff の場にも同じく現れる。本体は未変更 | 平滑化長の定義を変えるか(新しい数理方式になる) | 著者の判断まで現状維持 | 判断: 著者 | 著者が方式の見直しを決める | 別課題として設計・検証 |
| F | ほかの処理の同点順 | `13_llm_export.py`(`llm_context.json` の語・寄与文献)と `recluster.py`(`top_ngrams`)は numpy の同点順のまま。13_llm_export の候補数 600 は語彙数までに切り詰めた(範囲外は修正済み) | evidence_lib の規則をパイプラインの出力にも当てるか(同点で語が変わる) | B と合わせて判断 | 判断: 小松・川越 | B の方針が出る | 適用するなら実装・テスト・変更件数の記録 |

### A. ライセンスの比較

| 対象 | 公開 repo(`LICENSE`・`README.md`・`CITATION.cff`) | 投稿バンドル(`LICENSE_NOTE.md`) | 食い違い |
|---|---|---|---|
| コード | MIT(著作権者 4 名、2026)。`CITATION.cff` も `license: MIT` | 「著者が決める OSS ライセンス(MIT または BSD-3-Clause を提案)」 | 公開 repo では MIT に決まったが、バンドル側は未確定の書き方のまま |
| 設定 | 公開 repo には無い(設定値はコード内) | コードと同じ扱い | — |
| 派生データ | `data/`(Gram 行列・座標・SVD・参照指標・図)も MIT。README に「別のデータライセンスは無い」 | `results/`・`evidence/` は CC BY 4.0 | 同じ run の数値(座標・指標)が、配布物によって別の条件になる |
| 書誌リスト | `data/corpus_manifest.csv`(DOI・題名・年)は MIT の repo の一部 | `corpus_table_S1.csv` は「DOI で文献を特定。権利は各出版社に残る」 | 書誌情報の扱いの書き方が違う |
| LLM の応答 | 公開 repo には無い | 派生データとして公開(CC BY 4.0) | 公開 repo との直接の食い違いは無いが、バンドル公開時の条件は未確認 |
| 論文本文 | 含めない | 含めない | 一致 |

判断が要るのは、(1) 派生データを両配布物で MIT に揃えるか CC BY 4.0 に分けるか、(2) コードの MIT
をバンドル側の記述にも反映するか、(3) 書誌リストと LLM 応答の扱い、の 3 点。ライセンス本文は
変更しない。

### B. `recluster.py` の `top_ngrams`

| | `recluster.py` の `top_ngrams` | `13_llm_export.py` のクラスタ `top_ngrams` | `make_evidence.py`・`grid_scan.py` のレンズ語 |
|---|---|---|---|
| 元のベクトル | クラスタ内の L2 正規化 TF-IDF の平均(コーパス平均を引かない) | 同じ平均を正規化して、コーパス平均方向を引いた差 | SPH 重み付きの場(L2 は平均方向との差、L1 は KL 寄与) |
| 候補 | 上位 600(語彙が少なければ語彙数) | 上位 600(語彙数まで) | 上位 1500(語彙数まで) |
| 出力 | 生の char n-gram 8 個 | 生の char n-gram 15 個 | 語に復元した語(df・出典付き) |
| 代表文献 | 平均ベクトルとの cos が最大 | 2D 重心に最も近い文献 | — |
| 同点の順 | numpy の実装任せ | numpy の実装任せ | 規則で固定 |

- 現状を明確化して維持: 出力は原作と同じ。どのクラスタにも出る断片が上位に来るので命名には
  向かない(postprocessing_ja.md に明記済み)。追加作業なし。
- 別オプションを追加(例: 13_llm_export と同じ定義、または語への復元): 既定の出力は変わらない。
  キーや引数を足すのでテストと文書が要る。どの定義をクラスタの説明の根拠にするかを決める必要がある。
- 既定値を変更: `clusters_k{K}.json` の `top_ngrams` が変わり、原作の出力や既に作った図・資料と
  比べられなくなる。使っている資料があれば記述の更新が要る。

### C. k の推薦とキーワード指定クラスタリング(別々の機能)

どちらもマップ(座標)と標準の k=5(パイプライン・論文の図・`llm_context.json`)は変えない前提で、
`recluster.py` と同じく別ファイルに出す。クラスタ番号は k・方式ごとに固有とする。

| | C1. k の推薦 | C2. キーワード指定クラスタリング |
|---|---|---|
| 目的 | 表示に使う k を選ぶための比較材料を出す | 研究者が指定した語に沿って文献をまとめる |
| 入力 | `svd_scores[:, :10]`、k の範囲、指標(シルエット係数など、どれを使うかは未定)、seed と再起動回数 | キーワードの一覧、照合の方法(語の完全一致・n-gram・TF-IDF 列)、割当の規則(最大スコア、閾値) |
| 出力 | k ごとの指標の表と図。推薦は「候補」として示す | 文献ごとのラベルと根拠(どの語がどれだけ効いたか)、どの語にも当たらない文献の一覧 |
| 評価 | 合成データで既知のクラスタ数を回復するか、seed を変えても結論が保たれるか、指標どうしで結論が割れたときの扱い | 手作業の判定との一致、表記ゆれへの頑健性、割当なしの文献の割合 |
| 注意 | 推薦は研究上の正解ではない。複数の k を比べる材料として扱う | k-means の分割とは別物なので同じ `cluster` 列名・番号を使わない。語の選び方で結果が変わる(仮説を映す道具であって発見ではない) |

### D. 研究室コーパスでの格子走査(チェックリスト)

本文は公開 repo・PR に入れない。本文を正規に利用できる人が、Git 管理外の場所(`inputs/`・`outputs/`)で行う。

- [ ] 入力: 研究室コーパス 100 編の本文から `code/make_derived_input.py` で作った入力(`inputs/derived/…`)
- [ ] 環境: Python 3.11 と `requirements.txt` の固定版(公開座標の再構築は 3.11.15 で検証済み)
- [ ] run: `bash code/run_v50.sh --derived-input … --run-id … --outputs-root outputs`(`--quick` なし)と `code/validate.py` が PASS
- [ ] 必要な成果物: `coords.npy`・`coordinates_2d.csv`・`corpus_metadata.csv`・`corpus/docs_clean.json`・`manifest.json`・`work/{X_raw.npy, l2_norms.npy, vocab.json, gram_l2.npy, svd_scores.npy, gpr_model.pkl, cluster_labels.npy}`
- [ ] run の `coords.npy` が同梱の `data/derived/coords.npy` と一致するか(一致しなければ公開マップとは別のマップなので、以降の比較の意味をそこで記録する)
- [ ] `grid_scan.py --topk-l2 15 --topk-l1 15`(dx 0.1)と dx 0.05、`grid_viz.py` を実行
- [ ] meta の `consistency_checks` がすべて ok、`metric_kink_points` が 41(dx 0.1)、`lam2_clipped`・`e1_ill_defined` の件数
- [ ] `docs/USAGE_ja.md` 6.1 の例((-0.5, 0.0) の u 0.219・H 0.614・N_eff 16.9 とレンズ語)を `make_evidence.py`(`--readout auto`)で再現。数値は一致するはず。語は 2026-09-26 の同点規則で変わり得るので、変わった語が完全同点で説明できるかを確かめる
- [ ] `make_evidence.py --readout global --topk 15` と走査の details が一致
- [ ] 結果を worklog に記録し、差があれば原因を分類する
