# 後処理ツール: クラスタ列・再クラスタリング・全格子点走査

完成した run(`code/run_v50.sh` の出力 `KM_OUT` と `KM_OUT/work`)に対して、マップを
作り直さずに追加の集計をするツール群です。パイプライン本体の成果物と公開値
(`code/verify_reference.py` の 28 指標)には影響しません。どのツールも LLM を呼ばず、
外部にも接続しません。

| ツール | 役割 |
|---|---|
| `code/add_cluster_to_csv.py` | 座標 CSV にクラスタ列を付けた**新しい** CSV を書く |
| `code/recluster.py` | マップを固定したまま、任意の k で再クラスタリングする |
| `code/grid_scan.py` | [-1,1]² の全格子点で空白度・混合・支持・不確実性・計量テンソルを計算する |
| `code/grid_viz.py` | 走査結果からヒートマップと候補点一覧(暫定スクリーニング)を作る |

原作は共著者(川越)が 2026-09 に提供したスクリプトです(`add_cluster_to_csv.py`・
`recluster.py` は `repository_release_20260915.zip`、`k5_grid_scan.py`・`k5_grid_viz.py`
は `scan_code.zip`)。数値の定義は原作のまま保ち、入力検証・出力先・決定論性を直して
取り込みました。ZIP の hash とファイルごとの採否は
[`intake_2026-09_coauthor_tools.md`](intake_2026-09_coauthor_tools.md)、原作との違いは
本書の最後の節にあります。

## 0. 共通の約束

- **入力は書き換えない。** 出力はすべて新しいファイルです。標準の `cluster_labels.npy`・
  `coordinates_2d.csv`・`coords.npy` などには書きません(テストで hash を比較しています)。
- **既存の出力は上書きしない。** 出力先に同名ファイルがあれば止まります。作り直すときは
  `--force` を付けます。`--force` でも、前回の走査から作った図・候補一覧や、今回は書かない
  details.json が出力先に残っていれば止まります(新しい meta と食い違うため)。同梱データ
  `data/` の中には書きません(`--outdir` / `--output` で外を指定)。
- **足りない成果物は補わない。** 必要なファイルが無ければ、足りないものを全部並べて失敗します
  (終了コード 2)。別の run や別の解析結果から探して補うことはしません。
- **同じ run の成果物かを確かめる。** 手元にある成果物の組合せで次のつながりを照合し、
  食い違えば失敗します。照合できなかった組合せ(ファイルが無いなど)は meta の
  `consistency_checks` にそのまま記録します。

  | 照合 | 許容差 | 捕まえる取り違え |
  |---|---|---|
  | `coordinates_2d.csv` の x, y = `coords.npy` | 1e-8(CSV は小数 8 桁) | 座標 CSV と座標 |
  | `manifest.json` の `artifact_hashes` = `coordinates_2d.csv`・`corpus_metadata.csv` の sha256 | 一致 | 別 run の文献一覧、run 後の編集 |
  | `gpr_model.pkl` の学習点 = `coords.npy`、scale = std(`svd_scores[:, :10]`) | 一致 / 相対 1e-12 | 別 run の GPR、マップを作り直した後の古い GPR |
  | `svd_scores.npy` から Z Zᵀ = `gram_l2.npy` | 1e-10 | 別 run の SVD |
  | `X_raw.npy` から作った Gram = `gram_l2.npy`(語彙付き) | 1e-6 | 別 run の特徴行列 |
  | 各文献の TF-IDF 上位 5 n-gram が `docs_clean.json` の同じ番号の本文にある(語彙付き) | — | 別 run の本文 |
  | `corpus_metadata.csv` の `chars_cleaned` = 本文の文字数(本文を読むとき) | 一致 | 別 run の文献一覧 |
  | ラベル = この run の `kmeans(svd[:, :10], K, seed=0)` の分割(番号の付け方は問わない) | — | 別 run のラベル |

  最後の照合は標準ラベルにも `recluster.py` のラベルにも当てはまります。意図して別のラベルを
  使うときは `--custom-labels` を付けます(照合を省いたことが meta に残ります)。このほか
  文書の順序(`coords.npy` の行 i = `corpus_metadata.csv` の `doc_id` i = ラベルの i 番目 =
  `docs_clean.json` の i 番目、ファイル名の並び)も検査します。Polymer と J. Informetrics の
  run ではすべて ok でした。見分けられない例: `manifest.json` が無く、本文も照合に使わないとき
  (数値のみの走査、`docs_clean.json` の無い run での `recluster.py`)に `corpus_metadata.csv`
  だけを差し替えた場合。`add_cluster_to_csv.py` は `--run` のときに座標とラベルだけを照合し、
  結果を表示します(ファイルには残しません)。
- **クラスタ番号は run と k に固有です。** 別の run、別の k の番号とは意味が対応しません。
- **記録。** `grid_scan.py` と `recluster.py` は入力ファイルの sha256、設定、許容誤差、
  Python と numpy/scipy/scikit-learn/matplotlib の版、コードの git SHA を meta JSON に書きます。
  実行時刻・所要時間・コマンドラインは別ファイル(`*_runinfo.json`)に分けてあり、
  同じ入力・同じ環境なら、別プロセスで再実行しても CSV・details.json・meta.json はバイト単位で、
  npz は全配列が一致します(numpy 2.4.6 では npz ファイルもバイト一致)。

環境は本体と同じです(`requirements.txt`、Python 3.11)。以下の例の `outputs/mycorpus` は
`run_v50.sh --run-id mycorpus --outputs-root outputs` の出力です。

## 1. クラスタ列を付けた座標 CSV

```bash
python3 code/add_cluster_to_csv.py --run outputs/mycorpus
# -> outputs/mycorpus/coordinates_2d_clusters.csv
```

| 項目 | 既定 | 変更 |
|---|---|---|
| 座標 CSV | `<KM_OUT>/coordinates_2d.csv` | `--coords` |
| ラベル | `<KM_OUT>/work/cluster_labels.npy`(標準 k=5) | `--labels`(例: `recluster/cluster_labels_k7.npy`) |
| 出力 | `<KM_OUT>/coordinates_2d_clusters.csv` | `--output` |
| 列名 | `cluster` | `--column` |

行の並びではなく `doc_id` で突き合わせます(ラベルの i 番目 = `doc_id` i)。`doc_id` が
0..N-1 をちょうど 1 回ずつ含むこと(重複・欠損・範囲外なし)、ラベル数が行数と一致すること、
CSV にまだ同名の列が無いことを確かめ、どれかが崩れていれば何も書かずに失敗します。
`--run` を渡すと、CSV の x, y が `<KM_OUT>/coords.npy` と一致すること、ラベルがその run の
k-means 分割であることも照合します(`--custom-labels` で後者を省略)。`--coords`・`--labels`・
`--output` だけで呼ぶと run との照合はせず、その旨を表示します。元の列は文字列のまま写し、
改行は入力と同じ LF です。

## 2. 再クラスタリング(k の変更)

```bash
python3 code/recluster.py --out outputs/mycorpus --k 4 6 7
# -> outputs/mycorpus/recluster/
```

**定義(原作・パイプラインと同じ)。** SVD 潜在の先頭 10 次元(`work/svd_scores.npy[:, :10]`)
に `kmlib.kmeans(Z, k, seed=0)`(k-means++ 初期化、最大 100 反復)を当てます。標準の k=5
(`05_geodesics.py pairs`)と同じ呼び出しです。番号は文献数の多い順に 0, 1, 2, ... と振り直し、
文献数が同じクラスタは k-means の元の番号順に並べます(安定ソート。原作は既定ソートで、
同数の順は実装任せでした)。座標・アンカー・場は変わりません。

**k=5 と標準ラベル。** k=5 の結果は標準の `cluster_labels.npy` と同じ分割で、番号だけが
文献数順に付け替わります。`clusters_k5_meta.json` の `standard_partition` に一致の判定と
番号の対応表が入ります。同梱の `data/derived` でも確認できます(下の `--no-terms` の例。
テスト `test_k5_on_the_shipped_map_is_the_published_partition`)。

k ごとの出力:

| ファイル | 内容 |
|---|---|
| `cluster_labels_k{K}.npy` | ラベル(文献数順の番号) |
| `coordinates_2d_k{K}.csv` | `doc_id,x,y,is_anchor,cluster` |
| `clusters_k{K}.json` | 各クラスタの文献数・上位 n-gram・代表文献・2D 重心 |
| `clusters_k{K}_meta.json` | 設定・入力 hash・環境・標準分割との対応 |
| `fig_map_k{K}.png` / `.dat` | マップ図と gnuplot 用データ(`doc x y cluster is_anchor`) |

**`top_ngrams` と代表文献の定義に注意。** 原作どおり、`top_ngrams` はクラスタの平均
TF-IDF ベクトル(L2 正規化後の平均、コーパス平均を**引かない**)の値が大きい char n-gram
です。語への復元もしないため、`poly`・`with`・`ture` のようにどのクラスタにも出る断片が
上位に来ます。代表文献はその平均ベクトルとの cos が最大の文献です。`13_llm_export.py` が
`llm_context.json` に書くクラスタ特徴(コーパス平均方向を引いた差、2D 重心に最も近い文献)
とは別の定義なので、クラスタの命名にそのまま使わないでください。

`--no-terms` を付けると `X_raw.npy`・`vocab.json`・`corpus_metadata.csv` を読まず、
上位 n-gram と代表文献を省きます(`null`)。コーパスを持たなくても、同梱の派生データで
公開マップの k を変えた図を作れます。

```bash
python3 code/recluster.py --out data/derived --data data/derived --k 4 5 6 7 \
    --no-terms --outdir /tmp/recluster_published
```

(`data/derived` の中には書けないので、`--outdir` で外を指定します。)同梱データでは
`coordinates_2d.csv` と `manifest.json`、Z Zᵀ = Gram の照合が行われ、結果は
`clusters_k{K}_meta.json` の `consistency_checks` に残ります。

**k の範囲。** 2 ≤ k ≤ (SVD 空間で互いに異なる文献の数)。重複した k は受けません。
k-means が空のクラスタを残した k があれば、どの k も書かずに失敗します。k > 10 では
図の色(tab10)が繰り返すので凡例で区別してください。

**提供コードに無いもの。** シルエット係数による k の推薦と、キーワードを指定した
クラスタリングは提供コードに実装されていません。今回は取り込んでおらず、今後の課題です。

## 3. 全格子点走査

```bash
# 441 点 (dx = 0.1)。レンズ語は make_evidence.py の既定と同じ 15 語ずつ
python3 code/grid_scan.py --out outputs/mycorpus --topk-l2 15 --topk-l1 15
# -> outputs/mycorpus/grid_scan/k5_dx0.1_l2-15_l1-15/

# 全域を細かく: dx = 0.05 で 41 x 41 = 1681 点
python3 code/grid_scan.py --out outputs/mycorpus --dx 0.05 --topk-l2 15 --topk-l1 15
# -> outputs/mycorpus/grid_scan/k5_dx0.05_l2-15_l1-15/

# 数値のみ (X_raw.npy・vocab.json・docs_clean.json を読まない、数秒)
python3 code/grid_scan.py --out outputs/mycorpus --no-words
# -> outputs/mycorpus/grid_scan/k5_dx0.1_nowords/
```

既定の出力先は走査した run の中の `grid_scan/k{K}_dx{dx}_l2-{語数}_l1-{語数}/`(数値のみは
`k{K}_dx{dx}_nowords/`、δ を既定の 1e-3 から変えたときは末尾に `_delta{δ}`)で、run・k・dx・
語数・δ ごとに分かれます(原作は run の外の共通ディレクトリに書き、別 run の結果と衝突し得ました)。
`--outdir` で変えられます。所要時間の目安は 100 文献・語彙 25 万の run で dx=0.1 が約 10 秒、
dx=0.05 が約 25 秒(Apple M4、1 プロセス)です。

**必要な入力。** `coords.npy`・`corpus_metadata.csv`・`work/{gram_l2.npy, svd_scores.npy,
gpr_model.pkl, cluster_labels.npy}`、語彙付きではさらに `work/{X_raw.npy, l2_norms.npy,
vocab.json}` と `corpus/docs_clean.json`。文献数は 9 以上が必要です(`d_k8` が 8 番目、
`H_field_adaptive` の h が 9 番目の近傍を使うため)。

**格子。** `--dx` は 2/dx が整数になる値(0.1, 0.05, 0.04, 0.025, 0.02, 0.01 など)だけ
受けます。0.3 のように割り切れない値は、原作では `np.linspace` が黙って別の間隔
(0.2857…)に丸めていたので、拒否するようにしました。1 軸 201 点(dx ≥ 0.01)まで。
座標は dx に必要な桁数で書きます(0.1 なら小数 1 桁、0.05 なら 2 桁)。CSV の `x,y`、
details.json のキー `"x,y"`、各行の格子添字 `ix, iy` は一対一に対応します。行の並びは
x が外側(行 r で `ix = r // n`, `iy = r % n`)で、原作と同じです。

**計算する量(定義は原作のまま)。**

| 列 | 定義 |
|---|---|
| `d_nn, d_k5, d_k8`, `nearest_doc_id` | 1・5・8 番目に近い文献までの 2D 距離 |
| `p_C0..p_C{K-1}` | global SPH 重みのクラスタごとの和(和 = 1 を検査) |
| `H_cluster` | p のエントロピー / log K |
| `dominant_C, second_C, p_dom, p_second, p_gap` | 1・2 位のクラスタ(同値は番号の小さい方を先) |
| `H_field`, `N_eff = N^H_field` | global SPH 重みのエントロピー / log N、有効文献数 |
| `H_field_adaptive` | 読み出し層の adaptive 則(knn_k=8)での同じ量(参考) |
| `max_doc_weight, top3_weight_share` | 最大重み、上位 3 文献の重みの和 |
| `u_RBF` | run の `gpr_model.pkl`(ARD-RBF+White)の σ_post/σ_prior |
| `u_M32` | 同じターゲット(`svd_scores[:, :10]` / std)で Matérn 3/2+White を再学習した感度用 |
| `grad_norm = √tr g`, `lam1 ≥ lam2`, `aniso = lam1/lam2` | L2 場 v(P) の計量 g_ab = ⟨∂_a v, ∂_b v⟩(中心差分 δ=1e-3) |
| `theta1_deg, e1x, e1y` | 主方向(e1x ≥ 0 に符号を揃える) |
| `dirgrad1 = √lam1, dirgrad2 = √lam2` | 主方向・副方向へ単位距離動いたときの意味変化率 |

**フラグ(新設)。**

| 列 | 立つ条件 | 意味 |
|---|---|---|
| `lam2_clipped` | lam2 < 0 かつ \|lam2\| ≤ 1e-12·lam1 | 丸め誤差として 0 に切った。これより負なら入力が壊れているとみなして失敗する |
| `e1_ill_defined` | (lam1−lam2)/lam1 < 1e-3 | 主方向が数値的に定まらない。`theta1_deg`・`e1` を解釈しない |
| `metric_kink` | 中心差分のステンシル内で最遠文献が入れ替わる(同距離を含む) | 下記 |

`metric_kink` について。global SPH の平滑化長は h(P) = (P から最遠の文献までの距離)/1.98
です。四隅はアンカーで固定され、ほかの文献は箱拘束で内側にあるので、最遠の文献はいつも隅の
アンカーで、x = 0 と
y = 0 の線上で入れ替わります。そこで h は折れ、場 v(P) の左右の片側微分 ∂⁺v, ∂⁻v が一致
しません。中心差分 (v(P+δe) − v(P−δe))/(2δ) は δ → 0 で片側微分の平均 (∂⁺v + ∂⁻v)/2 に近づき
(誤差は O(δ))、g はその平均した微分どうしの内積から作られます。これは左右それぞれの片側計量
の平均とは一般に異なります(折れを横切る向きの成分では、片側計量の平均より |∂⁺v − ∂⁻v|²/4
だけ小さくなる)。δ を変えると g は O(δ) で動きます。Polymer の run の dx = 0.1 の全格子点
で測ると、線から外れた 400 点では δ を 1e-2→1e-3、1e-3→1e-4、1e-4→1e-5 と変えたときの
λ の相対変化の**最大**が 2.5e-4、2.5e-6、2.4e-8(O(δ²)。J. Informetrics も 2.0e-4、2.1e-6、
2.1e-8)で、主方向の変化は δ = 1e-3→1e-4 で最大 6e-4° でした。計量から予測した有限距離の
cos 変化(1−cos ≈ ½ s² eᵀ g e)とは、s = ±1e-4 で相対 3.4e-4 以内(J. Informetrics は 4.9e-4)
で合います。線上の 41 点では同じ相対変化が 5.8e-3〜1.3e-2、5.7e-4〜1.3e-3、5.8e-5〜1.3e-4
(O(δ))で、有限距離との食い違いは最大 24% でした。dx = 0.1 では 441 点のうち 41 点、
dx = 0.05 では 81 点が該当します。これらの点の
`lam*`・`theta1_deg`・`dirgrad*` は方向の議論に使わないでください。

global 則の `H_field`・`N_eff` のヒートマップには x = 0, y = 0 に沿った十字状の谷が見えます。
四隅にアンカーがあり、ほかの文献は箱拘束で内側にあるので、global 則の h(P) は文献の配置に
よらず隅までの最大距離 / 1.98 で決まり、x = 0, y = 0 の線上で最小(カーネルが最も狭い)に
なります。谷の位置はこの定義上の性質と一致するので、`H_field`・`N_eff` の空間的な形は、global
SPH の平滑化長の定義による影響とコーパス固有の構造とを切り分けて解釈してください。

Polymer と J. Informetrics の両 run(dx = 0.1)で `lam2_clipped` と `e1_ill_defined` は 0 点、
(lam1−lam2)/lam1 の最小値はそれぞれ 0.027 と 0.029 でした。1e-3 の閾値は、線から外れた点での
計量の誤差(δ = 1e-3 で相対 2.5e-6 以下)から見積もると、閾値ちょうどの点でも主方向の誤差が
0.2° 未満(≈ 誤差 / (lam1−lam2)/lam1 ラジアン)に収まる値です。

**原作と値が変わる場合。** lam2 ≤ 0 の点に限って、原作の列と表記が変わります。原作は lam2 を
負のまま(`-0.00000`)、`dirgrad2` を `nan`、`aniso` を lam1/1e-300(300 桁の数字)で書いて
いました。新版は丸め誤差の範囲の負値を 0 に切り(`lam2_clipped`)、`dirgrad2` = 0、
`aniso` = `inf` とします(lam2 がちょうど 0 のときの `aniso` も `inf`)。Polymer・
J. Informetrics の両 run には該当する点がありません(lam2 の最小 0.0038 / 0.0035)。

**クラスタ数 K。** K はラベルから決まります(0..K−1 がすべて出現することを検査)。既定は
標準の `work/cluster_labels.npy`(k=5)で、ファイル名は `k5_grid_scan_*` です。
`recluster.py` のラベルを `--labels outputs/mycorpus/recluster/cluster_labels_k7.npy` の
ように渡すと `p_C0..p_C6` と `k7_grid_scan_*` になります(K=3 でテスト済み)。

**語彙(語彙付きのとき)。** 各点のテーマレンズ(L2)と集中レンズ(L1)の語、上位 10 の
寄与文献を `k{K}_grid_scan_details.json` に書きます。n-gram の選び方、語への復元、寄与文献の
並べ方は `make_evidence.py` と共通のコード(`code/evidence_lib.py`)で、同点の扱いも同じです
(第 7 節)。`df ≤ 3` の語には出典文献が付きます。

**出力ファイル。**

| ファイル | 内容 |
|---|---|
| `k{K}_grid_scan_raw.csv` | 1 行 = 1 格子点。原作の列 + `ix, iy, lam_gap_rel, e1_ill_defined, lam2_clipped, metric_kink` |
| `k{K}_grid_scan_details.json` | キー `"x,y"`。各点のレンズ語・寄与文献・`grid_index` |
| `k{K}_grid_scan_values.npz` | 丸めない値(計量テンソル `metric_g`、固有値の生値 `lam_raw` を含む) |
| `k{K}_grid_scan_meta.json` | 格子定義・設定・許容誤差・フラグ件数・Matérn の学習結果・入力 hash・環境 |
| `k{K}_grid_scan_runinfo.json` | 実行時刻・所要時間・コマンドライン(実行ごとに変わる) |

CSV は小数 4〜5 桁に丸めてあります。検算や後段の計算には `values.npz` を使ってください。

## 4. ヒートマップと候補点一覧

```bash
python3 code/grid_viz.py --scan outputs/mycorpus/grid_scan/k5_dx0.1_l2-15_l1-15 --out outputs/mycorpus
```

走査ディレクトリに `heatmap_Hcluster.png`、`heatmap_white_support_uncertainty.png`
(d_nn・N_eff・u_RBF・grad_norm)、`k{K}_candidate_prelist.csv`、
`k{K}_candidate_prelist_meta.json` を書きます。

格子は meta.json の定義と各行の `ix, iy` から組み立て、行の欠落・重複・座標の食い違いが
あれば失敗します(原作は行数の平方根と並び順から推定していました)。重ねて描く文献座標と
ラベルは、走査に使ったものと sha256 が一致することを確かめます。画素の中心は格子点に
合わせてあります(原作は半セルずれていた)。

**候補の条件(原作のまま)。** 閾値はこの走査の格子点についてのパーセンタイルで、同時条件は

| 層 | H_cluster ≥ | d_nn ≥ | N_eff ≥ | u_RBF ≤ | grad_norm ≥ |
|---|---|---|---|---|---|
| B(緩) | p70 | p30 | p30 | p70 | p50 |
| A(厳) | p85 | p40 | p30 | p60 | p70 |

`region` は 4 近傍の連結成分です。**これは暫定条件によるスクリーニングで、最適解・研究上の
発見・最終的な Discovery Score ではありません。** 閾値が格子全体のパーセンタイルなので、
dx を変えると閾値も候補も変わります。global 則では H_field が高止まりし、緩い条件では候補が
一つの領域につながりやすいので、A 層か細かい格子で分解してください。

## 5. 数値計算・証拠抽出・文章化の区別

| 段階 | 誰が | 決定論的か |
|---|---|---|
| 数値計算(`grid_scan.py` の CSV・npz) | スクリプト | はい |
| 証拠抽出(details.json のレンズ語・寄与文献、`make_evidence.py` の証拠パッケージ) | スクリプト | はい(同じスコアなら、プロセス・numpy の版・CPU によらず同じ語と文献。スコア自体の環境差は数値計算の側の話) |
| 文章化(仮想論文・短い要約) | LLM(パイプラインの外で人が行う) | いいえ |

CSV の `verbalization_50w` 列は原作との互換のために残した**空欄**で、スクリプトは文章を
書きません。提供された Polymer 100 編の例では 441 行すべてに英文が入っていますが、あれは
走査のあとに LLM で書かれたもので、スクリプトの出力ではありません。細かい格子では近傍の
寄与文献がほとんど共通になり、点ごとの文章の差は小さくなります。例の文章は簡易的なものと
して扱ってください。

詳しく解釈したい点は、座標を指定して既存の手順に渡します。

```bash
# 1) 候補一覧や図から座標を選ぶ (例: -0.6, 0.3)
# 2) その点の証拠パッケージを作る
python3 code/make_evidence.py --x -0.6 --y 0.3 \
    --out outputs/mycorpus --data outputs/mycorpus/work --code code
# 3) evidence_point_-0.6_0.3.json と docs/verbalization_protocol.md を一緒に LLM へ渡す
```

`make_evidence.py` の読み出し層は既定で LOO による選択(`--readout auto`)なので、研究室
コーパスでは adaptive、E9 の 2 つのジャーナルコーパスでは global になります。走査の場は
すべて global 則です。走査の語と突き合わせたいときは `--readout global --topk 15` を付け、
走査は `--topk-l2 15 --topk-l1 15` で回してください。両者は同じコードで語と寄与文献を選ぶので、
この設定ならレンズ語とその並び・df・出典文献・上位 3 寄与文献は同点の点も含めて一致し、
H・N_eff・u は make_evidence.py の丸め(3 桁)の範囲で一致します(テスト
`test_matches_make_evidence_with_the_same_settings`)。語数や読み出し層を揃えずに比べた食い違いは、
不具合ではなく設定の差です。

LLM の呼び出しをこれらのツールに組み込むことはしていません。

## 6. 新しい run に適用したら確認すること

1. `k{K}_grid_scan_meta.json` の `grid.n_points` が (2/dx+1)² で、CSV の行数・details の
   キー数と一致する(ツールが保証し、`grid_viz.py` も検査する)
2. `eigen_summary` の `lam2_clipped_points`・`e1_ill_defined_points` が 0 か、0 でなければ
   その点を方向の議論から外す。`metric_kink_points` は隅のアンカーがある限り軸上の点数になる
3. `u_corr_RBF_M32`(RBF と Matérn 3/2 の u の相関)が高いこと。候補の上位が両カーネルで
   入れ替わらないかも見る
4. 気になる点は `make_evidence.py --readout global` で証拠を出して突き合わせる
5. 文献に近い点の δ 感度を見たいときは `--delta 1e-4` と `1e-2` で走査し直して比べる

## 7. 既知の制限

- **同点の扱い(2026-09-26 に固定)。** char n-gram の TF-IDF 列は、同じ語から切り出した n-gram
  どうしや、同じ文献に 1 回だけ現れる語どうしでしばしば完全に一致し(Polymer の run で 25 万列の
  うち 76%)、レンズ語の候補には必ず同点が含まれます。以前はその順を numpy の実装
  (`np.argpartition`/`np.argsort`)に任せていたため、numpy の版や CPU が変わると語が入れ替わり
  (提供された Polymer の例とは numpy 1.26.4 で全点一致、2.4.6 では L1 の語が 154 点で相違)、
  `make_evidence.py` は語を set で持つため Python の文字列 hash(プロセスごとに乱数化)でも
  代表語や語形の並びが変わりました。今は `code/evidence_lib.py` に規則を明示しています。

  | 対象 | 規則 |
  |---|---|
  | n-gram | スコアの降順。完全に同じスコアは特徴番号(vocab の列番号。TfidfVectorizer の n-gram の辞書順)の昇順。上位 1500 件の境界にかかる同点も同じ規則で選ぶので、候補集合は一意 |
  | 寄与文献 | 重みの降順。完全に同じ重みは doc_id の昇順 |
  | 代表語 | 含む文献が最多の語、同数なら短い語、最後は辞書順(語の併合を判定するときの各 n-gram の代表は、文献数のあと辞書順) |
  | 語形(`word_forms`) | 含む文献の数の降順、同数は辞書順で最大 4 つ |
  | 出典文献 | doc_id の昇順 |

  同点として扱うのは値が完全に一致するときだけで、丸めや許容誤差は入れていません。浮動小数点の
  微小な差はそのまま順位の差になります。したがって「同じスコアが与えられれば、どの numpy・どの
  プロセスでも同じ語と文献を選ぶ」ことは保証しますが、スコアそのものを計算する数値計算の環境差
  (BLAS の違いなど)まで含めた完全一致は保証しません。同点の中で番号の小さいものを採るのは
  再現のための約束で、意味上の優先順位ではありません。
- **以前の証拠パッケージとは語が変わり得ます。** 同点の解き方が変わったため、2026-09-26 より前に
  作った証拠パッケージや走査とは、同点がかかる所で語・語形・df・出典文献・寄与文献が変わり得ます。
  数値(H・N_eff・u、走査の CSV・npz)は変わりません。実測の件数は CHANGELOG.md にあります。
- **語彙が少ない run。** 候補は上位 1500 件か語彙数の少ない方です。語フィルタを通る候補が尽きれば、
  指定した語数より少なく返します(0 語もあり得る)。語彙が空、成果物の文献数が食い違う、
  `--topk` が 1 未満、スコアに非有限値がある、といった入力は理由を示して止まります。
- `metric_kink` の点(x = 0, y = 0)では、中心差分が片側微分の平均に近づき、その平均した微分の
  内積から計量を作ります。片側計量の平均とは一般に異なります(第 3 節)。
- Matérn 3/2 の再学習(再起動 10 回)は走査のたびに行います。
- `code/km_query.py`(提供 ZIP に含まれる対話クエリキット)は取り込んでいません。必要な
  `code/data/`(`X_sel.npy`, `vocab_sel.json`, `gpr.npz` など 10 ファイル)が同梱されておらず、
  生成するスクリプトもありません。この repo では 2026-08-18 に同じ理由で削除済みです
  (commit 749708b)。格子走査には不要です。

## 8. 原作からの移行

| 原作 | この repo | 主な違い |
|---|---|---|
| `code/add_cluster_to_csv.py` | `code/add_cluster_to_csv.py` | カレントディレクトリ依存をやめ `--run` などで指定。行順でなく doc_id で結合し、件数・重複・欠損・既存列を検査。改行は LF(原作は CRLF) |
| `code/recluster.py` | `code/recluster.py` | 同数クラスタの順を安定化。k の検査、空クラスタで失敗、語彙 600 未満に対応、`--no-terms`、上書き防止、`clusters_k{K}_meta.json`。`--code` は廃止 |
| `scan/k5_grid_scan.py` | `code/grid_scan.py` | dx の検査と桁数の自動決定、`ix, iy`、run 内の既定出力先(語数・δ ごと)、入力の整合検査、固有値の丸め処理とフラグ、語の整列、`values.npz`、meta と runinfo の分離、`--labels`・`--custom-labels`。`--code` は廃止 |
| `scan/k5_grid_viz.py` | `code/grid_viz.py` | meta と ix/iy から格子を組み立てて検査、座標・ラベルの sha256 照合、画素中心の補正、上書き防止、prelist に `ix, iy` |
| `scan/kmlib.py`, `scan/term_stoplist.txt` | 使わない | repo の `code/kmlib.py`(CORRECTIONS.md #4 の安定ソートを含む)と `code/term_stoplist.txt`(同一内容)を使う |
| `scan/README_scangrid.md` | 本書 | |

数値への影響: dx = 0.1 の走査では、原作と新版(PR #1 の時点)の出力は同じ環境で CSV の共通 35 列
(空欄の `verbalization_50w` を含む)が全点で文字列一致し、details.json も全点一致しました(Polymer・
J. Informetrics の両 run)。その後の同点規則(2026-09-26、第 7 節)で、details.json のレンズ語・
df・出典文献・寄与文献 top10 は同点の所で原作と異なるようになりました(件数は CHANGELOG.md。
CSV と npz の数値は変わらない)。再クラスタリングはラベル・CSV・gnuplot データがバイト一致、
`clusters_k{K}.json` も一致(k = 4〜7)。候補一覧も一致します。変わったのは追加した列・
ファイル・図の画素位置と、lam2 ≤ 0 の点の表記(第 3 節。両 run に該当点なし)です。CSV の
改行は `grid_scan.py` と `recluster.py` が原作どおり CRLF、`add_cluster_to_csv.py` が入力と
同じ LF です。
