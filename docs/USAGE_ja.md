# ローカル実行手順(LLM 不使用)

パイプライン本体は LLM を一切呼びません。`13_llm_export.py` も証拠パッケージを JSON に
書き出すだけで、外部 API には接続しません。`code/` 配下に `requests` / `urllib` / `httpx` /
`openai` / `anthropic` の import は存在しません。以下はすべてオフラインで完結します。

英語版の概要と、コーパスなしで公開値を再現する手順は [`../README.md`](../README.md) にあります。

## 0. 環境

- **Python 3.11**(下限ではなくサポート対象そのもの)。3.11.15 で検証。
- 必要パッケージ: `requirements.txt`(版を完全固定)
- 任意: `requirements-optional.txt`(`umap-learn`。E10 の UMAP ベースライン比較にのみ使用。
  無ければ自動でスキップされる)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-optional.txt   # 任意
```

**3.11 に限定しているのは、公開値の再現を端から端まで確認したのが 3.11.15 だけだからです。**
マップは線形代数スタックに敏感で、sparse な `X @ X.T` が BLAS ビルド間で相対 5e-9 ずれるだけで
アンカー割り当てが変わり得ます([`../CORRECTIONS.md`](../CORRECTIONS.md) の #2)。そのため
依存の版も同梱成果物を生成したものに固定しています。新しい Python でも import と単体テストは
通りますが、論文の数値が再現することは主張していません。また `umap-learn` は `numba` 依存の
ため 3.14 ではビルドできません。

**Python 3.10 について。** `requirements.txt` の固定版は 3.10 では解決できません(`numpy==2.4.6` が
Requires-Python >=3.11、3.10 の上限は numpy 2.2.6)。3.10 で入る最新スタック
(numpy 2.2.6 / scipy 1.15.3 / sklearn 1.7.2 / matplotlib 3.10.9、`umap-learn==0.5.12` も可)では
2026-08-18 実測で `pytest tests/` 32 passed、`code/verify_reference.py` が 28 指標すべて再現しました。
`verify_reference.py` は同梱 Gram 行列を読むだけで `X @ X.T` を再計算しないため、版固定が守っている
箇所を通りません。したがって**コーパス無しの照合は 3.10 でも可**です。3.10 でできないのは、
コーパスから公開座標を再構築することです(それには 3.11.15 固定環境が必要)。

**Python 3.9 では実行しないでください。** 15 ゲートは通りますが、アンカー割り当ての決着が
変わり E1 がずれます(spearman 0.687 → 0.676)。

## 1. 入力の作成

結合 Markdown をパイプラインが読む形式に展開します。

```bash
python3 code/make_derived_input.py \
    --input ./your_corpus.md \
    --output inputs/derived/mycorpus
```

`documents: 100 / with title: 100 / with doi: 100 / with year: 100` のように、
文書数とメタデータの充足数が表示されれば成功です。出力は次の構成になります。

```
inputs/derived/mycorpus/
  freeze_manifest.json
  texts/KM0001.md ... KM0100.md          TF-IDF の入力そのもの
  normalized_records/KM0001.json ...     title / doi / year
```

`[END OF DOCUMENT: ...]` フッターは既定で本文に残します。新規に前処理をやり直す場合のみ
`--strip-footer` を付けてください(結果は僅かに変わります)。

文書の順序は `sorted(texts/*.md)` で決まり、`KM0001` が map index 0 になります。
`data/corpus_manifest.csv` の `map_index` 列と同じ規則です。

**出力先が既に存在する場合は先に削除してください。** 同期フォルダ(Dropbox 等)では上書きが
弾かれることがあり、古い内容が残ると照合ができません。削除後、
`ls inputs/derived/mycorpus/texts | wc -l` が期待値になることを確認してから次に進みます。

## 2. パイプライン実行

```bash
bash code/run_v50.sh \
    --derived-input "$PWD/inputs/derived/mycorpus" \
    --run-id mycorpus \
    --outputs-root "$PWD/outputs"
```

`run_v50.sh` は冒頭で `code/` へ移動するため、`--derived-input` と `--outputs-root` は
**絶対パス**で渡します(`$PWD/` を付ける)。相対パスだと `code/` からの相対として解決され、
入力が見つかりません。

所要時間の目安は単一コアで 12〜20 分。測地線 34 本が大半を占めます。

`--quick` を付けると **E7/E8(頑健性・感度解析)を省略**します。κ スイープ、シードジッター、
アンカー数、GPR カーネル、平滑化長、差分スケール、コーパスサイズ依存性を調べる部分で、
同じ最適化を何十回も回すため時間の大半を占めます。ただし省略すると `validate.py` の一部
ゲートが FAIL します(E8 の成果物を参照するため)。初回はフル実行を推奨します。

中間ファイル(Gram 行列・GPR モデル・語彙・経路)は `outputs/mycorpus/work/` に置かれます。
実行 ID ごとに分かれるので複数の設定を並行して回しても混ざりません。第 6 節の仮想論文生成で
必要になるため、この `work/` は消さずに残してください。

## 3. 検証

```bash
KM_OUT="$PWD/outputs/mycorpus" \
KM_DATA="$PWD/outputs/mycorpus/work" \
python3 code/validate.py
```

15 のゲートがすべて PASS すれば正常です。1 つでも FAIL があれば終了コードが非ゼロになり、
`run_v50.sh` もそこで停止します。

## 4. 公開値との照合

同梱の派生成果物に対しては、コーパスなしで照合できます。

```bash
python3 code/verify_reference.py
```

`gram_l2.npy` と `coords.npy` から E1 の全指標と E2 の `loo_cosine_*` /
`nearest_paper_*` を再計算し、`data/derived/` の参照 JSON と突き合わせます(28 指標)。
恒等式と再現可能な範囲は [`../data/README.md`](../data/README.md) に書いてあります。

参照値(v5.2、100 論文コーパス): Spearman 0.6868、kNN@7 0.4271、
LOO cosine 0.4663(global)/ 0.4708(adaptive)。

**kNN 保存率は初期稿から値が変わっています。** `knn_preservation` が k+1 近傍を k で割って
いた不具合を修正したためで、k=7 は 0.504 → 0.427(k=5 は 0.492 → 0.388、k=10 は
0.513 → 0.463)。手法間の順位は 5 手法すべて不変です。詳細と E10 比較表の全手法の新旧値は
[`../CORRECTIONS.md`](../CORRECTIONS.md) の #1。

## 5. 図の出力

既定では `outputs/mycorpus/figures/` に 9 枚が出ます。

| ファイル | 内容 |
|---|---|
| `figA_knowledge_map.png` | 知識マップ |
| `figE_gpr_uncertainty.png` | GPR 不確かさ場 |
| `figF_sph_entropy.png` | SPH エントロピー場 |
| `figG_path_comparison.png` | 経路比較 |
| `figH_lambda_sensitivity.png` | λ 感度 |
| `figI_path_profile.png` | 経路プロファイル(論文と同じ 2 パネル) |
| `figR1_map_thumbnails.png` | データサイズ別マップ |
| `figR6_evaluation.png` | LOO 分布と不確かさとの関係(論文と同じ 2 パネル) |
| `figE10_embeddings.png` | 埋め込み手法の比較(`15_e10_embeddings.py` が出力) |

`--fullfig` を付けると、論文では使っていない補助図も生成します。

| ファイル | 内容 |
|---|---|
| `figC_L2_vs_L1_schematic.png` | L2/L1 幾何の概念図 |
| `figD_metric_tensor.png` | 計量テンソル場の 4 面図 |
| `figR1_scaling.png` | 計算時間・メモリのスケーリング |

100 論文コーパスで生成した 9 枚は `data/reference_figures/` に同梱してあります。

3 経路比較・アンカー比較・スケール並置(論文 Fig. 4、SI Fig. S1・S2)は査読対応の過程で
個別に作図したもので、**このパイプラインには含まれていません**。同梱もしていません。

クラスタの命名だけは LLM を使った作業ですが、パイプラインの実行には不要です。決定論的に
得られるのは各クラスタの上位 15 特徴語までで、これは `cluster_naming_evidence.json` に
出力されます。

## 6. 仮想論文の生成

マップ上の任意の点について、その位置の内容を言語化させることができます。証拠(特徴語・出典・
不確かさ)はパイプラインが決定論的に計算し、言語モデルはそれを読み出して文章にするだけ、
という役割分担を守るのが要点です。

### 6.1 証拠パッケージを作る

```bash
python3 code/make_evidence.py --x -0.5 --y 0.0 \
    --out outputs/mycorpus \
    --data outputs/mycorpus/work \
    --code code
```

実行すると、その点の不確かさ $u$・混合エントロピー・両レンズの語が表示され、
`evidence_point_-0.5_0.0.json` が書き出されます。同梱の 100 論文コーパスの参照 run では
次のようになります。

```
query      : (-0.5, 0.0)
u          : 0.219
entropy H  : 0.614  (effective 16.9 documents)
L2 lens    : breakage, kinking, feed, thrust, drilling, push-out, cfrtp, weibull, hole
L1 lens    : feed, drilling, breakage, thrust, push-out, spindle, unprocessed
written    : evidence_point_-0.5_0.0.json
```

座標を変えれば内容も変わります。上の位置は穴あけ加工・工具送りの領域です。

`--topk` で語数(既定 15)、`--outfile` で出力先を指定できます。

読み出し層の平滑化長 h は論文 Sec.3.3 の二層方針で決まります。幾何(場・微分・計量・測地線)は
常に global を使い、**読み出し層だけ**が選択対象です。`kmlib.select_readout()` が
`e2_loo_global.json` と `e2_loo_knn_adaptive.json`(第 5 段が両方式で出力)の LOO cosine を比較し、
高い方を採用します。研究室コーパスでは adaptive(0.4708 対 0.4663)、E9 の 2 つのジャーナル
コーパス(Polymer / J. Informetrics)では global が選ばれます(論文 Table 5)。
`--readout global|adaptive` で固定でき、`13_llm_export.py` は選択結果を標準出力に出します。
両ファイルが無い場合は adaptive です。なお adaptive の h は「9 番目に近い文書までの距離」で、
コード上の `knn_k=8` は 0-origin の添字なので論文の定義と一致します。

パッケージ中の語は、char n-gram を出典テキスト中の語へ復元したものです。断片(`ompto`)では
なく語(`compton`)で渡るため、モデルが人名と誤認する取り違えが起きません。

**座標と内容の対応は run に固定されたものです。** 環境が変わると同じ座標が別領域を指す
不具合がありましたが修正済みです([`../CORRECTIONS.md`](../CORRECTIONS.md) #2)。ただし
修正後も自由点の座標には環境差が残る(最大 0.35)ため、座標を根拠にした議論をする場合は
`requirements.txt` の環境を固定してください。

### 6.2 言語モデルに渡す

新規セッションを開き、次の 2 ファイルを添付します。

- `evidence_point_-0.5_0.0.json`(証拠パッケージ)
- [`verbalization_protocol.md`](verbalization_protocol.md)(言語化規約。E7 で 6 モデルに
  渡したものと同一)

指示は次の一文で足ります。

> 添付の規約に従い、証拠パッケージが示す位置の仮想論文を書いてください。

規約には、主題は L2 レンズと寄与文書から構成すること、L1 レンズだけに現れる語は df に応じて
扱いを変えること(df≥4 は萌芽的話題として提示可、df≤3 は出典を引用したうえで単一源の仮説と
してのみ言及)、証拠にない事実を補わないこと、末尾に使用した語と df を列挙することが
定められています。

**規約を渡さずに証拠だけ渡すと、監査可能性が失われます。** モデルが一般知識で補った内容と
証拠に基づく内容の区別がつかなくなるためです。必ず 2 ファイルをセットで渡してください。

ファイルをマウントできるツール(エディタ統合や CLI エージェント等)では、生成後にその場で
語の使われ方を検算できます。証拠に無い語が混じっていないか、df≤3 の語に出典が付いているかを
続けて確認させられます。

## 補足

- スクリプト名の `v50` や `12_map_v30.py` の `v30` は開発初期に付いた名前で、中身は v5.2
  です(`KAPPA=0.80` などの v5.2 確定値)。過去の成果物との対応を保つため据え置いています。
  **設定値はコード内に直接書かれており、設定ファイルは同梱していません。**
- 論文の評価番号 (E1〜E9) とファイル名の内部番号は一致しません(E1 と E2 が入れ替わり、論文 E8 が
  内部の `e7_*`)。対応表は [`../README.md`](../README.md) の "Correspondence with the manuscript"、
  投稿バンドル側の構成は [`submission_bundle_README.md`](submission_bundle_README.md) にあります。
- 環境変数: `KM_DERIVED`(入力)、`KM_OUT`(成果物)、`KM_DATA`(中間、既定は `KM_OUT/work`)、
  `KM_OVERRIDES`(κ や λ の JSON 上書き)、`KM_SUBSET_N`(部分コーパス)。
  `run_v50.sh` が前 3 つを設定します。
