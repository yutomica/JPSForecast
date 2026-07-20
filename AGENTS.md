# JPSForecast エージェント運用規約

## 1. 目的と適用範囲

本ファイルは、JPSForecastリポジトリで作業するすべてのCodexエージェントに適用する共通規約である。
JPSForecastは、日本株個別銘柄を対象とする金融機械学習モデル構築・運用プロジェクトである。個別のdomain、予測期間、target、universe、model、CV、metric、portfolio構築法、ensemble方式は研究仮説であり、本ファイルでは固定しない。
プロジェクトの目的は、日々の銘柄選定において低リスクかつハイリターンの可能性が高い日本株個別銘柄を特定し、信用取引にて当該銘柄のロングポジションを取ることである。これに向けて、時点整合性、再現性、統計的妥当性、売買可能性、本番安全性を満たし、売買コスト・流動性・投資容量・riskを考慮した後も価値を持つ意思決定システムを構築することを目指す。

## 2. 規則の階層

プロジェクトの規則を次の3階層へ分離する。

### 2.1 恒久的な研究統制

本ファイルに記載する。modelや実験結果が変わっても維持する。

- point-in-time整合性
- 将来情報・survivorship biasの禁止
- fold外評価とOOS統制
- 多重試行の記録
- 再現性とlineage
- 学習・推論整合性
- 独立した検証と職務分離
- 実行可能性と経済価値の確認

### 2.2 version管理された実験プロトコル

同一実験系列の開始前に固定し、結果確認後に暗黙変更しない。変更する場合は新しいprotocol versionと研究記録を作る。

- 仮説と推定対象
- 観測・判断・注文・約定・labelの時刻
- 開発期間と確認期間
- Step 1～4の設計
- benchmark
- primary metricとsecondary metrics
- CV契約
- search spaceとtrial budget
- sampling・weighting方針
- 合格・棄却条件

### 2.3 実験ごとの可変設定

Hydra設定、registry、manifestなどで管理し、本ファイルやagent定義へハードコードしない。

- domain名と予測期間
- target、label、正例の定義
- universeと流動性条件
- model、loss、feature set
- CV方式とwindow
- purge・embargoの具体値
- metric名と複合weight
- feature selection手法と閾値
- ensemble方式
- portfolio構築法
- runtime、device、並列数

実験仕様と合成済み設定が矛盾する場合は、優先順位で黙って解決せず、実行を停止して差異を報告する。

## 3. 金融MLの恒久的な不変条件

### 3.1 時点系譜

すべてのdata、feature、target、universe flag、weight、predictionについて、次を追跡可能にする。

- 観測時刻
- 公表・配信・決算発表時刻
- 最初に利用可能となる時刻
- feature計算時刻
- signalおよび意思決定時刻
- 想定注文・約定時刻
- label開始・終了時刻
- split、fold、経路への所属

因果性に影響する時刻が不明な値は、確認できるまで使用しない。

### 3.2 禁止事項

- 将来の価格、財務情報、universe、統計量を過去時点へ混入すること。
- 現在の生存銘柄集合を過去へ遡及適用すること。
- 評価用情報を使って前処理、feature、target、weight、閾値、modelを選択すること。
- split前に、学習が必要なimputer、scaler、encoder、feature selector、calibratorをfitすること。
- target依存のsamplingやweightingを、検証・確認用データへ事後適用すること。
- sequenceやrolling処理がentity境界または未来方向を跨ぐこと。
- stackingその他の学習型ensembleを、学習内のbase predictionでfitすること。
- 最終確認データを見た後、同じ確認データに合わせて設計を変更すること。
- 試行して失敗したtarget、feature、model、metric、portfolioを研究記録から除外すること。

### 3.3 OOFとOOS

- 学習型ensembleを使う場合、meta学習には厳密なOOF predictionだけを使用する。
- OOF成果物は、観測key、base model、fold・path、学習期間、予測時点を追跡可能にする。
- final refitの学習内predictionをmeta学習へ使用しない。
- 最終OOSまたは将来のpaper-trading期間は、候補を凍結した後にだけ評価する。
- 確認用データが設計変更へ影響した場合、その期間は以後のconfirmatory OOSではない。新しい未使用期間を設定する。

### 3.4 多重試行

research ledgerに、成功・失敗を問わず次を記録する。

- 仮説とprotocol version
- target・universe・feature setのvariant
- model・loss・metric・portfolioのvariant
- CVと期間
- seedとtrial数
- 人手による結果確認後の判断
- OOSを閲覧した日時と目的

単一の最良runだけを根拠に採用しない。必要に応じて、data snooping、selection bias、Probability of Backtest Overfitting、Deflated Sharpe Ratio、Reality Check、Superior Predictive Abilityなどを検討する。ただし、前提を満たさない手法を形式的に適用しない。

## 4. 実験工程

Step 1～4の目的と情報境界は固定するが、使用するalgorithm、model、metric、閾値はprotocolと設定から取得する。

### 4.1 Step 0：事前設計

実験開始前に次を固定する。

- 検証する仮説
- 推定対象と最終意思決定
- data・feature・signal・trade・labelの時刻
- universe
- benchmark
- 開発期間、確認期間、CV契約
- primary metric、secondary metrics、metric方向
- trial budgetとseed方針
- 合格・棄却条件
- portfolio上の評価条件

### 4.2 Step 1：候補特徴量の評価

- 開発データだけを使用する。
- data quality、coverage、時点整合性、冗長性、fold・期間安定性を評価する。
- 特定のmodel、importance手法、相関閾値を恒久規則にしない。
- 採用・不採用とその理由を保存する。

### 4.3 Step 2：粗いアンカー探索

- 単一の最良点ではなく、安定したmodel容量・学習条件の領域を探す。
- 学習lossと評価指標を区別する。
- 探索境界への到達、失敗trial、収束、fold分散、資源失敗を記録する。
- 結果確認後に探索範囲を変更する場合は、変更理由と新しいprotocol versionを記録する。

### 4.4 Step 3：条件付き増分価値と冗長性の評価

- 個別featureまたはfeature群の増分価値、代替可能性、相互作用、安定性を評価する。
- permutation、grouped permutation、conditional permutation、drop-column、ablationなどから、対象に適した方法をprotocolで選ぶ。
- 単一importance値または単一閾値だけで採否を決めない。
- 選択方法自体による不確実性とselection biasを記録する。

### 4.5 Step 4：凍結済み設計の最終HPO

- feature、target、universe、CV、objective、search space、trial budgetを開始前に固定する。
- failure・pruneを含む全trialを保存する。
- 境界到達、parameter識別性、seed感応度、fold・期間安定性を確認する。
- 同じ評価期間への反復適合を、真正OOS改善として扱わない。

### 4.6 確認評価

- 凍結済みcandidateを、未使用OOSまたはprospective paper tradingで評価する。
- predictive metricとeconomic metricを分離して報告する。
- benchmarkとの差、uncertainty、turnover、cost、liquidity、capacity、concentration、tail riskを確認する。

## 5. CV契約

特定のCV方式を常に正解としない。実験ごとに次を明示する。

- 予測・評価したいdeployment上の推定対象
- split単位とgroup単位
- train windowとvalidation window
- label information interval
- feature lookbackとavailability
- purgeの数理的根拠
- embargoを使う場合の目的と定義
- expanding・rolling・blockedなどのwindow方針
- OOF predictionの意味と一意性
- final refit方針
- 非定常性とregime changeに関する仮定

purgeは固定行数ではなく、学習labelの情報区間とvalidation情報集合の重複から定義する。embargoを慣習だけで追加しない。使用するsplitterは、合成日付・label intervalによる境界testへ合格させる。

評価用trainとvalidationは非重複とする。hyperparameterと学習手順を固定した後のfinal refitでは、過去のvalidation期間を学習へ再利用できるが、そのmodelの学習内scoreを評価値として使わない。

## 6. metric、sampling、weighting

### 6.1 metric

- primary metricは、推定対象と意思決定効用から導く。
- predictive metricとportfolio metricを区別する。
- 不均衡分類をaccuracyまたはROC-AUCだけで判断しない。
- prevalence依存metricを異なる期間・foldで比較する場合は、正例率を併記する。
- worst-foldは高分散になり得るため、単独objectiveへ固定しない。中央値、下位quantile、信頼区間下限、CVaR、paired benchmark差などを検討する。
- 重複labelのuncertaintyを独立同分布として推定しない。必要に応じてblock bootstrapやHACなどを使用する。
- metric名、合成weight、閾値は実験設定から取得する。

### 6.2 samplingとweighting

- target依存samplingは学習データだけで行う。
- validation・test・OOSを学習時のclass比率へ合わせて再samplingしない。
- 評価weightは、事前定義したestimandに必要で、未来labelに依存しない場合だけ使用する。
- weightの生成時刻、式、正規化、shape、index、dtype、有限性、有効標本数を検証する。
- pointwiseな正のsample weightでは、原則として加重lossをweight合計で正規化する。ただしpairwise・listwise・importance sampling・model内部weightへ機械的に一般化しない。
- 非有限loss・prediction・weightを黙ってskipまたはclipしない。原因と処理方針を記録する。

## 7. 経済性とportfolio検証

予測精度だけで採用しない。strategyまたは意思決定ruleへ変換した上で、少なくとも該当する次を評価する。

- signalからpositionへの写像
- execution lagと利用可能な約定価格
- turnover、手数料、slippage、market impact
- liquidity、ADV参加率、position limit、capacity
- sector、style、size、betaその他のexposure
- concentration、leverage、drawdown、tail loss
- score calibrationとposition sizing
- benchmark portfolioに対する増分価値
- 運用制約下での安定性

portfolio設計も研究仮説であり、特定のTop-N、long-short、leverage、risk overlayを恒久規則にしない。

## 8. 設定・実装・再現性

### 8.1 設定

- Hydraの階層設定とOmegaConfをプロジェクト標準とする。
- Structured Configまたはdataclassによる型付き設定を優先する。
- 結果を変える重要設定にsilent fallbackを設けない。
- 起動時にtarget、loss、metric方向、CV、label intervalなどの設定間制約を検証する。
- 実行ごとに合成・解決済み設定を保存する。

### 8.2 実装

- 承認済み仕様を満たす最小で一貫した変更を行い、無関係な変更を保持する。
- 構造的欠陥がある場合は、最小差分を絶対視せず、ADR、移行plan、回帰testを伴うrefactorを提案できる。
- preprocessing、model、predictionのshape、dtype、index、entity、time順序を検証する。
- fold内fit、save/load、train/serve parityを保証する。
- entityを扱う時系列modelでは、entity境界と時間方向を検証する。
- 認証情報や機種固有の絶対pathをrepositoryへ埋め込まない。

#### 8.2.1 変更スコープ統制

- 編集前に、依頼から必須変更、維持すべき既存挙動、完了条件を特定する。解釈によって結果、public interface、data contract、artifactが変わる不明点は、編集前にrootへ返す。
- 依頼達成に必要と確認できたものを除き、処理順序、data選択・filter、default値、入出力schema、例外、side effect、設定解決、数値式を変更しない。
- 既存コードに別の不具合、重複、非効率、構造的欠陥を見つけても、依頼達成に不可欠でなければ変更へ混在させず、別の指摘または提案として返す。提案は実装承認を意味しない。
- 明示要件を満たす最も単純な局所変更を優先し、依頼されていない将来拡張、一般化、抽象化、互換layer、fallback、option、helper、loggingを追加しない。
- 複数案の比較は、選択によって結果、interface、risk、保守費用が実質的に変わる場合、またはrootから求められた場合に限定する。

### 8.3 runtime

device、precision、worker数、thread数、memory手法、分散実行はruntime profileから取得する。特定hardwareの値をagent定義へ固定しない。

- correctnessとnumerical stabilityを性能より優先する。
- 最適化前後で数値同等性とsplit・index・metricの不変性を検証する。
- agent、HPO、model、BLAS、DataLoaderのnested parallelismを資源予算なしに乗算しない。

## 9. lineageとmodel governance

実行および昇格に必要な成果物を追跡可能にする。

- code revisionと未commit差分
- data snapshot・schema・availability
- target・universe・feature・protocol version
- 合成済みconfig
- train・validation・OOS期間
- CVとOOF provenance
- seedと全trial
- preprocessing stateとmodel state
- prediction schema
- metricとportfolio結果
- 承認・棄却判断

model昇格、deployment、外部通知、schedule変更、データ削除は、ユーザーの明示承認なしに実行しない。

本番では、data freshness、schema、feature distribution、score distribution、coverage、realized performance、portfolio risk、drift、model期限を監視する。状態名、registry方式、再学習頻度、alert閾値はproduction policyから取得する。

## 10. エージェントの役割分担

root agentが、task分割、protocol凍結、最終判断、ユーザーへの報告を担当する。subagentは証拠を返し、担当外の承認を行わない。

| 役割 | エージェント | 主な権限 |
| --- | --- | --- |
| 研究仮説・estimand・protocol設計 | `research_design_quant` | read-only、提案者 |
| data provenance・point-in-time・品質監査 | `data_provenance_auditor` | read-only、独立監査 |
| model・pipeline実装 | `modeling_systems_engineer` | workspace-write、主要writer |
| software test・再現性検証 | `software_verification_engineer` | test限定write、独立検証 |
| 統計的妥当性・multiple testing・OOS判定 | `independent_statistical_validator` | read-only、独立承認 |
| portfolio・execution・経済価値評価 | `portfolio_execution_validator` | read-only、独立評価 |
| model risk・本番昇格・monitoring | `model_risk_governor` | read-only、独立gate |
| 性能・資源・数値安定性分析 | `performance_engineer` | read-only、必要時のみ |
| 根拠文書・model card・ADR更新 | `evidence_documentation_curator` | 文書限定write、必要時のみ |

### 10.1 職務分離

- 仮説提案者と最終統計評価者を分ける。
- 実装者とsoftware verifierを分ける。
- 統計的有効性と経済的価値を別々に評価する。
- data provenance監査をmodel実装から独立させる。
- production昇格を研究者または実装者だけで承認しない。
- read-only agentは、承認と実装修正を同時に行わない。
- final OOSをcandidate凍結前に評価へ使用しない。
- 同一worktreeの主要writerは原則1名とし、重複編集を直列化する。

### 10.2 標準workflow

新しい研究仮説では、次の順序を基本とする。

1. `research_design_quant`が仮説とprotocol案を作る。
2. `data_provenance_auditor`がdata実現可能性を監査する。
3. `portfolio_execution_validator`が投資上の意味を事前審査する。
4. rootがprotocolを凍結する。
5. `modeling_systems_engineer`が実装する。
6. `software_verification_engineer`がコードとtestを検証する。
7. `independent_statistical_validator`が結果を独立評価する。
8. rootがcandidate採否を判断する。
9. `model_risk_governor`が本番昇格を審査する。

結果評価では、研究設計との整合性、統計的妥当性、経済的価値を別々に報告させ、rootが統合する。

### 10.3 限定的な保守変更

研究仮説、protocol、data contract、modelの意味を変更しない局所的なscript修正やbug fixでは、次の限定workflowを使用できる。

1. rootが、task開始時のbase revisionとstaged・unstaged・untracked差分を記録し、必須変更、維持すべき既存挙動、完了条件、編集対象を固定する。
2. `modeling_systems_engineer`が、固定されたscope内で最小の局所変更を実装する。
3. `software_verification_engineer`が、依頼、編集前の挙動、diff、testを対応付けてscopeと回帰を独立検証する。
4. 本番昇格の対象となる場合は、`model_risk_governor`がscope gateを含む独立判定と検証済みrevisionを確認する。

この限定workflowは、未承認の挙動変更、refactor、研究判断、独立評価の省略を許可しない。変更が研究結果または本番判断へ影響する場合は、該当する標準workflowの役割へ戻す。

## 11. 共通報告形式

各subagentは次を返す。

1. **判定：** 合格、条件付き合格、不合格、未評価。
2. **対象範囲：** file、config、run、期間、artifact。
3. **前提：** 結論に影響するものだけ。
4. **指摘事項：** P0～P3の重大度順。
5. **根拠：** file・line、command、run ID、metric、再現例。
6. **必須対応と任意改善。**
7. **実行した確認と結果。**
8. **残余リスクと未解決事項。**

重大度は次のように定義する。

- P0：将来情報、OOS汚染、data corruptionなど、研究結果を無効にする問題。
- P1：target・split・metric・実装・本番安全性に関する重大な問題。
- P2：頑健性、再現性、保守性、coverageの不足。
- P3：軽微な文書・可読性・記録不足。

実行していないtestを成功と書かない。証拠がない場合は未評価とし、合格を推定しない。

## 12. 完了条件

該当する次の条件を満たした場合だけtask完了とする。

- 研究仮説と実装変更の境界が明確である。
- data・feature・signal・trade・labelの時刻が明確である。
- protocolと合成済み設定が一致する。
- 未解決のP0またはP1がない。
- 必要なtestと独立評価が完了している。
- OOF、OOS、sampling、preprocessing、ensembleの情報境界が守られている。
- 全試行と成果物が追跡可能である。
- predictive valueとeconomic valueを混同していない。
- 本番変更に必要な承認、monitoring、rollbackがある。
- 無関係なユーザー変更を保持している。
- 明示承認なしにcommit、push、昇格、deployment、削除を行っていない。
- 必須変更、維持すべき既存挙動、完了条件が明確である。
- 各production変更hunkが、依頼要件またはその達成に直接必要な補助変更へ対応し、各test変更hunkが確認対象の要件または回帰riskへ対応しており、未承認の変更を含まない。
- 変更箇所に隣接して回帰riskがある既存挙動が、既存testまたは必要最小限のcharacterization testで確認されている。

## 13. プロジェクトフォルダ構成

本節は、現状のリポジトリ構成を説明するための運用ガイドである。実験protocol、target、model、metric、portfolio設計を固定するものではない。設定値の真実はHydra設定、registry、manifest、run artifactに置き、本節へ実験固有の値をハードコードしない。

### 13.1 主要な管理対象ファイル

- `AGENTS.md`：本リポジトリで作業するagent向けの恒久的な運用規約。
- `pyproject.toml`、`uv.lock`：Python実行環境と依存関係の定義。
- `train.py`：Hydra設定を入口とする学習・CV・HPO・artifact保存の主要entrypoint。
- `predict.py`：日次予測用のentrypoint。外部data取得、feature作成、model読込、prediction出力を扱う。
- `.gitignore`：MLflow、Hydra、data、report、local envなどの生成物・ローカル成果物の除外規則。
- `.codex/agents/`：本規約の役割分担に対応するCodex subagent定義。

### 13.2 source code

- `src/data_loader/`：市場data、財務data、universe関連dataの取得・読込処理。
- `src/features/`：feature engineeringの実装。時系列方向、entity境界、利用可能時刻を壊さないこと。
- `src/preprocess/`：model別の前処理、sampling、weighting、matrix化処理。fitが必要な処理は必ずfold内で扱う。
- `src/cv/`：purged KFold、CPCV、anchored walk-forward、RRVなどのsplitterとCV補助処理。
- `src/models/`：model wrapper、学習pipeline、custom loss・metric、pruning補助。
- `src/models/networks/`：TCN、N-BEATS、GANDALF、FT-Transformerなどのnetwork定義。
- `src/evaluation/`：predictive metric、optimization objective、PBOなどの評価処理。
- `src/utils/`：設定、MLflow、feature選択、training artifact、production training、stacking、weightなどの共通補助。

### 13.3 Hydra config

- `config/main.yaml`：Hydra defaultsと共通設定の入口。
- `config/data/`：使用するdata snapshotまたはdata profileの設定。
- `config/domain/`：domainと予測対象期間に関する設定。
- `config/target/`：target、label、metric方向などの設定。
- `config/features/`：feature set、rough/fixed/initなどのfeature list設定。
- `config/model/`：model familyとmodel wrapperの設定。
- `config/hparams/`：model別のhyperparameter設定。`anchor/`は粗い探索や固定候補の基準設定を置く。
- `config/cv/`：CV方式、window、purge、embargoなどのsplit契約設定。
- `config/period/`：開発期間、検証期間、確認期間などの期間設定。
- `config/experiment/`：実験単位の合成設定。protocolと矛盾する場合は実行を停止して差異を報告する。
- `config/sweep/`：HPO search space設定。`generated/`は探索結果から生成したrefined設定を置く。
- `config/promotion/`：候補選定、production training、model昇格候補に関する設定。

### 13.4 scriptsとnotebooks

- `scripts/data_prep/`：raw dataの標準化、master data作成、target追加、feature削除、stacking data作成などのdata preparation。
- `scripts/feature_select/`：feature screeningやrough selectionの補助script。
- `scripts/analysis/`：MLflow・Optuna結果抽出、holdout評価、相関確認、backtest、scoringなどの分析script。
- `scripts/pipeline/`：Step 0～5、data pipeline、daily prediction、promotion、production model作成などを連結する運用script。
- `notebooks/`：探索的分析、data確認、backtest確認用。notebook上の結果を採用判断に使う場合は、protocol、run、artifactへ根拠を移す。

### 13.5 data、artifact、生成物

次のdirectoryやfileは、原則としてローカルdata、実行結果、cache、検証成果物であり、source codeと同じ扱いにしない。削除、上書き、昇格、外部通知は明示承認なしに行わない。

- `data/`：raw、intermediate、master、master_select、sample、stacking_dir、backtest、temp_scodeなどのdata snapshotと中間成果物。
- `mlruns/`、`mlflow.db`：MLflow tracking dataとartifact。
- `outputs/`、`multirun/`：Hydra単発実行・multi-runの出力。
- `optuna.db`：Optuna studyのローカルDB。
- `reports/`：candidate selection、HPOなどのreport成果物。
- `predictions/`：日次推論結果。
- `logs/`、`log/`：実行log、profiling結果、runtime診断。
- `sandbox/`：一時検証や試作用の作業領域。恒久成果物は適切なsource、config、reportへ移す。
- `.venv/`、`.pytest_cache/`、`__pycache__/`：ローカル実行環境とcache。

### 13.6 配置原則

- 再利用する業務logicは`src/`へ置き、one-off orchestrationは`scripts/`へ置く。
- 実験ごとの差分は`config/`で表現し、Python sourceへtarget名、期間、metric weight、trial budgetを固定しない。
- dataの作成・更新処理は`data/`配下の実体だけでなく、生成script、入力snapshot、設定、実行時刻を追跡可能にする。
- reportやnotebookで得た判断は、採用・棄却理由、run ID、metric、period、artifact pathを研究記録へ移してから根拠として扱う。
- credentials、token、machine固有の絶対path、個人環境依存の設定をrepositoryへ追加しない。
