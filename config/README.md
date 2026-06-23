# Configuration

`vnm_ros` の設定ファイルと各パラメータの意味を説明します。
相対パスは基本的に `vnm_ros` パッケージのルートから解決されます。

## model.yaml

モデルの構造、重み、推論方法を設定します。`model_type` で使うモデルを
選び、`common` と選択したモデル専用セクションを結合して読み込みます。

### top level

| パラメータ | 意味 |
| --- | --- |
| `checkpoint_path` | 読み込むモデル重みの共通デフォルトです。モデル専用セクションの値が優先されます。 |
| `device` | 実行デバイスです。`auto` はCUDAが利用可能ならGPU、それ以外はCPUを使用します。 |
| `model_type` | 構築するモデル形式です。`vint` または `nomad` を指定します。 |

### common

ViNTとNoMaDで共通する設定です。

| パラメータ | 意味 |
| --- | --- |
| `obs_encoder` | 画像エンコーダです。 |
| `mha_num_attention_heads` | TransformerのMulti-Head Attentionのヘッド数です。 |
| `mha_num_attention_layers` | Transformer Encoderの層数です。 |
| `mha_ff_dim_factor` | Transformer内のFeed Forward層の拡大率です。 |
| `context_type` | コンテキスト形式を表す設定値です。現在の実装では未使用です。 |
| `normalize` | `true` の場合、ViNTのモデル出力WaypointのXYを実機用の距離へスケーリングします。 |
| `waypoint_index` | 予測されたWaypoint列のうち、制御に使用する番号です。0始まりです。 |

### vint

ViNT専用、またはViNT checkpointに合わせる設定です。

| パラメータ | 意味 |
| --- | --- |
| `checkpoint_path` | ViNTで読み込むモデル重みのパスです。 |
| `obs_encoding_size` | ViNTの画像特徴ベクトルの次元数です。 |
| `late_fusion` | 観測画像と目標画像を後段で融合するかを指定します。 |
| `context_size` | 現在画像より前に使う画像枚数です。 |
| `image_size` | モデル入力画像の `[幅, 高さ]` です。 |
| `len_traj_pred` | モデルが予測する将来Waypoint数です。 |
| `learn_angle` | `true` の場合、WaypointのXYに加えて向きのcos/sinも学習・出力します。 |

### nomad

NoMaD専用、またはNoMaD checkpointに合わせる設定です。

| パラメータ | 意味 |
| --- | --- |
| `checkpoint_path` | NoMaDで読み込むモデル重みのパスです。 |
| `encoding_size` | NoMaDの条件ベクトル次元数です。 |
| `context_size` | 現在画像より前に使う画像枚数です。 |
| `image_size` | モデル入力画像の `[幅, 高さ]` です。 |
| `len_traj_pred` | モデルが予測する将来Waypoint数です。 |
| `learn_angle` | NoMaDでは通常 `false` です。 |
| `down_dims` | NoMaD diffusion U-Netの各段の次元数です。 |
| `cond_predict_scale` | NoMaD diffusion U-Netで条件付きscale予測を使うかを指定します。 |
| `num_diffusion_iters` | NoMaD推論時の逆拡散ステップ数です。 |
| `num_action_samples` | NoMaDでゴール候補ごとにサンプルするAction数です。 |
| `action_sample_strategy` | 複数Actionサンプルの選び方です。`first` または `mean` を指定します。 |
| `action_stats` | NoMaDの正規化済みActionを実Actionへ戻すためのmin/maxです。 |

NoMaDを使う場合は `model_type: nomad`、NoMaD用checkpoint、`diffusers`、
`diffusion_policy` とその依存パッケージが必要です。現在の `scripts/train.py`
はViNT用の教師あり学習ループで、NoMaDのdiffusion学習には対応していません。

## topics.yaml

購読・配信するROSトピック名を設定します。

| パラメータ | 意味 |
| --- | --- |
| `image_topic` | 推論、Topomap作成、Dataset作成に使うカメラ画像です。 |
| `odometry_topic` | Datasetへ位置とyawを保存するためのオドメトリです。 |
| `amcl_pose_topic` | Datasetへ位置とyawを保存するためのAMCL自己位置です。 |
| `waypoint_topic` | 選択したWaypointの配信先です。 |
| `cmd_vel_topic` | ロボットへ送る速度指令の配信先です。 |
| `cmd_vel_debug_topic` | 実際の速度出力が無効でも配信される可視化用速度指令です。 |
| `reached_goal_topic` | 最終ノード到達状態を配信します。 |
| `marker_topic` | RViz用Waypoint Markerの配信先です。 |
| `topomap_image_topic` | 現在選択されているTopomap画像の配信先です。 |
| `annotated_image_topic` | カメラ画像へ速度矢印などを重ねた画像の配信先です。 |
| `frame_id` | Waypoint Markerの基準フレームです。 |

## runtime.yaml

実機実行、Topomap保存先、可視化などの実行時設定をまとめます。

### paths

| パラメータ | 意味 |
| --- | --- |
| `paths.topomap.topomap_dir` | Topomap画像の保存・読込先です。 |

### robot

| パラメータ | 意味 |
| --- | --- |
| `model_rate` | モデル推論と速度指令生成の周期 `[Hz]` です。 |
| `max_v` | 最大並進速度 `[m/s]` です。 |
| `max_w` | 最大角速度 `[rad/s]` です。 |
| `publish_cmd_vel` | `true` の場合、実際の速度指令を `cmd_vel_topic` へ送信します。 |
| `publish_waypoint` | `true` の場合、選択したWaypointを `waypoint_topic` へ送信します。 |

### topomap

| パラメータ | 意味 |
| --- | --- |
| `goal_node` | ゴールとするTopomapノード番号です。`-1` は最後のノードを意味します。 |
| `search_radius` | 現在位置として保持しているノードの前後何ノードを照合対象にするかを指定します。 |
| `close_threshold` | モデルが予測した目標画像までの距離がこの値以下なら、次のノードへ進めます。 |
| `sample_dt` | Topomap画像を保存する時間間隔 `[s]` です。 |
| `overwrite` | `true` の場合、既存のTopomapディレクトリを削除して作り直します。 |

### visualization

| パラメータ | 意味 |
| --- | --- |
| `dataset.dataset_type` | 再生するDatasetの種類です。`train` または `test` を指定します。 |
| `dataset.trajectory_name` | 再生する軌跡ディレクトリ名です。空文字の場合は対象Dataset内の最新ディレクトリを使用します。 |
| `dataset.frame_id` | RVizへ出すPathとPoseの基準フレームです。 |
| `dataset.rate` | Dataset画像とPoseの再生周期 `[Hz]` です。 |
| `dataset.loop` | `true` の場合、最後まで再生したあと先頭へ戻ります。 |
| `overlay.rate` | カメラ画像へ速度矢印とSubgoal画像を重ねる周期 `[Hz]` です。 |

## training.yaml

Dataset作成、ViNT学習、評価をまとめます。

### common

| パラメータ | 意味 |
| --- | --- |
| `seed` | Python、NumPy、PyTorchに設定する乱数シードです。 |
| `device` | 学習デバイスです。`auto`、`cuda`、`cuda:0`、`cpu`などを指定します。 |

### paths

| パラメータ | 意味 |
| --- | --- |
| `paths.rosbag.path` | Dataset作成とTopomap作成で使うrosbagパスです。 |
| `paths.dataset.train_data_dir` | 学習軌跡ディレクトリのパスです。 |
| `paths.dataset.test_data_dir` | テスト軌跡ディレクトリのパスです。 |

### dataset

| パラメータ | 意味 |
| --- | --- |
| `image_size` | 学習時の入力画像サイズ `[幅, 高さ]` です。 |
| `metric_waypoint_spacing` | 連続する収録画像間の想定移動距離 `[m]` です。 |
| `waypoint_spacing` | コンテキスト、行動、目標を何フレームおきに取り出すかを指定します。 |
| `context_size` | 現在画像より前に使う画像枚数です。 |
| `len_traj_pred` | 正解データとして生成する将来Waypoint数です。 |
| `min_goal_distance` | 現在フレームから目標画像までの最小間隔です。 |
| `max_goal_distance` | 現在フレームから目標画像までの最大間隔です。 |
| `min_action_distance` | Action lossを計算する目標距離の下限です。 |
| `max_action_distance` | Action lossを計算する目標距離の上限です。 |
| `normalize` | `true` の場合、正解WaypointのXYを `metric_waypoint_spacing * waypoint_spacing` で除算します。 |
| `learn_angle` | `true` の場合、正解Waypointへ向きのcos/sinを追加します。 |
| `negative_mining` | `true` の場合、学習データの約10%で無関係な目標画像を選びます。 |

### collection

| パラメータ | 意味 |
| --- | --- |
| `dataset_type` | 作成するDatasetの種類です。`train` または `test` を指定します。 |
| `trajectory_name` | 保存する軌跡ディレクトリ名です。空文字の場合は日時から自動生成します。 |
| `pose_source` | Datasetの軌跡に使う姿勢情報です。`odometry` または `amcl` を指定します。 |
| `sample_dt` | Datasetへ画像と選択した姿勢情報を保存する時間間隔 `[s]` です。 |
| `image_format` | 保存画像の拡張子です。例: `jpg`、`png`。 |

### training

| パラメータ | 意味 |
| --- | --- |
| `pretrained_checkpoint` | 新規学習時に初期重みとして読み込むチェックポイントです。 |
| `freeze_encoder` | `true` の場合、画像Encoderを固定して学習します。 |
| `use_test` | `true` の場合、各epochでtest Datasetを評価します。 |
| `tensorboard` | TensorBoardログを保存するかを指定します。 |
| `epochs` | 学習する総epoch数です。 |
| `batch_size` | 1回の更新で使用するサンプル数です。 |
| `num_workers` | PyTorch DataLoaderの並列読込プロセス数です。 |
| `learning_rate` | AdamW Optimizerの初期学習率です。 |
| `weight_decay` | AdamWのweight decay係数です。 |
| `alpha` | 距離lossとAction lossの重みです。 |
| `gradient_clip` | 勾配ノルムの最大値です。0以下にするとクリッピングしません。 |
| `scheduler` | 学習率Schedulerです。`cosine` の場合にCosine Annealingを使用します。 |
| `resume` | 学習を再開するチェックポイントのパスです。空文字なら新規学習です。 |
