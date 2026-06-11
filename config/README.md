# Configuration

`vnm_ros` の設定ファイルと各パラメータの意味を説明します。
相対パスは基本的に `vnm_ros` パッケージのルートから解決されます。

## model.yaml

モデルの構造、重み、推論方法を設定します。学習済み重みとモデル構造の
設定が一致している必要があります。

| パラメータ | 意味 |
| --- | --- |
| `model_name` | ログ表示などに使うモデル名です。 |
| `checkpoint_path` | 読み込むモデル重みのパスです。通常は `weights/best.pth` を指定します。 |
| `device` | 実行デバイスです。`auto` はCUDAが利用可能ならGPU、それ以外はCPUを使用します。`cuda`、`cuda:0`、`cpu`も指定できます。 |
| `model_type` | 構築するモデル形式です。現在は `vint` と `gnm` に対応しています。 |
| `obs_encoder` | ViNTの画像エンコーダです。例: `efficientnet-b0`。 |
| `obs_encoding_size` | 画像特徴ベクトルの次元数です。 |
| `mha_num_attention_heads` | TransformerのMulti-Head Attentionのヘッド数です。 |
| `mha_num_attention_layers` | Transformer Encoderの層数です。 |
| `mha_ff_dim_factor` | Transformer内のFeed Forward層の拡大率です。 |
| `late_fusion` | 観測画像と目標画像を後段で融合するかを指定します。 |
| `context_type` | コンテキスト形式を表す設定値です。現在の実装では未使用です。 |
| `context_size` | 現在画像より前に使う画像枚数です。モデルへの入力は現在画像を含むため、合計は `context_size + 1` 枚です。 |
| `normalize` | `true` の場合、モデル出力WaypointのXYを実機用の距離へスケーリングします。 |
| `image_size` | モデル入力画像の `[幅, 高さ]` です。 |
| `len_traj_pred` | モデルが予測する将来Waypoint数です。 |
| `learn_angle` | `true` の場合、WaypointのXYに加えて向きのcos/sinも学習・出力します。 |
| `waypoint_index` | 予測されたWaypoint列のうち、制御に使用する番号です。0始まりです。 |

## robot.yaml

Waypointから速度指令を生成するときの設定です。

| パラメータ | 意味 |
| --- | --- |
| `model_rate` | モデル推論と速度指令生成の周期 `[Hz]` です。Waypointを速度へ変換するときの時間幅にも使います。 |
| `max_v` | 最大並進速度 `[m/s]` です。生成された `linear.x` をこの範囲に制限します。 |
| `max_w` | 最大角速度 `[rad/s]` です。生成された `angular.z` をこの範囲に制限します。 |
| `publish_cmd_vel` | `true` の場合、実際の速度指令を `cmd_vel_topic` へ送信します。 |
| `publish_waypoint` | `true` の場合、選択したWaypointを `waypoint_topic` へ送信します。 |

テスト時は `publish_cmd_vel: false`、実際にロボットを走行させる場合は
`publish_cmd_vel: true` に設定して `navigate.launch` を起動します。

## topics.yaml

購読・配信するROSトピック名を設定します。

| パラメータ | 意味 |
| --- | --- |
| `image_topic` | 推論、Topomap作成、Dataset作成に使うカメラ画像です。型は `sensor_msgs/Image` です。 |
| `odometry_topic` | Datasetへ位置とyawを保存するためのオドメトリです。型は `nav_msgs/Odometry` です。 |
| `amcl_pose_topic` | Datasetへ位置とyawを保存するためのAMCL自己位置です。型は `geometry_msgs/PoseWithCovarianceStamped` です。 |
| `waypoint_topic` | 選択したWaypointの配信先です。 |
| `cmd_vel_topic` | ロボットへ送る速度指令の配信先です。型は `geometry_msgs/Twist` です。 |
| `cmd_vel_debug_topic` | 実際の速度出力が無効でも配信される可視化用速度指令です。 |
| `reached_goal_topic` | 最終ノード到達状態を配信します。型は `std_msgs/Bool` です。 |
| `marker_topic` | RViz用Waypoint Markerの配信先です。 |
| `topomap_image_topic` | 現在選択されているTopomap画像の配信先です。 |
| `annotated_image_topic` | カメラ画像へ速度矢印などを重ねた画像の配信先です。 |
| `frame_id` | Waypoint Markerの基準フレームです。通常は `base_link` を指定します。 |

## topomap.yaml

Topomapの作成とTopomap上のナビゲーションを設定します。

| パラメータ | 意味 |
| --- | --- |
| `topomap_dir` | Topomap画像の保存・読込先です。例: `topomaps/topomap`。 |
| `bag_path` | Topomap作成に使うrosbagです。必須です。 |
| `goal_node` | ゴールとするTopomapノード番号です。`-1` は最後のノードを意味します。 |
| `search_radius` | 現在位置として保持しているノードの前後何ノードを照合対象にするかを指定します。大きいほど探索範囲と計算量が増えます。 |
| `close_threshold` | モデルが予測した目標画像までの距離がこの値以下なら、そのノードへ十分近いと判断して次のノードへ進めます。単位はモデルが学習したノード間隔です。 |
| `sample_dt` | Topomap画像を保存する時間間隔 `[s]` です。 |
| `overwrite` | `true` の場合、作成開始時に既存のTopomapディレクトリを削除して作り直します。 |

## train.yaml

Dataset作成、ViNT学習、評価を設定します。

### 共通設定

| パラメータ | 意味 |
| --- | --- |
| `seed` | Python、NumPy、PyTorchに設定する乱数シードです。 |
| `device` | 学習デバイスです。`auto`、`cuda`、`cuda:0`、`cpu`などを指定します。 |

### dataset

| パラメータ | 意味 |
| --- | --- |
| `train_data_dir` | 学習軌跡ディレクトリのパスです。直下に複数の `traj_*` を配置できます。 |
| `test_data_dir` | テスト軌跡ディレクトリのパスです。 |
| `image_size` | 学習時の入力画像サイズ `[幅, 高さ]` です。通常は `model.yaml` と合わせます。 |
| `metric_waypoint_spacing` | 連続する収録画像間の想定移動距離 `[m]` です。Waypointを正規化するときに使います。実際の収録条件に合わせる必要があります。 |
| `waypoint_spacing` | コンテキスト、行動、目標を何フレームおきに取り出すかを指定します。 |
| `context_size` | 現在画像より前に使う画像枚数です。入力画像の合計は `context_size + 1` 枚です。 |
| `len_traj_pred` | 正解データとして生成する将来Waypoint数です。 |
| `min_goal_distance` | 現在フレームから目標画像までの最小間隔です。単位は `waypoint_spacing` で間引いたノード数です。 |
| `max_goal_distance` | 現在フレームから目標画像までの最大間隔です。 |
| `min_action_distance` | Action lossを計算する目標距離の下限です。判定はこの値より大きい場合です。 |
| `max_action_distance` | Action lossを計算する目標距離の上限です。判定はこの値より小さい場合です。 |
| `normalize` | `true` の場合、正解WaypointのXYを `metric_waypoint_spacing * waypoint_spacing` で除算します。 |
| `learn_angle` | `true` の場合、正解Waypointへ向きのcos/sinを追加します。 |
| `negative_mining` | `true` の場合、学習データの約10%で無関係な目標画像を選び、到達距離推定を学習させます。テスト時は無効です。 |

### collection

Dataset作成はrosbag入力専用で、`bag_path` は必須です。

| パラメータ | 意味 |
| --- | --- |
| `dataset_type` | 作成するDatasetの種類です。`train` または `test` を指定します。 |
| `trajectory_name` | 保存する軌跡ディレクトリ名です。空文字の場合は日時から自動生成します。 |
| `pose_source` | Datasetの軌跡に使う姿勢情報です。`odometry` は `odometry_topic`、`amcl` は `amcl_pose_topic` を使用します。 |
| `bag_path` | Dataset作成に使用するrosbagパスです。必須です。 |
| `sample_dt` | Datasetへ画像と選択した姿勢情報を保存する時間間隔 `[s]` です。 |
| `image_format` | 保存画像の拡張子です。例: `jpg`、`png`。 |

### training

| パラメータ | 意味 |
| --- | --- |
| `use_test` | `true` の場合、各epochでtest Datasetを評価します。`best.pth` はtest lossが最小のモデルになります。`false` の場合はtrain lossで選びます。 |
| `tensorboard` | TensorBoardログを日時ごとの `runs/YYYYMMDD_HHMMSS_ffffff/tensorboard/` へ保存するかを指定します。 |
| `epochs` | 学習する総epoch数です。 |
| `batch_size` | 1回の更新で使用するサンプル数です。 |
| `num_workers` | PyTorch DataLoaderの並列読込プロセス数です。 |
| `learning_rate` | AdamW Optimizerの初期学習率です。 |
| `weight_decay` | AdamWのweight decay係数です。 |
| `alpha` | 距離lossとAction lossの重みです。総lossは `alpha * 0.01 * distance_loss + (1 - alpha) * action_loss` です。 |
| `gradient_clip` | 勾配ノルムの最大値です。0以下にするとクリッピングしません。 |
| `scheduler` | 学習率Schedulerです。`cosine` の場合にCosine Annealingを使用します。それ以外はSchedulerなしです。 |
| `resume` | 学習を再開するチェックポイントのパスです。空文字なら新規学習です。 |

学習時の重みは `weights/latest.pth` と `weights/best.pth` に保存されます。
