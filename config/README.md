# Configuration

`vnm_ros` の設定ファイルと各パラメータの意味を説明します。
相対パスは基本的に `vnm_ros` パッケージのルートから解決されます。

## vint.yaml / nomad.yaml

モデルごとの構造、重み、Dataset作成、学習設定をまとめます。
どれを使うかは `runtime.yaml` の `model_type` で選びます。
`model_type: gnm` なら `gnm.yaml`、`model_type: vint` なら `vint.yaml`、
`model_type: nomad` なら `nomad.yaml` を読み込みます。

### top level

| パラメータ | 意味 |
| --- | --- |
| `seed` | Python、NumPy、PyTorchに設定する乱数シードです。 |
| `device` | 学習デバイスです。`auto` はCUDAが利用可能ならGPU、それ以外はCPUを使用します。 |

### paths

| パラメータ | 意味 |
| --- | --- |
| `paths.rosbag.path` | Dataset作成とTopomap作成で使うrosbagパスです。 |
| `paths.dataset.train_data_dir` | 学習軌跡ディレクトリのパスです。 |
| `paths.dataset.test_data_dir` | テスト軌跡ディレクトリのパスです。 |

### model

モデル構造、推論用checkpoint、推論時のAction選択方法を設定します。

| パラメータ | 意味 |
| --- | --- |
| `model_type` | この設定ファイルのモデル形式です。`gnm.yaml` は `gnm`、`vint.yaml` は `vint`、`nomad.yaml` は `nomad` です。 |
| `device` | 推論デバイスです。`auto` はCUDAが利用可能ならGPU、それ以外はCPUを使用します。 |
| `obs_encoder` | 画像エンコーダです。 |
| `mha_num_attention_heads` | TransformerのMulti-Head Attentionのヘッド数です。 |
| `mha_num_attention_layers` | Transformer Encoderの層数です。 |
| `mha_ff_dim_factor` | Transformer内のFeed Forward層の拡大率です。 |
| `context_type` | コンテキスト形式を表す設定値です。現在の実装では未使用です。 |
| `normalize` | `true` の場合、モデル出力WaypointのXYを実機用の距離へスケーリングします。 |
| `waypoint_index` | 予測されたWaypoint列のうち、制御に使用する番号です。0始まりです。 |
| `checkpoint_path` | 推論で読み込むモデル重みのパスです。 |

### gnm.yaml の model

GNM専用、またはGNM checkpointに合わせる設定です。公式GNM large checkpointは
`weights/gnm.pth` に置きます。GNMはViNTと同じく距離とWaypoint列を出力するため、
topomap navigationで使用できます。`direction_conditioning: true` の場合は、
goal画像encodingの代わりに `cmd_dir` から作った方向encodingを入力し、
goal画像なしのexploreで使用します。この方向encoderはGNM checkpointには含まれないため、
checkpoint読み込み時はGNM本体だけを事前学習済み重みから初期化し、方向encoderは新規初期化されます。

| パラメータ | 意味 |
| --- | --- |
| `obs_encoding_size` | GNMの観測画像特徴ベクトルの次元数です。公式GNM largeは `1024` です。 |
| `goal_encoding_size` | GNMの目標画像特徴ベクトルの次元数です。公式GNM largeは `1024` です。 |
| `context_size` | 現在画像より前に使う画像枚数です。公式GNM largeは `5` です。 |
| `image_size` | モデル入力画像の `[幅, 高さ]` です。公式設定は `[85, 64]` です。 |
| `len_traj_pred` | モデルが予測する将来Waypoint数です。公式GNM largeは `5` です。 |
| `learn_angle` | `true` の場合、WaypointのXYに加えて向きのcos/sinも出力します。 |
| `direction_conditioning` | `true` の場合、goal画像ではなく `cmd_dir` 方向encodingを入力します。 |
| `direction_num_commands` | 方向コマンド数です。通常はstraight/left/rightの3です。 |
| `direction_latent_dim` | コマンドごとの学習可能latent `z_i` の次元数です。 |
| `direction_hidden_dim` | `z_i` から方向encodingを作るMLPの隠れ層次元数です。 |

### vint.yaml の model

ViNT専用、またはViNT checkpointに合わせる設定です。

| パラメータ | 意味 |
| --- | --- |
| `obs_encoding_size` | ViNTの画像特徴ベクトルの次元数です。 |
| `late_fusion` | 観測画像と目標画像を後段で融合するかを指定します。 |
| `context_size` | 現在画像より前に使う画像枚数です。 |
| `image_size` | モデル入力画像の `[幅, 高さ]` です。 |
| `len_traj_pred` | モデルが予測する将来Waypoint数です。 |
| `learn_angle` | `true` の場合、WaypointのXYに加えて向きのcos/sinも学習・出力します。 |
| `direction_conditioning` | `true` の場合、`cmd_dir` のラベルindexから学習可能なlatent `z_i` を選び、MLPでTransformer入力用の方向tokenへ変換します。exploreではgoal画像tokenの代わりに `obs tokens + direction token` を使います。 |
| `direction_num_commands` | 方向コマンド数です。通常はstraight/left/rightの3です。 |
| `direction_latent_dim` | コマンドごとの学習可能latent `z_i` の次元数です。 |
| `direction_hidden_dim` | `z_i` から方向tokenを作るMLPの隠れ層次元数です。 |

### nomad.yaml の model

NoMaD専用、またはNoMaD checkpointに合わせる設定です。

| パラメータ | 意味 |
| --- | --- |
| `encoding_size` | NoMaDの条件ベクトル次元数です。 |
| `context_size` | 現在画像より前に使う画像枚数です。 |
| `image_size` | モデル入力画像の `[幅, 高さ]` です。 |
| `len_traj_pred` | モデルが予測する将来Waypoint数です。 |
| `normalize` | NoMaD公式checkpointと同じactionスケールを使う場合は `true` にします。学習時はDatasetのstep距離で正規化し、推論時は正規化解除後のWaypointを `max_v / model_rate` 倍してメートルへ戻します。 |
| `learn_angle` | NoMaDでは通常 `false` です。 |
| `down_dims` | NoMaD diffusion U-Netの各段の次元数です。 |
| `cond_predict_scale` | NoMaD diffusion U-Netで条件付きscale予測を使うかを指定します。 |
| `direction_conditioning` | `true` の場合、`cmd_dir` のラベルindexから学習可能なlatent `z_i` を選び、MLPで方向条件ベクトルへ変換します。 |
| `direction_conditioning_mode` | `residual`では公式NoMaDのgoal-masked条件ベクトルへ方向条件を加算します。`token`は方向条件をgoal token位置へ入れる旧方式です。両方式のcheckpointに互換性はありません。 |
| `direction_scale` | `residual`方式で、方向encoderの出力をNoMaD条件へ加算する強度です。`0`で元NoMaD条件をそのまま使い、既存の方向checkpointにはまず`0.1`程度を推奨します。 |
| `direction_num_commands` | 方向コマンド数です。通常はstraight/left/rightの3です。 |
| `direction_latent_dim` | コマンドごとの学習可能latent `z_i` の次元数です。 |
| `direction_hidden_dim` | `z_i` から方向tokenを作るMLPの隠れ層次元数です。 |
| `num_diffusion_iters` | NoMaD推論時の逆拡散ステップ数です。 |
| `num_action_samples` | NoMaDでゴール候補ごとにサンプルするAction数です。 |
| `action_noise_scale` | NoMaD diffusionの初期ノイズ倍率です。`1.0` が標準で、大きくすると候補のばらつきが増えます。 |
| `action_sample_strategy` | 複数Actionサンプルの選び方です。`first`、`mean`、探索モード用の `cmd_dir`、または障害物回避用の `care` を指定します。`care`は方向条件付きNoMaDへ`cmd_dir`を入力し、`care.yaml`で選んだmean/first軌道へAPF斥力回転とSafe-FOV制御を適用します。`direction_conditioning: true` なら `first`/`mean` でも `cmd_dir` はNoMaDのモデル入力として使われます。探索モードで `cmd_dir` の場合、各候補軌道の円周平均角を `cmd_dir_theta_threshold_deg` でleft/straight/rightへ分け、目標方向クラスタのmedoidを選びます。 |
| `cmd_dir_theta_threshold_deg` | `action_sample_strategy: cmd_dir` で候補軌道をleft/straight/rightに分ける代表方向角の閾値 `[deg]` です。 |
| `action_stats` | NoMaDの正規化済みActionを実Actionへ戻すためのmin/maxです。 |

NoMaDを使う場合は `model_type: nomad`、NoMaD用checkpoint、`diffusers`、
`diffusion_policy` とその依存パッケージが必要です。`scripts/train.py` は
`direction_conditioning: true` のNoMaDに対して、収録済み `cmd_dir` ラベルを
使ったdiffusion fine-tuningに対応しています。
`residual`方式の方向encoderは出力層をゼロ初期化するため、公式事前学習重みを
読み込んだ直後はgoal-masked NoMaDと同じ条件表現になります。方向encoderの学習に
よって、その条件表現へstraight/left/rightごとの補正が加わります。
`normalize: false` で学習した旧checkpointは公式スケールの
`normalize: true` と互換ではありません。保存済み設定が異なるcheckpointは
学習再開・推論時にエラーにし、意図しない距離スケールでの実行を防ぎます。

## topics.yaml

購読・配信するROSトピック名を設定します。

| パラメータ | 意味 |
| --- | --- |
| `image_topic` | 推論、Topomap作成、Dataset作成に使うカメラ画像です。 |
| `joy_topic` | 手動方向入力に使う `sensor_msgs/Joy` です。 |
| `cmd_dir_topic` | ViNT/NoMaDの方向conditioned exploreで使う `scenario_navigation_msgs/cmd_dir_intersection` です。 |
| `odometry_topic` | Datasetへ位置とyawを保存するためのオドメトリです。 |
| `amcl_pose_topic` | Datasetへ位置とyawを保存するためのAMCL自己位置です。 |
| `waypoint_topic` | 選択したWaypointの配信先です。 |
| `action_candidates_topic` | NoMaD探索モードで生成した全Action候補と選択候補番号の配信先です。 |
| `cmd_vel_topic` | ロボットへ送る速度指令の配信先です。 |
| `cmd_vel_debug_topic` | 実際の速度出力が無効でも配信される可視化用速度指令です。 |
| `reached_goal_topic` | 最終ノード到達状態を配信します。 |
| `marker_topic` | RViz用Waypoint Markerの配信先です。 |
| `topomap_image_topic` | 現在選択されているTopomap画像の配信先です。 |
| `annotated_image_topic` | カメラ画像へAction候補などを重ねた画像の配信先です。 |
| `frame_id` | Waypoint Markerの基準フレームです。 |

## care.yaml

方向条件付きNoMaDとUniDepthV2由来の障害物点群を組み合わせる設定です。
`care_navigation.launch`は`nomad.yaml`の方向条件付きcheckpointを使用し、
`/cmd_dir_intersection`のstraight/left/rightをNoMaDへ入力してからCARE補正を
適用します。

| パラメータ | 意味 |
| --- | --- |
| `runtime.stale_timeout_seconds` | 障害物点群を有効とみなす受信後の時間 `[s]` です。 |
| `runtime.robot_frame` | Actionと障害物を比較するロボット座標系です。 |
| `avoidance.require_obstacle_data` | `true`なら点群未受信・タイムアウト時に停止します。 |
| `avoidance.num_action_samples` | 1回のNoMaD推論で生成する候補軌道数です。 |
| `avoidance.waypoint_index` | 制御とSafe-FOVに用いるwaypoint番号です。論文の第2 waypointは0始まりで`1`です。 |
| `avoidance.base_action_strategy` | CARE補正前の基準軌道です。`mean`は全候補の平均、`first`は先頭候補を使用します。 |
| `avoidance.maximum_forward_range_m` | 軌跡を延長して障害物を評価する最大前方距離 `[m]` です。 |
| `avoidance.path_influence_radius_m` | 延長軌跡からこの距離以内にある障害物だけをAPF計算へ使用します。 |
| `avoidance.depth_offset_m` | ロボット寸法と深度誤差を補償するため障害物距離から差し引く値 `[m]` です。 |
| `avoidance.minimum_force_distance_m` | 斥力の発散を防ぐ最小距離 `[m]` です。 |
| `avoidance.force_balance_ratio_threshold` | 合力を個別斥力の大きさの総和で割った値がこの閾値未満なら、左右壁が釣り合っているとして軌道を補正しません。 |
| `avoidance.theta_clip_degrees` | APFによる軌道回転角の上限 `[deg]` です。 |
| `avoidance.safe_fov_threshold_degrees` | これを超える希望方位では前進を止め、その場旋回する閾値 `[deg]` です。 |
| `target_direction.stale_timeout_seconds` | `cmd_dir`を有効とみなす最終受信からの時間 `[s]` です。未受信、無効、タイムアウト時は停止します。 |

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
| `navigation_mode` | `topomap` の場合はTopomapのサブゴールへ向かい、`explore` の場合はNoMaDのgoal mask探索を使います。 |
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
| `dataset.trajectory_name` | 再生する軌跡ディレクトリ名です。空文字の場合は対象Dataset内の全軌跡を順番に再生します。 |
| `dataset.frame_id` | RVizへ出すPathとPoseの基準フレームです。 |
| `dataset.rate` | Dataset画像とPoseの再生周期 `[Hz]` です。 |
| `dataset.loop` | `true` の場合、最後まで再生したあと先頭へ戻ります。 |
| `overlay.rate` | カメラ画像へSubgoal画像、NoMaD Action候補を重ねる周期 `[Hz]` です。 |
| `overlay.image_size` | 可視化用にpublishするカメラ画像サイズ `[幅, 高さ]` です。モデル入力サイズには影響しません。 |
| `evaluation.dataset_type` | 教師軌跡とモデル予測を比較するDatasetの種類です。 |
| `evaluation.trajectory_name` | 評価対象の軌跡名です。空文字の場合はDataset全体からサンプルを選びます。 |
| `evaluation.sample_indices` | 評価するDataset sample indexのリストです。空リストの場合は方向ごとにランダム選択します。分岐の比較では対象画像に表示されたsample番号を指定します。 |
| `evaluation.output_dir` | 評価画像の出力先です。`auto` の場合はDataset、モデル、checkpoint名から決定します。 |
| `evaluation.samples_per_direction` | straight/left/rightごとに評価するサンプル数です。`-1`で全件を評価します。 |
| `evaluation.include_augmented` | `true` の場合は左右反転augmentationも評価対象に含めます。 |
| `evaluation.compare_all_commands` | `true` の場合、同一contextと同一diffusion乱数へstraight/left/rightを入力し、選択軌跡を1枚に重ねます。 |
| `evaluation.strategy` | 予測候補の選択方法です。`configured` はモデルの `action_sample_strategy` を使用します。 |
| `evaluation.sample_seed` | 方向ごとの評価サンプルをランダム選択するときのseedです。同じ値なら同じサンプルを選びます。 |
| `evaluation.seed` | diffusion予測の乱数seedです。 |
| `evaluation.preview_image_height` | 評価画像に並べるcontext画像の高さ `[px]` です。 |

`compare_all_commands: true` のとき、教師と異なる方向を入力した行のADE/FDEは
教師に対する正解率ではなく、コマンド変更による軌跡差の参考値です。
`metrics.csv` の `teacher_command_match` が `true` の行だけが通常の教師比較です。

## dataset / collection / training

`gnm.yaml`、`vint.yaml`、`nomad.yaml` は、それぞれモデルに対応した
`dataset`、`collection`、`training` を持ちます。`runtime.yaml` の
`model_type` で選ばれたファイルの設定だけが使われます。

### dataset

選択中モデルの学習サンプル生成方法を設定します。読み込み後のコード上では、
選択された設定が従来通り `dataset` として扱われます。

#### vint.yaml の dataset

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
| `cmd_dir_hold_samples_after_change` | オンライン収集で `cmd_dir` が切り替わったあと、このサンプル数だけ切替前のラベルを保持して `traj_data.pkl` へ保存します。`auto` の場合、選択中モデルの `len_traj_pred * waypoint_spacing` から自動計算します。 |

`model.direction_conditioning: true` のViNT方向fine-tuningでは目標画像を使わないため、
distance labelは予測する行動系列の終端までのtemporal distanceとして
`len_traj_pred` を使います。実時間に直す場合は
`len_traj_pred * waypoint_spacing * collection.sample_dt` 秒です。

#### nomad.yaml の dataset

| パラメータ | 意味 |
| --- | --- |
| `metric_waypoint_spacing` | 連続する収録画像間の平均移動距離 `[m]` です。`normalize: true` の場合、教師Action差分を `metric_waypoint_spacing * waypoint_spacing` で除算してから公式NoMaDの `action_stats` へ写します。 |
| `waypoint_spacing` | NoMaD方向fine-tuningで、行動系列を何フレームおきに取り出すかを指定します。`image_size`、`context_size`、`len_traj_pred` は `nomad.yaml` の `model` を使います。 |
| `cmd_dir_hold_samples_after_change` | オンライン収集で `cmd_dir` が切り替わったあと、このサンプル数だけ切替前のラベルを保持して `traj_data.pkl` へ保存します。`auto` の場合、選択中モデルの `len_traj_pred * waypoint_spacing` から自動計算します。手動値が必要値より小さい場合も必要値まで引き上げます。 |

### collection

| パラメータ | 意味 |
| --- | --- |
| `dataset_type` | 作成するDatasetの種類です。`train` または `test` を指定します。 |
| `trajectory_name` | 保存する軌跡ディレクトリ名です。空文字の場合は日時から自動生成します。 |
| `pose_source` | Datasetの軌跡に使う姿勢情報です。`odometry` または `amcl` を指定します。 |
| `sample_dt` | Datasetへ画像と選択した姿勢情報を保存する時間間隔 `[s]` です。 |
| `image_format` | 保存画像の拡張子です。例: `jpg`、`png`。 |
| `control_rate` | nav recovery収集時にcmd_velをpublishする周期 `[Hz]` です。 |
| `nav_path_topic` | 経路からの距離判定に使うnav stackのPath topicです。 |
| `recovery_start_distance` | 経路からこの距離以上離れたらnav recoveryへ切り替えます `[m]`。 |
| `vint_resume_distance` | 経路へこの距離以内に戻ったらVNM/NoMaD走行へ戻します `[m]`。 |
| `angular_recovery_error` | `abs(nav.angular.z - vnm.angular.z)` がこの値以上ならnav recoveryへ切り替えます `[rad/s]`。 |
| `min_trajectory_samples` | trajectoryを保存・切替する最小サンプル数です。`auto` の場合、選択中モデルの `(context_size + len_traj_pred) * waypoint_spacing + 1` から自動計算します。手動値が必要値より小さい場合も必要値まで引き上げます。 |

保存画像は生画像ではなく、現在選択中モデルの `image_size` に合わせて
resize を適用したRGB画像です。`model.image_center_crop: true` の場合だけ
4:3 center crop後にresizeします。NoMaDなら通常 `nomad.yaml` の
`model.image_size`、ViNTなら `vint.yaml` の `model.image_size` が使われます。正規化
（ImageNet mean/std）は画像ファイルには保存せず、学習・推論時にTensorへ
変換したあと適用します。

学習時は保存済み画像サイズが現在の設定と一致するか検証します。例えば
NoMaD用に `96x96` で作成したDatasetをViNTの `85x64` 学習へ流用すると
エラーになります。モデルを切り替える場合は、同じbagから
`roslaunch vnm_ros create_dataset.launch` を実行し直して、選択中モデル用の
Datasetを作成してください。
Datasetは `train_data_dir` または `test_data_dir` を基準ディレクトリとして、
`vint/<trajectory_name>`、`nomad/<trajectory_name>` の形で保存されます。
選択中のtrain Datasetへ左右反転augmentationを追加する場合は、次を実行します。

```bash
rosrun vnm_ros augment_dataset_horizontal_flip.py
```

出力先は `<train_data_dir>/<model_type>/aug/<trajectory_name>/` です。
画像を水平反転し、軌跡座標とyawを鏡映して、`cmd_dir` のleft/rightを交換します。
学習時はサブディレクトリも再帰的に探索するため、元データと `aug/` が同時に使われます。
既存の `aug/` は上書きしないため、作り直す場合は既存ディレクトリを確認してから削除します。

Datasetのcontext画像から得たモデル予測と教師軌跡を比較する場合は、次を実行します。

```bash
roslaunch vnm_ros evaluate_dataset_predictions.launch
```

各画像では、教師軌跡を緑と黒、全予測候補を薄青、実行時の
`action_sample_strategy` で選択される予測を赤で、同じロボット座標系へ描画します。
ADE、FDE、終点角度誤差は `metrics.csv`、方向別平均は `summary.csv` に保存します。
出力先は `plots/evaluation/<dataset>/<model>/<checkpoint>/<train|test>/` です。
評価条件は `runtime.yaml` の `visualization.evaluation`、checkpointは選択中モデルの
`model.checkpoint_path` から読み込みます。デフォルトでは `aug/` を除外します。

Dataset作成時に `cmd_dir_topic` がbagに含まれている場合、各保存サンプルへ
最新の `cmd_dir` one-hotラベルも保存します。方向fine-tuningではこの
保存済みラベルを使用します。未収録データは互換性のためstraight `[1, 0, 0]`
として扱われます。

nav recovery付きのオンライン収集は以下で実行します。

```bash
roslaunch vnm_ros collect_dataset_nav_recovery.launch
```

このlaunchは収集専用設定として、選択中モデル設定の
`training.pretrained_weights_path` を推論checkpointに使用し、
`robot.navigation_mode: explore`、`robot.publish_cmd_vel: false` を自動適用します。
公式事前学習checkpointには方向encoderが含まれないため、
`direction_conditioning: false` と `action_sample_strategy: cmd_dir` も適用し、
複数の生成軌跡から `cmd_dir` に合う候補を選択します。通常の
`navigate.launch` とYAMLファイル自体は変更しません。

この収集スクリプトは通常時に `/vnm/cmd_vel_debug`、復帰時に `/nav_vel` を
選んで `/cmd_vel` へpublishします。nav recoveryへの切替は、経路からの距離、または
navとVNM/NoMaDの角速度差で判定します。`recovery_start_distance` 以上、または
`abs(nav.angular.z - vnm.angular.z) >= angular_recovery_error` でnav recoveryへ入り、
`vint_resume_distance` 以下かつ角速度差が `angular_recovery_error` 未満になったら
VNM/NoMaD走行へ戻ります。Datasetとして保存するのはnav recovery中の区間だけです。
VNM/NoMaD走行中の区間は保存せず、nav recoveryが終わった時点でそのrecovery trajectoryを
保存して閉じます。`min_trajectory_samples` 未満の短すぎるrecoveryは学習サンプルを
作れないため破棄します。重複publishを避けるため、収集中はVNM本体の直接 `/cmd_vel`
publishを止め、debug topicだけを使う構成にしてください。

方向fine-tuning済みの `model.checkpoint_path` を使って収集する場合は、次のように
事前学習checkpointへの切替を止め、方向条件付けを有効にします。

```bash
roslaunch vnm_ros collect_dataset_nav_recovery.launch \
  use_pretrained_weights:=false \
  direction_conditioning_override:=true \
  action_sample_strategy_override:=mean
```

任意のcheckpointを使う場合は
`checkpoint_path_override:=weights/nomad/example.pth` を追加します。

収集後の `traj_data.pkl` はCSVへ変換できます。

```bash
rosrun vnm_ros pkl_to_csv.py dataset/my_dataset/train -r
```

Dataset軌跡をRVizなしで確認する場合は、地図画像上へtrajectoryをPNG出力できます。

```bash
roslaunch vnm_ros plot_dataset_trajectories.launch dataset_type:=train
```

出力先はデフォルトでDatasetディレクトリ名を使った
`vnm_ros/plots/dataset/<dataset>/<model_type>/<train|test>/` です。
`overview.png` に全軌跡、`trajectories/` に各軌跡ごとの画像を保存します。
方向conditioned Datasetでは、学習Datasetが保持するsampleの現在位置だけを描画します。
現在点を含む予測区間の `len_traj_pred + 1` 点で `cmd_dir` が混ざるsampleは、
最も多い方向をそのsampleの目標方向にします。
最多方向が同数のsampleだけを学習と表示から除外します。
軌跡画像にはデフォルトで1m間隔のworld座標グリッドを描画します。
間隔は `grid_spacing_m` で変更でき、0以下にすると非表示です。
方向conditioned Datasetの場合は `training_samples/` に実際に学習に使う
context画像列と教師軌跡も保存します。
`training_samples_per_direction` でstraight/left/rightそれぞれ何枚保存するかを
指定できます。`-1` なら各方向の学習サンプルを全件保存し、`0` なら保存しません。
目標方向と教師軌跡の向きが逆のサンプルは `opposite_direction_samples/` に保存します。
教師軌跡をロボット座標系へ変換し、終点角度が目標方向と反対側に
`opposite_direction_threshold_deg` 度以上あるものを対象とします。
`opposite_direction_samples_per_direction` はleft/rightそれぞれの画像保存数で、
`-1` なら全件、`0` なら画像保存を無効化します。該当サンプルの一覧は
画像保存数にかかわらず `opposite_direction_samples/samples.csv` に保存します。

### training

選択中モデルの学習方法を設定します。読み込み後のコード上では、
選択された設定が従来通り `training` として扱われます。

| パラメータ | 意味 |
| --- | --- |
| `model_name` | 学習済みcheckpointの名前です。例えば `encoder_lr` なら `encoder_lr_best.pth` と `encoder_lr_epoch005.pth` のように保存します。空文字なら従来の `best.pth`、`epoch005.pth` です。使用可能文字は英数字、`_`、`-`、`.` です。 |
| `pretrained_weights_path` | 新規学習時に初期重みとして読み込む事前学習済みモデルです。空文字の場合は初期重みを読み込みません。 |
| `freeze_dist_pred_net` | NoMaDで `true` の場合、距離予測headを固定します。 |
| `freeze_layers` | 固定するmodule名またはparameter prefixのリストです。`model.named_modules()` の名前を指定します。例: `vision_encoder.obs_encoder`、`vision_encoder.sa_encoder.layers.0`、`noise_pred_net`、`obs_encoder._blocks.0`。 |
| `use_test` | `true` の場合、各epochでtest Datasetを評価します。 |
| `tensorboard` | TensorBoardログを保存するかを指定します。 |
| `epoch` | 学習する総epoch数のリストです。`[5]` なら5 epochの学習を1回、`[5, 10, 20]` なら各値について同じ初期重み・同じseedから独立に学習します。`resume` は値が1つの場合だけ使用できます。 |
| `batch_size` | 1回の更新で使用するサンプル数です。 |
| `num_workers` | PyTorch DataLoaderの並列読込プロセス数です。 |
| `learning_rate` | AdamW Optimizerの初期学習率です。 |
| `layer_learning_rates` | module名またはparameter prefixごとの学習率です。未指定の層は `learning_rate` を使います。複数のprefixに一致する場合は最も具体的な長いprefixを優先します。NoMaDの例: `{vision_encoder.direction_encoder: 0.0001, vision_encoder.sa_encoder: 0.00002, noise_pred_net: 0.00002, vision_encoder.obs_encoder: 0.00001}`。 |
| `weight_decay` | AdamWのweight decay係数です。 |
| `alpha` | 距離lossとAction lossの重みです。総lossは `alpha * 1e-2 * distance_loss + (1 - alpha) * action_loss` です。 |
| `gradient_clip` | 勾配ノルムの最大値です。0以下にするとクリッピングしません。 |
| `cmd_dir_loss_weighting` | 方向fine-tuningで、straight/left/rightのデータ数に応じてlossへ逆頻度重みを掛けるかを指定します。DataLoaderのサンプリング確率は変えません。 |
| `cmd_dir_loss_weight_power` | `cmd_dir_loss_weighting: true` の重み付け強度です。`0.0` は重みなし、`1.0` は方向ごとのサンプル数の逆数を使います。 |
| `scheduler` | 学習率Schedulerです。`cosine` はCosine Annealing、`warmup_cosine` はwarmup後にcosine減衰します。`none` または空文字で無効化します。 |
| `warmup_epochs` | `scheduler: warmup_cosine` の場合に、何epochかけて学習率を立ち上げるかを指定します。内部epochは0始まりなので、`1` ならepoch 0だけwarmupです。 |
| `warmup_start_factor` | warmup開始時の学習率倍率です。`0.1` なら `learning_rate * 0.1` から始めます。 |
| `min_lr_factor` | cosine減衰後の最小学習率倍率です。`0.0` なら最終的に0まで下げます。 |
| `resume` | 学習を再開するチェックポイントのパスです。空文字なら新規学習です。 |

新規学習時の初期重みは `pretrained_weights_path` を使用します。
推論用の重みは選択中の `vint.yaml` または `nomad.yaml` の
`model.checkpoint_path` で別に指定します。
NoMaDなら `vision_encoder.obs_encoder`、`vision_encoder.goal_encoder`、
`vision_encoder.sa_encoder.layers.0`、`noise_pred_net`、`dist_pred_net` など、
ViNTなら `obs_encoder`、`goal_encoder`、`decoder.sa_decoder.layers.0`、
`direction_encoder`、`action_predictor` などを指定できます。
指定できるmodule名は以下で確認できます。

```bash
rosrun vnm_ros train.py --config-dir $(rospack find vnm_ros)/config --list-freeze-layers
```

方向conditioned Datasetで学習する場合、TensorBoardには全体lossに加えて
`train/direction/<straight|left|right>/...` と
`test/direction/<straight|left|right>/...` の方向別lossも記録します。
学習結果はアーキテクチャとDatasetごとに
`runs/<model_type>/<dataset_name>/<run_name>/` と
`weights/<model_type>/<dataset_name>/` へ保存されます。NoMaDは学習中にEMAを更新し、
checkpointの `state_dict` には推論用EMA重み、`model_state_dict` にはresume用の
生モデル重みを保存します。checkpointは各epochで
監視lossが改善した場合だけ `best.pth` を更新します。`use_test: true` ならtest loss、
`false`ならtrain lossを監視します。`latest.pth` は保存しません。
`epoch` の各値について独立したrunを作り、その最終epochのcheckpointも
`epochXXX.pth` として保存します。
`model_name` を指定した場合は、`<model_name>_best.pth` と
`<model_name>_epochXXX.pth` になります。
各checkpointと同名のYAMLへ実際に使用したモデル・Dataset・学習設定を保存し、
runディレクトリにも `training_config.yaml` を保存します。
例えばDataset名が `mix_0.8` のNoMaDなら
`weights/nomad/mix_0.8/best.pth` と `best.yaml` が生成されます。
学習再開時は `resume` が優先され、`pretrained_weights_path` は読み込みません。
`resume` ではcheckpoint作成時と同じ `freeze_layers`、`freeze_dist_pred_net`、
`layer_learning_rates` を指定してください。これらを変更して追加学習する場合は、
対象checkpointを `pretrained_weights_path` に指定してOptimizerを作り直します。
