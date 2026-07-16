# Configuration

`vnm_ros` の設定ファイルと各パラメータの意味を説明します。
相対パスは基本的に `vnm_ros` パッケージのルートから解決されます。

## vint.yaml / nomad.yaml

モデルごとの構造、重み、Dataset作成、学習設定をまとめます。
どちらを使うかは `runtime.yaml` の `model_type` で選びます。
`model_type: vint` なら `vint.yaml`、`model_type: nomad` なら
`nomad.yaml` を読み込みます。

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
| `model_type` | この設定ファイルのモデル形式です。`vint.yaml` は `vint`、`nomad.yaml` は `nomad` です。 |
| `device` | 推論デバイスです。`auto` はCUDAが利用可能ならGPU、それ以外はCPUを使用します。 |
| `obs_encoder` | 画像エンコーダです。 |
| `mha_num_attention_heads` | TransformerのMulti-Head Attentionのヘッド数です。 |
| `mha_num_attention_layers` | Transformer Encoderの層数です。 |
| `mha_ff_dim_factor` | Transformer内のFeed Forward層の拡大率です。 |
| `context_type` | コンテキスト形式を表す設定値です。現在の実装では未使用です。 |
| `normalize` | `true` の場合、モデル出力WaypointのXYを実機用の距離へスケーリングします。 |
| `waypoint_index` | 予測されたWaypoint列のうち、制御に使用する番号です。0始まりです。 |
| `checkpoint_path` | 推論で読み込むモデル重みのパスです。 |

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
| `normalize` | NoMaDでは通常 `false` です。NoMaD出力は `action_stats` で正規化解除されるため、`true` にすると追加で `max_v / model_rate` 倍されて低速になります。 |
| `learn_angle` | NoMaDでは通常 `false` です。 |
| `down_dims` | NoMaD diffusion U-Netの各段の次元数です。 |
| `cond_predict_scale` | NoMaD diffusion U-Netで条件付きscale予測を使うかを指定します。 |
| `direction_conditioning` | `true` の場合、`cmd_dir` のラベルindexから学習可能なlatent `z_i` を選び、MLPでTransformer入力用の方向tokenへ変換します。従来のgoal token位置に入り、token列は `obs tokens + direction token` になります。 |
| `direction_num_commands` | 方向コマンド数です。通常はstraight/left/rightの3です。 |
| `direction_latent_dim` | コマンドごとの学習可能latent `z_i` の次元数です。 |
| `direction_hidden_dim` | `z_i` から方向tokenを作るMLPの隠れ層次元数です。 |
| `num_diffusion_iters` | NoMaD推論時の逆拡散ステップ数です。 |
| `num_action_samples` | NoMaDでゴール候補ごとにサンプルするAction数です。 |
| `action_noise_scale` | NoMaD diffusionの初期ノイズ倍率です。`1.0` が標準で、大きくすると候補のばらつきが増えます。 |
| `action_sample_strategy` | 複数Actionサンプルの選び方です。`first`、`mean`、または探索モード用の `cmd_dir` を指定します。`direction_conditioning: true` なら `first`/`mean` でも `cmd_dir` はNoMaDのモデル入力として使われます。`mean` は `cmd_dir` 条件付きで生成した候補軌道の平均を使います。探索モードで `cmd_dir` の場合、各候補軌道の円周平均角を `cmd_dir_theta_threshold_deg` でleft/straight/rightへ分け、目標方向クラスタのmedoidを選びます。 |
| `cmd_dir_theta_threshold_deg` | `action_sample_strategy: cmd_dir` で候補軌道をleft/straight/rightに分ける代表方向角の閾値 `[deg]` です。 |
| `action_stats` | NoMaDの正規化済みActionを実Actionへ戻すためのmin/maxです。 |

NoMaDを使う場合は `model_type: nomad`、NoMaD用checkpoint、`diffusers`、
`diffusion_policy` とその依存パッケージが必要です。`scripts/train.py` は
`direction_conditioning: true` のNoMaDに対して、収録済み `cmd_dir` ラベルを
使ったdiffusion fine-tuningに対応しています。

## topics.yaml

購読・配信するROSトピック名を設定します。

| パラメータ | 意味 |
| --- | --- |
| `image_topic` | 推論、Topomap作成、Dataset作成に使うカメラ画像です。 |
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

## dataset / collection / training

`vint.yaml` と `nomad.yaml` は、それぞれモデルに対応した
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
| `nav_cmd_vel_topic` | 経路復帰時に使うnav stackのcmd_vel topicです。 |
| `vint_cmd_vel_topic` | 通常走行時に使うVNM/NoMaDのdebug cmd_vel topicです。通常は `/vnm/cmd_vel_debug` です。 |
| `publish_zero_when_idle` | 入力cmd_velが未到着のときにzero twistをpublishするかを指定します。 |
| `recovery_start_distance` | 経路からこの距離以上離れたらnav recoveryへ切り替えます `[m]`。 |
| `vint_resume_distance` | 経路へこの距離以内に戻ったらVNM/NoMaD走行へ戻します `[m]`。 |
| `angular_recovery_enabled` | `true` の場合、navとVNM/NoMaDの角速度差でもnav recoveryへ切り替えます。 |
| `angular_recovery_start_error` | `abs(nav.angular.z - vnm.angular.z)` がこの値以上ならnav recoveryへ切り替えます `[rad/s]`。 |
| `angular_recovery_resume_error` | 角速度差がこの値以下、かつ経路距離が `vint_resume_distance` 以下ならVNM/NoMaD走行へ戻します `[rad/s]`。 |
| `split_trajectory_on_mode_switch` | `true` の場合、nav recoveryからVNM/NoMaD走行へ戻った時点でtrajectoryを切り替えます。通常は `false` にして、連続走行を長いtrajectoryとして保存し、学習サンプル数を増やします。 |
| `split_trajectory_on_save_gap` | `true` の場合、保存できない期間が一定以上続いたあと次に保存するタイミングでtrajectoryを切り替えます。 |
| `trajectory_save_gap_factor` | `split_trajectory_on_save_gap` の閾値です。`sample_dt * trajectory_save_gap_factor` より保存間隔が空いたらtrajectoryを切り替えます。 |
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

Dataset作成時に `cmd_dir_topic` がbagに含まれている場合、各保存サンプルへ
最新の `cmd_dir` one-hotラベルも保存します。方向fine-tuningではこの
保存済みラベルを使用します。未収録データは互換性のためstraight `[1, 0, 0]`
として扱われます。

nav recovery付きのオンライン収集は以下で実行します。

```bash
roslaunch vnm_ros collect_dataset_nav_recovery.launch
```

この収集スクリプトは通常時に `/vnm/cmd_vel_debug`、復帰時に `/nav_vel` を
選んで `/cmd_vel` へpublishします。復帰への切替は経路からの距離、または
navとVNM/NoMaDの角速度差で判定できます。重複publishを避けるため、収集中は
VNM本体の直接 `/cmd_vel` publishを止め、debug topicだけを使う構成にしてください。

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
軌跡画像にはデフォルトで1m間隔のworld座標グリッドを描画します。
間隔は `grid_spacing_m` で変更でき、0以下にすると非表示です。
方向conditioned Datasetの場合は `training_samples/` に実際に学習に使う
context画像列と教師軌跡も保存します。
`training_samples_per_direction` でstraight/left/rightそれぞれ何枚保存するかを
指定できます。`-1` なら各方向の学習サンプルを全件保存し、`0` なら保存しません。

### training

選択中モデルの学習方法を設定します。読み込み後のコード上では、
選択された設定が従来通り `training` として扱われます。

| パラメータ | 意味 |
| --- | --- |
| `pretrained_weights_path` | 新規学習時に初期重みとして読み込む事前学習済みモデルです。空文字の場合は初期重みを読み込みません。 |
| `freeze_dist_pred_net` | NoMaDで `true` の場合、距離予測headを固定します。 |
| `freeze_layers` | 固定するmodule名またはparameter prefixのリストです。`model.named_modules()` の名前を指定します。例: `vision_encoder.obs_encoder`、`vision_encoder.sa_encoder.layers.0`、`noise_pred_net`、`obs_encoder._blocks.0`。 |
| `use_test` | `true` の場合、各epochでtest Datasetを評価します。 |
| `tensorboard` | TensorBoardログを保存するかを指定します。 |
| `epochs` | 学習する総epoch数です。 |
| `batch_size` | 1回の更新で使用するサンプル数です。 |
| `num_workers` | PyTorch DataLoaderの並列読込プロセス数です。 |
| `learning_rate` | AdamW Optimizerの初期学習率です。 |
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
学習結果はアーキテクチャごとに `runs/<model_type>/<run_name>/` と
`weights/<model_type>/` へ保存されます。各epochのcheckpointは
`model_type`、`learning_rate`、`batch_size`、`epochs`、`scheduler`、
`warmup_epochs`、`alpha`、`weight_decay`、`cmd_dir_loss_weighting`、
`freeze_dist_pred_net`、`freeze_layers` とepoch番号を含むファイル名で保存します。
互換性のため `latest.pth` と `best.pth` も更新します。例えばViNTは
`weights/vint/best.pth`、NoMaDは `weights/nomad/best.pth` がbest checkpointです。
学習再開時は `resume` が優先され、`pretrained_weights_path` は読み込みません。
