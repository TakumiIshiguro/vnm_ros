# vnm_ros

ROS inference package for Visual Navigation Models based on ViNT and NoMaD.

This package records datasets and topological maps, trains a ViNT model,
evaluates checkpoints, predicts local waypoints, and optionally publishes
velocity commands. NoMaD checkpoints are supported for navigation inference.

## Requirements

- ROS Noetic
- Python 3
- `torch`, `torchvision`, `Pillow`, `numpy`, `PyYAML`
- `efficientnet-pytorch`
- `warmup-scheduler` when loading the original legacy `weights/vint.pth`
- `diffusers`, `diffusion_policy`, and dependencies such as `einops` for NoMaD

Install ViNT runtime dependencies:

```bash
python3 -m pip install efficientnet-pytorch warmup-scheduler
```

NoMaD inference also needs the diffusion packages used by the upstream model:

```bash
python3 -m pip install diffusers einops
```

and a working `diffusion_policy` installation on `PYTHONPATH`.

The ViNT and NoMaD model components used by this package are included locally.
Installing `visualnav-transformer` or its `vint_train` package is not required.
See `THIRD_PARTY_NOTICES.md` for upstream attribution and license terms.
Checkpoints produced by `vnm_ros` store a plain state dictionary and do not
depend on `warmup-scheduler`; it is needed only because the original
`vint.pth` pickles the upstream training scheduler.

## Files

- `scripts/create_topomap.py`: save camera frames as a topomap.
- `scripts/create_dataset.py`: save synchronized camera and odometry trajectories.
- `scripts/train.py`: train ViNT from a processed dataset.
- `scripts/eval.py`: evaluate a checkpoint on the automatically held-out data.
- `scripts/vnm_node.py`: load model, select subgoal, publish waypoint/cmd_vel.
- `scripts/care_overlay.py`: overlay NoMaD candidates and robot-frame obstacles.
- `config/topics.yaml`: ROS topic names and frame id.
- `config/vint.yaml`: ViNT model, dataset collection, and training settings.
- `config/nomad.yaml`: NoMaD model, dataset collection, and training settings.
- `config/runtime.yaml`: robot, topomap, and visualization settings.

## Usage

Create a topomap:

```bash
roslaunch vnm_ros create_topomap.launch
```

Set `model_type` in `config/runtime.yaml`, then set the required rosbag path in
the selected `config/vint.yaml` or `config/nomad.yaml`.

Run navigation:

```bash
roslaunch vnm_ros navigate.launch
```

Set `robot.navigation_mode: explore` in `config/runtime.yaml` to run NoMaD in
goal-masked exploration mode. In that mode `vnm_node.py` samples multiple NoMaD
actions and selects the one closest to `cmd_dir` from `/cmd_dir_intersection`.
Set `robot.navigation_mode: topomap` to use the original topomap subgoal flow.

Publish manual direction commands from the keyboard:

```bash
roslaunch vnm_ros keyboard_cmd_dir.launch
```

Holding `a` publishes left, holding `d` publishes right, and no key publishes
straight. The `hold_timeout` launch parameter controls how quickly the command
returns to straight after key repeat stops.

Publish manual direction commands from a game controller:

```bash
roslaunch vnm_ros controller_cmd_dir.launch
```

The horizontal D-pad selects left/right and the left stick is used as a
fallback. Neutral input and controller timeout publish straight. Axis numbers,
thresholds, inversion, and optional button mappings are launch arguments. Do
not run the keyboard and controller publishers at the same time.

Run navigation with the camera/cmd_vel overlay viewer:

```bash
roslaunch vnm_ros navigate_visualization.launch
```

Run goal-free CARE exploration, which starts goal-masked NoMaD and metric
depth estimation. Direction conditioning and the `cmd_dir` input are disabled.
CARE applies APF repulsion to the first NoMaD trajectory, rotates the complete
trajectory away from obstacles, and publishes its avoidance command; velocity
output remains disabled by default:

```bash
roslaunch vnm_ros care_navigation.launch
```

After checking the candidates and obstacle point cloud, enable robot motion:

```bash
roslaunch vnm_ros care_navigation.launch publish_cmd_vel:=true
```

To show the common robot-coordinate BEV at the same time:

```bash
roslaunch vnm_ros care_visualization.launch
```

This launch opens both the CARE BEV and the colorized UniDepthV2 metric-depth image from
`/unidepth/depth_color`. Set `show_depth_image:=false` to hide only
the depth window.

The depth node publishes bin-selected CARE obstacles on
`/unidepth/obstacle_points` and all filtered points on
`/unidepth/obstacle_points_all`, both in `base_footprint`. The CARE
BEV draws all points in gray and the bin-selected CARE input in red, together
with `/vnm/action_candidates`. The selected path is yellow during normal
navigation and changes to magenta while CARE is applying avoidance rotation.
The default view is forward 1.2 m by lateral
3 m (-1.5 to +1.5 m), using the same pixels-per-metre scale on both axes. The result is
published as an image on `/vnm/care_bev`. Set `start_navigation:=false` or
`start_depth_estimator:=false` when those nodes are already running.

Avoidance parameters are under `avoidance` in `config/care.yaml`. Following the
CARE paper, the module finds the waypoint with the strongest inverse-distance
repulsive force, clips the trajectory rotation to 45 degrees, and uses the
Safe-FOV rule to suppress forward velocity above a 30-degree desired heading.
It publishes a zero trajectory when `require_obstacle_data: true` and the point
cloud is missing or stale.

Both launch files use the same files under `config/`. Set `publish_cmd_vel:
false` in `config/runtime.yaml` for model testing without moving the robot, or
set it to `true` for navigation.

Place checkpoints in `weights/`, for example:

```text
weights/best.pth
```

## Dataset collection

Set the camera and pose topics in `config/topics.yaml`, then select
`collection.pose_source` in the selected `config/vint.yaml` or
`config/nomad.yaml`:

```yaml
collection:
  pose_source: odometry  # odometry or amcl
```

Record one training trajectory:

```bash
roslaunch vnm_ros create_dataset.launch
```

Stop the node with Ctrl-C. The resulting trajectory contains numbered images and
`traj_data.pkl`:

```text
dataset/my_dataset/train/traj_000/
```

Set `collection.dataset_type` and `collection.trajectory_name` in
the selected `config/vint.yaml` or `config/nomad.yaml`, and set the bag path
there before launching.

Create a topomap and dataset at the same time:

```bash
roslaunch vnm_ros create_topomap_and_dataset.launch
```

For rosbag input, configure `runtime.yaml` and the selected model config before
launching.

Collection interval, dataset paths, and training parameters are configured in
`config/vint.yaml` or `config/nomad.yaml`. Topomap paths are configured in
`config/runtime.yaml`.

Plot dataset trajectories and training samples to PNG files:

```bash
roslaunch vnm_ros plot_dataset_trajectories.launch
```

The launch writes `overview.png`, per-trajectory plots, and direction training
sample previews under `plots/dataset/<dataset>/<model_type>/<train|test>/`.
Trajectory plots include a 1 m world-coordinate grid by default; override it
with `grid_spacing_m:=0.5` or disable it with `grid_spacing_m:=0`.

Create horizontal-flip augmentation for the selected training dataset:

```bash
rosrun vnm_ros augment_dataset_horizontal_flip.py
```

The command writes mirrored trajectories under
`<train_data_dir>/<model_type>/aug/`. It flips every image, mirrors trajectory
positions and yaw, and swaps the left/right command labels. Training discovers
nested trajectory directories, so the original and augmented trajectories are
loaded together. The command refuses to overwrite an existing `aug/` directory.

Compare predictions from dataset context images with their teacher trajectories:

```bash
roslaunch vnm_ros evaluate_dataset_predictions.launch
```

Each preview shows the context images, the teacher trajectory, every model
candidate, and the candidate selected by the configured action strategy in the
same robot coordinate frame. Per-sample ADE/FDE values are written to
`metrics.csv`, with direction summaries in `summary.csv`, under
`plots/evaluation/<dataset>/<model>/<checkpoint>/<train|test>/`. Augmented
trajectories are excluded by default; set `include_augmented:=true` to evaluate
them too in `runtime.yaml` under `visualization.evaluation`.

With `visualization.evaluation.compare_all_commands: true`, each preview uses
the same context and diffusion noise for straight, left, and right, then overlays
the three selected trajectories. Set `sample_indices` to the sample numbers of
the junction images to restrict the comparison to those observations.

## Training

```bash
rosrun vnm_ros train.py --config-dir $(rospack find vnm_ros)/config
```

Training and test datasets use separate directories:

```yaml
paths:
  dataset:
    train_data_dir: dataset/my_dataset/train
    test_data_dir: dataset/my_dataset/test
```

Test evaluation can be enabled or disabled:

```yaml
training:
  use_test: true
  tensorboard: true
```

It can also be overridden per run:

```bash
rosrun vnm_ros train.py \
  --config-dir $(rospack find vnm_ros)/config \
  --use-test false
```

When test evaluation is disabled, `best.pth` is selected using the training loss.

Checkpoints are grouped by dataset under
`weights/<model_type>/<dataset_name>/`. `best.pth` is updated when the monitored
loss improves, and an epoch sweep also writes its final `epochXXX.pth`
checkpoint. Set `training.model_name` to add a prefix to both checkpoint names:

```text
weights/nomad/mix_0.8/epoch010.pth
weights/nomad/mix_0.8/best.pth
weights/nomad/mix_0.8/encoder_lr_epoch010.pth
weights/nomad/mix_0.8/encoder_lr_best.pth
```

Each checkpoint has a same-name YAML file containing the effective model,
dataset, and training configuration. Metrics, TensorBoard logs, and another
copy of the configuration are written to:

```text
weights/nomad/mix_0.8/epoch010.yaml
runs/nomad/mix_0.8/20260611_153000_123456_ep010/training_config.yaml
runs/nomad/mix_0.8/20260611_153000_123456_ep010/metrics.jsonl
runs/nomad/mix_0.8/20260611_153000_123456_ep010/tensorboard/
```

Start TensorBoard while training or after training:

```bash
tensorboard \
  --logdir "$(rospack find vnm_ros)/runs" \
  --port 6006
```

Then open `http://localhost:6006`. The following values are recorded:

- `train/loss`, `train/distance_loss`, `train/action_loss`
- `train/distance_mae`, `train/position_error`
- corresponding `test/*` metrics when test evaluation is enabled
- `training/learning_rate`

If TensorBoard is not installed:

```bash
python3 -m pip install tensorboard
```

Evaluate a checkpoint:

```bash
rosrun vnm_ros eval.py \
  --config-dir $(rospack find vnm_ros)/config \
  --checkpoint weights/best.pth
```

To deploy the best trained model:

```bash
roslaunch vnm_ros navigate.launch
```
