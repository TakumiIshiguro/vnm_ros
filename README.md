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
- `config/topics.yaml`: ROS topic names and frame id.
- `config/model.yaml`: model architecture and checkpoint path.
- `config/runtime.yaml`: robot, topomap, and visualization settings.
- `config/training.yaml`: dataset collection, rosbag, and training settings.

## Usage

Create a topomap:

```bash
roslaunch vnm_ros create_topomap.launch
```

Set the required rosbag path in `config/training.yaml`.

Run navigation:

```bash
roslaunch vnm_ros navigate.launch
```

Run navigation with the camera/cmd_vel overlay viewer:

```bash
roslaunch vnm_ros navigate_visualization.launch
```

Both launch files use the same files under `config/`. Set `publish_cmd_vel:
false` in `config/runtime.yaml` for model testing without moving the robot, or
set it to `true` for navigation.

Place checkpoints in `weights/`, for example:

```text
weights/best.pth
```

## Dataset collection

Set the camera and pose topics in `config/topics.yaml`, then select
`collection.pose_source` in `config/training.yaml`:

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
`config/training.yaml`, and set the bag path in `config/training.yaml` before
launching.

Create a topomap and dataset at the same time:

```bash
roslaunch vnm_ros create_topomap_and_dataset.launch
```

For rosbag input, configure `runtime.yaml` and `training.yaml` before launching.

Collection interval, dataset paths, and training parameters are configured in
`config/training.yaml`. Topomap paths are configured in `config/runtime.yaml`.

Visualize the latest training trajectory in RViz:

```bash
roslaunch vnm_ros visualize_dataset.launch
```

The launch displays the recorded images, full XY trajectory, and current pose.
Select a test trajectory, a specific trajectory name, or playback rate in
`config/runtime.yaml`.

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

Checkpoints are written directly to `weights/` so the navigation launch files can
use them immediately:

```text
weights/latest.pth
weights/best.pth
```

Metrics and TensorBoard logs are written to:

```text
runs/20260611_153000_123456/metrics.jsonl
runs/20260611_153000_123456/tensorboard/
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
