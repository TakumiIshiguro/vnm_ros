# vnm_ros

ROS inference package for Visual Navigation Models based on `visualnav-transformer`.

This package records datasets and topological maps, trains a ViNT model, evaluates
checkpoints, predicts local waypoints, and optionally publishes velocity commands.

## Requirements

- ROS Noetic
- Python 3
- `torch`, `torchvision`, `Pillow`, `numpy`, `PyYAML`
- `vint_train` importable from `visualnav-transformer/train`
- Model-specific packages such as `efficientnet-pytorch` for ViNT

Install the model package from the original repository:

```bash
cd /home/takumi/catkin_ws/src/visualnav-transformer
python3 -m pip install -e train/
```

Install ViNT runtime dependencies:

```bash
python3 -m pip install efficientnet-pytorch warmup-scheduler
```

## Files

- `scripts/create_topomap.py`: save camera frames as a topomap.
- `scripts/create_dataset.py`: save synchronized camera and odometry trajectories.
- `scripts/train.py`: train ViNT from a processed dataset.
- `scripts/eval.py`: evaluate a checkpoint on the automatically held-out data.
- `scripts/vnm_node.py`: load model, select subgoal, publish waypoint/cmd_vel.
- `config/topics.yaml`: ROS topic names and frame id.
- `config/robot.yaml`: control rate and velocity limits.
- `config/model.yaml`: model architecture and checkpoint path.
- `config/topomap.yaml`: topomap directory and subgoal selection settings.

## Usage

Create a topomap:

```bash
roslaunch vnm_ros create_topomap.launch topomap_name:=topomap
```

Create a topomap directly from a rosbag:

```bash
roslaunch vnm_ros create_topomap.launch \
  topomap_name:=topomap \
  bag_path:=/path/to/topomap.bag
```

Run navigation:

```bash
roslaunch vnm_ros navigate.launch topomap_name:=topomap
```

Place checkpoints in `weights/`, for example:

```text
weights/vint.pth
```

## Dataset collection

Set `image_topic` and `odometry_topic` in `config/topics.yaml`, then record one
training trajectory:

```bash
roslaunch vnm_ros create_dataset.launch \
  dataset:=train \
  trajectory_name:=traj_000
```

Stop the node with Ctrl-C. The resulting trajectory contains numbered images and
`traj_data.pkl`:

```text
dataset/my_dataset/train/traj_000/
```

Record test trajectories separately:

```bash
roslaunch vnm_ros create_dataset.launch \
  dataset:=test \
  trajectory_name:=traj_test_000
```

Create a dataset directly from a rosbag:

```bash
roslaunch vnm_ros create_dataset.launch \
  dataset:=train \
  trajectory_name:=traj_000 \
  bag_path:=/path/to/train.bag
```

Collection interval, dataset paths, and training parameters are configured in
`config/train.yaml`.

Bag paths can also be set in YAML. Use `config/topomap.yaml` for topomap
creation and `config/train.yaml` `collection.bag_path`, `train_bag_path`, or
`test_bag_path` for dataset creation.

## Training

```bash
rosrun vnm_ros train.py --config-dir $(rospack find vnm_ros)/config
```

Training and test datasets use separate directories:

```yaml
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
runs/my_vint/metrics.jsonl
runs/my_vint/tensorboard/
```

Start TensorBoard while training or after training:

```bash
tensorboard \
  --logdir "$(rospack find vnm_ros)/runs/my_vint/tensorboard" \
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
roslaunch vnm_ros navigate.launch checkpoint:=$(rospack find vnm_ros)/weights/best.pth
```
