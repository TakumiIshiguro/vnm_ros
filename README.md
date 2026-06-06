# vnm_ros

ROS inference package for Visual Navigation Models based on `visualnav-transformer`.

This package intentionally contains no training code. It records image topological
maps, loads a pre-trained ViNT/GNM-style checkpoint, predicts local waypoints from
the current camera image and topomap, and optionally publishes velocity commands.

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

Run navigation:

```bash
roslaunch vnm_ros navigate.launch topomap_name:=topomap
```

Place checkpoints in `weights/`, for example:

```text
weights/vint.pth
```

