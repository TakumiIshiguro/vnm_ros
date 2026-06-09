cmake_minimum_required(VERSION 3.0.2)
project(vnm_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/create_dataset.py
  scripts/create_topomap.py
  scripts/eval.py
  scripts/overlay_view.py
  scripts/train.py
  scripts/vnm_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  config
  weights
  topomaps
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
