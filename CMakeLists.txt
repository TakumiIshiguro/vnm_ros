cmake_minimum_required(VERSION 3.0.2)
project(vnm_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  scenario_navigation_msgs
  std_msgs
  std_srvs
  geometry_msgs
  nav_msgs
  visualization_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/collect_dataset_nav_recovery.py
  scripts/create_dataset.py
  scripts/create_topomap.py
  scripts/eval.py
  scripts/keyboard_cmd_dir.py
  scripts/overlay_view.py
  scripts/pkl_to_csv.py
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

install(FILES
  LICENSE
  THIRD_PARTY_NOTICES.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
