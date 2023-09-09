# ABYSS Coding Challenge

This ROS package provides a node `image_fusion_node.py` that takes a image topics and merges them into a concatenated image.

The image fusion node is located in the `scripts` directory.

While a homography blend was trialed, I wasnt able to debug it in the time required.

## Install dependencies

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git-all
```

## Setting up ROS

```bash
# Set up source list
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Set up keys
sudo apt install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Installation
sudo apt-get update
sudo apt install ros-noetic-desktop-full

# Environment setup  if using bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Environment setup if using zsh
echo "source /opt/ros/noetic/setup.zsh" >> ~/.zshrc
source ~/.zshrc

# Dependencies for building packages
sudo apt-get install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential python3-catkin-tools
sudo apt install python3-rosdep
sudo rosdep init
rosdep update
```

## Setting up ROS workspace

```bash
mkdir ~/catkin_ws
cd ~/catkin_ws
mkdir src
cd src
# Initialise workspace
catkin_init_workspace
git clone <this-repo>
cd ~/catkin_ws

# Build all packages
catkin_make

# Environment setup if using bash
echo "source $(pwd)/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Environment setup if using zsh
echo "source $(pwd)/devel/setup.zsh" >> ~/.zshrc
source ~/.zshrc
```
