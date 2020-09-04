# thing-gym-ros
A repository containing gym-based envs built on ROS to be combined with the Thing mobile manipulation platform at UTIAS. 

## Requirements
This package is designed to be combined with the [thing ROS package](https://github.com/utiasSTARS/thing), which currently must be built and run in ROS Indigo and Ubuntu 14.04. Assuming that your computer is running a later version of Ubuntu, the thing packages can be built and run in a [docker image](https://github.com/trevorablett/rosdocked-thing/tree/nvidia-fixes).

The environments are tested with ROS Noetic and Ubuntu 20.04. Since ROS Noetic is built with python3, you can directly call these gym envs from your learning code built on python3.

## Installation
1. `thing_gym_ros` is technically a ROS package, but only to give the running code access to custom messages (and possibly services in the future). To ensure the thing environments have access to these custom components, this repository needs to be under a `src` folder in a catkin workspace (e.g. `~/catkin_ws/src`). Before you can use the environments here, you must call `catkin_make` and then `source devel/setup.bash`.

2. *(optional)* To reduce unnecessary warning messages and allow gripper control, under your `catkin_ws/src` folder containing `thing-gym-ros`, you also need to clone:

    a) the `warning-fix` branch of our [fork of geometry2](https://github.com/trevorablett/geometry2/tree/warning-fix).

2. If you're using virtualenv/conda for python, you should be okay as long as the following considerations are met:
    - Your `PYTHONPATH` should contain `/opt/ros/ros_dist/lib/python3/dist-packages`, by default this is done when you call `source /opt/ros/ros_dist/setup.bash`, which should be in your `.bashrc` after installing ROS.
    - Some python packages (`PyKDL`, `cv2`) are not installed in `/opt/ros/...`, but rather `/usr/lib/python3/dist-packages`. To access these libraries in your virtual/conda env, you need to install them manually (since adding `/usr/lib/python3/dist-packages` to your `PYTHONPATH` is a doesn't work):
        - *cv2*: `pip install cv2`
        - *PyKDL*: `pip install PyKDL` does NOT work, instead, install the custom wheel for your machine and python version from [here](https://rospypi.github.io/simple/pykdl/). Download the `.whl` file and install with `pip install`. Thanks to [@otamachan](https://github.com/otamachan) for setting this up.

3. In your learning python environment, install the package with
    ```
    pip install -e .
    ```
    The `-e` ([editable](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs)) option allows you to modify files without having to reinstall.

## Usage

1. **Sim**: (Probably in a docker image), bring up a thing simulation with a kinect:
    ```
    roslaunch thing_gazebo thing_world.launch gui:=False kinect:=True
    ```
   **Real**: TODO

2. In the catkin workspace where `thing-gym-ros` (the top-level folder of this repo) is contained, call `source devel/bash`.

3. In your learning code, import whatever environments you want from this package, for example:
    ```
    from thing_gym_ros.envs import ThingRosReaching
    ```
  
4. Create an environment in your learning code and use it as you would any other gym env:
    ```
    env = ThingRosReaching()
    obs = env.reset()
    obs, rew, done, _ = env.step(env.action_space.sample())
    ```
## A note on rotations
For internal calculations and action specifications, wxyz quaternions are used. However, for ease of interpretability, poses in config files are represented with 6 components, with the first 3 being the translational component relative to the reference frame, and the last 3 as euler angles specified relative to the static (extrinsic, non-rotating) reference frame, and rotated about x, y, z (i.e. 'sxyz' from tf.transformations or transforms3d, see [this reference](https://matthew-brett.github.io/transforms3d/reference/transforms3d.euler.html#specifying-angle-conventions) for more info).