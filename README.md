# thing-gym-ros
A repository containing gym-based envs built on ROS to be combined with the Thing mobile manipulation platform at UTIAS. 

## Requirements
This package is designed to be combined with the [thing ROS package](https://github.com/utiasSTARS/thing), which currently must be built and run in ROS Indigo and Ubuntu 14.04. Assuming that your computer is running a later version of Ubuntu, the thing packages can be built and run in a [docker image](https://github.com/trevorablett/rosdocked-thing/tree/nvidia-fixes).

The environments are tested with ROS Noetic and Ubuntu 20.04. Since ROS Noetic is built with python3, you can directly call these gym envs from your learning code built on python3.

## Dependencies
- [ROS](http://wiki.ros.org/noetic/Installation)
    - Currently only tested with Noetic on Ubuntu 20.04.
- [ros-np-tools](https://github.com/utiasSTARS/ros-np-tools)
- [realsense-ros](https://github.com/IntelRealSense/realsense-ros)

## Installation
1. `thing_gym_ros` is technically a ROS package, but only to give the running code access to custom messages (and possibly services in the future). To ensure the thing environments have access to these custom components, this repository needs to be under a `src` folder in a catkin workspace (e.g. `~/catkin_ws/src`). Before you can use the environments here, you must call `catkin_make` and then `source devel/setup.bash`.

2. *(optional)* To reduce unnecessary warning messages, under your `catkin_ws/src` folder containing `thing-gym-ros`, you also need to clone the `warning-fix` branch of our [fork of geometry2](https://github.com/trevorablett/geometry2/tree/warning-fix).

3. *(optional)* If you're using virtualenv/conda for python, you should be okay as long as the following considerations are met:
    - Your `PYTHONPATH` should contain `/opt/ros/ros_dist/lib/python3/dist-packages`, by default this is done when you call `source /opt/ros/ros_dist/setup.bash`, which should be in your `.bashrc` after installing ROS.
    - Some python packages are not installed in `/opt/ros/...`, but rather `/usr/lib/python3/dist-packages`. To access these libraries in your virtual/conda env, you need to install them manually (since adding `/usr/lib/python3/dist-packages` to your `PYTHONPATH` doesn't work, nor should it). Most are installed automatically via the `setup.py` requirements, but unfortunately the version of `PyKDL` automatically installed via `pip` is the wrong one, so you must install a custom wheel as follows:
    ```
    pip install --extra-index-url https://rospypi.github.io/simple/_pre PyKDL
    ``` 
    Thanks to [@otamachan](https://github.com/otamachan) for setting this up.

4. In your learning python environment, install the package with
    ```
    pip install -e .
    ```
    The `-e` ([editable](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs)) option allows you to modify files without having to reinstall.

## Usage

1. 
    **Sim**: (Probably in a docker image), bring up a thing simulation with a kinect:
    ```
    roslaunch thing_gazebo thing_world.launch gui:=False kinect:=True
    ```
    **Real**:
    
    a) Turn on the Thing robot, plug into computer (probably Monolith), connect via ssh, and run
    ```
    source trevor_ws/devel/setup.bash
    roslaunch thing_control ur10+FT+gripper_ros_control.launch
    ```
    Currently using `trevor_ws` which is on branch `feature/mobile-base-demo-collection` since it has diverged from the   master branch. See the [thing repository](https://github.com/utiasSTARS/thing) for more info.
    
    b) Plug realsense into computer (probably Monolith), and run (in another terminal)
    ```
    source REALSENSE_CATKIN_WS/devel/setup.bash
    roslaunch realsense2_camera rs_aligned_depth.launch
    ```
   
    c) *(optional for mobile base view-agnostic envs)* First calibrate the realsense to the UR10. See [instructions](https://github.com/utiasSTARS/thing/tree/feature/mobile-base-demo-collection/thing_utils). Then, **in your docker image with your thing installation**, run
    ```
    source THING_CATKIN_WS/devel/setup.bash
    roslaunch thing_utils realsense_publish_calibration.launch
    ```
   
   d) *(optional for viewing, though recommended for view-agnostic envs)* Launch rviz, **in your docker image**, with
   ```
   source THING_CATKIN_WS/devel/setup.bash
   roslaunch thing_utils thing_gym_ros_rviz.launch
   ```
   
2. With your system ROS installation (not in the docker-env), call `catkin_make` in the directory containing `src/thing-gym-ros`.

3. In the same folder, call `source devel/bash`.

4. In your learning code, import whatever environments you want from this package, for example:
    ```
    from thing_gym_ros.envs.reaching.visual import ThingRosReachAndGraspXYImageMB
    ```
  
5. Create an environment in your learning code and use it as you would any other gym env (ensuring you have completed step 3 in whatever environment you are calling this in):
    ```
    env = ThingRosReachAndGraspXYImageMB()
    obs = env.reset()
    obs, rew, done, _ = env.step(env.action_space.sample())
    ```
## A note on rotations
For internal calculations and action specifications, xyzw quaternions are used. However, for ease of interpretability, poses in config files are represented with 6 components, with the first 3 being the translational component relative to the reference frame, and the last 3 as euler angles specified relative to the static (extrinsic, non-rotating) reference frame, and rotated about x, y, z (i.e. 'sxyz' from tf.transformations or transforms3d, see [this reference](https://matthew-brett.github.io/transforms3d/reference/transforms3d.euler.html#specifying-angle-conventions) for more info).
