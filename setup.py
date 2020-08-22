from setuptools import setup, find_packages

setup(
    name='thing_gym_ros',
    version='0.0.1',
    description='Package for getting data from the Thing robot (in the real world or through gazebo)'
                'in a OpenAI gym-friendly way. Requires regular ros tools to be installed in system wide'
                'directory (/opt/ros/DIST/...).',
    author='Trevor Ablett',
    author_email='trevor.ablett@robotics.utias.utoronto.ca',
    license='MIT',
    packages=find_packages(),
    install_requires=['rospkg', 'rosdep', 'ros_np_tools'],
    include_package_data=True
)
