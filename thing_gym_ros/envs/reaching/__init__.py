from gym.envs.registration import registry, register, make, spec

# Visual
# -----------------------------------------------------------------------------------------------------------

# Reaching
# -----------------------------------------------------------------------------------------------------------

register(
    id='ThingRosReachAndGrasp6DOFImage-v0',
    entry_point='thing_gym_ros.envs.reaching.visual:ThingRosReachAndGrasp6DOFImage'
)

register(
    id='ThingRosReachAndGrasp6DOFImageMB-v0',
    entry_point='thing_gym_ros.envs.reaching.visual:ThingRosReachAndGrasp6DOFImageMB'
)