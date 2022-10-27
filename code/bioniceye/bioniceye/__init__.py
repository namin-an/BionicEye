from gym.envs.registration import register

register(
    id='BionicEyeEnv-v0',
    entry_point='bioniceye.envs:BionicEyeEnv_v0',
)

register(
    id='BionicEyeEnv-v1',
    entry_point='bioniceye.envs:BionicEyeEnv_v1',
)
