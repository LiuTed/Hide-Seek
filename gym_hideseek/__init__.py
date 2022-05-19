from gym.envs.registration import register

register(
    id = 'Hider-v0',
    entry_point = 'gym_hideseek.env:Hider',
    max_episode_steps = 1000
)

register(
    id = 'Seeker-v0',
    entry_point = 'gym_hideseek.env:Seeker',
    max_episode_steps = 2000
)
