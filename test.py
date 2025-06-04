from env2 import HideAndSeekEnv

env = HideAndSeekEnv(
    render_mode="human",
)
env.reset(seed=42)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
