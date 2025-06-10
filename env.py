import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import AECEnv
from gymnasium.utils import seeding
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector


def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env

    return env


class SimpleEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 30,
    }

    def __init__(
        self,
        scenario,
        world,
        max_cycles,
        render_mode=None,
        continuous_actions=False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions

        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((world.width, world.height))
            self.clock = pygame.time.Clock()

        self._seed()

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {agent.name: i for i, agent in enumerate(self.world.agents)}
        self._agent_selector = AgentSelector(self.agents)

        self.action_spaces = {}
        self.observation_spaces = {}
        for agent in self.world.agents:
            if self.continuous_actions:
                action_shape = (self.world.dim_p,)
                self.action_spaces[agent.name] = spaces.Box(
                    low=-1, high=+1, shape=action_shape
                )
            else:
                self.action_spaces[agent.name] = spaces.Discrete(5)

            obs_dim = len(self.scenario.observation(agent, self.world))
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32
            )

        self.reset()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.possible_agents = [agent.name for agent in self.world.agents]
        self.agents = self.possible_agents[:]

        self._index_map = {agent.name: i for i, agent in enumerate(self.world.agents)}

        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.current_actions = [None] * len(self.agents)
        self.steps = 0

        if self.agents:
            self.agent_selection = self._agent_selector.reset()
        else:
            self.agent_selection = None

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        cur_agent_idx = self._index_map[self.agent_selection]
        self.current_actions[cur_agent_idx] = action

        if self._agent_selector.is_last():
            for i, agent in enumerate(self.world.agents):
                agent_action = self.current_actions[i]
                move_vector = np.zeros(self.world.dim_p)
                if agent_action is not None:
                    if self.continuous_actions:
                        move_vector = np.clip(agent_action, -1.0, +1.0)
                    else:  # 对于离散动作，进行映射
                        if agent_action == 1:
                            move_vector[1] = -1.0  # 上
                        elif agent_action == 2:
                            move_vector[1] = +1.0  # 下
                        elif agent_action == 3:
                            move_vector[0] = -1.0  # 左
                        elif agent_action == 4:
                            move_vector[0] = +1.0  # 右
                agent.action.u = move_vector

            self.world.step()

            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True

            active_agents = []
            agents_to_remove = []
            for agent in self.world.agents:
                if hasattr(agent, "caught") and agent.caught:
                    agents_to_remove.append(agent.name)
                else:
                    active_agents.append(agent)
                    self.rewards[agent.name] = self.scenario.reward(agent, self.world)

            for agent_name in agents_to_remove:
                if agent_name in self.agents:
                    self.agents.remove(agent_name)
                for state_dict in [
                    self.rewards,
                    self._cumulative_rewards,
                    self.terminations,
                    self.truncations,
                    self.infos,
                ]:
                    state_dict.pop(agent_name, None)

            self.world.agents = active_agents

            if self.agents:
                self.agents.sort()
                self._index_map = {
                    agent.name: i for i, agent in enumerate(self.world.agents)
                }
                self._agent_selector = AgentSelector(self.agents)
            else:
                self._index_map = {}
                self._agent_selector = None

            if self.agent_selection not in self.agents and self.agents:
                self.agent_selection = self._agent_selector.next()
            elif not self.agents:
                self.agent_selection = None

            self.current_actions = [None] * len(self.agents)

            if not any(not agent.is_seeker for agent in self.world.agents):
                for a in self.agents:
                    self.truncations[a] = True
                    self.terminations[a] = True
                self._accumulate_rewards()
                return

            self._accumulate_rewards()

        if len(self.agents) > 0:
            self.agent_selection = self._agent_selector.next()
        else:
            self.agent_selection = None

        if self.render_mode == "human":
            self.render()

    def draw(self):
        self.screen.fill((240, 240, 240))

        for obstacle in self.world.obstacles:
            color = tuple((np.clip(obstacle.color, 0, 1) * 255).astype(int))
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(obstacle.x, obstacle.y, obstacle.width, obstacle.height),
            )

        for agent in self.world.agents:
            rgb_color = tuple((np.clip(agent.color, 0, 1) * 255).astype(int))
            position = (int(agent.state.p_pos[0]), int(agent.state.p_pos[1]))
            pygame.draw.circle(self.screen, rgb_color, position, int(agent.radius))

    def render(self):
        if self.render_mode is None:
            return

        self.draw()
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
