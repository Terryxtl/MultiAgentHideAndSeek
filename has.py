import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn

from env import SimpleEnv, make_env
from scenario import BaseScenario
from core import Agent, World, Obstacle


OBSTACLE_CONFIG = [
    Obstacle(380, 200, 800 - 330, 20),
    Obstacle(250, 200, 70, 20),
    Obstacle(250, 220, 20, 50),
    Obstacle(250, 330, 20, 600 - 280),
]


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_hider=2,
        num_seeker=2,
        obstacle_config=None,
        max_cycles=100,
        continuous_actions=True,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            num_hider=num_hider,
            num_seeker=num_seeker,
            max_cycles=max_cycles,
            obstacle_config=obstacle_config,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_hider, num_seeker, obstacle_config)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "multi_agent_hide_and_seek"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_hider=2, num_seeker=2, obstacle_config=None):
        world = World(width=800, height=600)
        world.dim_c = 2
        world.num_seekers = num_seeker
        world.num_hiders = num_hider

        num_agents = num_seeker + num_hider

        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.is_seeker = True if i < num_seeker else False
            base_name = "seeker" if agent.is_seeker else "hider"
            base_index = i if i < num_seeker else i - num_seeker
            agent.name = f"{base_name}_{base_index}"
            agent.radius = 20.0
            agent.silent = True
            agent.accel = 3000 if agent.is_seeker else 4000
            agent.max_speed = 10000 if agent.is_seeker else 12000

        if obstacle_config is None:
            world.obstacles = OBSTACLE_CONFIG
        else:
            world.obstacles = obstacle_config

        world.num_agents_initial = len(world.agents)

        for obstacle in world.obstacles:
            obstacle.color = np.array([0.5, 0.5, 0.5])

        return world

    def reset_world(self, world, np_random):
        num_seekers = world.num_seekers
        num_hiders = world.num_hiders

        world.agents = []
        for i in range(num_seekers + num_hiders):
            agent = Agent()
            agent.is_seeker = i < num_seekers
            base_name = "seeker" if agent.is_seeker else "hider"
            base_index = i if agent.is_seeker else i - num_seekers
            agent.name = f"{base_name}_{base_index}"
            agent.radius = 20.0
            agent.silent = True
            agent.accel = 3000 if agent.is_seeker else 4000
            agent.max_speed = 10000 if agent.is_seeker else 12000
            world.agents.append(agent)

        world_width = world.width
        world_height = world.height

        for agent in world.agents:
            agent.color = (
                np.array([0.85, 0.35, 0.35])
                if agent.is_seeker
                else np.array([0.35, 0.85, 0.35])
            )
            agent.caught = False

        for obstacle in world.obstacles:
            obstacle.color = np.array([0.5, 0.5, 0.5])

        def is_agent_overlapping(new_agent, existing_agents):
            for existing_agent in existing_agents:
                dist = np.linalg.norm(
                    new_agent.state.p_pos - existing_agent.state.p_pos
                )

                if dist < (new_agent.radius + existing_agent.radius) * 1.2:
                    return True
            return False

        def is_obstacle_overlapping(agent, obstacles):
            for obs in obstacles:
                obs_left = obs.x
                obs_right = obs.x + obs.width
                obs_top = obs.y
                obs_bottom = obs.y + obs.height

                closest_x = max(obs_left, min(agent.state.p_pos[0], obs_right))
                closest_y = max(obs_top, min(agent.state.p_pos[1], obs_bottom))

                distance = np.sqrt(
                    (agent.state.p_pos[0] - closest_x) ** 2
                    + (agent.state.p_pos[1] - closest_y) ** 2
                )

                if distance < agent.radius:
                    return True
            return False

        seekers = self.seekers(world)
        hiders = self.hiders(world)

        placed_seekers = []
        for agent in seekers:
            is_safe_pos = False
            while not is_safe_pos:
                x = np_random.uniform(agent.radius, world_width * 0.4)
                y = np_random.uniform(agent.radius, world_height * 0.4)
                agent.state.p_pos = np.array([x, y], dtype=np.float32)

                if not is_agent_overlapping(
                    agent, placed_seekers
                ) and not is_obstacle_overlapping(agent, world.obstacles):
                    is_safe_pos = True

            agent.state.p_vel = np.zeros(world.dim_p, dtype=np.float32)
            agent.state.c = np.zeros(world.dim_c, dtype=np.float32)
            placed_seekers.append(agent)

        placed_hiders = []
        for agent in hiders:
            is_safe_pos = False
            while not is_safe_pos:
                x = np_random.uniform(world_width * 0.6, world_width - agent.radius)
                y = np_random.uniform(world_height * 0.6, world_height - agent.radius)
                agent.state.p_pos = np.array([x, y], dtype=np.float32)

                if not is_agent_overlapping(
                    agent, placed_hiders
                ) and not is_obstacle_overlapping(agent, world.obstacles):
                    is_safe_pos = True

            agent.state.p_vel = np.zeros(world.dim_p, dtype=np.float32)
            agent.state.c = np.zeros(world.dim_c, dtype=np.float32)
            placed_hiders.append(agent)

    def benchmark_data(self, agent, world):
        if agent.is_seeker:
            collisions = 0
            for a in self.hiders(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        dist = np.linalg.norm(agent1.state.p_pos - agent2.state.p_pos)
        return dist < (agent1.radius + agent2.radius)

    def hiders(self, world):
        return [agent for agent in world.agents if not agent.is_seeker]

    def seekers(self, world):
        return [agent for agent in world.agents if agent.is_seeker]

    def reward(self, agent, world):
        main_reward = (
            self.seeker_reward(agent, world)
            if agent.is_seeker
            else self.hider_reward(agent, world)
        )
        return main_reward

    def hider_reward(self, agent, world):
        rew = 0
        shape = False
        seekers = self.seekers(world)
        if shape:
            for s in seekers:
                rew += 0.1 * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - s.state.p_pos))
                )

        if agent.collide:
            for s in seekers:
                if self.is_collision(s, agent):
                    rew = -10
                    agent.caught = True

        return rew

    def seeker_reward(self, agent, world):
        rew = 0
        shape = False
        hiders = self.hiders(world)
        seekers = self.seekers(world)
        if shape:
            for s in seekers:
                rew -= 0.1 * min(
                    np.sqrt(np.sum(np.square(h.state.p_pos - s.state.p_pos)))
                    for h in hiders
                )
        if agent.collide:
            for h in hiders:
                for s in seekers:
                    if self.is_collision(h, s):
                        rew += 10
        return rew

    def observation(self, agent, world):
        num_obstacles = len(world.obstacles)
        max_agents = world.num_agents_initial

        obs_size = 4 + 4 * num_obstacles + 4 * (max_agents - 1)
        obs = np.zeros(obs_size, dtype=np.float32)

        obs[0:2] = agent.state.p_vel
        obs[2:4] = agent.state.p_pos

        obs_idx = 4
        for i, obstacle in enumerate(world.obstacles):
            if i >= num_obstacles:
                break
            obs[obs_idx] = obstacle.x - agent.state.p_pos[0]
            obs[obs_idx + 1] = obstacle.y - agent.state.p_pos[1]
            obs[obs_idx + 2] = obstacle.width
            obs[obs_idx + 3] = obstacle.height
            obs_idx += 4

        other_agents = [a for a in world.agents if a != agent]
        for i, other in enumerate(other_agents):
            if i >= max_agents - 1:
                break
            obs[obs_idx] = other.state.p_pos[0] - agent.state.p_pos[0]
            obs[obs_idx + 1] = other.state.p_pos[1] - agent.state.p_pos[1]
            if not other.is_seeker:
                obs[obs_idx + 2] = other.state.p_vel[0]
                obs[obs_idx + 3] = other.state.p_vel[1]
            else:
                obs[obs_idx + 2 : obs_idx + 4] = 0
            obs_idx += 4

        return obs
