import logging
import numpy as np
import pygame

from gymnasium.spaces import Box
from gymnasium.utils import EzPickle, seeding

from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test
from pettingzoo.utils.conversions import parallel_wrapper_fn
from agents import Seeker, Hider
from obstacles import Obstacle
import constants as const

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def env(render_mode=None, **kwargs):
    initial_position_arg = kwargs.pop("initial_position", None)
    final_kwargs = {
        "world_size": const.WORLD_SIZE,
        "max_cycles": const.MAX_CYCLES,
        "num_seekers": const.NUM_SEEKERS,
        "num_hiders": const.NUM_HIDERS,
        "agent_radius": const.AGENT_RADIUS,
        "seeker_speed": const.SEEKER_SPEED,
        "hider_speed": const.HIDER_SPEED,
        "catch_radius_multiplier": const.CATCH_RADIUS_MULTIPLIER,
        **kwargs,
    }
    env_instance = raw_env(
        render_mode=render_mode, initial_position=initial_position_arg, **final_kwargs
    )
    return env_instance


parallel_env = parallel_wrapper_fn(env)


class raw_env(ParallelEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "multi_agent_hide_and_seek",
        "render_fps": const.RENDER_FPS,
    }

    def __init__(
        self,
        world_size=const.WORLD_SIZE,
        max_cycles=const.MAX_CYCLES,
        render_mode=None,
        num_seekers=const.NUM_SEEKERS,
        num_hiders=const.NUM_HIDERS,
        agent_radius=const.AGENT_RADIUS,
        seeker_speed=const.SEEKER_SPEED,
        hider_speed=const.HIDER_SPEED,
        catch_radius_multiplier=const.CATCH_RADIUS_MULTIPLIER,
        initial_position=None,
    ):
        EzPickle.__init__(
            self,
            world_size=world_size,
            max_cycles=max_cycles,
            render_mode=render_mode,
            num_seekers=num_seekers,
            num_hiders=num_hiders,
            agent_radius=agent_radius,
            seeker_speed=seeker_speed,
            hider_speed=hider_speed,
            catch_radius_multiplier=catch_radius_multiplier,
            initial_position=initial_position,
        )

        self.world_width, self.world_height = world_size
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.num_seekers = num_seekers
        self.num_hiders = num_hiders
        self.agent_radius = float(agent_radius)
        self.seeker_speed = float(seeker_speed)
        self.hider_speed = float(hider_speed)
        self.catch_distance = self.agent_radius * 2 * catch_radius_multiplier
        self.obstacles = []
        self._initialize_obstacles()
        self.initial_position_config = (
            initial_position if initial_position is not None else {}
        )
        self.current_episode_num = 0
        self.font = None

        logger.info(
            f"Initializing {self.metadata['name']}: world_size={world_size}, "
            f"num_seekers={num_seekers}, num_hiders={num_hiders}, agent_radius={self.agent_radius}"
        )

        self.seeker_ids = [f"seeker_{i}" for i in range(self.num_seekers)]
        self.hider_ids = [f"hider_{i}" for i in range(self.num_hiders)]
        self.possible_agents = self.seeker_ids + self.hider_ids
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agents = []

        self.agent_objs = {}
        for agent_id in self.seeker_ids:
            self.agent_objs[agent_id] = Seeker(
                agent_id,
                self.agent_radius,
                self.seeker_speed,
                self.world_width,
                self.world_height,
            )
        for agent_id in self.hider_ids:
            self.agent_objs[agent_id] = Hider(
                agent_id,
                self.agent_radius,
                self.hider_speed,
                self.world_width,
                self.world_height,
            )

        self._action_spaces = {
            agent: Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self._hiders_caught_status = {}
        self._current_cycle = 0

        self.screen = None
        self.clock = None
        self.np_random = None

        num_obstacle_features = 0
        if self.obstacles:
            num_obstacle_features = 4 * len(self.obstacles)

        obs_dim = (
            2 * len(self.possible_agents) + len(self.hider_ids) + num_obstacle_features
        )
        self._observation_spaces = {
            agent: Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def _initialize_obstacles(self):
        self.obstacles = []
        self.obstacles.append(Obstacle(330, 150, 400, 20))
        self.obstacles.append(Obstacle(200, 150, 70, 20))
        self.obstacles.append(Obstacle(200, 170, 20, 50))
        self.obstacles.append(Obstacle(200, 280, 20, 230))

    def _render_frame(self, to_rgb_array=False):
        if self.render_mode == "human" and not to_rgb_array:
            if self.screen is None:
                pygame.init()
                pygame.display.init()
                if pygame.font:
                    self.font = pygame.font.SysFont(None, const.FONT_SIZE)
                else:
                    self.font = None

                self.screen = pygame.display.set_mode(
                    (int(self.world_width), int(self.world_height))
                )
                pygame.display.set_caption(self.metadata["name"])
            if self.clock is None:
                self.clock = pygame.time.Clock()

        canvas = pygame.Surface((int(self.world_width), int(self.world_height)))
        canvas.fill(const.BACKGROUND_COLOR)

        for obs in self.obstacles:
            pygame.draw.rect(canvas, obs.color, obs.rect)

        for agent_id, agent_obj in self.agent_objs.items():
            if agent_id in self.agents:
                color = const.SEEKER_COLOR if agent_obj.is_seeker else const.HIDER_COLOR
                if not agent_obj.is_seeker and self._hiders_caught_status.get(
                    agent_id, False
                ):
                    continue
                pygame.draw.circle(
                    canvas,
                    color,
                    (int(agent_obj.position[0]), int(agent_obj.position[1])),
                    int(agent_obj.radius),
                )

        if self.render_mode == "human" and not to_rgb_array and self.font:
            episode_text = f"Episode: {self.current_episode_num}"
            text_surface = self.font.render(episode_text, True, (10, 10, 10))
            canvas.blit(text_surface, (10, 10))

        if self.render_mode == "human" and not to_rgb_array:
            if self.screen:
                self.screen.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()
            if self.clock:
                self.clock.tick(self.metadata["render_fps"])

        if to_rgb_array:
            return np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2))
        return None

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame(to_rgb_array=True)
        elif self.render_mode == "human":
            self._render_frame()
            return None
        return None

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
        logger.info("Closed.")

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reinit_game_state(self):
        self.agents = self.possible_agents[:]
        self._current_cycle = 0
        self._hiders_caught_status = {hider: False for hider in self.hider_ids}

        placed_agent_specs = {}

        for agent_id in self.possible_agents:
            agent_obj = self.agent_objs[agent_id]
            placed_this_agent = False

            if agent_id in self.initial_position_config:
                pos_config = self.initial_position_config[agent_id]
                candidate_pos = np.array(pos_config, dtype=np.float32)

                candidate_pos[0] = np.clip(
                    candidate_pos[0],
                    agent_obj.radius,
                    self.world_width - agent_obj.radius,
                )
                candidate_pos[1] = np.clip(
                    candidate_pos[1],
                    agent_obj.radius,
                    self.world_height - agent_obj.radius,
                )
                agent_obj.set_position(candidate_pos)
                logger.info(
                    f"Agent {agent_id} placed at specified position: {agent_obj.position}"
                )

            else:
                placed = False
                attempts = 0
                while not placed and attempts < const.AGENT_PLACEMENT_ATTEMPTS:
                    attempts += 1
                    candidate_pos_array = np.array(
                        [
                            self.np_random.uniform(
                                agent_obj.radius, self.world_width - agent_obj.radius
                            ),
                            self.np_random.uniform(
                                agent_obj.radius, self.world_height - agent_obj.radius
                            ),
                        ],
                        dtype=np.float32,
                    )
                    overlap_with_placed_agent = False
                    for other_pos_tuple in placed_agent_specs.values():
                        if (
                            np.linalg.norm(
                                candidate_pos_array - np.array(other_pos_tuple)
                            )
                            < agent_obj.radius * const.AGENT_PLACEMENT_OVERLAP_FACTOR
                        ):
                            overlap_with_placed_agent = True
                            break
                    if overlap_with_placed_agent:
                        continue

                    overlap_with_obstacle = False

                    agent_rect_potential = pygame.Rect(
                        candidate_pos_array[0] - agent_obj.radius,
                        candidate_pos_array[1] - agent_obj.radius,
                        agent_obj.radius * 2,
                        agent_obj.radius * 2,
                    )
                    for obs in self.obstacles:
                        if agent_rect_potential.colliderect(obs.rect):
                            overlap_with_obstacle = True
                            break

                    if overlap_with_obstacle:
                        continue

                    agent_obj.set_position(candidate_pos_array)
                    placed_agent_specs[agent_id] = tuple(candidate_pos_array)
                    placed = True

                if not placed:
                    logger.warning(
                        f"Could not place agent {agent_id} without overlap (randomly). Placing at default or potentially overlapping."
                    )
                    fallback_pos = np.array(
                        [
                            self.np_random.uniform(
                                agent_obj.radius, self.world_width - agent_obj.radius
                            ),
                            self.np_random.uniform(
                                agent_obj.radius, self.world_height - agent_obj.radius
                            ),
                        ],
                        dtype=np.float32,
                    )
                    agent_obj.set_position(fallback_pos)

        logger.info(f"Environment reinitialized. Cycle: {self._current_cycle}")
        for agent_id in self.possible_agents:
            logger.info(
                f"  Agent {agent_id} final position: {self.agent_objs[agent_id].position}"
            )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed)
        elif self.np_random is None:
            self._seed()

        self.reinit_game_state()
        observations = self._get_obs()
        infos = self._get_infos()

        if self.render_mode == "human":
            self._render_frame()
        return observations, infos

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def _normalize_pos(self, pos_vec):
        return np.array(
            [pos_vec[0] / self.world_width, pos_vec[1] / self.world_height],
            dtype=np.float32,
        )

    def _get_obs(self):
        all_agent_pos_norm_flat = []
        for agent_id in self.possible_agents:
            agent_obj = self.agent_objs[agent_id]
            pos_norm = self._normalize_pos(agent_obj.position)
            all_agent_pos_norm_flat.extend(pos_norm)

        hider_statuses_flat = [
            1.0 if self._hiders_caught_status[h_id] else 0.0 for h_id in self.hider_ids
        ]

        obstacle_features_flat = []
        for obs in self.obstacles:
            norm_x = obs.x / self.world_width
            norm_y = obs.y / self.world_height
            norm_w = obs.width / self.world_width
            norm_h = obs.height / self.world_height
            obstacle_features_flat.extend([norm_x, norm_y, norm_w, norm_h])

        base_obs_vector = np.array(
            all_agent_pos_norm_flat + hider_statuses_flat + obstacle_features_flat,
            dtype=np.float32,
        )

        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = base_obs_vector.copy()
        return observations

    def _get_infos(self):
        infos = {}
        for agent_id in self.agents:
            agent_obj = self.agent_objs[agent_id]
            infos[agent_id] = {
                "hiders_remaining": sum(
                    not s for s in self._hiders_caught_status.values()
                ),
                "position": agent_obj.position.tolist(),
                "cycle": self._current_cycle,
                "is_seeker": agent_obj.is_seeker,
                "is_caught": self._hiders_caught_status.get(agent_id, None)
                if not agent_obj.is_seeker
                else None,
            }
        return infos

    def _move_agent(self, agent_id, action_vector, other_agents_for_collision):
        agent_obj = self.agent_objs[agent_id]
        old_pos = agent_obj.position.copy()

        agent_obj.move_and_clip(
            action_vector, self.obstacles, other_agents_for_collision
        )

        if not np.array_equal(old_pos, agent_obj.position):
            logger.debug(
                f"  Env: Agent {agent_id} action: {np.around(action_vector, 2).tolist()}, from {np.around(old_pos, 1).tolist()} to {np.around(agent_obj.position, 1).tolist()}"
            )

        logger.debug(
            f"  Env: Agent {agent_id} action: {action_vector}, from {old_pos.round(1)} to {agent_obj.position.round(1)}"
        )

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        self._current_cycle += 1
        logger.info(f"--- Cycle {self._current_cycle}/{self.max_cycles} ---")

        rewards = {agent: 0.0 for agent in self.agents}
        agents_at_step_start = self.agents[:]

        for agent_id, action in actions.items():
            if agent_id in self.agents:
                other_agents_for_collision_check = [
                    other_agent_obj
                    for other_id, other_agent_obj in self.agent_objs.items()
                    if other_id != agent_id and other_id in self.possible_agents
                ]
                self._move_agent(agent_id, action, other_agents_for_collision_check)

        for seeker_id in self.seeker_ids:
            if seeker_id not in self.agents:
                continue
            seeker_obj = self.agent_objs[seeker_id]
            for hider_id in self.hider_ids:
                if hider_id in self.agents and not self._hiders_caught_status[hider_id]:
                    hider_obj = self.agent_objs[hider_id]
                    distance = np.linalg.norm(seeker_obj.position - hider_obj.position)
                    if distance < self.catch_distance:
                        self._hiders_caught_status[hider_id] = True
                        rewards[seeker_id] += const.SEEKER_CATCH_REWARD
                        rewards[hider_id] += const.HIDER_CAUGHT_PENALTY
                        logger.info(
                            f"  CATCH! {seeker_id} caught {hider_id} at dist: {distance:.2f}"
                        )

        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}

        all_hiders_caught_or_terminated = True
        active_hider_count = 0
        for h_id in self.hider_ids:
            if h_id in agents_at_step_start:
                active_hider_count += 1
                if self._hiders_caught_status[h_id]:
                    terminations[h_id] = True
                else:
                    all_hiders_caught_or_terminated = False

        if active_hider_count == 0 and self.num_hiders > 0:
            all_hiders_caught_or_terminated = True
        elif self.num_hiders == 0:
            all_hiders_caught_or_terminated = True

        time_up = self._current_cycle >= self.max_cycles
        episode_is_over = all_hiders_caught_or_terminated or time_up

        if episode_is_over:
            reason = (
                "all hiders caught/terminated"
                if all_hiders_caught_or_terminated
                else "time up"
            )
            logger.info(f"  Episode ending: {reason}")
            for agent_id in agents_at_step_start:
                if time_up and not all_hiders_caught_or_terminated:
                    truncations[agent_id] = True
                else:
                    terminations[agent_id] = True

            if all_hiders_caught_or_terminated:
                for seeker_id in self.seeker_ids:
                    if seeker_id in agents_at_step_start:
                        rewards[seeker_id] += const.SEEKER_ALL_HIDERS_CAUGHT_BONUS
            if time_up and not all_hiders_caught_or_terminated:
                for h_id in self.hider_ids:
                    if (
                        h_id in agents_at_step_start
                        and not self._hiders_caught_status[h_id]
                    ):
                        rewards[h_id] += const.HIDER_SURVIVED_BONUS

        for seeker_id in self.seeker_ids:
            if seeker_id in agents_at_step_start:
                rewards[seeker_id] += const.SEEKER_TIMESTEP_PENALTY

        active_seeker_objs = [
            self.agent_objs[s_id]
            for s_id in self.seeker_ids
            if s_id in agents_at_step_start
        ]
        for h_id in self.hider_ids:
            if h_id in agents_at_step_start and not self._hiders_caught_status[h_id]:
                hider_obj = self.agent_objs[h_id]
                if active_seeker_objs:
                    min_dist_to_seeker = min(
                        np.linalg.norm(hider_obj.position - s_obj.position)
                        for s_obj in active_seeker_objs
                    )
                    rewards[h_id] += const.HIDER_DISTANCE_REWARD_FACTOR * (
                        min_dist_to_seeker / max(self.world_width, self.world_height)
                    )
                else:
                    rewards[h_id] += const.HIDER_SAFE_REWARD

        next_active_agents = []
        for agent_id in agents_at_step_start:
            if not (terminations[agent_id] or truncations[agent_id]):
                next_active_agents.append(agent_id)
        self.agents = next_active_agents

        observations = self._get_obs()
        infos = self._get_infos()

        logger.debug(f"  Rewards: {rewards}")
        logger.info(f"  Hiders caught: {self._hiders_caught_status}")
        logger.info(f"  Terminations: {terminations}, Truncations: {truncations}")
        logger.info(f"  Active agents for next step: {self.agents}")

        if self.render_mode == "human":
            self._render_frame()

        final_rewards = {
            agent: rewards.get(agent, 0.0) for agent in agents_at_step_start
        }

        if not self.agents:
            return {}, final_rewards, terminations, truncations, {}
        return observations, final_rewards, terminations, truncations, infos

    def state(self):
        state_positions = [
            self.agent_objs[agent_id].position.tolist()
            for agent_id in self.possible_agents
        ]
        state_caught_status = [
            1.0 if self._hiders_caught_status[h_id] else 0.0 for h_id in self.hider_ids
        ]
        return {"positions": state_positions, "caught_status": state_caught_status}


HideAndSeekEnv = raw_env

if __name__ == "__main__":
    logger.info("--- Running PettingZoo parallel_api_test ---")

    env_test_params = {
        "world_size": (200, 150),
        "max_cycles": 60,
        "num_seekers": const.NUM_SEEKERS,
        "num_hiders": 1,
        "agent_radius": 10.0,
    }

    test_env = env(render_mode=None, **env_test_params)

    try:
        parallel_api_test(test_env, num_cycles=200)
        logger.info("--- PettingZoo parallel_api_test passed ---")
    except Exception as e:
        logger.error(
            f"--- PettingZoo parallel_api_test FAILED : {e} ---",
            exc_info=True,
        )
    finally:
        test_env.close()

    logger.info("\n--- Starting manual interaction loop  ---")

    agents_initial_positions = {
        "seeker_0": (50, 100),
        "seeker_1": (100, 50),
        "hider_0": (500, 400),
        "hider_1": (400, 300),
    }

    env_instance = env(
        render_mode="human",
        max_cycles=1000,
        num_seekers=2,
        num_hiders=2,
        agent_radius=20,
        world_size=(700, 500),
        seeker_speed=6,
        hider_speed=5,
        initial_position=agents_initial_positions,
    )

    num_episodes = 9999

    for episode in range(num_episodes):
        logger.info(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        observations, infos = env_instance.reset(
            seed=42 + episode if episode < 10 else None
        )

        for step_count in range(env_instance.max_cycles + 10):
            if not env_instance.agents:
                logger.info(
                    f"Episode {episode + 1} finished at step {step_count} (all agents done)."
                )
                break

            actions = {
                agent_id: env_instance.action_space(agent_id).sample()
                for agent_id in env_instance.agents
            }

            next_observations, rewards, terminations, truncations, next_infos = (
                env_instance.step(actions)
            )

            for agent_id in actions.keys():
                if (
                    rewards.get(agent_id, 0) != 0
                    or terminations.get(agent_id, False)
                    or truncations.get(agent_id, False)
                ):
                    logger.debug(
                        f"  Agent {agent_id}: Reward={rewards.get(agent_id, 0):.2f}, "
                        f"Term={terminations.get(agent_id, False)}, Trunc={truncations.get(agent_id, False)}, "
                        f"Info: {next_infos.get(agent_id, {})}"
                    )
            observations = next_observations
            infos = next_infos
            if step_count >= env_instance.max_cycles - 1:
                logger.info(
                    f"Episode {episode + 1} reached max_cycles ({env_instance.max_cycles}) at step {step_count + 1}."
                )

        if env_instance.agents:
            logger.warning(
                f"Manual loop for episode {episode + 1} ended by step limit, but agents {env_instance.agents} might still be active."
            )
        elif step_count < env_instance.max_cycles:
            logger.info(f"Episode {episode + 1} ended early at step {step_count}.")

    env_instance.close()
    logger.info("--- Manual interaction loop finished ---")
