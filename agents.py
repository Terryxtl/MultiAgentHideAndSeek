import numpy as np
import pygame


class Agent:
    def __init__(self, agent_id, radius, speed, world_width, world_height):
        self.agent_id = agent_id
        self.radius = float(radius)
        self.speed = float(speed)
        self.world_width = float(world_width)
        self.world_height = float(world_height)
        self.position = np.array([0.0, 0.0], dtype=np.float32)

    def set_position(self, position_array):
        self.position = np.array(position_array, dtype=np.float32)

    def move_and_clip(self, action_vector, obstacles_list=None, other_agents_list=None):
        direction = np.array(action_vector, dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 1e-4:
            direction = direction / norm
        else:
            direction = np.array([0.0, 0.0], dtype=np.float32)

        velocity = direction * self.speed
        old_pos = self.position.copy()
        candidate_pos = self.position + velocity

        allow_move = True

        agent_rect_config = pygame.Rect(
            float(candidate_pos[0] - self.radius),
            float(candidate_pos[1] - self.radius),
            float(self.radius * 2),
            float(self.radius * 2),
        )

        for obs in obstacles_list:
            if agent_rect_config.colliderect(obs.rect):
                allow_move = False
                break

        if allow_move and other_agents_list:
            for other_agent in other_agents_list:
                if other_agent.agent_id == self.agent_id:
                    continue

                dist_sq = (candidate_pos[0] - other_agent.position[0]) ** 2 + (
                    candidate_pos[1] - other_agent.position[1]
                ) ** 2

                sum_radii = self.radius + other_agent.radius
                min_dist_sq = sum_radii**2

                if dist_sq < min_dist_sq:
                    allow_move = False
                    break

        if allow_move:
            self.position = candidate_pos

        self.position[0] = np.clip(
            self.position[0], self.radius, self.world_width - self.radius
        )
        self.position[1] = np.clip(
            self.position[1], self.radius, self.world_height - self.radius
        )
        return old_pos


class Seeker(Agent):
    def __init__(self, agent_id, radius, speed, world_width, world_height):
        super().__init__(agent_id, radius, speed, world_width, world_height)
        self.is_seeker = True


class Hider(Agent):
    def __init__(self, agent_id, radius, speed, world_width, world_height):
        super().__init__(agent_id, radius, speed, world_width, world_height)
        self.is_seeker = False
