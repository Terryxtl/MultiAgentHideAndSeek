import numpy as np


class EntityState:
    def __init__(self):
        self.p_pos = np.zeros(2)
        self.p_vel = np.zeros(2)


class AgentState(EntityState):
    def __init__(self):
        super().__init__()
        self.c = np.zeros(2)


class Action:
    def __init__(self):
        self.u = np.zeros(2)
        self.c = np.zeros(2)


class Entity:
    def __init__(self):
        self.name = ""
        self.movable = False
        self.collide = True
        self.density = 25.0
        self.radius = 20.0
        self.color = np.zeros(3)
        self.max_speed = None
        self.accel = 10.0
        self.state = EntityState()
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


class Obstacle:
    def __init__(self, x, y, width, height):
        self.x, self.y, self.width, self.height = x, y, width, height
        self.color = np.array([0.5, 0.5, 0.5])


class Agent(Entity):
    def __init__(self):
        super().__init__()
        self.silent = False
        self.movable = True
        self.blind = False
        self.is_seeker = False
        self.caught = False
        self.u_noise = None
        self.c_noise = None
        self.u_range = 1.0
        self.state = AgentState()
        self.action = Action()
        self.action_callback = None


class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []
        self.obstacles = []
        self.dim_c = 0
        self.dim_p = 2
        self.dim_color = 3
        self.dt = 1 / 30
        self.damping = 0.25

    @property
    def entities(self):
        return self.agents

    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self):
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        p_force = [np.zeros(2) for _ in self.entities]
        p_force = self._apply_action_force(p_force)
        self._integrate_state(p_force)
        self._resolve_collisions()
        self._check_catches()
        for agent in self.agents:
            self.update_agent_state(agent)

    def _apply_action_force(self, p_force):
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.zeros(2)
                if agent.u_noise:
                    noise = np.random.randn(2) * agent.u_noise

                force = (agent.action.u + noise) * agent.accel * agent.mass

                p_force[i] += force
        return p_force

    def _apply_collision_force(self, p_force):
        # Agent-Agent 碰撞
        for i, agent_a in enumerate(self.agents):
            for j, agent_b in enumerate(self.agents):
                if j <= i:
                    continue
                dist = np.linalg.norm(agent_a.state.p_pos - agent_b.state.p_pos)
                if dist < agent_a.radius + agent_b.radius:
                    # 如果发生碰撞，为了简单起见，让他们速度归零
                    agent_a.state.p_vel = np.zeros(2)
                    agent_b.state.p_vel = np.zeros(2)

        # Agent-Obstacle 碰撞
        for i, agent in enumerate(self.agents):
            for obstacle in self.obstacles:
                if self._is_agent_obstacle_collision(agent, obstacle):
                    agent.state.p_vel = np.zeros(2)

        return p_force

    def _check_catches(self):
        seekers = [agent for agent in self.agents if agent.is_seeker]
        hiders = [
            agent for agent in self.agents if not agent.is_seeker and not agent.caught
        ]
        if not seekers or not hiders:
            return

        seeker_pos = np.array([s.state.p_pos for s in seekers])
        hider_pos = np.array([h.state.p_pos for h in hiders])
        seeker_rad = np.array([s.radius for s in seekers])
        hider_rad = np.array([h.radius for h in hiders])

        diff = seeker_pos[:, np.newaxis] - hider_pos
        sq_dists = np.einsum("ijk,ijk->ij", diff, diff)

        rad_sums = seeker_rad[:, np.newaxis] + hider_rad
        sq_rad_sums = rad_sums**2

        caught_mask = np.any(sq_dists < sq_rad_sums, axis=0)
        for idx, hider in enumerate(hiders):
            if caught_mask[idx]:
                hider.caught = True

    def _get_agent_obstacle_collision_info(self, agent, obstacle):
        obs_left, obs_right = obstacle.x, obstacle.x + obstacle.width
        obs_top, obs_bottom = obstacle.y, obstacle.y + obstacle.height

        closest_x = max(obs_left, min(agent.state.p_pos[0], obs_right))
        closest_y = max(obs_top, min(agent.state.p_pos[1], obs_bottom))

        closest_point = np.array([closest_x, closest_y])
        distance = np.linalg.norm(agent.state.p_pos - closest_point)

        is_collision = distance < agent.radius
        return is_collision, closest_point

    def _resolve_collisions(self):
        active_agents = [
            agent
            for agent in self.agents
            if agent.collide and not getattr(agent, "caught", False)
        ]

        for i, agent_a in enumerate(active_agents):
            for j, agent_b in enumerate(active_agents):
                if j <= i:
                    continue
                delta_pos = agent_a.state.p_pos - agent_b.state.p_pos
                dist = np.linalg.norm(delta_pos)
                min_dist = agent_a.radius + agent_b.radius
                if dist < min_dist:
                    overlap = min_dist - dist
                    direction = delta_pos / dist if dist != 0 else np.array([1.0, 0.0])
                    correction_a = direction * overlap * 0.5
                    correction_b = -direction * overlap * 0.5
                    agent_a.state.p_pos += correction_a
                    agent_b.state.p_pos += correction_b
                    agent_a.state.p_vel *= 0
                    agent_b.state.p_vel *= 0

        for agent in active_agents:
            for obstacle in self.obstacles:
                is_collision, closest_point = self._get_agent_obstacle_collision_info(
                    agent, obstacle
                )
                if is_collision:
                    delta_pos = agent.state.p_pos - closest_point
                    dist = np.linalg.norm(delta_pos)
                    overlap = agent.radius - dist
                    direction = delta_pos / dist if dist != 0 else np.array([1.0, 0.0])
                    agent.state.p_pos += direction * overlap
                    agent.state.p_vel *= 0

    def _is_agent_obstacle_collision(self, agent, obstacle):
        obs_left = obstacle.x
        obs_right = obstacle.x + obstacle.width
        obs_top = obstacle.y
        obs_bottom = obstacle.y + obstacle.height

        closest_x = max(obs_left, min(agent.state.p_pos[0], obs_right))
        closest_y = max(obs_top, min(agent.state.p_pos[1], obs_bottom))

        distance = np.sqrt(
            (agent.state.p_pos[0] - closest_x) ** 2
            + (agent.state.p_pos[1] - closest_y) ** 2
        )

        return distance < agent.radius

    def _integrate_state(self, p_force):
        for i, agent in enumerate(self.agents):
            if not agent.movable or getattr(agent, "caught", False):
                continue
            agent.state.p_vel *= 1 - self.damping
            if p_force[i] is not None:
                agent.state.p_vel += (p_force[i] / agent.mass) * self.dt
            speed = np.linalg.norm(agent.state.p_vel)
            if agent.max_speed is not None and speed > agent.max_speed:
                agent.state.p_vel = (agent.state.p_vel / speed) * agent.max_speed
            agent.state.p_pos += agent.state.p_vel * self.dt
            agent.state.p_pos[0] = np.clip(
                agent.state.p_pos[0], agent.radius, self.width - agent.radius
            )
            agent.state.p_pos[1] = np.clip(
                agent.state.p_pos[1], agent.radius, self.height - agent.radius
            )

    def update_agent_state(self, agent):
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise
