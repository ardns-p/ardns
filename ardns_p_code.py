import numpy as np
import matplotlib.pyplot as plt

# ARDNS-P Model with Piaget's Developmental Stages
class ARDNS_P:
    def __init__(self, state_dim=2, action_dim=4, ms_dim=5, ml_dim=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ms_dim = ms_dim
        self.ml_dim = ml_dim
        self.W_s = np.random.randn(ms_dim, state_dim) * 0.01
        self.W_l = np.random.randn(ml_dim, state_dim) * 0.01
        self.W_a = np.random.randn(action_dim, ms_dim + ml_dim) * 0.01
        self.M_s = np.zeros(ms_dim)
        self.M_l = np.zeros(ml_dim)
        self.mu = [0.0, 1.0]
        self.sigma = [0.1, 0.2]
        self.pi = [0.5, 0.5]
        self.variances = []
        self.episode = 0
        self.disequilibrium_count = 0
        self.params = {
            'eta': 0.05,
            'eta_r': 0.05,
            'alpha_s': 0.7,
            'alpha_l': 0.95,
            'beta': 1.5,
            'gamma': 0.5,
            'tau': 1.2,
            'epsilon': 1.0,
            'epsilon_min': 0.1,
            'epsilon_decay': 0.99999,
            'weight_clip': 10.0,
            'sigma_min': 1e-3,
            'exp_clip': 500.0,
            'reward_clip': 10.0
        }

    def get_stage_params(self):
        sensorimotor_end = 400
        preoperational_end = 600
        concrete_end = 750
        transition_window = 50

        if self.episode <= sensorimotor_end:
            eta_base = self.params['eta'] * 2.0
            epsilon_base = max(0.8, self.params['epsilon'] * (0.994 ** self.episode))
            alpha_s_base, alpha_l_base = 0.95, 0.7
            curiosity_bonus = 1.0
        elif self.episode <= preoperational_end:
            eta_base = self.params['eta'] * 1.5
            epsilon_base = max(0.6, self.params['epsilon'] * (0.996 ** (self.episode - 400)))
            alpha_s_base, alpha_l_base = 0.85, 0.8
            curiosity_bonus = 0.5
        elif self.episode <= concrete_end:
            eta_base = self.params['eta'] * 1.2
            epsilon_base = max(0.4, self.params['epsilon'] * (0.998 ** (self.episode - 600)))
            alpha_s_base, alpha_l_base = 0.75, 0.9
            curiosity_bonus = 0.2
        else:
            eta_base = self.params['eta']
            epsilon_base = max(0.01, self.params['epsilon'] * (0.9995 ** (self.episode - 750)))
            alpha_s_base, alpha_l_base = 0.65, 0.95
            curiosity_bonus = 0.0

        if sensorimotor_end < self.episode <= sensorimotor_end + transition_window:
            t = (self.episode - sensorimotor_end) / transition_window
            eta = (1 - t) * (self.params['eta'] * 2.0) + t * (self.params['eta'] * 1.5)
            epsilon = (1 - t) * max(0.8, self.params['epsilon'] * (0.994 ** sensorimotor_end)) + t * max(0.6, self.params['epsilon'] * (0.996 ** (self.episode - 400)))
            alpha_s = (1 - t) * 0.95 + t * 0.85
            alpha_l = (1 - t) * 0.7 + t * 0.8
            curiosity_bonus = (1 - t) * 1.0 + t * 0.5
        elif preoperational_end < self.episode <= preoperational_end + transition_window:
            t = (self.episode - preoperational_end) / transition_window
            eta = (1 - t) * (self.params['eta'] * 1.5) + t * (self.params['eta'] * 1.2)
            epsilon = (1 - t) * max(0.6, self.params['epsilon'] * (0.996 ** (preoperational_end - 400))) + t * max(0.4, self.params['epsilon'] * (0.998 ** (self.episode - 600)))
            alpha_s = (1 - t) * 0.85 + t * 0.75
            alpha_l = (1 - t) * 0.8 + t * 0.9
            curiosity_bonus = (1 - t) * 0.5 + t * 0.2
        elif concrete_end < self.episode <= concrete_end + transition_window:
            t = (self.episode - concrete_end) / transition_window
            eta = (1 - t) * (self.params['eta'] * 1.2) + t * self.params['eta']
            epsilon = (1 - t) * max(0.4, self.params['epsilon'] * (0.998 ** (concrete_end - 600))) + t * max(0.01, self.params['epsilon'] * (0.9995 ** (self.episode - 750)))
            alpha_s = (1 - t) * 0.75 + t * 0.65
            alpha_l = (1 - t) * 0.9 + t * 0.95
            curiosity_bonus = (1 - t) * 0.2 + t * 0.0
        else:
            eta, epsilon, alpha_s, alpha_l = eta_base, epsilon_base, alpha_s_base, alpha_l_base

        if self.disequilibrium_count > 0:
            eta *= 1.5
            self.disequilibrium_count -= 1
        return eta, epsilon, alpha_s, alpha_l, curiosity_bonus

    def update_memory(self, state):
        eta, _, alpha_s, alpha_l, _ = self.get_stage_params()
        self.M_s = alpha_s * self.M_s + (1 - alpha_s) * np.maximum(0, self.W_s @ state)
        self.M_l = alpha_l * self.M_l + (1 - alpha_l) * np.maximum(0, self.W_l @ state)

    def update_reward(self, reward):
        reward = np.clip(reward, -self.params['reward_clip'], self.params['reward_clip'])
        self.sigma = [max(s, self.params['sigma_min']) for s in self.sigma]
        exponents = [-0.5 * ((reward - self.mu[i]) / self.sigma[i])**2 for i in range(2)]
        exponents = [np.clip(exp, -self.params['exp_clip'], self.params['exp_clip']) for exp in exponents]
        gamma = [self.pi[i] * np.exp(exp) / self.sigma[i] for i, exp in enumerate(exponents)]
        gamma_sum = sum(gamma)
        if gamma_sum < 1e-10:
            gamma_sum = 1e-10
        gamma = [g / gamma_sum for g in gamma]
        for i in range(2):
            self.mu[i] += self.params['eta_r'] * gamma[i] * (reward - self.mu[i])
            self.sigma[i] = np.sqrt(self.sigma[i]**2 + self.params['eta_r'] * gamma[i] *
                                   ((reward - self.mu[i])**2 - self.sigma[i]**2))
            self.pi[i] = (gamma[i] + self.pi[i]) / 2
        sigma2 = sum(p * s**2 for p, s in zip(self.pi, self.sigma))
        self.variances.append(sigma2)

    def update_weights(self, state, action, reward, prev_state, curiosity_bonus=0):
        self.update_reward(reward + curiosity_bonus)
        sigma2 = sum(p * s**2 for p, s in zip(self.pi, self.sigma))
        delta_S = np.linalg.norm(state - prev_state)**2
        eta, _, _, _, _ = self.get_stage_params()
        mod = eta * (reward + curiosity_bonus) / max(0.1, 1 - self.params['beta'] * sigma2) * np.exp(-self.params['gamma'] * delta_S)
        M = np.concatenate([self.M_s, self.M_l])
        if reward < -0.3 and self.episode > 100:
            self.disequilibrium_count = 20
        self.W_a[action] += mod * M
        self.W_s += mod * np.outer(self.M_s, state)
        self.W_l += mod * np.outer(self.M_l, state)
        self.W_a = np.clip(self.W_a, -self.params['weight_clip'], self.params['weight_clip'])
        self.W_s = np.clip(self.W_s, -self.params['weight_clip'], self.params['weight_clip'])
        self.W_l = np.clip(self.W_l, -self.params['weight_clip'], self.params['weight_clip'])

    def choose_action(self, state, visited_states):
        self.update_memory(state)
        _, epsilon, _, _, curiosity_bonus = self.get_stage_params()
        M = np.concatenate([self.M_s, self.M_l])
        V = self.W_a @ M
        scaled_V = self.params['tau'] * V
        max_V = np.max(scaled_V)
        exp_V = np.exp(scaled_V - max_V)
        probs = exp_V / (np.sum(exp_V) + 1e-10)
        if np.random.rand() < epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = np.argmax(V)
        actual_curiosity_bonus = curiosity_bonus if tuple(state) not in visited_states else 0
        return action, actual_curiosity_bonus

# DQN Baseline with Piaget-Inspired Adjustments
class DQN:
    def __init__(self, state_dim=2, action_dim=4, hidden_dim=64, buffer_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        limit = np.sqrt(6 / (hidden_dim + state_dim))
        self.W1 = np.random.uniform(-limit, limit, (hidden_dim, state_dim))
        self.W2 = np.random.uniform(-limit, limit, (hidden_dim, hidden_dim))
        self.W3 = np.random.uniform(-limit, limit, (action_dim, hidden_dim))
        self.W1_target = self.W1.copy()
        self.W2_target = self.W2.copy()
        self.W3_target = self.W3.copy()
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99999
        self.gamma = 0.99
        self.eta = 0.05
        self.buffer = []
        self.buffer_size = buffer_size
        self.clip_value = 1.0
        self.update_freq = 100
        self.step_count = 0
        self.episode = 0
        self.disequilibrium_count = 0
        self.weight_clip = 10.0

    def get_stage_params(self):
        sensorimotor_end = 400
        preoperational_end = 600
        concrete_end = 750
        transition_window = 50

        if self.episode <= sensorimotor_end:
            eta_base = self.eta * 2.0
            epsilon_base = max(0.8, self.epsilon * (0.994 ** self.episode))
            curiosity_bonus = 1.0
        elif self.episode <= preoperational_end:
            eta_base = self.eta * 1.5
            epsilon_base = max(0.6, self.epsilon * (0.996 ** (self.episode - 400)))
            curiosity_bonus = 0.5
        elif self.episode <= concrete_end:
            eta_base = self.eta * 1.2
            epsilon_base = max(0.4, self.epsilon * (0.998 ** (self.episode - 600)))
            curiosity_bonus = 0.2
        else:
            eta_base = self.eta
            epsilon_base = max(0.01, self.epsilon * (0.9995 ** (self.episode - 750)))
            curiosity_bonus = 0.0

        if sensorimotor_end < self.episode <= sensorimotor_end + transition_window:
            t = (self.episode - sensorimotor_end) / transition_window
            eta = (1 - t) * (self.eta * 2.0) + t * (self.eta * 1.5)
            epsilon = (1 - t) * max(0.8, self.epsilon * (0.994 ** sensorimotor_end)) + t * max(0.6, self.epsilon * (0.996 ** (self.episode - 400)))
            curiosity_bonus = (1 - t) * 1.0 + t * 0.5
        elif preoperational_end < self.episode <= preoperational_end + transition_window:
            t = (self.episode - preoperational_end) / transition_window
            eta = (1 - t) * (self.eta * 1.5) + t * (self.eta * 1.2)
            epsilon = (1 - t) * max(0.6, self.epsilon * (0.996 ** (preoperational_end - 400))) + t * max(0.4, self.epsilon * (0.998 ** (self.episode - 600)))
            curiosity_bonus = (1 - t) * 0.5 + t * 0.2
        elif concrete_end < self.episode <= concrete_end + transition_window:
            t = (self.episode - concrete_end) / transition_window
            eta = (1 - t) * (self.eta * 1.2) + t * self.eta
            epsilon = (1 - t) * max(0.4, self.epsilon * (0.998 ** (concrete_end - 600))) + t * max(0.01, self.epsilon * (0.9995 ** (self.episode - 750)))
            curiosity_bonus = (1 - t) * 0.2 + t * 0.0
        else:
            eta, epsilon = eta_base, epsilon_base

        if self.disequilibrium_count > 0:
            eta *= 1.5
            self.disequilibrium_count -= 1
        return eta, epsilon, curiosity_bonus

    def forward(self, state, target=False):
        W1 = self.W1_target if target else self.W1
        W2 = self.W2_target if target else self.W2
        W3 = self.W3_target if target else self.W3
        h1 = np.maximum(0, W1 @ state)
        h2 = np.maximum(0, W2 @ h1)
        return W3 @ h2

    def choose_action(self, state, visited_states):
        eta, epsilon, curiosity_bonus = self.get_stage_params()
        if np.random.rand() < epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = np.argmax(self.forward(state))
        actual_curiosity_bonus = curiosity_bonus if tuple(state) not in visited_states else 0
        return action, actual_curiosity_bonus

    def update(self, state, action, reward, next_state, curiosity_bonus=0):
        self.buffer.append((state, action, reward + curiosity_bonus, next_state))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        if len(self.buffer) < 32:
            return

        batch = np.random.choice(len(self.buffer), 32, replace=False)
        eta, _, _ = self.get_stage_params()
        for idx in batch:
            s, a, r, ns = self.buffer[idx]
            h1 = np.maximum(0, self.W1 @ s)
            h2 = np.maximum(0, self.W2 @ h1)
            q_current = self.W3 @ h2
            q_next = np.max(self.forward(ns, target=True))
            target = r + self.gamma * q_next
            error = target - q_current[a]

            delta3 = np.zeros(self.action_dim)
            delta3[a] = error
            grad_W3 = np.outer(delta3, h2)
            delta2 = (self.W3.T @ delta3) * (h2 > 0)
            grad_W2 = np.outer(delta2, h1)
            delta1 = (self.W2.T @ delta2) * (h1 > 0)
            grad_W1 = np.outer(delta1, s)

            if np.any(np.abs(grad_W1) > self.clip_value):
                grad_W1 = np.clip(grad_W1, -self.clip_value, self.clip_value)
            if np.any(np.abs(grad_W2) > self.clip_value):
                grad_W2 = np.clip(grad_W2, -self.clip_value, self.clip_value)
            if np.any(np.abs(grad_W3) > self.clip_value):
                grad_W3 = np.clip(grad_W3, -self.clip_value, self.clip_value)

            self.W3 += eta * grad_W3
            self.W2 += eta * grad_W2
            self.W1 += eta * grad_W1

            self.W1 = np.clip(self.W1, -self.weight_clip, self.weight_clip)
            self.W2 = np.clip(self.W2, -self.weight_clip, self.weight_clip)
            self.W3 = np.clip(self.W3, -self.weight_clip, self.weight_clip)

        self.step_count += 1
        if self.step_count % self.update_freq == 0:
            self.W1_target = self.W1.copy()
            self.W2_target = self.W2.copy()
            self.W3_target = self.W3.copy()
        if reward < -0.3 and self.episode > 100:
            self.disequilibrium_count = 20

# Grid-World Environment
class GridWorld:
    def __init__(self, size=10):
        self.size = size
        self.goal = (size-1, size-1)
        self.obstacles = set()
        self.reset()

    def reset(self):
        self.state = (0, 0)
        self.visited_states = set()
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        x, y = self.state
        if action == 0: y = min(y+1, self.size-1)  # Up
        elif action == 1: y = max(y-1, 0)          # Down
        elif action == 2: x = max(x-1, 0)          # Left
        elif action == 3: x = min(x+1, self.size-1) # Right

        next_state = (x, y)
        if next_state in self.obstacles:
            next_state = self.state

        self.state = next_state
        self.visited_states.add(tuple(next_state))
        reward = 1.0 if next_state == self.goal else 0.0
        reward += np.random.normal(0, 0.2)  # Add noise
        done = next_state == self.goal
        return np.array(next_state, dtype=np.float32), reward, done

    def update_obstacles(self):
        self.obstacles = set()

# Simulation
def run_simulation(model, env, episodes=3000, max_steps=150):
    rewards = []
    steps_to_goal = []
    goals_reached = 0
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        if episode % 100 == 0 and episode > 0:
            env.update_obstacles()

        model.episode = episode
        while not done and steps < max_steps:
            prev_state = state.copy()
            action, curiosity_bonus = model.choose_action(state, env.visited_states)
            next_state, reward, done = env.step(action)

            if isinstance(model, ARDNS_P):
                model.update_weights(state, action, reward, prev_state, curiosity_bonus)
            else:  # DQN
                model.update(state, action, reward, next_state, curiosity_bonus)

            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_to_goal.append(steps if done else max_steps)
        if done:
            goals_reached += 1

    print(f"Goals reached: {goals_reached}/{episodes}")
    return rewards, steps_to_goal

# Main Execution
np.random.seed(42)
env = GridWorld(size=10)
ardns_p = ARDNS_P()
dqn = DQN()

ardns_rewards, ardns_steps = run_simulation(ardns_p, env)
dqn_rewards, dqn_steps = run_simulation(dqn, env)

# Plotting Results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(np.convolve(ardns_rewards, np.ones(50)/50, mode='valid'), label='ARDNS-P')
plt.plot(np.convolve(dqn_rewards, np.ones(50)/50, mode='valid'), label='DQN')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Learning Curve')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(np.convolve(ardns_steps, np.ones(50)/50, mode='valid'), label='ARDNS-P')
plt.plot(np.convolve(dqn_steps, np.ones(50)/50, mode='valid'), label='DQN')
plt.xlabel('Episode')
plt.ylabel('Steps to Goal')
plt.title('Steps to Goal')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(np.convolve(ardns_p.variances, np.ones(50)/50, mode='valid'), label='Variance')
plt.plot(np.convolve(ardns_rewards[49:], np.ones(50)/50, mode='valid'), label='ARDNS-P Reward')
plt.xlabel('Episode')
plt.ylabel('Value')
plt.title('Reward vs. Variance (ARDNS-P)')
plt.legend()

plt.tight_layout()
plt.savefig('ardns_p_results.png')
plt.show()

# Print Metrics (last 200 episodes)
print("ARDNS-P Mean Reward (last 200):", np.mean(ardns_rewards[-200:]), "±", np.std(ardns_rewards[-200:]))
print("DQN Mean Reward (last 200):", np.mean(dqn_rewards[-200:]), "±", np.std(dqn_rewards[-200:]))
print("ARDNS-P Steps to Goal (last 200):", np.mean(ardns_steps[-200:]), "±", np.std(ardns_steps[-200:]))
print("DQN Steps to Goal (last 200):", np.mean(dqn_steps[-200:]), "±", np.std(dqn_steps[-200:]))