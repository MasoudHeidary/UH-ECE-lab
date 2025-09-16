import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3 import PPO, SAC

GAME_NAME = "LunarLander-v3"

# === Callback to store rewards during training ===
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        # Collect episode rewards whenever an episode ends
        if "episode" in self.locals["infos"][0]:
            r = self.locals["infos"][0]["episode"]["r"]
            self.rewards.append(r)
        return True

# === Create environment ===
# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make(GAME_NAME)
env = Monitor(env)  # Monitor helps track episode rewards

# === Train DQN ===
callback = RewardCallback()
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=5e-4,
    buffer_size=50000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.15,
    target_update_interval=1000,

    policy_kwargs=dict(net_arch=[256, 256])
)


print("Training DQN agent...")
model.learn(total_timesteps=1_000_000, callback=callback)

# === Plot training rewards ===
plt.plot(callback.rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training Rewards on CartPole")
plt.show()

# === Test trained agent with graphical rendering ===
env = gym.make(GAME_NAME, render_mode="human")

total_rewards = []
for episode in range(10):
    obs, _ = env.reset()
    ep_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
    total_rewards.append(ep_reward)

print("Average reward over 10 episodes:", sum(total_rewards)/len(total_rewards))