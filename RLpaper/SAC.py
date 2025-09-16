import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

GAME_NAME = "LunarLander-v3"  # SAC works best with continuous actions, but LunarLander is discrete. We'll use LunarLanderContinuous-v2
GAME_NAME = "LunarLanderContinuous-v3"

# === Callback to store rewards during training ===
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        if "episode" in self.locals["infos"][0]:
            r = self.locals["infos"][0]["episode"]["r"]
            self.rewards.append(r)
        return True

# === Create environment ===
env = gym.make(GAME_NAME)
env = Monitor(env)

# === Train SAC ===
callback = RewardCallback()
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
)
print("Training SAC agent...")
model.learn(total_timesteps=300_000, callback=callback)

# === Plot rewards ===
plt.plot(callback.rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("SAC Training Rewards on LunarLanderContinuous")
plt.show()

# === Test the trained SAC agent ===
env = gym.make(GAME_NAME, render_mode="human")
obs, _ = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
