
"""DQN RL with manipulated reward"""

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

GAME_NAME           = "LunarLander-v3"
TOTAL_TRAIN_TIMELAPSE = 400_000
TOTAL_INFERENCE = 20
MODEL_FILENAME      = f"lunar_lander.DQN.tf{TOTAL_TRAIN_TIMELAPSE//1_000_000}M"

MODEL_TRAIN         = True
CONFIRM_TO_TRAIN    = True
TRAIN_GUI           = False
INFERENCE_GUI       = True
TRAIN_RECORD        = ...   #TODO
INFERENCE_RECORD    = False



# === Custom Wrapper to manipulate rewards ===
class RewardDebugWrapper(gym.Wrapper):
    def __init__(self, env, time_penalty=0.005, engine_penalty=0.2):
        super().__init__(env)
        self.time_penalty = time_penalty
        self.engine_penalty = engine_penalty
        self.timestep = 0

    def reset(self, **kwargs):
        self.timestep = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.timestep += 1

        # Print action + original reward
        # print(f"Timestep={self.timestep}, Action={action}, OriginalReward={reward:.3f}")

        # === Add penalties ===
        # 1. Penalize each step (forces faster landing)
        # reward -= self.time_penalty
        # print(f"   → TimePenalty: -{self.time_penalty:.3f}, NewReward={reward:.3f}")

        # 2. Penalize engine usage (optional, keeps thruster use efficient)
        # if action != 0:
        #     reward -= self.engine_penalty
        #     print(f"   → EnginePenalty: -{self.engine_penalty:.3f}, NewReward={reward:.3f}")

        return obs, reward, done, truncated, info


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

# train
if MODEL_TRAIN:
    if CONFIRM_TO_TRAIN:
        conf = input("confirm to train? [y(es)/n(o)]\n")
        if not (conf.lower() in ['y', 'yes']):
            exit()

    env = gym.make(GAME_NAME, render_mode=("human" if TRAIN_GUI else None))
    env = RewardDebugWrapper(env)
    env = Monitor(env)

    callback = RewardCallback()
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        buffer_size=50000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.20,
    )

    model.learn(total_timesteps=TOTAL_TRAIN_TIMELAPSE, callback=callback)
    model.save(MODEL_FILENAME)

    plt.plot(callback.rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training Rewards on CartPole")
    plt.show()


### inference
if INFERENCE_GUI and INFERENCE_RECORD:
    raise ValueError("inference GUI and inference Record are not available at the same time")

render = None
if INFERENCE_GUI:
    render = "human"
elif INFERENCE_RECORD:
    render = "rgb_array"

env = gym.make(GAME_NAME, render_mode=render)
if INFERENCE_RECORD:
    env = RecordVideo(env, video_folder="videos", episode_trigger=lambda t: True, fps=60)

model = DQN.load(MODEL_FILENAME)


total_rewards = []
for episode in range(TOTAL_INFERENCE):
    obs, _ = env.reset()
    ep_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
    total_rewards.append(ep_reward)

    if INFERENCE_RECORD:
        env.close()
print("Average reward:", sum(total_rewards)/len(total_rewards))

