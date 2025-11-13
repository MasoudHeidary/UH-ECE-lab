

from stable_baselines3 import DQN
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np, random, torch
from stable_baselines3.common.monitor import Monitor

from HEnvDQN import SystolicArrayEnv, MAX_STEP

TOTAL_TRAIN_TIMESTEPS = 1000_000
TRAIN_CPU = 1
TOTAL_INFERENCE_EPOCH = 100
MODEL_FILENAME = f"{__file__}.model"
DEVICE = "cuda:0"
SEED = 42

MODEL_TRAIN = True
MODEL_INFERENCE = True


if __name__ == "__main__":
    def make_env():
        def _init():
            env = SystolicArrayEnv()
            env.reset()
            return env
        return _init

    def train(timesteps, filename, v_env, device):
        env = DummyVecEnv([make_env() for _ in range(TRAIN_CPU)])  # single env is fine for DQN        
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=1000,
            batch_size=64,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.5,
            exploration_final_eps=0.05,
            # seed=SEED,
            device=device
        )

        model.learn(total_timesteps=timesteps)
        model.save(filename)
        env.close()


    def inference(model_filename, image_filename, device):
        env = SystolicArrayEnv()
        obs = env.reset()
        model = DQN.load(model_filename, env=env, device=device)

        freq, max_freq = [], []
        vdd, latency, energy = [], [], []
        backlog_not_active, backlog_ok, backlog_linear, backlog_crashed = [], [], [], []
        model_flops, computing_flops = [], []
        reward_lst = []

        for step in range(MAX_STEP):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action), inference=True)

            freq.append(env.freq)
            max_freq.append(env.hardware.get_max_freq())

            vdd.append(env.vdd)
            latency.append(env.latency / env.max_latency)
            energy.append(env.energy / env.max_energy)

            backlog_not_active.append(env.backlog_not_active)
            backlog_ok.append(env.backlog_ok)
            backlog_linear.append(env.backlog_linear)
            backlog_crashed.append(env.backlog_crashed)

            model_flops.append(env.inference_flops)
            computing_flops.append(env.hardware_flops)

            reward_lst.append(reward)

        plt.figure(figsize=(12,8))

        plt.subplot(5,1,1)
        plt.plot(freq, label='Frequency (MHz)')
        plt.plot(max_freq, label='max_freq (MHz)')
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)

        plt.subplot(5,1,2)
        plt.plot(vdd, label='vdd (v)')
        plt.plot(latency, label='Latency (ps)')
        plt.plot(energy, label='Energy (uw * MHz)')
        plt.ylabel("derived parameter")
        plt.legend()
        plt.grid(True)

        plt.subplot(5,1,3)
        plt.plot(backlog_ok, label='backlog (ok)')
        plt.ylabel("xxx")
        plt.legend()
        plt.grid(True)

        plt.subplot(5,1,4)
        plt.plot(backlog_linear, label='backlog (linear)')
        plt.plot(backlog_crashed, label='backlog (crashed)')
        plt.ylabel("xxx")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)

        plt.subplot(5,1,5)
        plt.plot(model_flops, label="inference flops")
        plt.plot(computing_flops, label="computing flops")
        plt.ylabel("xxx")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)


        plt.tight_layout()
        plt.savefig(image_filename)
    

    if MODEL_TRAIN:
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        train(TOTAL_TRAIN_TIMESTEPS, MODEL_FILENAME, TRAIN_CPU, DEVICE)
    
    if MODEL_INFERENCE:
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        for epoch in range(TOTAL_INFERENCE_EPOCH):
            inference(MODEL_FILENAME, f"pic.jpg", DEVICE)
            if 'n' in input("go to next inference [ENTER/n]..."):
                exit()