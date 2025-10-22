

from stable_baselines3 import DQN
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np, random, torch
from stable_baselines3.common.monitor import Monitor

from HEnvDQN import SystolicArrayEnv, MAX_STEP

TOTAL_TRAIN_TIMESTEPS = 200_000
TRAIN_CPU = 1
TOTAL_INFERENCE_EPOCH = 100
MODEL_FILENAME = f"{__file__}.model"
DEVICE = "cuda:1"
SEED = 42

MODEL_TRAIN = True
MODEL_INFERENCE = True


if __name__ == "__main__":
    def make_env():
        def _init():
            env = SystolicArrayEnv()
            # env = Monitor(env)  # monitoring wrapper
            env.reset()
            return env
        return _init

    def train(timesteps, filename, v_env, device):
        # env = SubprocVecEnv([make_env for _ in range(v_env)])
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

        voltages, freqs, latencies, powers = [], [], [], []
        backlogs, instr_rates = [], []
        reward_lst, power_penalty, backlog_penalty = [], [], []
        max_freq = []

        for step in range(MAX_STEP):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action), inference=True)

            voltages.append(env.vdd)
            freqs.append(env.current_frequency)
            latencies.append(env.latency)
            powers.append(env.power)

            backlogs.append(env.backlog_running)
            instr_rates.append(env.backlog_crashed)
            max_freq.append(info['max_freq'])

            reward_lst.append(reward)
            power_penalty.append(0)
            backlog_penalty.append(0)

            # if done:
            #     obs = env.reset()

            

        plt.figure(figsize=(12,8))

        plt.subplot(5,1,1)
        plt.plot(freqs, label='Frequency (MHz)')
        plt.plot(max_freq, label='max_freq (MHz)')
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)

        plt.subplot(5,1,2)
        plt.plot(latencies, label='Latency (ps)')
        plt.plot(powers, label='Power (uw * MHz)')
        plt.ylabel("Latency / Power")
        plt.legend()
        plt.grid(True)

        plt.subplot(5,1,3)
        plt.plot(voltages, label='Voltage (V)')
        plt.ylabel("Voltage")
        plt.legend()
        plt.grid(True)

        plt.subplot(5,1,4)
        plt.plot(backlogs, label='running inst')
        plt.ylabel("...")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)

        plt.subplot(5,1,5)
        plt.plot(instr_rates, label='crashed inst')
        # plt.plot(reward_lst, label='reward')
        # plt.plot(power_penalty, label='power penalty')
        # plt.plot(backlog_penalty, label='backlog penalty')
        plt.ylabel("...")
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