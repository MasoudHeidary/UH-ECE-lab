

from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np, random, torch


from HEnv import SystolicArrayEnv, MAX_STEP

TOTAL_TRAIN_TIMESTEPS = 100_000
TRAIN_CPU = 32
TOTAL_INFERENCE_EPOCH = 100
MODEL_FILENAME = f"ppo.{__file__}.model"
DEVICE = "cuda:1"
SEED = 42

MODEL_TRAIN = True
MODEL_INFERENCE = True


if __name__ == "__main__":
    def make_env():
        env = SystolicArrayEnv()
        env.reset()
        return env

    def train(timesteps, filename, v_env, device):
        # env = SubprocVecEnv([make_env for _ in range(v_env)])
        env = SubprocVecEnv([lambda: make_env() for _ in range(v_env)])
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            batch_size=64,
            device=device,
            ent_coef=0.05, 
            learning_rate=3e-4,
            n_steps=1024,
        )

        model.learn(total_timesteps=timesteps)
        model.save(filename)
        env.close()

    def inference(model_filename, image_filename, device):
        env = SystolicArrayEnv()
        obs = env.reset()
        model = PPO.load(model_filename, env=env, device=device)

        voltages, freqs, latencies, powers = [], [], [], []
        backlogs, instr_rates = [], []
        reward_lst, power_penalty, backlog_penalty = [], [], []

        for step in range(MAX_STEP):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action, inference=True)

            voltages.append(env.current_voltage)
            freqs.append(env.current_frequency)
            latencies.append(env.latency)
            powers.append(env.power)
            backlogs.append(env.backlog_size)
            instr_rates.append(env.instt)

            reward_lst.append(reward)
            power_penalty.append(info['pow'])
            backlog_penalty.append(info['bkl'])

            # if done:
            #     obs = env.reset()

            

        plt.figure(figsize=(12,8))

        plt.subplot(5,1,1)
        plt.plot(freqs, label='Frequency (MHz)')
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)

        plt.subplot(5,1,2)
        plt.plot(latencies, label='Latency (ns)')
        plt.plot(powers, label='Power (arb.)')
        plt.ylabel("Latency / Power")
        plt.legend()
        plt.grid(True)

        plt.subplot(5,1,3)
        plt.plot(voltages, label='Voltage (V)')
        plt.ylabel("Voltage")
        plt.legend()
        plt.grid(True)

        plt.subplot(5,1,4)
        plt.plot(backlogs, label='Backlog Size')
        plt.plot(instr_rates, label='Instruction Rate')
        plt.ylabel("Backlog / Instr Rate")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)

        plt.subplot(5,1,5)
        plt.plot(reward_lst, label='reward')
        plt.plot(power_penalty, label='power penalty')
        plt.plot(backlog_penalty, label='backlog penalty')
        plt.ylabel("reward")
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