import gymnasium as gym
import numpy as np



class NoisyCartPoleEnv(gym.Wrapper):
    def __init__(self, env, noise_level=0.1):
        super().__init__(env)
        self.noise_level = noise_level

    def noise_step(self, action):
        next_state, reward, done, truncate, info = self.env.step(action)
        noise = np.random.normal(0, self.noise_level, size=next_state.shape)
        noisy_next_state = next_state + noise
        return noisy_next_state, reward, done, truncate, info
    
   
