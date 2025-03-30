import gymnasium as gym
import torch        
import torch.nn as nn 
import random
import numpy as np
from collections import deque
from constants import *

LEARNING_RATE = 0.00063
TRAINING_SIZE = 32
START_RANDOM_EPSILON = 1
RANDOM_EPSILON_DECAY = 0.99
BATCH_SIZE = 16
MAX_EXP = 50000

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human") 

def sigmoid(x: float):
    return 1 / (1 + np.exp(-x))
def minmax(x: float, minimum: float, maximum: float):
    return (x - minimum) / (maximum - minimum)
def normalize_cartpole(state):
    state_max = 4.8
    state2_max = 0.418
    return np.array([
        minmax(state[0], -state_max, state_max),
        sigmoid(state[1]),
        minmax(state[2], -state2_max, state2_max),
        sigmoid(state[3])
    ])
env2norm = {"CartPole-v1": normalize_cartpole,}
normalize = env2norm[env_name]

env_dimension = { 
    "state_size": env.observation_space.shape[0],
    "action_size": env.action_space.n, 
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def greedy(state, model):
    tensor_state = torch.from_numpy(state.reshape((1, env.observation_space.shape[0])).astype(np.float32)).to(device)
    action_value_tensor = model(tensor_state)
    action = action_value_tensor.argmax(1).detach().cpu().numpy()
    return action[0] 

def epsilon_greedy(state, epsilon, model):
    ep = np.random.random(1)[0] 
    if ep < epsilon:
        return env.action_space.sample() 
    else:
        return greedy(state, model) 

class DQN(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, name="DQN"):
        super().__init__()
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.network = nn.Sequential(
            nn.Linear(np.array(self.input_shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_shape),
        )
    def forward(self, x):
        return self.network(x)
    def copy(cls, other, name="target"):
        return cls(input_shape=other.input_shape, output_shape=other.output_shape, name=name)
    
def generate_dqn_model_pair(input_shape, output_shape, typ=""):
    names = ["net", "targetnet"]      
    models = []     
    for n in names: 
        models.append(DQN(input_shape=input_shape, output_shape=output_shape, name=typ + n).to(device))
    return models 

net, targetnet = generate_dqn_model_pair(input_shape=env_dimension["state_size"], output_shape=env_dimension["action_size"])

def copy_net(net: DQN, target: DQN):
    target.load_state_dict(net.state_dict())
    target.eval()      

class ExpPool:
    def __init__(self, state_size, action_size):
        self.pool = deque(maxlen=MAX_EXP)
        self.state_size = state_size
        self.action_size = int(action_size/2)

    def sample(self):
        exps = random.sample(self.pool, BATCH_SIZE)
        state_batch = []
        action_batch = []
        reward_batch = []
        done_batch = []
        next_state_batch = []
        for e in exps:
            state_batch.append(e[0])
            action_batch.append(e[1])
            reward_batch.append(e[2])
            done_batch.append(e[3])
            next_state_batch.append(e[4])
        state_batch = np.array(state_batch).reshape((BATCH_SIZE, self.state_size)).astype(np.float32)
        action_batch = np.array(action_batch).reshape((BATCH_SIZE, self.action_size)).astype(np.int64)
        reward_batch = np.array(reward_batch).reshape((BATCH_SIZE, 1)).astype(np.float32)
        done_batch = np.array(done_batch).reshape((BATCH_SIZE, 1))
        next_state_batch = np.array(next_state_batch).reshape((BATCH_SIZE, self.state_size)).astype(np.float32)
        return state_batch, action_batch, reward_batch, done_batch, next_state_batch
    
    def append(self, x):
        self.pool.append(x)


epool = ExpPool(**env_dimension)

def compute_y(reward_batch, done_batch, next_action_value_batch_tensor):
    reward_batch_tensor = torch.from_numpy(reward_batch).to(device)
    done_mask_batch = 1 - done_batch.sum(axis=1).reshape((BATCH_SIZE, 1))
    done_mask_batch_tensor = torch.from_numpy(done_mask_batch).to(device)
    max_action_value_tensor = torch.max(next_action_value_batch_tensor, dim=1)[0]       
    decayed_maction_value_tensor = torch.mul(max_action_value_tensor, GAMMA).reshape((BATCH_SIZE, 1))
    masked_action_value_tensor = torch.mul(done_mask_batch_tensor, decayed_maction_value_tensor)
    return masked_action_value_tensor + reward_batch_tensor

def decay_epsilon(epsilon):
    return max(epsilon * RANDOM_EPSILON_DECAY, MINIMUM_EPSILON) 

def train(models: tuple, episodes):
    model, target_model = models
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    training_epoch = -TRAINING_SIZE
    e = 0
    epsilon = START_RANDOM_EPSILON
    losses = []
    Q = []
    episodic_rewards = []
    while e < episodes:     
        episodic_seed = random.randint(a=0, b=episodes)
        state = env.reset(seed=episodic_seed)   
        print(state)      
        norm_state = normalize(state[0])        
        done = False
        truncate = False
        episodic_reward = 0         
        while not done and not truncate:        
            action = epsilon_greedy(norm_state, epsilon, model)         
            next_state, reward, done, truncate, info = env.step(action)     
            episodic_reward += reward      
            norm_next_state = normalize(next_state)        
            experience = [norm_state, action, reward, done or truncate, norm_next_state]       
            epool.append(experience) 
            if training_epoch > 0: 
                state_batch, action_batch, reward_batch, done_batch, next_state_batch = epool.sample()
                state_batch_tensor = torch.from_numpy(state_batch).to(device)
                action_value_batch = model(state_batch_tensor)      
                action_value_batch = torch.gather(
                    action_value_batch,
                    dim=1,
                    index=torch.from_numpy(action_batch).to(device)
                )   
                next_state_batch_tensor = torch.from_numpy(next_state_batch).to(device)
                next_action_value_batch_tensor = target_model(next_state_batch_tensor)
                y_batch = compute_y(reward_batch, done_batch, next_action_value_batch_tensor) 
                criterion = nn.MSELoss()  # nn.SmoothL1Loss()
                loss = criterion(action_value_batch, y_batch)    
                action_value_batch_sum = action_value_batch.sum().detach().cpu().numpy()      
                loss_value = loss.detach().cpu().numpy()     
                losses.append(float(loss_value))      
                Q.append(float(action_value_batch_sum))       
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if not training_epoch % TARGET_COPY:        
                    copy_net(model, target_model)
                if not training_epoch % EPSILON_DECAY_STEP:        
                    epsilon = decay_epsilon(epsilon)
            state = next_state  
            norm_state = norm_next_state  
            training_epoch += 1 
        episodic_rewards.append(episodic_reward)        
        e += 1           

def main():
    episodes = 1000
    models = (net, target_net)
    train(models, episodes)
    
main()