import gymnasium as gym         
import torch        
import torch.nn as nn  
import numpy as np      
from collections import deque       
import random       
import matplotlib.pyplot as plt     
from torch.utils.tensorboard import SummaryWriter     
import datetime     
import pathlib     
import json     
from typing import Union       
import csv
import os



from constants import *
from algorithms.dqn import DQN
from utils.experience_pool import ExpPool
from utils.normalize import *
from utils.neural_network_funcs import difference
from utils.add_noise import *
from utils.noise_env import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def copy_net(net: DQN, target: DQN):
    target.load_state_dict(net.state_dict())
    target.eval()       


def init_weights(m):
    if isinstance(m, nn.Linear):   
        nn.init.normal_(m.weight.data, mean=0, std=1)  

def init_weights2(m):
    if isinstance(m, nn.Linear):   
        nn.init.normal_(m.weight.data, mean=0, std=0.5) 



def generate_dqn_model_pair(input_shape, output_shape, typ=""):
    names = ["Qnet", "targetQnet"]     
    models = []     
    for n in names: 
        models.append(DQN(input_shape=input_shape, output_shape=output_shape, name=typ + n).to(device))
    return models 



env2norm = {
    "CartPole-v1": normalize_cartpole,
    "Acrobot-v1": normalize_acrobot, 
    "MountainCar-v0": normalize_mountain,
    "CliffWalking-v0": normalize_cliff,
}

env_name = "CartPole-v1"
# env_name = "Acrobot-v1"
# env_name = "MountainCar-v0"
# env_name = "CliffWalking-v0"

env = gym.make(env_name, render_mode="human") 
mutual_env = gym.make(env_name, render_mode="human")
def basic_info(env_name, env):
    print(f"""
Env name: {env_name},
Observation space: {env.observation_space.shape},
Action space: {env.action_space.n}
""")

basic_info(env_name, env)
algorithm = "DQN" 
version = 1  
normalize = env2norm[env_name] 

def seed_all(seed: int):
    torch.manual_seed(seed)    
    #env.seed(seed)
    random.seed(seed) 
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True  
    torch.manual_seed(seed)

env_dimension = { 
    "state_size": env.observation_space.shape[0],
    "action_size": env.action_space.n, 
}

if env_dimension["state_size"] == (): 
    env_dimension["state_size"] = 1

mutual_coef = 1

epool = ExpPool(**env_dimension) 
writer = SummaryWriter(log_dir="run/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ "-" + str(mutual_coef)+"-0") # 为TensorBoard编写日志



episodes = 1200 
seed = 2        


seed_all(seed) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Qnet, targetQnet = generate_dqn_model_pair(input_shape=env_dimension["state_size"], output_shape=env_dimension["action_size"])
mutualQnet, mutualtargetQnet = generate_dqn_model_pair(input_shape=env_dimension["state_size"], output_shape=env_dimension["action_size"],typ="mutual")#定义模型类型不同


mutual_epool = ExpPool(**env_dimension)
Qnet.apply(init_weights)
copy_net(Qnet, targetQnet) 
mutualQnet.apply(init_weights2)
copy_net(mutualQnet, mutualtargetQnet)



#保存已训练模型的路径
model_path = pathlib.Path(f"models/{env_name}/{algorithm}/{env_name}-{algorithm}-{version}-{seed}-{mutual_coef}.pth")
model_path.parent.mkdir(parents=True, exist_ok=True)   

model_path1 = pathlib.Path(f"model/{env_name}/{algorithm}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{'t'}-{1}-{seed}-{mutual_coef}-{'model'}.pth")
model_path1.parent.mkdir(parents=True, exist_ok=True)   

model_path2 = pathlib.Path(f"mutual_model/{env_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{algorithm}/{'t'}-{1}-{seed}-{mutual_coef}-{'mutual_model'}.pth")
model_path2.parent.mkdir(parents=True, exist_ok=True)   

def negetive_sigmoid(x, mu=0, sigma=1):
    return -(1 / (1 + np.exp(-(x + mu)/sigma))) + 1


def negetive_sigmoid2(x, mu=2, sigma=1):
    return - 1 / (1 + 1/np.exp(-(x - mu)/sigma)) + 1


def mutual_coef_decay(mutual_coef, episode, episodes):
    x = 8 * episode/episodes - 4  # [0, episodes] -> [-4, 4]
    return mutual_coef * negetive_sigmoid(x)


def mutual_coef_change(mutual_coef, episode, episodes):
    # [0, episodes] -> [-6, 6]
    x = 12 * episode/episodes - 6
    return mutual_coef * negetive_sigmoid2(x, mu=2, sigma=2)




def decay_epsilon(epsilon):
    return max(epsilon * RANDOM_EPSILON_DECAY, MINIMUM_EPSILON) 


def greedy(state, model):
    tensor_state = torch.from_numpy(
        state.reshape((1, env.observation_space.shape[0])).astype(np.float32)
    ).to(device)
    action_value_tensor = model(tensor_state)
    action = action_value_tensor.argmax(1).detach().cpu().numpy()
    return action[0] 
def epsilon_greedy1(state, epsilon, model):
    ep = np.random.random(1)[0] 
    if ep < epsilon:
        return env.action_space.sample() 
    else:
        return greedy(state, model) 

epsilon_random_sequence = np.random.random(MAX_TRAINING*2)
ep1 = -1
ep2 = -1
action_random_sequence = np.random.randint(2, size=MAX_TRAINING*2)


def epsilon_greedy(state, epsilon, model):
    global ep1
    ep1 += 1
    ep = epsilon_random_sequence[ep1] 
    if ep < epsilon:
        return action_random_sequence[ep1]
    else:
        return greedy(state, model) 
    
def mutual_epsilon_greedy(state, epsilon, model):
    global ep2
    ep2 += 1
    ep = epsilon_random_sequence[ep2] 
    if ep < epsilon:
        return action_random_sequence[ep2]
    else:
        return greedy(state, model) 

def load_model(env):
    model = DQN(env) 
    model.load_state_dict(torch.load(model_path)) 
    model.eval() 
    return model


def evaluate(model, episodes, seeds):
    env = gym.make(env_name, render_mode="human") 
    e = 0
    while e < episodes:
        state = env.reset(seed=seeds[e])  
        norm_state = normalize(state[0])    
        done = False 
        truncate = False
        episodic_reward = 0
        while not done and not truncate:
            action = greedy(norm_state, model) 
            next_state, reward, done, truncate, info = env.step(action) 
            state = next_state 
            norm_state = normalize(state)
            episodic_reward += reward
        writer.add_scalar(f"evaluation/{model.name}/episodic_reward-parameter_noise", episodic_reward, e) 
        e += 1


def evaluate_env(model, episodes, seeds, noise_scale):
    env_noise = NoisyCartPoleEnv(gym.make(env_name), noise_level=noise_scale) 
    e = 0
    while e < episodes: 
        state = env_noise.reset(seed=seeds[e]) 
        norm_state = normalize(state[0]) 
        done = False 
        truncate = False 
        episodic_reward = 0
        while not done and not truncate:
            action = greedy(norm_state, model) 
            next_state, reward, done, truncate, info = env_noise.noise_step(action) 
            state = next_state 
            norm_state = normalize(state)
            episodic_reward += reward
        writer.add_scalar(f"evaluation/{model.name}/episodic_reward-env_noise", episodic_reward, e) 
        e += 1


def compute_y(reward_batch, done_batch, next_action_value_batch_tensor):
    reward_batch_tensor = torch.from_numpy(reward_batch).to(device)
    done_mask_batch = 1 - done_batch.sum(axis=1).reshape((BATCH_SIZE, 1))
    done_mask_batch_tensor = torch.from_numpy(done_mask_batch).to(device)
    max_action_value_tensor = torch.max(next_action_value_batch_tensor, dim=1)[0]      
    decayed_maction_value_tensor = torch.mul(max_action_value_tensor, GAMMA).reshape((BATCH_SIZE, 1))
    masked_action_value_tensor = torch.mul(done_mask_batch_tensor, decayed_maction_value_tensor)
    return masked_action_value_tensor + reward_batch_tensor



def show_loss(losses: list):
    plt.figure()
    plt.plot(losses)
    plt.show()


def kl_divergence(pred1, pred2):
    kl = torch.sum(pred2 * (torch.log(pred2 + 1e-8) - torch.log(pred1 + 1e-8)), dim=0) #1
    return torch.mean(kl)

def train(models: tuple, episodes: int = episodes):
    model, target_model = models
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    training_epoch = -TRAINING_SIZE     
    e = 0
    epsilon = START_RANDOM_EPSILON
    losses = []    
    Q = []  
    episodic_rewards = [] 
    while e < episodes:     
        episodic_seed = random.randint(a=0, b=episodes)
        writer.add_scalar(f"training/{model.name}/episodic_seed", episodic_seed, e)
        state = env.reset(seed=episodic_seed)    
        norm_state = normalize(state[0])       
        done = False
        truncate = False
        episodic_reward = 0         
        while not done and not truncate:        
            action = epsilon_greedy1(norm_state, epsilon, model)         
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
                criterion = nn.MSELoss()  
                loss = criterion(action_value_batch, y_batch)      
                action_value_batch_sum = action_value_batch.sum().detach().cpu().numpy()        
                loss_value = loss.detach().cpu().numpy()       
                losses.append(float(loss_value))        
                Q.append(float(action_value_batch_sum))         
                writer.add_scalar(f"training/{model.name}/loss", loss_value, training_epoch)
                writer.add_scalar(f"training/{model.name}/epsilon", epsilon, training_epoch)
                writer.add_scalar(f"training/{model.name}/Q", action_value_batch_sum, training_epoch)
                writer.add_scalar(f"training/{model.name}/difference", difference(model, target_model), training_epoch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if not training_epoch % TARGET_COPY:     
                    copy_net(model, target_model)
                if not training_epoch % EPSILON_DECAY_STEP:       
                    epsilon = decay_epsilon(epsilon)
                if not training_epoch % SAVING_STEP:          
                    torch.save(model.state_dict(), model_path)
            state = next_state         
            norm_state = norm_next_state        
            training_epoch += 1    
        writer.add_scalar(f"training/{model.name}/episodic_reward", episodic_reward, e)
        episodic_rewards.append(episodic_reward)        
        e += 1           
    torch.save(model.state_dict(), model_path)
    save_path = pathlib.Path(f"results/{version}.dat")
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({
            "loss": losses,
            "Q": Q,
            "episodic_rewards": episodic_rewards
        }, f, ensure_ascii=True, indent=0)



def mutual_training(
    models: tuple,
    mutual_models: tuple,
    envs: tuple,
    episodes: int = episodes
):
    model, target_model = models       
    mutual_model, mutual_target_model = mutual_models    
    env, mutual_env = envs    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)     
    mutual_optimizer = torch.optim.Adam(mutual_model.parameters(), lr=LEARNING_RATE)       
    training_epoch = mutual_training_epoch = -TRAINING_SIZE        
    e = mutual_e = 0      
    epsilon = mutual_epsilon = START_RANDOM_EPSILON         
    random_sequence = [random.randint(0, MAX_TRAINING) for _ in range(MAX_TRAINING + 1)]
    episodic_seed = random.randint(a=0, b=episodes)       
    state_flag=0
    mutual_state_flag=0
    state = env.reset(seed=random_sequence[state_flag])

    evaluation_norm_state = norm_state = normalize(state[0])     
    mutual_state = mutual_env.reset(seed=random_sequence[mutual_state_flag])

    mutual_norm_state = normalize(mutual_state[0])
    episodic_reward = mutual_episodic_reward = 0

    csv_file = "——————.csv"
    with open(csv_file, mode='w', newline='') as file:
        writerc = csv.writer(file)
        writerc.writerow(['Epoch', 'Env Loss', 'Difference', 'Loss', 'Mutual Env Loss', 'Mutual Difference', 'Mutual Loss'])

    while True:
        action = epsilon_greedy(norm_state, epsilon, model)
        mutual_action = mutual_epsilon_greedy(mutual_norm_state, mutual_epsilon, mutual_model)
        next_state, reward, done, truncate, info = env.step(action)
        mutual_next_state, mutual_reward, mutual_done, mutual_truncate, mutual_info = mutual_env.step(mutual_action)
        

        episodic_reward += reward
        mutual_episodic_reward += mutual_reward
        norm_next_state = normalize(next_state)
        mutual_norm_next_state = normalize(mutual_next_state)
        experience = [norm_state, action, reward, done or truncate, norm_next_state]
        mutual_experience = [
            mutual_norm_state,
            mutual_action,
            mutual_reward,
            mutual_done or mutual_truncate,
            mutual_norm_next_state
        ]
        epool.append(experience)
        mutual_epool.append(mutual_experience)
        if training_epoch > 0:
            current_mutual_coef = mutual_coef
            state_batch, action_batch, reward_batch, done_batch, next_state_batch = epool.sample()
            state_batch_tensor = torch.from_numpy(state_batch).to(device)
            action_value_batch = model(state_batch_tensor)
            mutual_action_value_batch = mutual_model(state_batch_tensor)
            
            p1=nn.functional.softmax(action_value_batch,dim=0)
            q1=nn.functional.softmax(mutual_action_value_batch,dim=0)
            difference_tensor=kl_divergence(p1,q1)
            Q_difference = difference_tensor.detach().cpu().numpy()         
            action_value_batch = torch.gather(action_value_batch,dim=1,index=torch.from_numpy(action_batch).to(device))      
            action_value_batch_sum = action_value_batch.sum().detach().cpu().numpy()      
            next_state_batch_tensor = torch.from_numpy(next_state_batch).to(device)         
            next_action_value_batch_tensor = target_model(next_state_batch_tensor)         
            y_batch = compute_y(reward_batch, done_batch, next_action_value_batch_tensor)
            # Compute loss = MSELoss + Difference
            criterion = nn.MSELoss()  
            env_loss = criterion(action_value_batch, y_batch)      
            loss = env_loss + current_mutual_coef * difference_tensor       
            env_loss_value = env_loss.detach().cpu().numpy()        
            loss_value = loss.detach().cpu().numpy()       
            writer.add_scalar(f"{model.name}/env_loss", env_loss_value, training_epoch)
            writer.add_scalar(f"{model.name}/loss", loss_value, training_epoch)
            writer.add_scalar(f"{model.name}/epsilon", epsilon, training_epoch)
            writer.add_scalar(f"{model.name}/Q", action_value_batch_sum, training_epoch)
            writer.add_scalar(f"{model.name}/difference", Q_difference, training_epoch)
            optimizer.zero_grad()
            loss.backward()         
           
            optimizer.step()
            if not training_epoch % TARGET_COPY:
                copy_net(model, target_model)
            if not training_epoch % EPSILON_DECAY_STEP:
                epsilon = decay_epsilon(epsilon)
            #saving model
            if not training_epoch % SAVING_STEP:
                torch.save(model.state_dict(), model_path)
            # Training Mutual Model
            mutual_state_batch, mutual_action_batch, mutual_reward_batch, mutual_done_batch,mutual_next_state_batch = mutual_epool.sample()
            mutual_state_batch_tensor = torch.from_numpy(mutual_state_batch).to(device)
            mutual_action_value_batch1 = mutual_model(mutual_state_batch_tensor)
            # Compute the Q value from model on the same state batch
            mutual_mutual_action_value_batch = model(mutual_state_batch_tensor)
            p2=nn.functional.softmax(mutual_action_value_batch1,dim=0)
            q2=nn.functional.softmax(mutual_mutual_action_value_batch,dim=0)
            mutual_difference_tensor=kl_divergence(p2,q2)
            mutual_difference = mutual_difference_tensor.detach().cpu().numpy()         

            mutual_action_value_batch1 = torch.gather(
                mutual_action_value_batch1,
                dim=1,
                index=torch.from_numpy(mutual_action_batch).to(device)
            )
            mutual_action_value_batch_sum = mutual_action_value_batch1.sum().detach().cpu().numpy()
            mutual_next_state_batch_tensor = torch.from_numpy(mutual_next_state_batch).to(device)
            mutual_next_action_value_batch_tensor = mutual_target_model(mutual_next_state_batch_tensor)
            mutual_y_batch = compute_y(mutual_reward_batch,mutual_done_batch,mutual_next_action_value_batch_tensor)

            mutual_criterion = nn.MSELoss()  
            mutual_env_loss = mutual_criterion(mutual_action_value_batch1, mutual_y_batch)
            mutual_loss = mutual_env_loss + current_mutual_coef * mutual_difference_tensor
            mutual_env_loss_value = mutual_env_loss.detach().cpu().numpy()
            mutual_loss_value = mutual_loss.detach().cpu().numpy()
            writer.add_scalar(f"{mutual_model.name}/env_loss", mutual_env_loss_value, mutual_training_epoch)  #mutual
            writer.add_scalar(f"{mutual_model.name}/loss", mutual_loss_value, mutual_training_epoch)
            writer.add_scalar(f"{mutual_model.name}/epsilon", mutual_epsilon, mutual_training_epoch)
            writer.add_scalar(
                f"{mutual_model.name}/Q",
                mutual_action_value_batch_sum,
                mutual_training_epoch
            )
            writer.add_scalar(f"{mutual_model.name}/difference", mutual_difference, mutual_training_epoch)

            mutual_optimizer.zero_grad()
            mutual_loss.backward()
            mutual_optimizer.step()
            if not mutual_training_epoch % TARGET_COPY: 
                copy_net(mutual_model, mutual_target_model)
            if not mutual_training_epoch % EPSILON_DECAY_STEP: 
                mutual_epsilon = decay_epsilon(mutual_epsilon)
            writer.add_scalar(f"general/mutual_coef", current_mutual_coef, training_epoch)
            evaluation_action_value = model(torch.from_numpy(evaluation_norm_state.astype(np.float32)).to(device))
            evaluation_mutual_action_value = mutual_model(
                torch.from_numpy(evaluation_norm_state.astype(np.float32)).to(device))
            action_value_difference = (evaluation_action_value - evaluation_mutual_action_value).detach().cpu().numpy()         #两个模型在动作值上差异程度的指标
            action_value_difference = np.abs(action_value_difference).sum()
            writer.add_scalar(
                f"general/action_value_difference",
                action_value_difference,
                training_epoch
            )
        
            with open(csv_file, mode='a', newline='') as file:
                writerc = csv.writer(file)
                writerc.writerow([training_epoch, env_loss_value, Q_difference, loss_value, mutual_env_loss_value, mutual_difference, mutual_loss_value])

        training_epoch += 1
        mutual_training_epoch +=1
        if done or truncate:
            state_flag +=1
            state = env.reset(seed=random_sequence[state_flag])
            norm_state = normalize(state[0])        
            writer.add_scalar(f"{model.name}/episodic_reward", episodic_reward, e)
            episodic_reward = 0
            e += 1
        else:
            state = next_state
            norm_state = norm_next_state
        if mutual_done or mutual_truncate:
            mutual_state_flag +=1
            mutual_state = mutual_env.reset(seed=random_sequence[mutual_state_flag])
            mutual_norm_state = normalize(mutual_state[0])
            writer.add_scalar(f"{mutual_model.name}/mutual_episodic_reward", mutual_episodic_reward, mutual_e)
            mutual_episodic_reward = 0
            mutual_e += 1
        else:
            mutual_state = mutual_next_state
            mutual_norm_state = mutual_norm_next_state


        if training_epoch == MAX_TRAINING :
            evaluation_action_value = model(torch.from_numpy(evaluation_norm_state.astype(np.float32)).to(device))
            evaluation_mutual_action_value = mutual_model(torch.from_numpy(evaluation_norm_state.astype(np.float32)).to(device))
            print(evaluation_action_value)
            print(evaluation_mutual_action_value)
            print(evaluation_action_value - evaluation_mutual_action_value)
            break

    torch.save(model.state_dict(), model_path1)
    torch.save(mutual_model.state_dict(),model_path2)
    


def main():
    # training process
    models = (Qnet, targetQnet)
    mutual_models = (mutualQnet, mutualtargetQnet)
    envs = (env, mutual_env)
    mutual_training(models, mutual_models, envs)
    seeds = [
        random.randint(a=0, b=episodes) for _ in range(episodes)
    ]
    evaluate(model=Qnet, seeds=seeds, episodes=500)
    evaluate(model=mutualQnet, seeds=seeds,episodes=500)


    # evaluation process
    # load model
    # model=DQN(input_shape=env_dimension["state_size"], output_shape=env_dimension["action_size"])
    # model_path = pathlib.Path(f"model/{env_name}/{algorithm}/——————.pth")  
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    
    #environment noise
    # evaluate_env(model=model, seeds=seeds,episodes=500,noise_scale=0.05)
    
    #parameter noise
    # add_noise(model)
    # evaluate(model=model, seeds=seeds, episodes=200)

    writer.close()
    
main()