import torch
import torch.nn as nn
import numpy as np
from utils.experience_pool import ExpPool
from algorithms.abc import ABCTrainer
from torch.utils.tensorboard import SummaryWriter       
import datetime     
from constants import *


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

    #目标网络的更新
    def copy(cls, other, name="target"):
        return cls(input_shape=other.input_shape, output_shape=other.output_shape, name=name)

class DQNTrainer:
    def _initialize(self):
        self.env_dimension = {
            "state_size": self.env.observation_space.shape[0],
            "action_size": self.env.action_space.n,
        }
        self.model = DQN(input_shape=self.env_dimension["state_size"], output_shape=self.env_dimension["action_size"])
        self.target_model = self.model.copy(self.model)
        self.pool = ExpPool(**self.env_dimension)
        self.log_dir = "run/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.epsilon = 1

    def __init__(self, env):
        self.env = env
        self._initialize()

    def epsilon_greedy(self, state):
        ep = np.random.random(1)[0]
        if ep < self.epsilon:
            return np.random.choice([0, 1])
        else:
            tensor_state = torch.from_numpy(state.reshape((1, 4)).astype(np.float32)).to(self.device)
            action_value_tensor = self.model(tensor_state)
            action = action_value_tensor.argmax(1).detach().cpu().numpy()
            return action[0]

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * RANDOM_EPSILON_DECAY, MINIMUM_EPSILON)
        return self.epsilon


# class Trainer:

#     def train(self, episodes: int = episodes):
#         model, target_model = models
#         optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#         training_epoch = -TRAINING_SIZE
#         e = 0
#         epsilon = START_RANDOM_EPSILON
#         losses = []
#         Q = []
#         episodic_rewards = []
#         while e < episodes:
#             episodic_seed = random.randint(a=0, b=episodes)
#             writer.add_scalar(f"training/{model.name}/episodic_seed", episodic_seed, e)
#             state = env.reset()
#             norm_state = normalize(state[0])
#             done = False
#             truncate = False
#             episodic_reward = 0
#             while not done and not truncate:
#                 action = epsilon_greedy(norm_state, epsilon, model)
#                 next_state, reward, done, truncate, info = env.step(action)
#                 episodic_reward += reward
#                 norm_next_state = normalize(next_state)
#                 experience = [norm_state, action, reward, done or truncate, norm_next_state]
#                 epool.append(experience)
#                 if training_epoch > 0:
#                     state_batch, action_batch, reward_batch, done_batch, next_state_batch = epool.sample()
#                     state_batch_tensor = torch.from_numpy(state_batch).to(device)
#                     action_value_batch = model(state_batch_tensor)
#                     action_value_batch = torch.gather(
#                         action_value_batch,
#                         dim=1,
#                         index=torch.from_numpy(action_batch).to(device)
#                     )
#                     next_state_batch_tensor = torch.from_numpy(next_state_batch).to(device)
#                     next_action_value_batch_tensor = target_model(next_state_batch_tensor)
#                     y_batch = compute_y(reward_batch, done_batch, next_action_value_batch_tensor)
#                     # Compute Huber loss
#                     criterion = nn.MSELoss()  # nn.SmoothL1Loss()
#                     loss = criterion(action_value_batch, y_batch)
#                     action_value_batch_sum = action_value_batch.sum().detach().cpu().numpy()
#                     loss_value = loss.detach().cpu().numpy()
#                     losses.append(float(loss_value))
#                     Q.append(float(action_value_batch_sum))
#                     writer.add_scalar(f"training/{model.name}/loss", loss_value, training_epoch)
#                     writer.add_scalar(f"training/{model.name}/epsilon", epsilon, training_epoch)
#                     writer.add_scalar(f"training/{model.name}/Q", action_value_batch_sum, training_epoch)
#                     writer.add_scalar(f"training/{model.name}/difference", difference(model, target_model), training_epoch)
#                     # Optimize the model
#                     optimizer.zero_grad()
#                     loss.backward()
#                     # for param in Qnet.parameters():
#                     #     param.grad.data.clamp_(-1, 1)
#                     optimizer.step()
#                     if not training_epoch % TARGET_COPY:
#                         copy_net(model, target_model)
#                     if not training_epoch % EPSILON_DECAY_STEP:
#                         epsilon = decay_epsilon(epsilon)
#                     if not training_epoch % SAVING_STEP:
#                         torch.save(model.state_dict(), model_path)
#                     # if not training_epoch % INFO_STEP:
#                     #     print("")
#                 # print(next_state, reward, done, truncate, info)
#                 state = next_state
#                 norm_state = norm_next_state
#                 training_epoch += 1
#             writer.add_scalar(f"training/{model.name}/episodic_reward", episodic_reward, e)
#             episodic_rewards.append(episodic_reward)
#             e += 1
#         torch.save(model.state_dict(), model_path)
#         save_path = pathlib.Path(f"results/{version}.dat")
#         save_path.parent.mkdir(exist_ok=True)
#         with open(save_path, "w") as f:
#             json.dump({
#                 "loss": losses,
#                 "Q": Q,
#                 "episodic_rewards": episodic_rewards
#             }, f, ensure_ascii=True, indent=0)

