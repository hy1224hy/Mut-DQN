import pathlib
import numpy as np
import json
import matplotlib.pyplot as plt

result_path = pathlib.Path("results")

losses = []
Q = []
episodic_rewards = []
ind = 0
loss_minimum = 0

choose_version = [10, 11, 12, 13, 14]

for v in choose_version:
    file_name = result_path / f"{v}.dat"
    with open(file_name, "r") as f:
        res = json.load(f)
        loss_len = len(res["loss"])
        if ind == 0:
            loss_minimum = loss_len
        else:
            loss_minimum = min(loss_len, loss_minimum)
        losses.append(res["loss"])
        Q.append(res["Q"])
        episodic_rewards.append(res["episodic_rewards"])
        ind += 1


for i in range(ind):
    losses[i] = losses[i][:loss_minimum]
    Q[i] = Q[i][:loss_minimum]


loss_array = np.array(losses)
Q_array = np.array(Q)
rewards_array = np.array(episodic_rewards)

average_loss = np.average(loss_array, axis=0)
std_loss = np.std(loss_array, axis=0)
max_loss = np.max(loss_array, axis=0)
min_loss = np.min(loss_array, axis=0)
# print(average_loss.shape, std_loss.shape)

loss_bottom = min_loss
loss_above = max_loss

average_Q = np.average(Q_array, axis=0)
std_Q = np.std(Q_array, axis=0)
max_Q = np.max(Q_array, axis=0)
min_Q = np.min(Q_array, axis=0)

average_reward = np.average(rewards_array, axis=0)
std_reward = np.std(rewards_array, axis=0)
max_reward = np.max(rewards_array, axis=0)
min_reward = np.min(rewards_array, axis=0)
reward_x = list(range(len(average_reward)))
reward_bottom = average_reward - std_reward
reward_above = average_reward + std_reward

x = list(range(len(average_Q)))
bottom_line = average_Q - std_Q
above_line = average_Q + std_Q

fig, ax1 = plt.subplots()
ax1.plot(average_Q, color="b", alpha=0.6)
# ax.plot(above_line, color="b", alpha=0.3)
# ax.plot(bottom_line, color="b", alpha=0.3)
ax1.fill_between(x, above_line, bottom_line, color="b", alpha=0.3)

fig2, ax2 = plt.subplots()
ax2.plot(average_reward, color="r", alpha=0.6)
ax2.fill_between(reward_x, reward_bottom, reward_above, color="r", alpha=0.3)


fig3, ax3 = plt.subplots()
ax3.plot(average_loss, color="g", alpha=0.6)
ax3.fill_between(x, loss_bottom, loss_above, color="g", alpha=0.3)

plt.show()

