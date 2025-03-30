

class RewardCounter:
    def __init__(self) -> None:
        self.rewards = []
        self.epoch = 0
    
    def add(self, reward):
        self.rewards.append(reward)
        self.epoch += 1

    @property
    def episodic_reward(self):
        return sum(self.rewards) / self.epoch
    
    def reset(self):
        self.epoch = 0
