import numpy as np


def sigmoid(x: float):
    return 1 / (1 + np.exp(-x))


def minmax(x: float, minimum: float, maximum: float):
    return (x - minimum) / (maximum - minimum)


def normalize_cartpole(state):
    '''
    :param state:
    :return:
    '''
    state_max = 4.8
    state2_max = 0.418
    return np.array([
        minmax(state[0], -state_max, state_max),
        sigmoid(state[1]),
        minmax(state[2], -state2_max, state2_max),
        sigmoid(state[3])
    ])


def normalize_acrobot(state):
    return np.array([
        state[0],
        state[1],
        state[2],
        state[3],
        minmax(state[4], -4*np.pi, 4*np.pi),
        minmax(state[5], -9*np.pi, 9*np.pi),
    ])


def normalize_mountain(state):
    return np.array([
        sigmoid(state[0]),
        sigmoid(state[1]),
    ])


def normalize_cliff(state):
    return np.array(state)
