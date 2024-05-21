import random
import collections
import glob
import numpy as np

class ReplayBuffer:
    def __init__(self, path, capacity=50000):
        self.buffer = collections.deque(maxlen=capacity) 
        self.load_data(path)
    
    '''
    original .npz file stores the transitions as following format;
        'observation': maybe current state
        'action': action
        'reward': reward of the action
        'discount': discounting the reward
        'physics': don't know
    '''
    def load_data(self, path):
        """
        An example function to load the episodes in the 'data_path'.
        """
        epss = sorted(glob.glob(f'{path}/*.npz'))
        for eps in epss:
            with open(eps, 'rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
                obs, act, r, _, _ = episode.values()
                for transition in zip(obs, act, r, obs[1:]):
                    self.buffer.append(transition)

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, r, next_state = zip(*transitions)
        return state, action, r, next_state

    def size(self): 
        return len(self.buffer) 