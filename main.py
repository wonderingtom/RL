import torch
import numpy as np
from CQL_agent import CQL
import random
from rl_utils import ReplayBuffer
from tqdm import tqdm
import dmc
import glob
from dataloader import ReplayBuffer


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def load_data(data_path):
    """
    An example function to load the episodes in the 'data_path'.
    """
    epss = sorted(glob.glob(f'{data_path}/*.npz'))
    episodes = []
    for eps in epss:
        with open(eps, 'rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
            episodes.append(episode)
    print(len(episodes))
    return episodes

def eval(eval_env, agent, eval_episodes):
    """
    An example function to conduct online evaluation for some agentin eval_env.
    """
    returns = []
    for episode in range(eval_episodes):
        time_step = eval_env.reset()
        cumulative_reward = 0
        while not time_step.last():
            action = agent.act(time_step.observation)
            time_step = eval_env.step(action.detach().cpu().numpy())
            cumulative_reward += time_step.reward
        returns.append(cumulative_reward)
    return sum(returns) / eval_episodes


path = "./collected_data/walker_run-td3-medium/data"
replay_buffer = ReplayBuffer(path)

task_name = "walker_run"
seed = 42
eval_env = dmc.make(task_name, seed=seed)

state_dim = 24
action_dim = 6
action_bound = 1
actor_lr = 3e-4
critic_lr = 3e-3
alpha_lr = 3e-4
num_episodes = 100
hidden_dim = 128
gamma = 0.99
tau = 0.005  # 软更新参数
buffer_size = 100000
minimal_size = 1000
batch_size = 64
target_entropy = -6
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
beta = 5.0
num_random = 5
num_epochs = 100
num_trains_per_epoch = 500

agent = CQL(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
            critic_lr, alpha_lr, target_entropy, tau, gamma, device, beta,
            num_random)

return_list = []
for i in range(10):
    with tqdm(total=int(num_epochs / 10), desc='Iteration %d' % i) as pbar:
        for i_epoch in range(int(num_epochs / 10)):
            return_list.append(eval(eval_env=eval_env, agent=agent, eval_episodes=1))

            for _ in range(num_trains_per_epoch):
                b_s, b_a, b_r, b_ns = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': np.zeros(len(b_s))
                }
                agent.update(transition_dict)

            if (i_epoch + 1) % 10 == 0:
                pbar.set_postfix({
                    'epoch':
                    '%d' % (num_epochs / 10 * i + i_epoch + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
            
