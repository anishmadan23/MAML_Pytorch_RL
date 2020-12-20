#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import json
import os
import matplotlib.pyplot as plt
import time
import sys
sys.argv = ['']
import numpy as np
import scipy.stats as st
import torch
from tensorboardX import SummaryWriter
from metalearner_ppo2 import MetaLearnerPPO
from copy import deepcopy
import utils
from arguments import parse_args
from baseline import LinearFeatureBaseline
from metalearner import MetaLearner
from policies.categorical_mlp import CategoricalMLPPolicy
from policies.normal_mlp import NormalMLPPolicy, CaviaMLPPolicy
from sampler import BatchSampler


# In[2]:


os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[3]:



def get_returns(episodes_per_task):

    # sum up for each rollout, then take the mean across rollouts
    returns = []
    for task_idx in range(len(episodes_per_task)):
        curr_returns = []
        episodes = episodes_per_task[task_idx]
        for update_idx in range(len(episodes)):
            # compute returns for individual rollouts
            ret = (episodes[update_idx].rewards * episodes[update_idx].mask).sum(dim=0)
            curr_returns.append(ret)
        # result will be: num_evals * num_updates
        returns.append(torch.stack(curr_returns, dim=1))

    # result will be: num_tasks * num_evals * num_updates
    returns = torch.stack(returns)
    returns = returns.reshape((-1, returns.shape[-1]))

    return returns


def total_rewards(episodes_per_task, interval=False):

    returns = get_returns(episodes_per_task).cpu().numpy()

    mean = np.mean(returns, axis=0)
    conf_int = st.t.interval(0.95, len(mean) - 1, loc=mean, scale=st.sem(returns, axis=0))
    conf_int = mean - conf_int
    if interval:
        return mean, conf_int[0]
    else:
        return mean


# In[4]:


def main(args):
    print('starting....')

    utils.set_seed(args.seed, cudnn=args.make_deterministic)
    args.maml=True
    assert(args.restore_model is not None)
    base_save_dir = os.path.dirname(args.restore_model)
    print(base_save_dir)
    train_config_file = open(os.path.join(base_save_dir,'config.json'),'r')
    train_config = json.load(train_config_file)
    args.env_name = train_config['env_name']
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
                                            'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
                                            '2DNavigation-v0'])

    # subfolders for logging
    method_used = 'maml' if args.maml else 'cavia'
    num_context_params = str(args.num_context_params) + '_' if not args.maml else ''
    # output_name = num_context_params + 'lr=' + str(args.fast_lr) + 'tau=' + str(args.tau)
    # output_name += '_' + datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    project_base_path = '/home/anish/projects/proj/cavia/rl/'
    assert(args.restore_model is not None)
    base_save_dir = os.path.dirname(args.restore_model)

    train_config_file = open(os.path.join(base_save_dir,'config.json'),'r')
    train_config = json.load(train_config_file)
    train_env_name = train_config['env_name']

    test_log_folder = base_save_dir+'/'
    test_save_folder = base_save_dir

    # log_folder = os.path.join(os.path.join(dir_path, 'logs'), args.env_name, method_used, output_name)
    # save_folder = os.path.join(os.path.join(dir_path, 'saves'), output_name)
    # if not os.path.exists(save_folder):
    # 	os.makedirs(save_folder)
    # if not os.path.exists(log_folder):
    #     os.makedirs(log_folder)

    # initialise tensorboard writer
    # writer = SummaryWriter(log_folder)

    # save config file
    with open(os.path.join(test_save_folder, 'config_test.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)
    with open(os.path.join(test_log_folder, 'config_test.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size, num_workers=args.num_workers,
                           device=args.device, seed=args.seed)

    if continuous_actions:
        if not args.maml:
            policy = CaviaMLPPolicy(
                int(np.prod(sampler.envs.observation_space.shape)),
                int(np.prod(sampler.envs.action_space.shape)),
                hidden_sizes=(args.hidden_size,) * args.num_layers,
                num_context_params=args.num_context_params,
                device=args.device
            )
            
        else:
            policy = NormalMLPPolicy(
                int(np.prod(sampler.envs.observation_space.shape)),
                int(np.prod(sampler.envs.action_space.shape)),
                hidden_sizes=(args.hidden_size,) * args.num_layers
            )
    else:
        if not args.maml:
            raise NotImplementedError
        else:
            policy = CategoricalMLPPolicy(
                int(np.prod(sampler.envs.observation_space.shape)),
                sampler.envs.action_space.n,
                hidden_sizes=(args.hidden_size,) * args.num_layers)
    policy2 = deepcopy(policy)
    ### load policy 
    policy.load_state_dict(torch.load(args.restore_model))
    policy2.load_state_dict(torch.load(args.restore_model2))
    
#     policy.eval()   # check this function
    print('Loaded Policy')

    # initialise baseline
    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))

    # initialise meta-learner
    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau,
                              device=args.device)
    
    metalearner2 = MetaLearnerPPO(sampler, policy, baseline, gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau,
                              device=args.device)
    print(args.num_batches,args.meta_batch_size,args.num_test_steps,args.test_batch_size,args.halve_test_lr)
    # for batch in range(args.num_batches):

    test_tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
    print('Tasks Sampled!')
    

    test_episodes = metalearner.test(test_tasks, num_steps=args.num_test_steps,
                                     batch_size=args.test_batch_size, halve_lr=args.halve_test_lr)
    
    test_episodes2 = metalearner2.test(test_tasks, num_steps=args.num_test_steps,
                                     batch_size=args.test_batch_size, halve_lr=args.halve_test_lr)
    print('Tested!')
    all_returns = total_rewards(test_episodes, interval=True)
    all_returns2 = total_rewards(test_episodes2, interval=True)

    for num in range(args.num_test_steps + 1):
        print('evaluation_rew/avg_rew '+str(num), all_returns[0][num])
        print('evaluation_cfi/avg_rew ' + str(num), all_returns[1][num])
    return all_returns,test_tasks,test_episodes,all_returns2,test_episodes2


# In[5]:


restore_model= '/home/anish/projects/proj/cavia/rl/plotting_files/AntVel/trpo_policies/policy-430.pt'
fast_lr = 0.1
meta_batch_size = 40


# In[6]:


args = parse_args()
args.restore_model = restore_model
args.restore_model2 = '/home/anish/projects/proj/cavia/rl/plotting_files/AntVel/ppo_policies/policy-430.pt'
args.fast_lr = fast_lr
args.meta_batch_size = meta_batch_size

avg_returns,test_tasks,test_eps,avg_returns2,test_eps2 = main(args)


# In[7]:


print(test_tasks)


# In[8]:


base_save_dir = '/home/anish/projects/proj/cavia/rl/plotting_files/'
title = 'AntVel- Goal_Velocity'
plt.figure(figsize=(12,7),dpi=250)
plt.title(title)
# plt.plot(returns[0],color='r', linewidth=3, label='Ground Truth',zorder=1)
plt.plot(avg_returns[0],c='#294E88',marker='o',label='MAML-TRPO')
plt.plot(avg_returns2[0],c='#165D0B',marker='s',label='MAML-PPO')

plt.legend()
plt.xlabel('Number of Gradient Steps')
plt.ylabel('Average Return')
plt.grid(True)
plt.savefig(base_save_dir+title+'.png')
plt.show()


# In[ ]:




