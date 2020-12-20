import json
import matplotlib.pyplot as plt 
import os
import numpy as np 
import glob

base_path = '/home/anishmadan23/Desktop/Sem9/RL/project/plotting_files/'
env_names = ['2DNav','AntDir','AntVel','HalfCheetahDir','HalfCheetahVel']

env_idx=4
avg_0 = glob.glob(base_path+env_names[env_idx]+'/*maml_lr*avg_rew_0.json')[0]
avg_0_ppo = glob.glob(base_path+env_names[env_idx]+'/*ppo_*avg_rew_0.json')[0]
avg_1 = glob.glob(base_path+env_names[env_idx]+'/*maml_lr*avg_rew_1.json')[0]
avg_1_ppo = glob.glob(base_path+env_names[env_idx]+'/*ppo_*avg_rew_1.json')[0]

### bsl
avg_rew0 = np.array(json.load(open(avg_0,'r')))
avg_rew0_ppo = np.array(json.load(open(avg_0_ppo,'r')))
avg_rew1 = np.array(json.load(open(avg_1,'r')))
avg_rew1_ppo = np.array(json.load(open(avg_1_ppo,'r')))


labels = ['Baseline(MAML-TRPO)','MAML-PPO']
colors = ['#294E88','#165D0B']

fig = plt.figure(figsize=(10,9),dpi=180)
ax1=fig.add_subplot(2, 1, 1)
ax1.plot(avg_rew0[:,1],avg_rew0[:,2],c=colors[0],label=labels[0])
ax1.plot(avg_rew0_ppo[:,1],avg_rew0_ppo[:,2],c=colors[1],label=labels[1])
ax1.set_title(env_names[env_idx]+'-Before Update')
ax1.grid(True)
ax1.set_xlabel('Num iterations')
ax1.set_ylabel('Avg Returns')
ax1.legend()

ax2 = fig.add_subplot(2,1,2)
ax2.plot(avg_rew1[:,1],avg_rew1[:,2],c=colors[0],label=labels[0])
ax2.plot(avg_rew1_ppo[:,1],avg_rew1_ppo[:,2],c=colors[1],label=labels[1])
ax2.set_title(env_names[env_idx]+'-After Update')
ax2.grid(True)
ax2.set_xlabel('Num iterations')
ax2.set_ylabel('Avg Returns')
ax2.legend()
fig.subplots_adjust(hspace=0.4)
plt.show()

fig.savefig(base_path+env_names[env_idx]+'_avg_reward_num_iters_plot.png')

