import datetime
import json
import os
import matplotlib.pyplot as plt
import time

import numpy as np
import scipy.stats as st
import torch
from tensorboardX import SummaryWriter
import sys
sys.path.append('/home/anishmadan23/Desktop/Sem9/RL/project/rl')
import utils
from arguments import parse_args
from baseline import LinearFeatureBaseline
from metalearner import MetaLearner
from policies.categorical_mlp import CategoricalMLPPolicy
from policies.normal_mlp import NormalMLPPolicy, CaviaMLPPolicy
from sampler import BatchSampler
import moviepy.editor as mpy
# import logger
import cv2 

import moviepy.video.io.ImageSequenceClip

def save_vid(fname,frames,fps=8.0):
# image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".png")]
	clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=fps)
	clip.write_videofile(fname)


def _save_video(fname, frames, fps=8.0):
        path = fname

        def f(t):
            frame_length = len(frames)
            new_fps = 1.0 / (1.0 / fps + 1.0 / frame_length)
            idx = min(int(t * new_fps), frame_length - 1)
            return frames[idx]

        video = mpy.VideoClip(f, duration=len(frames) / fps + 2)

        video.write_videofile(path, fps, verbose=False, logger=None)
        print("Video saved: {}".format(path))

# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)  # Create a window to init GLFW.
args = parse_args()
args.seed=0
# args.restore_model='/home/anishmadan23/Desktop/Sem9/RL/project/rl/saves/lr=0.1tau=1.0_17_12_2020_20_57_41/policy-420.pt'
args.restore_model='/home/anishmadan23/Desktop/Sem9/RL/project/rl/saves/lr=0.1tau=1.0_17_12_2020_20_59_32/policy-480.pt'
utils.set_seed(args.seed, cudnn=args.make_deterministic)

env_name = 'HalfCheetahDir-v1'
sampler = BatchSampler(env_name, batch_size=1, num_workers=args.num_workers,
                       device=args.device, seed=args.seed)

policy = NormalMLPPolicy(
                int(np.prod(sampler.envs.observation_space.shape)),
                int(np.prod(sampler.envs.action_space.shape)),
                hidden_sizes=(args.hidden_size,) * args.num_layers)


policy.load_state_dict(torch.load(args.restore_model))
policy = policy.to(args.device)
	# policy.eval()   # check this function
print('Loaded Policy')

test_tasks = sampler.sample_tasks(num_tasks=1)
print('Test Task',test_tasks,args.test_batch_size)
baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))

metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau,
                              device=args.device)

_,frames = metalearner.test_sample_mj(test_tasks, num_steps=args.num_test_steps,
                                     batch_size=args.test_batch_size, halve_lr=args.halve_test_lr)

#### saving frames : Sanity check #####
# os.makedirs('frames',exist_ok=True)
# for idx,frame in enumerate(frames):
# 	frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
# 	cv2.imwrite('frames/'+str(idx)+'.png',frame)


#### Uncomment for saving video : Problem of color of frames retreived exists in Mujoco

# print('Rendering Video...')
# save_vid('Backward_Half_Cheetah.mp4',frames)
# print('Done!')
# print('frames length',len(frames))
# print(frames[0].shape)

