import multiprocessing as mp

import gym
import torch
import matplotlib.pyplot as plt
from envs.subproc_vec_env import SubprocVecEnv
from episode import BatchEpisodes


def make_env(env_name):
    def _make_env():
        return gym.make(env_name)

    return _make_env


class BatchSampler(object):
    def __init__(self, env_name, batch_size, device, seed, num_workers=mp.cpu_count() - 1):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)], queue=self.queue)
        self.envs.seed(seed)
        self._env = gym.make(env_name)
        self._env.seed(seed)

    def sample(self, policy, params=None, gamma=0.95, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        episodes = BatchEpisodes(batch_size=batch_size, gamma=gamma, device=self.device)
        for i in range(batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            # self._env.render()

            # self._env.render('rgb_array')
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def sample_mj(self, policy, params=None, gamma=0.95, batch_size=None):
        self.num_workers=1
        if batch_size is None:
            batch_size = self.batch_size
        episodes = BatchEpisodes(batch_size=batch_size, gamma=gamma, device=self.device)
        for i in range(batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            # self._env.render()
            self._env.render('human')
            self._env.render('rgb_array')
            print("RUNNING")
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def test_sample(self, policy, params=None, gamma=0.95, batch_size=None):
        my_env = gym.make(self.env_name)
        frames = []
        # episodes = BatchEpisodes(batch_size=batch_size, gamma=gamma, device=self.device)
        # render = lambda : plt.imshow(my_env.render(mode='rgb_array'))
        # for i in range(batch_size//4):
        for i in range(4):

            print(i)
            observations = my_env.reset()
            for t in range(80):
                # render()
                # if t==0 and i==0:
                #     my_env.render('human')
                # else:
                my_env.render('human')
                frame = my_env.render('rgb_array')
                print(frame.shape)
                # with torch.no_grad():
                frames.append(frame)
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
                new_observations, rewards, dones, _ = my_env.step(actions)
                # episodes.append(observations, actions, rewards, 0)
                # print(observations)
                observations = new_observations
    
        # my_env.monitor.close()
        # if batch_size is None:
        #     batch_size = self.batch_size
        # episodes = BatchEpisodes(batch_size=batch_size, gamma=gamma, device=self.device)
        # for i in range(batch_size):
        #     self.queue.put(i)
        # for _ in range(self.num_workers):
        #     self.queue.put(None)
        # observations, batch_ids = self.envs.reset()
        # dones = [False]
        # while (not all(dones)) or (not self.queue.empty()):
        #     with torch.no_grad():
        #         observations_tensor = torch.from_numpy(observations).to(device=self.device)
        #         actions_tensor = policy(observations_tensor, params=params).sample()
        #         actions = actions_tensor.cpu().numpy()
        #     new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
        #     episodes.append(observations, actions, rewards, batch_ids)
        #     observations, batch_ids = new_observations, new_batch_ids
        # return episodes
        return frames
    # def play_sample(self, policy, params=None, gamma=0.95, batch_size=None):


    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks
