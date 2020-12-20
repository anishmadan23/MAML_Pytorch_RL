import numpy as np
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv as HalfCheetahEnv_
# from PIL import Image

class HalfCheetahEnv(HalfCheetahEnv_):
	def _get_obs(self):
		return np.concatenate([
			self.sim.data.qpos.flat[1:],
			self.sim.data.qvel.flat,
			self.get_body_com("torso").flat,
		]).astype(np.float32).flatten()

	def viewer_setup(self):
		camera_id = self.model.camera_name2id('track')
		self.viewer.cam.type = 2
		self.viewer.cam.fixedcamid = camera_id
		self.viewer.cam.distance = self.model.stat.extent * 0.35
		# Hide the overlay
		self.viewer._hide_overlay = True

	def render(self, mode='human'):
		if mode == 'rgb_array':
			width, height = 1280,720
			self.task_type='goal'
			self._get_viewer(mode).render(width,height)
			# window size used for old mujoco-py:
			data = self._get_viewer(mode).read_pixels(width, height, depth=False)
			data = np.rot90(np.rot90(data))
			data = np.fliplr(data)
			# data = np.asarray(data[::-1, :, :], dtype=np.uint8)
			# from PIL import Image, ImageFont, ImageDraw
			# data = Image.fromarray(data)
#########################################################################################################################
			# draw = ImageDraw.Draw(data)
			# # font = ImageFont.truetype("/System/Library/fonts/SFNSText.ttf", 50)
			# font=ImageFont.load_default()
			# # draw.text((x, y),"Sample Text",(r,g,b))

			# y_offset = 30
			# # draw.text((1200, 0+y_offset), "Number of updates: {}".format(self.num_updates), (255, 255, 255), font=font)
			# draw.text((1200, 0+y_offset), "Number of updates: {}".format(10), (255, 255, 255), font=font)

			# # add task-relevant text
			# if self.task_type == 'goal':
			#     if self._task == -1:
			#         draw.text((300, 100+y_offset), "Task: Walk backwards", (255, 255, 255), font=font)
			#     elif self._task == +1:
			#         draw.text((300, 100+y_offset), "Task: Walk forwards", (255, 255, 255), font=font)
			#     # prediction
			#     go_left_prob = self.direction_pred
			#     # go_left_prob = self._goal_dir

			#     # draw.text((1500, 100+y_offset), "Predictions from context parameters:", (255, 255, 255), font=font)
			#     if self._task == -1:
			#         draw.text((600, 170+y_offset), "Backwards: {} %".format(np.round(100*go_left_prob[0][0], 2)), (0, 255, 0), font=font)
			#         draw.text((600, 240+y_offset), "Forwards:  {} %".format(np.round(100*go_left_prob[0][1], 2)), (255, 0, 0), font=font)
			#     else:
			#         draw.text((600, 170+y_offset), "Backwards: {} %".format(np.round(100*go_left_prob[0][0], 2)), (255, 0, 0), font=font)
			#         draw.text((600, 240+y_offset), "Forwards:  {} %".format(np.round(100*go_left_prob[0][1], 2)), (0, 255, 0), font=font)

			# elif self.task_type == 'vel':
			#     draw.text((300, 100+y_offset), "Task: Walk at velocity {}".format(self._task), (255, 255, 255), font=font)
			#     draw.text((300, 170+y_offset), "Current velocity:      {}".format(np.round(float(self.forward_vel), 2)), (255, 255, 255), font=font)

			# # add reward text
			# # draw.text((700, 170+y_offset), "Return (total):   {}".format(np.round(float(self.collected_return), 2)), (255, 255, 255), font=font)
			# # draw.text((700, 240+y_offset), "Return (forward): {}".format(np.round(float(self.forward_return), 2)), (255, 255, 255), font=font)
			# draw.text((300, 170+y_offset), "Return (total):   {}".format(np.round(float(self.infos['reward_forward']+self.infos['reward_ctrl']), 2)), (255, 255, 255), font=font)
			# draw.text((300, 240+y_offset), "Return (forward): {}".format(np.round(float(self.infos['reward_forward']), 2)), (255, 255, 255), font=font)


####################################################################################################################
			# draw = ImageDraw.Draw(data)
			# # font = ImageFont.truetype("/System/Library/fonts/SFNSText.ttf", 50)
			# font=ImageFont.load_default()
			# # draw.text((x, y),"Sample Text",(r,g,b))

			# y_offset = 300
			# # draw.text((1200, 0+y_offset), "Number of updates: {}".format(self.num_updates), (255, 255, 255), font=font)
			# draw.text((1200, 0+y_offset), "Number of updates: {}".format(10), (255, 255, 255), font=font)

			# # add task-relevant text
			# if self.task_type == 'goal':
			#     if self.task == -1:
			#         draw.text((700, 100+y_offset), "Task: Walk backwards", (255, 255, 255), font=font)
			#     elif self.task == +1:
			#         draw.text((700, 100+y_offset), "Task: Walk forwards", (255, 255, 255), font=font)
			#     # prediction
			#     go_left_prob = self.direction_pred
			#     draw.text((1500, 100+y_offset), "Predictions from context parameters:", (255, 255, 255), font=font)
			#     if self.task == -1:
			#         draw.text((1600, 170+y_offset), "Backwards: {} %".format(np.round(100*go_left_prob[0][0], 2)), (0, 255, 0), font=font)
			#         draw.text((1600, 240+y_offset), "Forwards:  {} %".format(np.round(100*go_left_prob[0][1], 2)), (255, 0, 0), font=font)
			#     else:
			#         draw.text((1600, 170+y_offset), "Backwards: {} %".format(np.round(100*go_left_prob[0][0], 2)), (255, 0, 0), font=font)
			#         draw.text((1600, 240+y_offset), "Forwards:  {} %".format(np.round(100*go_left_prob[0][1], 2)), (0, 255, 0), font=font)

			# elif self.task_type == 'vel':
			#     draw.text((700, 100+y_offset), "Task: Walk at velocity {}".format(self._task), (255, 255, 255), font=font)
			#     draw.text((700, 170+y_offset), "Current velocity:      {}".format(np.round(float(self.forward_vel), 2)), (255, 255, 255), font=font)

			# # add reward text
			# # draw.text((700, 170+y_offset), "Return (total):   {}".format(np.round(float(self.collected_return), 2)), (255, 255, 255), font=font)
			# # draw.text((700, 240+y_offset), "Return (forward): {}".format(np.round(float(self.forward_return), 2)), (255, 255, 255), font=font)
			# draw.text((700, 170+y_offset), "Return (total):   {}".format(np.round(float(self.infos['reward_forward']+self.infos['reward_ctrl']), 2)), (255, 255, 255), font=font)
			# draw.text((700, 240+y_offset), "Return (forward): {}".format(np.round(float(self.infos['reward_forward']), 2)), (255, 255, 255), font=font)

			data = np.array(data)

			return data
		elif mode == 'human':
			self._get_viewer(mode).render()

	# def render(self, mode='human'):
	# 	if mode == 'rgb_array':
	# 		width, height = 500, 500

	# 		self._get_viewer(mode).render(width,height)
	# 		# window size used for old mujoco-py:
	# 		data = self._get_viewer(mode).read_pixels(width, height, depth=False)
	# 		return data
	# 	elif mode == 'human':
	# 		width, height = 500, 500
	# 		self._get_viewer(mode).render()
			
	# 		data = self._get_viewer(mode).read_pixels(width, height, depth=False)
	# 		data = np.rot90(np.rot90(data))
	# 		data = np.fliplr(data)


class HalfCheetahVelEnv(HalfCheetahEnv):
	"""Half-cheetah environment with target velocity, as described in [1]. The 
	code is adapted from
	https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py

	The half-cheetah follows the dynamics from MuJoCo [2], and receives at each 
	time step a reward composed of a control cost and a penalty equal to the 
	difference between its current velocity and the target velocity. The tasks 
	are generated by sampling the target velocities from the uniform 
	distribution on [0, 2].

	[1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
		Meta-Learning for Fast Adaptation of Deep Networks", 2017 
		(https://arxiv.org/abs/1703.03400)
	[2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for 
		model-based control", 2012 
		(https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
	"""

	def __init__(self, task={}):
		self._task = task
		self._goal_vel = task.get('velocity', 0.0)
		super(HalfCheetahVelEnv, self).__init__()

	def step(self, action):
		xposbefore = self.sim.data.qpos[0]
		self.do_simulation(action, self.frame_skip)
		xposafter = self.sim.data.qpos[0]

		forward_vel = (xposafter - xposbefore) / self.dt
		self.forward_vel = forward_vel
		forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
		ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

		observation = self._get_obs()
		reward = forward_reward - ctrl_cost
		done = False
		self.infos = dict(reward_forward=forward_reward,
					 reward_ctrl=-ctrl_cost, task=self._task)
		return (observation, reward, done, self.infos)

	def sample_tasks(self, num_tasks):
		velocities = self.np_random.uniform(0.0, 2.0, size=(num_tasks,))
		tasks = [{'velocity': velocity} for velocity in velocities]
		return tasks

	def reset_task(self, task):
		self._task = task
		self._goal_vel = task['velocity']


class HalfCheetahDirEnv(HalfCheetahEnv):
	"""Half-cheetah environment with target direction, as described in [1]. The 
	code is adapted from
	https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand_direc.py

	The half-cheetah follows the dynamics from MuJoCo [2], and receives at each 
	time step a reward composed of a control cost and a reward equal to its 
	velocity in the target direction. The tasks are generated by sampling the 
	target directions from a Bernoulli distribution on {-1, 1} with parameter 
	0.5 (-1: backward, +1: forward).

	[1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
		Meta-Learning for Fast Adaptation of Deep Networks", 2017 
		(https://arxiv.org/abs/1703.03400)
	[2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for 
		model-based control", 2012 
		(https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
	"""

	def __init__(self, task={}):
		self._task = task
		self._goal_dir = task.get('direction', 1)
		super(HalfCheetahDirEnv, self).__init__()

	def step(self, action):
		xposbefore = self.sim.data.qpos[0]
		self.do_simulation(action, self.frame_skip)
		xposafter = self.sim.data.qpos[0]

		forward_vel = (xposafter - xposbefore) / self.dt
		forward_reward = self._goal_dir * forward_vel
		ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

		observation = self._get_obs()
		reward = forward_reward - ctrl_cost
		done = False
		infos = dict(reward_forward=forward_reward,
					 reward_ctrl=-ctrl_cost, task=self._task)
		return (observation, reward, done, infos)

	def sample_tasks(self, num_tasks):
		directions = 2 * self.np_random.binomial(1, p=0.5, size=(num_tasks,)) - 1
		tasks = [{'direction': direction} for direction in directions]
		return tasks

	def reset_task(self, task):
		self._task = task
		self._goal_dir = task['direction']
