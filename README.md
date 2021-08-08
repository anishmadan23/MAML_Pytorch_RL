# MAML_Pytorch_RL
This repo contains code for the RL experiments of [Model Agnostic Meta-Learning](https://arxiv.org/abs/1703.03400). Make sure you install requirements and get your Mujoco License to run the experiments here. Additionally, there is an implementation of PPO as the meta-optimizer instead of TRPO as used by the authors. This work is done as part of the RL Course Project (Monsoon 2020) [Project Report](https://docs.google.com/presentation/d/1kY24neJ085exBcUqox6RlMSgbUEI6LWGWHM8U_ksUi0/edit?usp=sharing).

![Alt text](Backward_Half_Cheetah_3.gif)


## Usage
#### Training for Navigation Task. Replace environment to switch experiments
    python main.py --env-name 2DNavigation-v0 --fast-lr 0.1  --maml 

#### Training for Locomotion Task using PPO as meta-optimizer.

    python main_ppo2.py --env-name HalfCheetahVel-v1 --fast-lr 0.1  --maml  --meta-lr 0.1 --critic_weight 0.005 --eps_clip 0.2

#### Testing 
This script is used for testing our meta-trained policies and plots the avg returns vs number of gradient steps taken for adaptation at test time.

    python test_and_plot.py

### Other Scripts 

 - **plot_eval_curves.py** :  Used for plotting avg returns vs number of iterations. Use this after downloading testing curves from tensorboard in JSON format.
 - **demo_cheetah.py** : Used for visualizing (Mujoco) the performance of trained policies for the HalfCheetah Environment. Saves a video of the visualization.
### Acknowledgement
- This code is an extension of the [repo](https://github.com/lmzintgraf/cavia) by  Luisa M Zintgraf.

 
