import torch
import torch.nn as nn
import wandb
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.memories.torch import RandomMemory
from skrl.utils import set_seed
from skrl.trainers.torch import SequentialTrainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

set_seed(42)
run = wandb.init()
torch.cuda.empty_cache()

class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
            nn.ReLU(),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
    
    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}
    
class Critic(DeterministicMixin,Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
        )
    
    def compute(self,inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}

env = load_isaaclab_env(task_name="Isaac-Lift-Cube-Franka-v0", headless=True, num_envs=1024)
env = wrap_env(env)

device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

memory = RandomMemory(memory_size=100000, num_envs=64, device=device)

models = {}
models["policy"] = StochasticActor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 4096
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["random_timesteps"] = 1000
cfg["learning_starts"] = 1000
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["mixed_precision"] = True
cfg["entropy_learning_rate"] = 5e-3
cfg["initial_entropy_value"] = 1.0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["experiment"]["wandb"] =  True
cfg["experiment"]["write_interval"] = 336
cfg["experiment"]["checkpoint_interval"] = 3360
cfg["experiment"]["directory"] = "/home/kranade/runs/torch/Isaac-Lift-Franka-v0-SAC"




agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space= env.observation_space,
            action_space=env.action_space,
            device=device)

cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

trainer.train()
torch.cuda.empty_cache()
run.finish()