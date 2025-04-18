import os
import torch
import torch.nn as nn
import wandb
from skrl.resources.noises.torch import GaussianNoise
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.models.torch import Model, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.resources.preprocessors.torch import RunningStandardScaler
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
set_seed(42)
run = wandb.init()
torch.cuda.empty_cache()
env = load_isaaclab_env(task_name="Isaac-Lift-Cube-Franka-v0", num_envs=2048, headless=True)
env = wrap_env(env)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ReLU(),
                                 nn.Linear(256,128),
                                 nn.ReLU(),
                                 nn.Linear(128,64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions),
                                 nn.Tanh(),
                                 )                                 

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64,1)
                                 )

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


models = {}
models["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device)

cfg = TD3_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
cfg["smooth_regularization_noise"] = GaussianNoise(0, 0.2, device=device)
cfg["smooth_regularization_clip"] = 0.5
cfg["gradient_steps"] = 1
cfg["random_timesteps"] = 500
cfg["learning_starts"] = 500
cfg["timesteps_per_update"] = 1
cfg["batch_size"] = 4096
cfg["policy_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["discount_factor"] = 0.99
cfg["tau"] = 0.005
cfg["policy_noise"] = 0.2
cfg["noise_clip"] = 0.5
cfg["policy_update_frequency"] = 2
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["experiment"]["write_interval"] = 336
cfg["experiment"]["checkpoint_interval"] = 3360
cfg["experiment"]["directory"] = "/home/kranade/runs/torch/Isaac-Lift-Franka-v0-TD3"

agent = TD3(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
torch.cuda.empty_cache()
trainer.train()
run.finish()