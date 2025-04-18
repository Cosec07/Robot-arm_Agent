import torch
import torch.nn as nn
import wandb
# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

set_seed(42)
run = wandb.init()

# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU())

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}



env = load_isaaclab_env(task_name="Isaac-Lift-Cube-Franka-v0", headless=True)
env = wrap_env(env)

device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



memory = RandomMemory(memory_size=96, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 96  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4  
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01, "min_lr": 1e-5}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.01
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = True
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["wandb"] =  True
cfg["experiment"]["write_interval"] = 336
cfg["experiment"]["checkpoint_interval"] = 3360
cfg["experiment"]["directory"] = "runs/torch/Isaac-Lift-Franka-v0"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 67000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)


trainer.train()


'''
The following code is for Agent Evaluation only
from skrl.utils.huggingface import download_model_from_huggingface
 
#path = PATH_TO_checkpointfile.pth
#agent.load(path)

# start evaluation
#trainer.eval()
'''
run.finish()