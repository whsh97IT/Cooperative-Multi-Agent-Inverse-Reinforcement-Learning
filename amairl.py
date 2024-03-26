import maxent

import argparse
import os
import sys
import pickle
import time
import numpy as np
import shutil
import optimizers as O

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWrite

def maxent(trajectories):

    # set up features: we use one feature vector per state
    features =

    # choose our parameter initialization strategy:
    #   initialize parameters with constant
    init = O.Constant(1.0)

    # choose our optimization strategy:
    #   we select exponentiated gradient descent with linear learning-rate decay
    optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

    # actually do some inverse reinforcement learning
    reward = maxent.irl_main(trajectories, optim, init)

    return reward

from utils import *


# Updating the SimplifiedAMAIL class to include MaxEntIRL calls

use_gpu = True


class AMAIL:
    def __init__(self, n_agents, n_features, expert_trajs, learning_rate=0.01, max_cycles=100):
        self.n_agents = n_agents
        self.n_features = n_features
        self.expert_trajs = expert_trajs  # Expert trajectories for MaxEnt IRL
        self.learning_rate = learning_rate
        self.max_cycles = max_cycles

        # Initialize reward parameters for each agent using MaxEnt IRL
        self.theta = maxent(expert_trajs)

    def update_theta(self, agent_index, learned_feature_expectations):
        gradient = self.expert_feature_expectations[agent_index] - learned_feature_expectations
        self.theta[agent_index] += self.learning_rate * gradient

    def train(self):
        for cycle in range(self.max_cycles):
            learned_feature_expectations = np.random.rand(self.n_agents, self.n_features)

            for agent_index in range(self.n_agents):
                self.update_theta(agent_index, learned_feature_expectations[agent_index])
            with torch.no_grad():
                gen_batch_value = self.V(gen_batch_state)
                gen_batch_reward = self.D(gen_batch_state, gen_batch_action)

            gen_batch_advantage, gen_batch_return = estimate_advantages(gen_batch_reward, gen_batch_mask,
                                                                        gen_batch_value, self.config["gae"]["gamma"],
                                                                        self.config["gae"]["tau"],
                                                                        self.config["jointpolicy"]["trajectory_length"])

            ppo_optim_epochs = self.config["ppo"]["ppo_optim_epochs"]
            ppo_mini_batch_size = self.config["ppo"]["ppo_mini_batch_size"]
            gen_batch_size = gen_batch_state.shape[0]
            optim_iter_num = int(math.ceil(gen_batch_size / ppo_mini_batch_size))

            for _ in range(ppo_optim_epochs):
                perm = torch.randperm(gen_batch_size)

                for i in range(optim_iter_num):
                    ind = perm[slice(i * ppo_mini_batch_size,
                                     min((i + 1) * ppo_mini_batch_size, gen_batch_size))]
                    mini_batch_state, mini_batch_action, mini_batch_next_state, mini_batch_advantage, mini_batch_return, \
                        mini_batch_old_log_prob = gen_batch_state[ind], gen_batch_action[ind], gen_batch_next_state[
                        ind], \
                        gen_batch_advantage[ind], gen_batch_return[ind], gen_batch_old_log_prob[ind]

                    v_loss, p_loss = ppo_step(self.P, self.V, self.optimizer_policy, self.optimizer_value,
                                              states=mini_batch_state,
                                              actions=mini_batch_action,
                                              next_states=mini_batch_next_state,
                                              returns=mini_batch_return,
                                              old_log_probs=mini_batch_old_log_prob,
                                              advantages=mini_batch_advantage,
                                              ppo_clip_ratio=self.config["ppo"]["clip_ratio"],
                                              value_l2_reg=self.config["value"]["l2_reg"])

                    self.writer.add_scalar('train/loss/p_loss', p_loss, epoch)
                    print(f" Training episode:{epoch} ".center(80, "#"))
                    print('gen_r:', gen_r.mean().item())
                    print('expert_r:', expert_r.mean().item())
                    print('d_loss', d_loss.item())
                    self.writer.add_scalar('train/loss/v_loss', v_loss, epoch)



        def eval(self, epoch):
            self.P.eval()
            self.D.eval()
            self.V.eval()

            gen_batch = self.P.collect_samples(self.config["ppo"]["sample_batch_size"])
            gen_batch_state = torch.stack(gen_batch.state)
            gen_batch_action = torch.stack(gen_batch.action)

            gen_r = self.D(gen_batch_state, gen_batch_action)
            for expert_batch_state, expert_batch_action in self.expert_data_loader:
                expert_r = self.D(expert_batch_state.to(device), expert_batch_action.to(device))

                print(f" Evaluating episode:{epoch} ".center(80, "-"))
                print('validate_gen_r:', gen_r.mean().item())
                print('validate_expert_r:', expert_r.mean().item())

            self.writer.add_scalar("validate/reward/gen_r", gen_r.mean().item(), epoch)
            self.writer.add_scalar("validate/reward/expert_r", expert_r.mean().item(), epoch)

        def save_model(self, save_path):
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            # dump model from pkl file
            # torch.save((self.D, self.P, self.V), f"{save_path}/{self.exp_name}.pt")
            torch.save(self.D, f"{save_path}/{self.exp_name}_Discriminator.pt")
            torch.save(self.P, f"{save_path}/{self.exp_name}_JointPolicy.pt")
            torch.save(self.V, f"{save_path}/{self.exp_name}_Value.pt")

        def load_model(self, model_path):
            # load entire model
            # self.D, self.P, self.V = torch.load((self.D, self.P, self.V), f"{save_path}/{self.exp_name}.pt")
            self.D = torch.load(f"{model_path}_Discriminator.pt", map_location=device)
            self.P = torch.load(f"{model_path}_JointPolicy.pt", map_location=device)
            self.V = torch.load(f"{model_path}_Value.pt", map_location=device)
