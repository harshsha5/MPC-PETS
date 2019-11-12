import os
import tensorflow as tf
import numpy as np
import gym
import copy

from agent import Agent, RandomPolicy, CEMPolicy
import ipdb

# INITIAL_MU = 0
# INITIAL_SIGMA = 0.5


class MPC:
    def __init__(self, env, plan_horizon, model, popsize, num_elites, max_iters,
                 initial_mu,
                 initial_sigma,
                 num_particles=6,
                 use_gt_dynamics=True,
                 use_mpc=True,
                 use_random_optimizer=False):
        """

        :param env:
        :param plan_horizon:
        :param model: The learned dynamics model to use, which can be None if use_gt_dynamics is True
        :param popsize: Population size
        :param num_elites: CEM parameter
        :param max_iters: CEM parameter
        :param num_particles: Number of trajectories for TS1
        :param use_gt_dynamics: Whether to use the ground truth dynamics from the environment
        :param use_mpc: Whether to use only the first action of a planned trajectory
        :param use_random_optimizer: Whether to use CEM or take random actions
        """
        self.env = env
        action_dim = len(env.action_space.low)
        state_dim = len(env.observation_space.low)
        self.use_gt_dynamics, self.use_mpc, self.use_random_optimizer = use_gt_dynamics, use_mpc, use_random_optimizer
        self.num_particles = num_particles
        self.plan_horizon = plan_horizon
        self.num_nets = None if model is None else model.num_nets

        self.state_dim, self.action_dim = 8, env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low

        self.initial_sigma = initial_sigma
        self.initial_mu = initial_mu

        self.popsize = popsize
        self.max_iters = max_iters

        # Set up optimizer
        self.model = model

        if use_gt_dynamics:
            self.predict_next_state = self.predict_next_state_gt
            assert num_particles == 1
        else:
            self.predict_next_state = self.predict_next_state_model

        # TODO: write your code here
        # Initialize your planner with the relevant arguments.
        # Write different optimizers for cem and random actions respectively
        if(not self.use_random_optimizer):
            print("Using CEM Policy")
            self.num_elites = num_elites
            self.policy = CEMPolicy(self.env,self.action_dim,self.initial_mu,self.initial_sigma,self.plan_horizon,self.popsize,self.num_elites,self.max_iters,self.ac_ub,self.ac_lb,self.use_gt_dynamics)
        else:
            print("Using Random Policy")
            self.policy = RandomPolicy(self.env,self.action_dim,self.initial_mu,self.initial_sigma,self.plan_horizon,self.popsize,self.max_iters,self.ac_ub, self.ac_lb,self.use_gt_dynamics)

    def predict_next_state_model(self, states, actions):
        """ Given a list of state action pairs, use the learned model to predict the next state"""
        # TODO: write your code here
        raise NotImplementedError

    def predict_next_state_gt(self, states, actions):
        """ Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
        # TODO: write your code here
        raise NotImplementedError

    def train(self, obs_trajs, acs_trajs, rews_trajs, epochs=5):
        """
        Take the input obs, acs, rews and append to existing transitions the train model.
        Arguments:
          obs_trajs: states
          acs_trajs: actions
          rews_trajs: rewards (NOTE: this may not be used)
          epochs: number of epochs to train for
        """
        # TODO: write your code here
        raise NotImplementedError

    def reset(self):
        # TODO: write your code here
        print("Resetting MPC policy")
        self.policy.reset()

    def act(self, state, present_timestep):
        """
        Use model predictive control to find the action give current state.

        Arguments:
          state: current state
          present_timestep: current timestep
        """
        if(present_timestep==0):
            self.policy.goal = state[[-2, -1]]

        if(self.use_random_optimizer):
            best_trajectory = self.policy.train(state)
            return best_trajectory[0,:].tolist()
        else:
            mu = self.policy.train(state)
            self.policy.mu = np.vstack((mu[1:,:],np.zeros((1,self.action_dim))))
            return mu[0,:].tolist()
