import numpy as np
import ipdb


class Agent:
    def __init__(self, env):
        self.env = env

    def sample(self, horizon, policy):
        """
        Sample a rollout from the agent.

        Arguments:
          horizon: (int) the length of the rollout
          policy: the policy that the agent will use for actions
        """
        rewards = []
        states, actions, reward_sum, done = [self.env.reset()], [], 0, False

        policy.reset()
        for t in range(horizon):
            # print('time step: {}/{}'.format(t, horizon))
            actions.append(policy.act(states[t], t))
            state, reward, done, info = self.env.step(actions[t])
            states.append(state)
            reward_sum += reward
            rewards.append(reward)
            if done:
                # print(info['done'])
                break

        # print("Rollout length: ", len(actions))

        return {
            "obs": np.array(states),
            "ac": np.array(actions),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
        }


class RandomPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def reset(self):
        pass

    def act(self, arg1, arg2):
        return np.random.uniform(size=self.action_dim) * 2 - 1

class CEMPolicy:
    def __init__(self, action_dim,initial_mu_val,initial_sigma_val,plan_horizon,popsize, num_elites, max_iters):
        self.action_dim = action_dim
        self.initial_mu_val = initial_mu_val
        self.initial_sigma_val = initial_sigma_val
        self.plan_horizon = plan_horizon
        self.popsize = popsize
        self.num_elites = num_elites
        self.max_iters = max_iters
        self.mu,self.sigma = self.reset()

    def reset(self):
        mu = self.initial_mu_val*np.ones(self.action_dim)
        sigma = self.initial_sigma_val*np.identity(self.action_dim)
        ipdb.set_trace()
        return mu,sigma

    def act(self, arg1, arg2):
        return np.random.uniform(size=self.action_dim) * 2 - 1
