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

def cost_fn(state):
    """ Cost function of the current state """
    # Weights for different terms
    W_PUSHER = 1
    W_GOAL = 2
    W_DIFF = 5

    pusher_x, pusher_y = state[0], state[1]
    box_x, box_y = state[2], state[3]
    goal_x, goal_y = self.goal[0], self.goal[1]

    pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
    box_goal = np.array([goal_x - box_x, goal_y - box_y])
    d_box = np.sqrt(np.dot(pusher_box, pusher_box))
    d_goal = np.sqrt(np.dot(box_goal, box_goal))
    diff_coord = np.abs(box_x / box_y - goal_x / goal_y)
    # the -0.4 is to adjust for the radius of the box and pusher
    return W_PUSHER * np.max(d_box - 0.4, 0) + W_GOAL * d_goal + W_DIFF * diff_coord

def generate_action_sequences(mu,sigma,plan_horizon,population_size,action_upper_bound,action_lower_bound):
    for i in range(population_size):
        if(i==0):
            action_sequences = np.clip(np.random.normal(mu, sigma),action_lower_bound[0],action_upper_bound[0])
        else:
            ipdb.set_trace()
            action_sequences = np.dstack((action_sequences,np.clip(np.random.normal(mu, sigma),action_lower_bound[0],action_upper_bound[0])))

    ipdb.set_trace()
    return action_sequences

class CEMPolicy:
    def __init__(self, action_dim,initial_mu_val,initial_sigma_val,plan_horizon,popsize, num_elites, max_iters,action_upper_bound,action_lower_bound):
        self.action_dim = action_dim
        self.initial_mu_val = initial_mu_val
        self.initial_sigma_val = initial_sigma_val
        self.plan_horizon = plan_horizon
        self.popsize = popsize
        self.num_elites = num_elites
        self.max_iters = max_iters
        self.action_upper_bound = action_upper_bound
        self.action_lower_bound = action_lower_bound
        self.mu,self.sigma = self.reset()
        self.train()

    def reset(self):
        mu = self.initial_mu_val*np.ones((self.plan_horizon,self.action_dim))
        sigma = self.initial_sigma_val*np.ones((self.plan_horizon,self.action_dim)) #using a simplified notation of (sigma1,sigma2) for each time step instead of np.identity*self.initial_sigma_val per time_step
        return mu,sigma

    def train(self):
        for i in range(self.max_iters):
            action_sequences = generate_action_sequences(self.mu,self.sigma,self.plan_horizon,self.popsize,self.action_upper_bound,self.action_lower_bound)

    def act(self, arg1, arg2):
        return np.random.uniform(size=self.action_dim) * 2 - 1
