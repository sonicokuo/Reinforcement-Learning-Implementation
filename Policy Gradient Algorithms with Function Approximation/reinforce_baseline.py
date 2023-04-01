# Spring 2023, 535515 Reinforcement Learning
# HW1: REINFORCE and baseline

import os
import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter
# device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" )
device = torch.device("cpu")

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_2")
        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        self.double()
        
        ########## YOUR CODE HERE (5~10 lines) ##########

        self.linear1_for_shared = nn.Linear(self.observation_dim, self.hidden_size).to(device)
        # print(self.linear1_for_shared)
        self.dropout = nn.Dropout(p = 0.5)
        self.linear2_for_policy = nn.Linear(self.hidden_size, self.action_dim).to(device)
        self.linear2_for_value = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.valuehead = nn.Linear(self.hidden_size, 1).to(device)
        ########## END OF YOUR CODE ##########
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########

        shared_linear = self.linear1_for_shared(state)
        # shared_linear = self.dropout(shared_linear)
        shared_linear = F.relu(shared_linear)
        action_scores = self.linear2_for_policy(shared_linear)
        action_prob = F.softmax(action_scores, dim=1)
        state_value = self.linear2_for_value(shared_linear)
        state_value = self.valuehead(state_value)

        ########## END OF YOUR CODE ##########

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        # print(state)
        state = torch.from_numpy(state).unsqueeze(0).to(device)#.float().unsqueeze(0)
        # print(state)
        probs, state_value = self.forward(state)
        # print(probs)
        m = Categorical(probs)
        action = m.sample()
        # print(m.log_prob(action))
        # print(action)
    

        ########## END OF YOUR CODE ##########
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.999):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        reversed_returns = []
        current_gamma = 1
        # print(self.rewards)
        for i in range(len(self.rewards)):
            R = self.rewards[len(self.rewards) -1 - i] + gamma * R
            reversed_returns.append(R)
        returns = torch.tensor(list(reversed(reversed_returns)))
        
        #use whitening transformation to modify Gt as the result of Gt - B(st)
        # returns = (returns - returns.mean()) / (returns.std())  

        for log_prob_and_value, R_i in zip(saved_actions, returns):
            #minus predicted V to reduce variance
            advantage = R_i - log_prob_and_value.value.item()

            #calculate actor loss, multiplying a negtive sign to change the update direction
            policy_losses.append( -log_prob_and_value.log_prob * advantage * current_gamma)

            #calculate critic loss with smooth_l1_loss
            sampled_value = torch.tensor([R_i])
            # print(F.mse_loss(log_prob_and_value.value, sampled_value))
            # print(F.smooth_l1_loss(log_prob_and_value.value, sampled_value))
            # print(log_prob_and_value.value.item())
            # print(log_prob_and_value.value, sampled_value)
            value_losses.append(F.smooth_l1_loss(log_prob_and_value.value[0], sampled_value) * current_gamma)

            current_gamma *= gamma
        # print(policy_losses)
        policy_losses = torch.stack(policy_losses).sum()
        # print(policy_losses)
        value_losses = torch.stack(value_losses).sum()
        # print(value_losses)
        # print(policy_losses, value_losses)

        loss = policy_losses , value_losses
        # print(loss)


        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
    # """
    #     Implement Generalized Advantage Estimation (GAE) for your value prediction
    #     TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
    #     TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
    # """

        ########## YOUR CODE HERE (8-15 lines) ##########

        pass

        
        ########## END OF YOUR CODE ##########

def train(lr=0.01):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode, 
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """
    
    # Instantiate the policy model and the optimizer
    model = Policy()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()
        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        ########## YOUR CODE HERE (10-15 lines) ##########
        for t in range(1, 10000):
            action = model.select_action(state)
            # print(env.step(action))
            state, reward, done, _ = env.step(action)
            # env.render()
            model.rewards.append(reward/100)
            ep_reward += reward
            if done:
                break
        # print(model.rewards)
        optimizer.zero_grad()
        policy_losses, value_losses = model.calculate_loss()
        # if i_episode % 200 == 0 :print(value_losses)
        loss = value_losses + policy_losses
        loss.backward()
        optimizer.step()
        model.clear_memory()
        
        
        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        #Try to use Tensorboard to record the behavior of your implementation 
        ########## YOUR CODE HERE (4-5 lines) ##########
        writer.add_scalar('train/length', t, i_episode)
        writer.add_scalar('train/policy_losses', policy_losses, i_episode)
        writer.add_scalar('train/value_losses', value_losses, i_episode)
        writer.add_scalar('train/ewma_reward', ewma_reward, i_episode)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], i_episode)

        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        # if ewma_reward > env.spec.reward_threshold:
        if ewma_reward > 130:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """     
    model = Policy()
    # model.to(device)
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    lr = 0.002
    # lr = 0.03
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train(lr)
    test(f'LunarLander_{lr}.pth')
