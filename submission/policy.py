from re import L
import sys
from pathlib import Path
# from turtle import forward
from typing import Any, Dict
# from xml.parsers.expat import model


# for TD3 policy
import os
import random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from math import pi

seed = 1024
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# To import submission folder
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

class BasePolicy:
    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        raise NotImplementedError


def submitted_wrappers():
    """Return environment wrappers for wrapping the evaluation environment.
    Each wrapper is of the form: Callable[[env], env]. Use of wrappers is
    optional. If wrappers are not used, return empty list [].

    Returns:
        List[wrappers]: List of wrappers. Default is empty list [].
    """

    from action import Action as DiscreteAction
    from observation import Concatenate, FilterObs, SaveObs

    from smarts.core.controllers import ActionSpaceType
    from smarts.env.wrappers.format_action import FormatAction
    from smarts.env.wrappers.format_obs import FormatObs
    from smarts.env.wrappers.frame_stack import FrameStack

    # fmt: off
    wrappers = [
        FormatObs,
        lambda env: FormatAction(env=env, space=ActionSpaceType["TargetPose"]),
        SaveObs,
        DiscreteAction,
        FilterObs,
        lambda env: FrameStack(env=env, num_stack=3),
        lambda env: Concatenate(env=env, channels_order="first"),
    ]
    # fmt: on

    return wrappers

# class Policy(BasePolicy):
class st3__PPO_Policy(BasePolicy):
    """Policy class to be submitted by the user. This class will be loaded
    and tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """

        import stable_baselines3 as sb3lib
        import network

        model_path = Path(__file__).resolve().parents[0] / "saved_model.zip"
        self.model = sb3lib.PPO.load(model_path)

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            action, _ = self.model.predict(observation=agent_obs, deterministic=True)
            wrapped_act.update({agent_id: action})

        return wrapped_act


# class Policy(BasePolicy):
class RandomPolicy(BasePolicy):
    """A sample policy with random actions. Note that only the class named `Policy`
    will be tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """
        import gym

        self._action_space = gym.spaces.Discrete(4)

    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.

        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            action = self._action_space.sample()
            wrapped_act.update({agent_id: action})

        return wrapped_act


"""Replay Buffery"""
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity 
        self.buffer = [] 
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) 
        state, action, reward, next_state, done =  zip(*batch) 
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


"""D3QN Policy"""

class CNN_ReLU_BatchNorm(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1):
        super(CNN_ReLU_BatchNorm, self).__init__()
        """3dcnn->BN->relu"""
        self.cnn = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
            #         kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                    kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99),
            nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.99),
            nn.ReLU(),
        )

    def forward(self, feature):
        feature = self.cnn(feature)
        return feature

class FeatureExtracter(nn.Module):
    def __init__(self, in_channels=3, out_channels=[32, 64, 64, 128], feat_dim=512):
        super().__init__()
        self.batchnorm = nn.BatchNorm3d(in_channels, eps=0.001, momentum=0.99)
        self.cnn_relu_batchnorm1 = CNN_ReLU_BatchNorm(in_channels=in_channels, out_channels=out_channels[0], kernel_size=(3,3,3), stride=(1,2,2))
        self.cnn_relu_batchnorm2 = CNN_ReLU_BatchNorm(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=(3,3,3), stride=(1,2,2))
        self.cnn_relu_batchnorm3 = CNN_ReLU_BatchNorm(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=(3,3,3), stride=(1,2,2))
        self.cnn_relu_batchnorm4 = CNN_ReLU_BatchNorm(in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=(3,3,3), stride=(1,2,2))
        
        self.cnn = nn.Sequential(
            self.batchnorm,
            self.cnn_relu_batchnorm1,
            self.cnn_relu_batchnorm2,
            self.cnn_relu_batchnorm3,
            self.cnn_relu_batchnorm4
        )
        self.linear = nn.Linear(128*3*7*7 + 3*2, feat_dim)

    def forward(self, rgb, goal_distance, goal_heading):
        #rgb <class 'numpy.ndarray'> (bs, 3, 3, 112, 112)
        #goal_distance <class 'numpy.ndarray'> (bs, 3)
        #goal_heading <class 'numpy.ndarray'> (bs, 3)

        rgb = self.cnn(rgb)
        rgb = rgb.view(rgb.shape[0], -1) # (bs, 18816)

        input_ = torch.cat((rgb, goal_distance, goal_heading), dim=-1)
        feat   = self.linear(input_)
        return feat

class DuelingNet(nn.Module):
    """docstring for DuelingNet for DuelingDQN"""
    def __init__(self, action_dim, feat_dim=512, head_dim=512, device=None):
        super(DuelingNet, self).__init__()
        self.device = device
        self.feature_extracter = FeatureExtracter(feat_dim=512)
        
        self.advantage_head = nn.Sequential(
            nn.Linear(feat_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, action_dim),
        )
        self.state_value_head = nn.Sequential(
            nn.Linear(feat_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )
        

    def forward(self, state):
        #  state: goal_distance, goal_heading, rgb
        rgb           = state["rgb"]            # (bs,9,112,112)
        goal_distance = state["goal_distance"]  # (bs,3,1)
        goal_heading  = state["goal_heading"]   # (bs,3,1)
        
        rgb = rgb.view(rgb.shape[0], -1, 3, rgb.shape[-2], rgb.shape[-1]) #
        rgb = rgb.permute(0, 2, 1, 3, 4) # 
        goal_distance = goal_distance.view(goal_distance.shape[0], -1)
        goal_heading  = goal_heading.view(goal_heading.shape[0], -1)
    
        feat = self.feature_extracter(rgb, goal_distance, goal_heading) 
        feat = feat.view(feat.shape[0], -1) # (batch_size, feat_dim)
        
        adv = self.advantage_head(feat)        # (batch_size, action_num)
        val = self.state_value_head(feat)      # (batch_size, 1)
        
        mean_adv  = torch.mean(adv, dim=-1, keepdim=True)  # (batch_size, 1)
        
        Q_value = val + adv - mean_adv # (batch_size, action_num)
        
        return Q_value


class Policy(BasePolicy):
# class DuelingDQN_Policy(BasePolicy):

    def __init__(self, device="-1", writer=None, model_path=None, test=True):
        # super(DQN_Policy, self).__init__()  
        self.test = test    
          
        # network structure
        self.action_dim = 4         

        # gpu/cpu
        if device == "-1":
            self.device = torch.device("cpu")  # check GPU
        else:
            self.device = torch.device(
                f"cuda:{device}" if torch.cuda.is_available() else "cpu")  # 检测GPU
            
        
        # network
        self.eval_net   = DuelingNet(self.action_dim, feat_dim=512, head_dim=512, device=self.device).to(self.device)
        self.target_net = DuelingNet(self.action_dim, feat_dim=512, head_dim=512, device=self.device).to(self.device)

        self.target_net.load_state_dict(self.eval_net.state_dict())

        
        """ for training """
        self.lr    = 1e-3 #1e-4 #1e-3
        self.gamma = 0.99 #0.95
        
        self.memory_capacity = 200000
        self.batch_size      = 128

        self.act_step_counter     = 0 # the number of steps to act
        self.param_update_counter = 0 # the number of steps to learn the network
        self.memory = ReplayBuffer(self.memory_capacity)
        
        self.init_epsilon  = 0.1   # EPSILON-greedy random exploration
        self.epsilon_steps = 100000 #10000 # epsilon linearly increase during {epsilon_steps} steps 
        self.start_step    = 1000  # the step when the agent start to learn network
        
        self.target_update_freq = 1000 #10000 # target Q-net update frequence
        self.learning_freq = 4
        
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.writer    = writer # tensorboard
        
        """ for test """    
        if self.test:
            # model_path = "eval_net.pt"
            model_path = Path(__file__).resolve().parents[0] / "eval_net.pt"
            self.load(model_path)
            self.target_net.load_state_dict(self.eval_net.state_dict())

    def choose_action(self, state):
        with torch.no_grad():
            # state preprocess
            rgb           = state["rgb"].data            # (bs,9,112,112)
            goal_distance = state["goal_distance"]  # (bs,3,1)
            goal_heading  = state["goal_heading"]   # (bs,3,1)
            
            rgb = torch.FloatTensor(rgb).unsqueeze(0).to(self.device) / 255.0
            goal_distance = torch.FloatTensor(goal_distance).unsqueeze(0).to(self.device)
            goal_heading  = torch.FloatTensor(goal_heading).unsqueeze(0).to(self.device)
            
            state = {"rgb":rgb, "goal_distance":goal_distance, "goal_heading":goal_heading}
            
            if self.test:
                action_value = self.eval_net.forward(state)
                action = torch.max(action_value, 1)[1].cpu().data.numpy()
                action = action[0]
            else:
                if self.act_step_counter <= self.start_step:
                    self.epsilon = self.init_epsilon
                elif self.act_step_counter-self.start_step <= self.epsilon_steps:
                    self.epsilon = self.init_epsilon - self.init_epsilon*(self.act_step_counter-self.start_step) / self.epsilon_steps
                else:
                    self.epsilon = 0.
                self.writer.add_scalar('Epsilon', self.epsilon, global_step=self.act_step_counter)
                
                if np.random.rand() >= self.epsilon: # epsilon-greedy policy
                    action_value = self.eval_net.forward(state)
                    action = torch.max(action_value, 1)[1].cpu().data.numpy()
                    action = action[0]
                else: # random policy
                    action = np.random.randint(0, self.action_dim)
                    print(f"Random action {action} choosed!")
            
        return action


        
    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.
        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.
        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        self.act_step_counter += 1
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            # agent_obs = self.obs_dict_to_np(agent_obs) # get numpy obs
            action = self.choose_action(agent_obs)     # get action
            wrapped_act.update({agent_id: action})
            # print(action)
        return wrapped_act


    def store_transition(self, states, actions, rewards, next_states, dones):
        assert len(states.keys()) == 1
        for agent_id in states.keys():
            state      = states[agent_id]
            next_state = next_states[agent_id]
            action     = actions[agent_id]
            reward     = rewards[agent_id]
            done       = dones[agent_id]
            
            self.memory.push(state, action, reward, next_state, done)
    

    def get_state(self, states):
        """"""
        rgbs = torch.FloatTensor(np.array([state["rgb"].data for state in states])).to(self.device) / 255.0
        goal_distances = torch.FloatTensor(np.array([state["goal_distance"] for state in states])).to(self.device)
        goal_headings  = torch.FloatTensor(np.array([state["goal_heading"] for state in states])).to(self.device)

        return {"rgb":rgbs, "goal_distance":goal_distances, "goal_heading":goal_headings}
        
            
        
    def learn(self):
        """update the network"""
        if self.act_step_counter >= self.start_step and self.act_step_counter%self.learning_freq == 0:
            
            # update the parameters
            if self.param_update_counter % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.param_update_counter += 1
            
            #sample a batch of transition from memory
            states, action, reward, next_states, dones = self.memory.sample(self.batch_size)
            
            # transfer to torch.tensor
            states      = self.get_state(states)
            next_states = self.get_state(next_states)
            
            action = torch.LongTensor(np.array(action).astype(int)).unsqueeze(1).to(self.device)
            reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
            dones   = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(self.device)
            
            # update 
            q_eval = self.eval_net(states).gather(1, action) 
            
            q_next = self.eval_net(next_states).detach() 
            q_next_target = self.target_net(next_states).detach()
            
            next_action = q_next.argmax(dim=1, keepdim=True)
            q_next_action_target = q_next_target.gather(1, next_action)
            
            q_target = reward + ((1-dones) * self.gamma * q_next_action_target.view(self.batch_size, 1)) 
            
            loss = self.loss_func(q_eval, q_target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.writer.add_scalar('Loss', loss, global_step=self.param_update_counter)


    
    def save(self, path, epoch):
        os.makedirs(path, exist_ok=True)
        eval_path = os.path.join(path, f"{epoch}_eval_net.pt")
        
        torch.save(self.eval_net.state_dict(), eval_path)
        print("\nSaved trained model.\n")
    
    
    def load(self, eval_path):        
        self.eval_net.load_state_dict(torch.load(eval_path, map_location=self.device)) 
   

if __name__ == "__main__":
    agent = Policy(test=False)
    print(agent.eval_net)

    