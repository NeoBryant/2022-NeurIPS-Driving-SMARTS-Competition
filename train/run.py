from math import pi
import os
from re import I
from tabnanny import check

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import warnings
import sys
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Any, Dict

import gym
# import stable_baselines3 as sb3lib
import torch as th
import yaml
# from ruamel.yaml import YAML
# from stable_baselines3.common.callbacks import CheckpointCallback
# from stable_baselines3.common.evaluation import evaluate_policy
from train import env as multi_scenario_env

# To import submission folder
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission"))
import network
import policy

# For TD3
from tensorboardX import SummaryWriter
import numpy as np

print("\nTorch cuda is available: ", th.cuda.is_available(), "\n")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
# yaml = YAML(typ="safe")


def set_logdir(args):
    # Setup logdir.
    if not args.logdir:
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logdir = Path(__file__).resolve().parents[0] / "logs" / time
    else:
        logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    # config["logdir"] = logdir
    print("\nLogdir:", logdir, "\n")
    return logdir


def main(args: argparse.Namespace):
    # Load config file.
    # config_file = yaml.load((args.config))
    with open(args.config, 'r', encoding='utf-8') as rf:
        config_file = yaml.load(rf.read(), Loader=yaml.FullLoader)  
         
    # Load env config.
    config = config_file["smarts"]
    config["mode"] = args.mode
    config["head"] = args.head

    # Setup logdir.
    logdir = set_logdir(args)
    config["logdir"] = logdir

    # Setup model.
    if config["mode"] == "evaluate":
        # Begin evaluation.
        config["model"] = args.model
        print("\nModel:", config["model"], "\n")
    elif config["mode"] == "train" and not args.model:
        # Begin training.
        pass
    else:
        raise KeyError(f'Expected \'train\' or \'evaluate\', but got {config["mode"]}.')

    # Make training and evaluation environments.
    envs_train = {}
    envs_eval = {}
    wrappers = multi_scenario_env.wrappers(config=config)
    for scen in config["scenarios"]:
        envs_train[f"{scen}"] = multi_scenario_env.make(
            config=config, scenario=scen, wrappers=wrappers
        )
        
        
        # envs_eval[f"{scen}"] = multi_scenario_env.make(
        #     config=config, scenario=scen, wrappers=wrappers
        # )

    # Run training or evaluation.
    run(envs_train=envs_train, envs_eval=envs_eval, config=config)

    # Close all environments
    for env in envs_train.values():
        env.close()
    # for env in envs_eval.values():
    #     env.close()


def run(
    envs_train: Dict[str, gym.Env],
    envs_eval: Dict[str, gym.Env],
    config: Dict[str, Any],
):

    # tensorboard
    tensorboard_dir = os.path.join(config["logdir"], "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    gpu = "0" if th.cuda.is_available() else "-1"
    
    # build model
    # model = policy.DQN_Policy(device=gpu, writer=writer, model_path=None, test=False)
    model = policy.Policy(device=gpu, writer=writer, model_path=None, test=False)
    
    print(model)
    
    # fine-tuning
    # pretrained_model = ""
    # model.load(pretrained_model)
    # model.init_epsilon = 0.1
    
    # training hyper-parameters
    total_epochs = 5000
    train_steps  = 5000
    
    if config["mode"] == "train":
        print("\nStart training.\n")
        scenarios_iter = cycle(config["scenarios"])

        global_step = 0
        for epoch in range(total_epochs):
            scen = next(scenarios_iter)
            env_train = envs_train[scen]
            # env_eval  = envs_eval[scen]
            print(f"\n ======== Training on {scen}. ======== \n")        
            global_step = train(model, env_train, epoch, 
                                train_steps=train_steps, 
                                global_step=global_step, writer=writer)        
            model.save(os.path.join(config["logdir"], "checkpoint"), epoch)
           
    
    if config["mode"] == "evaluate":
        # print("\nEvaluate policy.\n")
        # model = getattr(sb3lib, config["alg"]).load(
        #     config["model"], print_system_info=True
        # )
        # for env_name, env_eval in envs_eval.items():
        #     print(f"\nEvaluating env {env_name}.")
        #     mean_reward, std_reward = evaluate_policy(
        #         model, env_eval, n_eval_episodes=config["eval_eps"], deterministic=True
        #     )
        #     print(f"Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}\n")
        print("\nFinished evaluating.\n")


def obs_dict_to_np(state):
    """transfer the observation from the dict type to numpy type"""
    # state(dict): 
    #goal_distance <class 'numpy.ndarray'> (1, 3, 1)
    #goal_heading <class 'numpy.ndarray'> (1, 3, 1)
    #rgb <class 'numpy.ndarray'> (1, 9, 112, 112)
    np_obs = []
    for _, value in state.items():
        np_obs.extend(value.flatten())
    return np.array(np_obs) # shape=(112902,)


def train(agent, env, epoch, train_steps, global_step=0, writer=None):
    """training the agent in one scene"""    
    state = env.reset()    
    
    episode = 1
    episode_timesteps = 0
    episode_reward = 0
    episode_rewards = list()
    
    dones = {"__all__": False}
    for t_step in range(train_steps):
        t_step += 1 # 使t_step从1开始计数
        episode_timesteps += 1
        global_step += 1
        
        actions = agent.act(state)
        
        next_state, rewards, dones, infos = env.step(actions)        

        for agent_reward in rewards.values():
            episode_reward += agent_reward
        # infos.keys: {'Agent_0: dict_keys(['score', 'env_obs', 'is_success'])}
        # dones: {'Agent_0': False, '__all__': False}
        
        agent.store_transition(state, actions, rewards, next_state, dones)
        
        agent.learn()
        
        state = next_state
        
        
        writer.add_scalar('Action', actions['Agent_0'], global_step=global_step)
        
        """print training information"""
        if t_step % 100 == 0 or t_step == 1:
            mean_episode_reward = sum(episode_rewards)/len(episode_rewards) if episode_rewards else episode_reward
            print(f"[Epoch {epoch}] step={t_step}, episode={episode}, memory={len(agent.memory)}, mean_episode_reward={mean_episode_reward:.2f}")
        
        if dones["__all__"]:
            state = env.reset()
            
            writer.add_scalar('Episode_reward', episode_reward, global_step=global_step)
            writer.add_scalar('Success', infos['Agent_0']['is_success'], global_step=global_step)
            writer.add_scalar('Epoch', epoch, global_step=global_step)
            writer.add_scalar('Episode', episode+1, global_step=global_step)
            
            episode_rewards.append(episode_reward)
            
            episode += 1
            episode_timesteps = 0
            episode_reward = 0
    
    return global_step

if __name__ == "__main__":
    program = Path(__file__).stem
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "--mode",
        help="`train` or `evaluate`. Default is `train`.",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--logdir",
        help="Directory path for saving logs.",
        type=str,
        default="", 
    )
    parser.add_argument(
        "--config",
        help="Directory path to config yaml file",
        type=str,
        default="config.yaml",
    )
    parser.add_argument(
        "--model",
        help="Directory path to saved RL model. Required if `--mode=evaluate`.",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--head", help="Display the simulation in Envision.", action="store_true"
    )
    
    args = parser.parse_args()

    if args.mode == "evaluate" and args.model is None:
        raise Exception("When --mode=evaluate, --model option must be specified.")

    main(args)
