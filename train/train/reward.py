from typing import Any, Dict

import gym
import numpy as np


class Reward(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Adapts the wrapped environment's step.

        Note: Users should not directly call this method.
        """
        obs, reward, done, info = self.env.step(action)
        wrapped_reward = self._reward(obs, reward)

        for agent_id, agent_done in done.items():
            if agent_id != "__all__" and agent_done == True:
                if obs[agent_id]["events"]["reached_goal"]:
                    print(f"{agent_id}: Hooray! Reached goal.")
                elif obs[agent_id]["events"]["reached_max_episode_steps"]:
                    print(f"{agent_id}: Reached max episode steps.")
                elif (
                    obs[agent_id]["events"]["collisions"]
                    | obs[agent_id]["events"]["off_road"]
                    | obs[agent_id]["events"]["off_route"]
                    | obs[agent_id]["events"]["on_shoulder"]
                    | obs[agent_id]["events"]["wrong_way"]
                ):
                    pass
                else:
                    print("Events: ", obs[agent_id]["events"])
                    raise Exception("Episode ended for unknown reason.")

        return obs, wrapped_reward, done, info

    def _reward(
        self, obs: Dict[str, Dict[str, Any]], env_reward: Dict[str, np.float64]
    ) -> Dict[str, np.float64]:
        reward = {agent_id: np.float64(0) for agent_id in env_reward.keys()}

        for agent_id, agent_reward in env_reward.items():
            # ! Penalty for time
            # reward[agent_id] -= np.float64(0.1)


            # Penalty for colliding
            if obs[agent_id]["events"]["collisions"]:
                # ! 原来是10，->修改为50
                # reward[agent_id] -= np.float64(10)
                reward[agent_id] -= np.float64(50)
                print(f"{agent_id}: Collided.")
                break
            
            # Penalty for driving off road
            if obs[agent_id]["events"]["off_road"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went off road.")
                break

            # Penalty for driving off route
            if obs[agent_id]["events"]["off_route"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went off route.")
                break
                
            # ! 本来是注释掉的，我解注释训练看看
            # Penalty for driving on road shoulder
            # if obs[agent_id]["events"]["on_shoulder"]:
            #     reward[agent_id] -= np.float64(1)
            #     print(f"{agent_id}: Went on shoulder.")
            #     break
            
            # Penalty for driving on wrong way
            if obs[agent_id]["events"]["wrong_way"]:
                reward[agent_id] -= np.float64(10)
                print(f"{agent_id}: Went wrong way.")
                break

            # Reward for reaching goal
            if obs[agent_id]["events"]["reached_goal"]:
                # ! 原来是30，->修改为100
                # reward[agent_id] += np.float64(30)
                reward[agent_id] += np.float64(100)

            # Reward for distance travelled
            reward[agent_id] += np.float64(agent_reward)
            
            

        return reward
