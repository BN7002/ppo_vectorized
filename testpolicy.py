import gymnasium as gym
import numpy as np
from ppo import Agent

def test_policy(env_name, policy_path, num_episodes=2000):
    max_episode_steps = 400
    env = gym.make_vec(env_name, max_episode_steps=max_episode_steps, render_mode='human', num_envs=1, continuous = True)
    agent = Agent(n_actions=env.action_space, action_space=env.action_space, num_envs=1, batch_size=64, 
                    alpha=0.0003, n_epochs=4, 
                    input_dims=env.observation_space.shape)
    agent.load_models()
    
    for episode in range(num_episodes):
        observation = env.reset()[0]
        done = [False]
        score = 0
        steps = 0
        while not all(done) and steps < max_episode_steps:
            action, _, _ = agent.choose_action(observation)
            # fix if num_env = 1 => add dim in order to prevent error that stays that dimension are not correct
            if(action.shape == (env.action_space.shape[1], )): # type: ignore
                action = np.expand_dims(action, axis=0)
            observation, reward, done, _, _ = env.step(action)
            score += reward
            steps+=1
            
        print(f"Episode {episode + 1}: Score = {score}")

if __name__ == '__main__':
    test_policy(env_name='LunarLander-v3', policy_path='path_to_saved_policy')
