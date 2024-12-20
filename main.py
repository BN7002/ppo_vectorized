import math
import gymnasium as gym
import numpy as np
from ppo import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    max_episode_steps=400
    num_envs: int = 16
    env = gym.make_vec('LunarLander-v3', max_episode_steps=max_episode_steps, num_envs=num_envs, continuous=True)
    print(f"\n\n========== env spec==========\n\
          enviroment: {env}\n\
          observation_space: {env.observation_space}\n\
          action_space: {env.action_space}\n")
    
    print("obs: ", env.observation_space.shape)
    N = 80
    batch_size = 10
    n_epochs = 8
    alpha = 0.0003
    
    agent = Agent(n_actions=env.action_space, num_envs=num_envs, action_space=env.action_space, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=env.observation_space.shape)
    n_games = 1600
    #print(f"\n[Debug]\n\
    #        {agent.actor}\n\
    #        {agent.critic}")

    figure_file = 'plots/cartpole.png'

    best_score = -math.inf #env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        # .reset() returns obs, info so we have to specify what is the observation
        observation = env.reset()[0]
        done = [False]
        score = 0
        steps = 0
        while not all(done) and steps < max_episode_steps:
            action, prob, val = agent.choose_action(observation)
            print(f"action: {action}")
            observation_, reward, done, terminated, info = env.step(action)
            #print(f"action: {action}, prob: {prob}, val: {val}, observation: {observation}, reward: {reward}, done: {done}, terminated: {terminated}, info: {info}")
            n_steps += 1
            score += reward
            #print(f"score: {score}")
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
            #print(f"new observations: {observation}")
            steps+=1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score', score, 'score mean', score.mean(), 'avg score %.1f' % avg_score,
                'time_steps', n_steps * num_envs, 'learning_steps', learn_iters)
        x = [i+1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)