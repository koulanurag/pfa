import random
import time
import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.distributions import Categorical
from tools import plot_data


class PolicyGraident:
    """
    Policy Gradient Implementation
    Implementation Reference : https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    """

    def __init__(self, vis = None):
        self.vis = vis
        self.__prob_bar_window = None
        pass

    def __get_plot_data_dict(self, train_data, test_data):
        data_dict = [
            {'title': "Train_Performance_vs_Epoch", 'data': train_data, 'y_label': 'Episode Reward', 'x_label': 'Time'},
            {'title': "Test_Performance_vs_Epoch", 'data': test_data, 'y_label': 'Episode Reward', 'x_label': 'Time'},
        ]
        return data_dict

    def train(self, net, env_fn, net_path, plots_dir, args):
        optimizer = Adam(net.parameters(), lr=args.lr)

        test_perf_data = []
        train_perf_data = []
        best = None
        n_trajectory_loss = []
        for episode in range(args.train_episodes):
            net.train()
            env = env_fn()

            # Gather data for a single episode
            done = False
            total_reward = 0
            log_probs = []
            ep_rewards = []
            obs = env.reset()
            while not done:
                obs = Variable(torch.FloatTensor(obs.tolist())).unsqueeze(0)
                action_probs = net(obs)
                m = Categorical(action_probs)
                action = m.sample()
                log_probs.append(m.log_prob(Variable(action.data)))

                action = int(action.data[0])
                obs, reward, done, info = env.step(action)
                ep_rewards.append(reward)
                total_reward += reward

            train_perf_data.append(total_reward)

            # Estimate the Gradients
            R = 0
            discounted_returns = []
            for r in ep_rewards[::-1]:
                R = r + args.gamma * R
                discounted_returns.insert(0, R)

            discounted_returns = torch.FloatTensor(discounted_returns)
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (
                    discounted_returns.std() + np.finfo(np.float32).eps)

            policy_loss = []
            for log_prob, score in zip(log_probs, discounted_returns):
                policy_loss.append(-log_prob * score)
            n_trajectory_loss.append(policy_loss)  # collect n-trajectories

            # Update the network after collecting n trajectories
            if episode % 10 == 0:
                optimizer.zero_grad()
                sample_loss = 0
                for _loss in n_trajectory_loss:
                    sample_loss += torch.cat(_loss).sum()
                sample_loss = sample_loss / 10
                sample_loss.backward()
                optimizer.step()
                n_trajectory_loss = []
            print('Train=> Episode:{} Reward:{} Length:{}'.format(episode, total_reward, len(ep_rewards)))

            # test and log
            if episode % 50 == 0:
                test_reward = self.test(net, env_fn, 10, log=True)
                test_perf_data.append(test_reward)
                print('Test Performance:', test_reward)
                if best is None or best <= test_reward:
                    torch.save(net.state_dict(), net_path)
                    best = test_reward
                    print('Model Saved!')
                if best == env.reward_threshold:
                    print('Optimal Performance achieved!!')
                    break
            if episode % 200 == 0:
                plot_data(self.__get_plot_data_dict(train_perf_data, test_perf_data), plots_dir)
        return net

    def test(self, net, env_fn, episodes, log=False, render=False, sleep=0):
        net.eval()
        all_episode_rewards = 0
        for episode in range(episodes):
            env = env_fn()
            done = False
            episode_reward = 0
            obs = env.reset()
            ep_actions = []  # just to exit early if the agent is stuck
            steps = 0
            while not done:
                obs = Variable(torch.FloatTensor(obs.tolist())).unsqueeze(0)
                action_probs = net(obs)
                m = Categorical(action_probs)
                action = m.sample().data[0]

                if render:
                    env.render()
                    if self.__prob_bar_window is None:
                        self.__prob_bar_window = self.vis.bar(
                                                    X = action_probs.data.numpy()[0],
                                                    opts = dict(rownames=['UP', 'RIGHT', 'DOWN', 'LEFT'])
                                                )
                    else:
                        self.vis.bar(X = action_probs.data.numpy()[0],
                                     opts = dict(rownames=['UP', 'RIGHT', 'DOWN', 'LEFT']),
                                     win = self.__prob_bar_window)

                obs, reward, done, info = env.step(action)
                episode_reward += reward

                # exit if stuck
                if len(ep_actions) > 100:  # same action being taken continuously for last n steps
                    done = True
                    print('Early Exit', ep_actions[-1])
                elif (len(ep_actions) == 0) or (len(ep_actions) > 0 and ep_actions[-1] != action):
                    ep_actions = [action]
                else:
                    ep_actions.append(action)
                time.sleep(sleep)

                steps += 1
            env.close()
            all_episode_rewards += episode_reward
            if log:
                print('Test => Episode:{} Reward:{} Length:{}'.format(episode, episode_reward, steps))
            if render:
                self.vis.close(win=self.__prob_bar_window)
        return all_episode_rewards / episodes
