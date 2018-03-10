import random
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

    def __init__(self):
        pass

    def __get_plot_data_dict(self, train_data, test_data):
        data_dict = [
            {'title': "Train_Performance_vs_Epoch", 'data': train_data, 'y_label': 'Episode Reward', 'x_label': 'Time'},
            {'title': "Test_Performance_vs_Epoch", 'data': test_data, 'y_label': 'Episode Reward', 'x_label': 'Time'},
        ]
        return data_dict

    def train(self, net, env_fn, net_path, plots_dir, args):
        optimizer = Adam(net.parameters(), lr=args.lr)
        random.seed(args.seed)

        test_perf_data = []
        train_perf_data = []
        best = None
        for episode in range(args.train_episodes):
            random.seed(random.randint(1000, 100000))
            net.train()
            env = env_fn()

            # Gather data for a single episode
            done = False
            total_reward = 0
            log_probs = []
            ep_rewards = []
            obs = env.reset()
            while not done:
                obs = Variable(torch.Tensor(obs.tolist())).unsqueeze(0)
                action_probs = net(obs)
                m = Categorical(action_probs)
                action = m.sample()
                log_probs.append(m.log_prob(action))

                action = int(action.data[0])
                obs, reward, done, info = env.step(action)
                ep_rewards.append(reward)
                total_reward += reward

            train_perf_data.append(total_reward)

            # Estimate the Gradients and update the network
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

            optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()
            print('Train=> Episode:{} Reward:{} Length:{}'.format(episode, total_reward, len(ep_rewards)))
            # test and log
            if episode % 50 == 0:
                test_reward = self.test(net, env_fn, 10, log=True)
                test_perf_data.append(test_reward)
                print('Performance:', test_reward)
                if best is None or best <= test_reward:
                    torch.save(net.state_dict(), net_path)
                    best = test_reward
                    print('Model Saved!')
            if episode % 200 == 0:
                plot_data(self.__get_plot_data_dict(train_perf_data, test_perf_data),plots_dir)
        return net

    def test(self, net, env_fn, episodes, log=False, render=False):
        net.eval()
        all_episode_rewards = 0
        for episode in range(episodes):
            random.seed(episode)
            env = env_fn()
            done = False
            episode_reward = 0
            obs = env.reset()
            ep_actions = []  # just to exit early if the agent is stuck
            steps = 0
            while not done:
                obs = Variable(torch.Tensor(obs.tolist())).unsqueeze(0)
                action_probs = net(obs)
                action = int(np.argmax(action_probs.cpu().data.numpy()[0]))
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

                steps += 1
            all_episode_rewards += episode_reward
            if log:
                print('Test => Episode:{} Reward:{} Length:{}'.format(episode, episode_reward, steps))
        return all_episode_rewards / episodes
