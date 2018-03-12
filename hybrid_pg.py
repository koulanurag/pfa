import random
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.distributions import Categorical
from tools import plot_data

class HybridPolicyGraident:
    def __init__(self, reward_types, decompose = False, vis = None):
        self.reward_types = reward_types
        self.decompose = decompose
        self.vis = vis
        self.__prob_bar_window = {i: None for i in range(self.reward_types)}
        self.__combined_prob_bar_window = None

    def softmax(self, w, t = 1.0):
        e = np.exp(w / t)
        dist = e / np.sum(e)
        return dist

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
        n_trajectory_type_loss = []
        for episode in range(args.train_episodes):
            episode_start_time = time.time()
            net.train()
            env = env_fn()

            # Gather data for a single episode
            done = False
            total_reward = 0
            log_probs = []
            reward_type_log_probs = {i: [] for i in range(self.reward_types)}

            ep_decomposed_rewards = []
            obs = env.reset()
            while not done:
                obs = Variable(torch.Tensor(obs.tolist())).unsqueeze(0)
                action_probs, reward_type_action_probs = net(obs)
                m = Categorical(action_probs)
                action = m.sample()
                log_probs.append(m.log_prob(action))
                for reward_type_i in range(self.reward_types):
                    m = Categorical(reward_type_action_probs[reward_type_i])
                    reward_type_log_probs[reward_type_i].append(m.log_prob(Variable(action.data)))

                action = int(action.data[0])
                obs, reward, done, info = env.step(action)

                ep_decomposed_rewards.append(reward)
                total_reward += sum(reward)

            train_perf_data.append(total_reward)

            # Estimate the Gradients and update the network
            R_total = 0
            R_decomposed = {i: 0 for i in range(self.reward_types)}
            discounted_total_returns = []
            discounted_decomposed_returns = {i: [] for i in range(self.reward_types)}
            for r in ep_decomposed_rewards[::-1]:
                R_total = sum(r) + args.gamma * R_total
                discounted_total_returns.insert(0, R_total)
                for i, r_d in enumerate(r):
                    R_decomposed[i] = r_d + args.gamma * R_decomposed[i]
                    discounted_decomposed_returns[i].insert(0, R_decomposed[i])

            discounted_total_returns = torch.FloatTensor(discounted_total_returns)
            discounted_total_returns = (discounted_total_returns - discounted_total_returns.mean()) / (
                    discounted_total_returns.std() + np.finfo(np.float32).eps)

            for i in discounted_decomposed_returns:
                discounted_decomposed_returns[i] = torch.FloatTensor(discounted_decomposed_returns[i])
                discounted_decomposed_returns[i] = (discounted_decomposed_returns[i] - discounted_decomposed_returns[i].mean()) / (
                        discounted_decomposed_returns[i].std() + np.finfo(np.float32).eps)

            policy_loss = []
            policy_type_losses = {i: [] for i in range(self.reward_types)}
            for log_prob, score in zip(log_probs, discounted_total_returns):
                policy_loss.append(-log_prob * score)

            for type_i in range(self.reward_types):
                for log_prob, score in zip(reward_type_log_probs[type_i], discounted_decomposed_returns[type_i]):
                    policy_type_losses[type_i].append(-log_prob * score)

            n_trajectory_loss.append(policy_loss)
            n_trajectory_type_loss.append(policy_type_losses)

            if episode % 10 == 0:
                start_time = time.time()
                optimizer.zero_grad()
                sample_loss = 0

                for _loss in n_trajectory_loss:
                    sample_loss += torch.cat(_loss).sum()

                if self.decompose:
                    for _loss in n_trajectory_type_loss:
                        for type_i in range(self.reward_types):
                            sample_loss += torch.cat(_loss[type_i]).sum()

                end_time = time.time()
                print("Loss Time", end_time - start_time)

                sample_loss = sample_loss / 10
                start_time = time.time()
                sample_loss.backward()
                optimizer.step()
                end_time = time.time()
                n_trajectory_loss = []
                n_trajectory_type_loss = []

                print("Update Network Time", end_time - start_time)



            episode_end_time = time.time()
            print('Episode:{} Reward:{} Length:{} Time:{}'.format(episode, total_reward, len(ep_decomposed_rewards), episode_end_time - episode_start_time))

            # test and log
            if episode % 50 == 0:
                test_reward = self.test(net, env_fn, 10, log=True, render=False)
                test_perf_data.append(test_reward)
                print('Performance:', test_reward)
                if best is None or best <= test_reward:
                    torch.save(net.state_dict(), net_path)
                    best = test_reward
                    print('Model Saved!')
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
                obs = Variable(torch.Tensor(obs.tolist())).unsqueeze(0)
                action_probs, typed_action_probs = net(obs)
                m = Categorical(action_probs)
                action = m.sample().data[0]

                if render:
                    env.render()
                    for reward_type in range(self.reward_types):
                        logits = typed_action_probs[reward_type].data.numpy()[0]
                        probs = self.softmax(logits)
                        x, y = env._fruit_positions[reward_type]

                        if self.__prob_bar_window[reward_type] is None:
                            self.__prob_bar_window[reward_type] = self.vis.bar(
                                                        X = probs,
                                                        opts = dict(rownames = ['UP', 'RIGHT', 'DOWN', 'LEFT'],
                                                                    title   = "Action Prob for reward type ({}, {})".format(x, y))
                                                    )
                        else:
                            self.vis.bar(X = probs,
                                         opts = dict(rownames = ['UP', 'RIGHT', 'DOWN', 'LEFT'],
                                                     title   = "Action Prob for reward type ({}, {})".format(x, y)
                                                    ),
                                         win = self.__prob_bar_window[reward_type])

                    if self.__combined_prob_bar_window is None:
                        self.__combined_prob_bar_window = self.vis.bar(
                                                            X = action_probs.data.numpy(),
                                                            opts = dict(rownames=['UP', 'RIGHT', 'DOWN', 'LEFT'],
                                                                        title   = "Combined Action Prob")
                                                        )
                    else:
                        self.vis.bar(X = action_probs.data.numpy(),
                                     opts = dict(rownames =['UP', 'RIGHT', 'DOWN', 'LEFT'],
                                                 title    = "Combined Action Prob"),
                                     win = self.__combined_prob_bar_window)

                obs, reward, done, info = env.step(action)
                episode_reward += sum(reward)

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
        return all_episode_rewards / episodes
