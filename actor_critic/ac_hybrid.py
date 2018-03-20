import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.distributions import Categorical
from tools import plot_data


class HybridActorCritic:
    """
    Hybrid Actor Critic Implementation
    """

    def __init__(self, reward_types, decompose=False, vis=None):
        self.reward_types = reward_types
        self.decompose = decompose
        self.vis = vis
        self.__prob_bar_window = {i: None for i in range(self.reward_types)}
        self.__value_bar_window = {i: None for i in range(self.reward_types)}
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
        mse_loss = nn.MSELoss.cuda() if args.cuda else nn.MSELoss()

        test_perf_data = []
        train_perf_data = []
        best = None
        n_trajectory_info = []
        for episode in range(1, args.train_episodes + 1):
            net.train()
            env = env_fn()

            # Gather data for a single episode
            done = False
            total_reward = 0
            log_probs = []
            reward_type_log_probs = {i: [] for i in range(self.reward_types)}
            reward_type_critic_info = {i: [] for i in range(self.reward_types)}
            ep_decomposed_rewards = []
            ep_obs = []
            obs = env.reset()
            while not done:
                ep_obs.append(obs)
                obs = Variable(torch.FloatTensor(obs.tolist())).unsqueeze(0)
                action_probs, reward_type_action_probs, reward_type_critic = net(obs)
                m = Categorical(action_probs)
                action = m.sample()
                log_probs.append(m.log_prob(Variable(action.data)))
                for reward_type_i in range(self.reward_types):
                    m = Categorical(reward_type_action_probs[reward_type_i])
                    reward_type_log_probs[reward_type_i].append(m.log_prob(Variable(action.data)))
                    reward_type_critic_info[reward_type_i].append(reward_type_critic[reward_type_i])

                action = int(action.data[0])
                obs, reward, done, info = env.step(action)
                ep_decomposed_rewards.append(reward)
                total_reward += sum(reward)
            train_perf_data.append(total_reward)
            n_trajectory_info.append((ep_obs, ep_decomposed_rewards,
                                      reward_type_critic_info, reward_type_log_probs, log_probs))

            print('Train=> Episode:{} Reward:{} Length:{}'.format(episode, total_reward, len(ep_obs)))

            # Update the network after collecting n trajectories
            if episode % args.batch_size == 0:
                optimizer.zero_grad()
                critic_loss = 0
                for trajectory_info in n_trajectory_info:
                    obs, _decomposed_reward, _type_critic_info, _, _ = trajectory_info
                    # import pdb; pdb.set_trace()
                    for step_i in range(len(obs)):
                        for i, r in enumerate(_decomposed_reward[step_i]):
                            critic = _type_critic_info[i][step_i]
                            if step_i != (len(obs) - 1):
                                target_critic = r + Variable(_type_critic_info[i][step_i + 1].data)
                            else:
                                target_critic = Variable(torch.Tensor([[r]]))
                            critic_loss += mse_loss(critic, target_critic)
                critic_loss = critic_loss / args.batch_size
                critic_loss.backward(retain_graph=True)
                optimizer.step()

                optimizer.zero_grad()
                actor_loss = 0
                for trajectory_info in n_trajectory_info:
                    obs, _decomposed_reward, _, _type_log_probs, _ = trajectory_info
                    # import pdb; pdb.set_trace()
                    for i in range(len(obs)):
                        _, _, type_v_state = net(Variable(torch.FloatTensor(obs[i].tolist())).unsqueeze(0))
                        if i != len(obs) - 1:
                            _, _, type_v_next_state = net(Variable(torch.FloatTensor(obs[i + 1].tolist())).unsqueeze(0))
                        else:
                            type_v_next_state = None

                        for r_i, r in enumerate(_decomposed_reward[i]):
                            v_state = Variable(type_v_state[r_i].data)
                            if type_v_next_state is not None:
                                v_next_state = Variable(type_v_next_state[r_i].data)
                            else:
                                v_next_state = 0
                            advantage = r + v_next_state - v_state
                            actor_loss -= _type_log_probs[r_i][i] * advantage

                actor_loss = actor_loss / args.batch_size
                actor_loss.backward()
                optimizer.step()

                n_trajectory_info = []

            # test and log
            if episode % 20 == 0:
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
            if episode % 10 == 0:
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
                action_probs, typed_action_probs, reward_type_critic = net(obs)
                # print(action_probs, critic)
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
                                                        opts = dict(rownames = ['RIGHT', 'LEFT'],
                                                                    title   = "Action Prob for reward type ({}, {})".format(x, y))
                                                    )
                        else:
                            self.vis.bar(X = probs,
                                         opts = dict(rownames = ['RIGHT', 'LEFT'],
                                                     title   = "Action Prob for reward type ({}, {})".format(x, y)
                                                    ),
                                         win = self.__prob_bar_window[reward_type])
                        #VAlue estimate
                        print(reward_type_critic[reward_type])
                        value = [reward_type_critic[reward_type].data.numpy()[0].tolist()[0], 0]
                        if self.__value_bar_window[reward_type] is None:
                            self.__value_bar_window[reward_type] = self.vis.bar(
                                                        X = value,
                                                        opts = dict(title   = "Value for reward type ({}, {})".format(x, y))
                                                    )
                        else:
                            self.vis.bar(X = value,
                                         opts = dict(title   = "Value for reward type ({}, {})".format(x, y)),
                                         win = self.__value_bar_window[reward_type])

                    if self.__combined_prob_bar_window is None:
                        self.__combined_prob_bar_window = self.vis.bar(
                                                            X = action_probs.data.numpy(),
                                                            opts = dict(rownames=['RIGHT', 'LEFT'],
                                                                        title   = "Combined Action Prob")
                                                        )
                    else:
                        self.vis.bar(X = action_probs.data.numpy(),
                                     opts = dict(rownames =['RIGHT', 'LEFT'],
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
            if render:
                self.vis.close(win=self.__prob_bar_window)
        return all_episode_rewards / episodes
