import random
import time
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical
from tools import plot_data


class ActorHybridCritic:
    """
    Actor Critic Implementation
    """

    def __init__(self, vis=None, reward_types=1):
        self.reward_types = reward_types
        self.vis = vis
        self.__prob_bar_window = None
        self.__value_bar_window = None

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
        # n_trajectory_loss = []
        n_trajectory_info = []
        for episode in range(1, args.train_episodes + 1):
            net.train()
            env = env_fn()

            # Gather data for a single episode
            done = False
            total_reward = 0
            log_probs = []
            ep_rewards = []
            critic_info = []
            ep_obs = []
            obs = env.reset()
            entropies = []
            while not done:
                ep_obs.append(obs)
                obs = Variable(torch.FloatTensor(obs.tolist())).unsqueeze(0)
                logit, critic = net(obs)

                action_probs = F.softmax(logit)
                action_log_prob = F.log_softmax(logit)
                entropy = -(action_log_prob * action_probs).sum(1)
                entropies.append(entropy)

                m = Categorical(action_probs)
                action = m.sample()
                log_probs.append(m.log_prob(Variable(action.data)))
                action = int(action.data[0])

                obs, reward, done, info = env.step(action)
                ep_rewards.append(reward)
                critic_info.append(critic)
                total_reward += sum(reward)
            train_perf_data.append(total_reward)
            n_trajectory_info.append((ep_obs, ep_rewards, critic_info, log_probs, entropies))

            # Update the network after collecting n trajectories
            if episode % args.batch_size == 0:
                # critic update
                # TODO: Optimize critic update by calculating MSE once for everything
                optimizer.zero_grad()
                critic_loss = 0
                for trajectory_info in n_trajectory_info:
                    obs, _rewards, _critic_info, _log_probs, _ = trajectory_info
                    for i in range(len(obs)):
                        critic = _critic_info[i]
                        target_critic = []
                        for r_i, r in enumerate(_rewards[i]):
                            if i != len(obs) - 1:
                                target_critic.append(r + args.gamma * _critic_info[i + 1].data.numpy()[0][r_i])
                            else:
                                target_critic.append(r)
                        target_critic = Variable(torch.FloatTensor(target_critic)).unsqueeze(0)
                        critic_loss += mse_loss(critic, target_critic)
                critic_loss = critic_loss / args.batch_size
                critic_loss.backward(retain_graph=True)
                optimizer.step()

                optimizer.zero_grad()
                actor_loss = 0
                for trajectory_info in n_trajectory_info:
                    obs, _rewards, _critic_info, _log_probs, _entropies = trajectory_info
                    # gae = [0 for _ in range(self.reward_types)]
                    for i in range(len(obs)):
                        _, v_state = net(Variable(torch.FloatTensor(obs[i].tolist())).unsqueeze(0))
                        v_state = v_state.data.numpy()[0]
                        if i != len(_rewards) - 1:
                            _, v_next_state = net(Variable(torch.FloatTensor(obs[i + 1].tolist())).unsqueeze(0))
                            v_next_state = v_next_state.data.numpy()[0]
                        else:
                            v_next_state = [0 for _ in range(len(_rewards))]

                        advantage = 0
                        for r_i, r in enumerate(_rewards[i]):
                            advantage += r + args.gamma * v_next_state[r_i] - v_state[r_i]
                        actor_loss -= _log_probs[i] * advantage - args.beta * _entropies[i]

                actor_loss = actor_loss / args.batch_size
                actor_loss.backward()
                optimizer.step()

                n_trajectory_info = []
            print('Train=> Episode:{} Reward:{} Length:{}'.format(episode, total_reward, len(ep_rewards)))

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
        pause, stop = False, False
        # if render:
        #     pause = True
        #     stop = False
        #     txt = 'Press Enter to Make Next Step'
        #     callback_text_window = self.vis.text(txt)
        #
        #     def type_callback(event):
        #         global pause, stop
        #         if event['event_type'] == 'KeyPress':
        #             curr_txt = event['pane_data']['content']
        #             if event['key'] == 'Enter':
        #                 pause = False
        #                 sleep(1)
        #                 curr_txt += '<br>' + 'Press Enter to Make Next Step / BackSpace to Stop the game'
        #                 pause = True
        #             elif event['key'] == 'Backspace':
        #                 curr_txt += '<br>' + 'Stopped!'
        #                 stop = True
        #             self.vis.text(curr_txt, win=callback_text_window)
        #
        #     self.vis.register_event_handler(type_callback, callback_text_window)

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
                logits, critic = net(obs)
                prob = F.softmax(logits)
                action = int(prob.max(1)[1].data.numpy()[0])

                if render:
                    env.render()
                    if self.__prob_bar_window is None:
                        self.__prob_bar_window = self.vis.bar(
                            X=prob.data.numpy()[0],
                            opts=dict(rownames=env.get_action_meanings, title="Action Prob")
                        )
                    else:
                        self.vis.bar(X=prob.data.numpy()[0],
                                     opts=dict(rownames=env.get_action_meanings, title="Action Prob"),
                                     win=self.__prob_bar_window)

                    # Value estimate
                    value = critic.data.numpy()[0].reshape(net.reward_types, 1).tolist()
                    if self.__value_bar_window is None:
                        self.__value_bar_window = self.vis.bar(
                            X=value,
                            opts=dict(title="Value for state", stacked=False,
                                      rownames=[str(i) for i in range(net.reward_types)])
                        )
                    else:
                        self.vis.bar(X=value,
                                     opts=dict(title="Value for state", stacked=False,
                                               rownames=[str(i) for i in range(net.reward_types)]),
                                     win=self.__value_bar_window)
                    # to control the environment
                    while pause and not stop:
                        time.sleep(0.5)
                    if stop:
                        sys.exit(0)

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
