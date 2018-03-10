import random
import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.distributions import Categorical


class HybridPolicyGraident:
    def __init__(self, reward_types):
        self.reward_types = reward_types

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

            policy_loss = []
            policy_type_losses = {i: [] for i in range(self.reward_types)}
            for log_prob, score in zip(log_probs, discounted_total_returns):
                policy_loss.append(-log_prob * score)
            for type_i in range(self.reward_types):
                for log_prob, score in zip(reward_type_log_probs[type_i], discounted_decomposed_returns[type_i]):
                    policy_type_losses[type_i].append(-log_prob * score)

            optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            for type_i in range(self.reward_types):
                policy_loss += torch.cat(policy_type_losses[type_i]).sum()
            policy_loss.backward()
            optimizer.step()
            print('Episode:{} Reward:{} Length:{}'.format(episode, total_reward, len(ep_decomposed_rewards)))

            # test and log
            if episode % 50 == 0:
                test_reward = self.test(net, env_fn, 10, log=True)
                test_perf_data.append(test_reward)
                print('Performance:', test_reward)
                if best is None or best <= test_reward:
                    torch.save(net.state_dict(), net_path)
                    best = test_reward
                    print('Model Saved!')

        return net

    def test(self, net, env_fn, episodes, log=False, render=False):
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
                action_probs, _ = net(obs)
                action = int(np.argmax(action_probs.cpu().data.numpy()))
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

                steps += 1
            all_episode_rewards += episode_reward
            if log:
                print('Test => Episode:{} Reward:{} Length:{}'.format(episode, episode_reward, steps))
        return all_episode_rewards / episodes
