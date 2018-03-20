import argparse
import os
import torch
import visdom
import torch.nn as nn
import torch.nn.functional as F
from envs import FruitCollection1D, FruitCollection2D
from tools import ensure_directory_exits, weights_init, normalized_columns_initializer
from .actor_decomposed_critic import ActorHybridCritic as AHC


class ActorHybridCriticNet2D(nn.Module):
    def __init__(self, input_size, actions, reward_types):
        super(ActorHybridCriticNet2D, self).__init__()
        self.reward_types = reward_types
        self.actions = actions
        self.state_actor = nn.Sequential(nn.Linear(input_size, 128),
                                         nn.ReLU())
        self.state_critic = nn.Sequential(nn.Linear(input_size, 128),
                                          nn.ReLU())
        self.decomposed_critic = nn.Linear(128, self.reward_types)
        self.actor = nn.Linear(128, actions)
        self.actor_dropout = nn.Dropout(p=0.5)

        self.apply(weights_init)
        self.decomposed_critic.weight.data.fill_(0)
        self.decomposed_critic.bias.data.fill_(0)
        self.actor.bias.data.fill_(0)
        self.actor.weight.data = normalized_columns_initializer(self.actor.weight.data, 0.01)

    def forward(self, input):
        state_actor = self.state_actor(input)
        state_critic = self.state_critic(input)
        actor_linear = self.actor_dropout(self.actor(state_actor))
        critic = self.decomposed_critic(state_critic)
        return actor_linear, critic


class ActorHybridCriticNet1D(nn.Module):
    def __init__(self, input_size, actions, reward_types):
        super(ActorHybridCriticNet1D, self).__init__()
        self.reward_types = reward_types
        self.actions = actions

        #Critic
        self.fc0  = nn.Linear(input_size, 128)
        self.decomposed_critic = nn.Linear(128, self.reward_types)

        #Actor
        self.actor = nn.Linear(128, actions)
        self.actor_dropout = nn.Dropout(p=0.5)

        self.apply(weights_init)

        self.decomposed_critic.weight.data.fill_(0)
        self.decomposed_critic.bias.data.fill_(0)

        self.actor.bias.data.fill_(0)
        self.actor.weight.data = normalized_columns_initializer(self.actor.weight.data, 0.01)

    def forward(self, input):
        state = F.relu(self.fc0(input))
        actor_linear = self.actor_dropout(self.actor(state))
        critic = self.decomposed_critic(state)
        return actor_linear, critic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Policy Fusion Architecture - PyTorch')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=10, metavar='N', help='random seed (default: 10)')
    parser.add_argument('--render', action='store_true', default=False, help='render the environment')
    parser.add_argument('--hybrid', action='store_true', default=False, help='use hybrid reward')
    parser.add_argument('--beta', type=float, default=0.001, help='Rate for Entropy')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch Size(No. of Episodes) for Training')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='interval between training status logs (default: 5)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables Cuda Usage')
    parser.add_argument('--train', action='store_true', default=False, help='Train the network')
    parser.add_argument('--test', action='store_true', default=False, help='Test the network')
    parser.add_argument('--train_episodes', type=int, default=500, help='Episode count for training')
    parser.add_argument('--test_episodes', type=int, default=100, help='Episode count for testing')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate for Training (Adam Optimizer)')
    parser.add_argument('--scratch', action='store_true', default=False,
                        help='Train the network from scratch ( or Does not load pre-trained model)')
    parser.add_argument('--env', default="1D",
                        help='Train the network from scratch ( or Does not load pre-trained model)')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and (not args.no_cuda)

    vis = visdom.Visdom() if args.render else None

    if args.env == "1D":
        env_fn = lambda: FruitCollection1D(hybrid=True, vis = vis)
    else:
        env_fn = lambda: FruitCollection2D(hybrid=True, vis = vis)


    _env = env_fn()
    total_actions = _env.total_actions

    if args.env == "1D":
        print("Creating network for 1D FruitCollection...")
        net = ActorHybridCriticNet1D(_env.reset().shape[0], total_actions, _env.total_fruits)
    else:
        print("Creating network for 2D FruitCollection...")
        net = ActorHybridCriticNet2D(_env.reset().shape[0], total_actions, _env.total_fruits)



    # create directories to store results
    result_dir = ensure_directory_exits(os.path.join(os.getcwd(), 'results'))
    env_dir = ensure_directory_exits(os.path.join(result_dir, _env.name, 'Actor_Hybrid_Critic'))
    plots_dir = ensure_directory_exits(os.path.join(env_dir, 'Plots'))
    net_path = os.path.join(env_dir, 'model.p')

    # Let the game begin !! Broom.. Broom..
    if args.cuda:
        net = net.cuda()
    if os.path.exists(net_path) and not args.scratch:
        net.load_state_dict(torch.load(net_path))

    hac = AHC(vis)

    if args.train:
        net.train()
        net = hac.train(net, env_fn, net_path, plots_dir, args)
        net.load_state_dict(torch.load(net_path))
    if args.test:
        net.eval()
        print('Average Performance:',
              hac.test(net, env_fn, args.test_episodes, log=True, sleep=args.sleep, render=args.render))
