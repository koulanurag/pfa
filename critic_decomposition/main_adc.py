import argparse
import os
import torch
import visdom
import torch.nn as nn
import torch.nn.functional as F
from envs import FruitCollection1D, FruitCollection2D
from tools import ensure_directory_exits, weights_init, normalized_columns_initializer
from critic_decomposition.actor_decomposed_critic import ActorHybridCritic as AHC


class ActorHybridCriticNet(nn.Module):
    def __init__(self, input_size, actions, reward_types):
        super(ActorHybridCriticNet, self).__init__()
        self.reward_types = reward_types
        self.actions = actions
        self.state = nn.Sequential(nn.Linear(input_size, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
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
        state = self.state(input)
        actor_linear = self.actor_dropout(self.actor(state))
        critic = self.decomposed_critic(state)
        return actor_linear, critic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Policy Fusion Architecture - PyTorch')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=10, metavar='N', help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--hybrid', action='store_true', help='use hybrid reward')
    parser.add_argument('--beta', type=float, default=0.01, help='Rate for Entropy')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage')
    parser.add_argument('--train', action='store_true', default=False, help='Train')
    parser.add_argument('--test', action='store_true', default=False, help='Test')
    parser.add_argument('--train_episodes', type=int, default=20000, help='Test')
    parser.add_argument('--test_episodes', type=int, default=100, help='Test')
    parser.add_argument('--lr', type=float, default=0.01, help='Test')
    parser.add_argument('--scratch', action='store_true', help='scratch')
    parser.add_argument('--sleep', type=int, help='Sleep time for render', default=1)
    parser.add_argument('--batch_size', type=int, help='Sleep time for render', default=32)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and (not args.no_cuda)

    vis = visdom.Visdom() if args.render else None

    env_fn = lambda: FruitCollection2D(hybrid=True, vis=vis)

    _env = env_fn()
    total_actions = _env.total_actions
    net = ActorHybridCriticNet(_env.reset().shape[0], total_actions, _env.total_fruits)

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
