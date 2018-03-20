import argparse
import os
import torch
import visdom
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from envs.fruit_collection_1d import FruitCollection1D
from envs.fruit_collection_2d import FruitCollection2D
from tools import ensure_directory_exits, weights_init, normalized_columns_initializer
from .hybrid_pg import HybridPolicyGraident


class PolicyNet(nn.Module):
    def __init__(self, input_size, actions, reward_types):
        super(PolicyNet, self).__init__()
        self.reward_types = reward_types

        for network_i in range(reward_types):
            layer = nn.Sequential(nn.Linear(input_size, 50),
                                  nn.ReLU(),
                                  nn.Linear(50, actions))
            setattr(self, 'policy_{}'.format(network_i), layer)
        self.actor_input_size = reward_types * actions
        self.actor_linear = nn.Linear(self.actor_input_size, actions)
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

    def forward(self, input):
        comb_policies = None
        for network_i in range(self.reward_types):
            out = getattr(self, 'policy_{}'.format(network_i))(input)
            # out = F.softmax(out)
            if network_i == 0:
                comb_policies = [out]
            else:
                comb_policies.append(out)
        comb_policies = torch.stack(comb_policies)
        x = F.softmax(comb_policies, dim = 2)
        x = comb_policies
        x = x.view(self.actor_input_size)
        x = Variable(x.data)
        actor_linear = self.actor_linear(x)
        return actor_linear, comb_policies


class PolicyNet1D(nn.Module):
    def __init__(self, input_size, actions, reward_types):
        super(PolicyNet1D, self).__init__()
        self.reward_types = reward_types

        for network_i in range(reward_types):
            layer = nn.Sequential(nn.Linear(input_size, 120),
                                  nn.ReLU(),
                                  nn.Linear(120, actions))
            setattr(self, 'policy_{}'.format(network_i), layer)
        self.actor_input_size = reward_types * actions
        self.actor_linear = nn.Linear(self.actor_input_size, actions)
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

    def forward(self, input):
        comb_policies = None
        for network_i in range(self.reward_types):
            out = getattr(self, 'policy_{}'.format(network_i))(input)
            # out = F.softmax(out)
            if network_i == 0:
                comb_policies = [out]
            else:
                comb_policies.append(out)
        comb_policies = torch.stack(comb_policies)
        x = comb_policies
        x = x.view(self.actor_input_size)
        x = Variable(x.data)
        prob = F.softmax(self.actor_linear(x), dim=0)
        return prob, comb_policies


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Decompose Policy Gradient - PyTorch')
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
    parser.add_argument('--decompose', action='store_true', help='Decompose reward type')
    parser.add_argument('--sleep', type=int, help='Sleep time for render', default=1)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and (not args.no_cuda)

    vis = visdom.Visdom() if args.render else None

    env_fn = lambda: FruitCollection1D(hybrid=True, vis = vis)

    _env = env_fn()
    total_actions = _env.total_actions
    policy_net = PolicyNet1D(_env.reset().shape[0], total_actions, _env.total_fruits)

    # create directories to store results
    result_dir = ensure_directory_exits(os.path.join(os.getcwd(), 'results'))
    env_dir = ensure_directory_exits(os.path.join(result_dir, _env.name, 'Hybrid_Pg_decompose' if args.decompose else "Hybrid_Pg"))
    plots_dir = ensure_directory_exits(os.path.join(env_dir, 'Plots'))
    policy_net_path = os.path.join(env_dir, 'model.p')

    # Let the game begin !! Broom.. Broom..
    if args.cuda:
        policy_net = policy_net.cuda()
    if os.path.exists(policy_net_path) and not args.scratch:
        policy_net.load_state_dict(torch.load(policy_net_path))


    pg = HybridPolicyGraident(_env.total_fruits, args.decompose, vis)

    if args.train:
        policy_net.train()
        policy_net = pg.train(policy_net, env_fn, policy_net_path, plots_dir, args)
        policy_net.load_state_dict(torch.load(policy_net_path))
    if args.test:
        policy_net.eval()
        print('Average Performance:', pg.test(policy_net, env_fn, args.test_episodes, log=True, sleep=args.sleep, render=True))
