import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from env import FruitCollection
from tools import ensure_directory_exits, weights_init, normalized_columns_initializer
from pg import PolicyGraident


class PolicyNet(nn.Module):
    def __init__(self, input_size, actions):
        super(PolicyNet, self).__init__()
        self.fc0 = nn.Linear(input_size, 256)
        self.fc1 = nn.Linear(256, 64)
        self.actor_linear = nn.Linear(64, actions)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

    def forward(self, input):
        x = F.relu(self.fc0(input))
        x = F.relu(self.fc1(x))
        prob = F.softmax(self.actor_linear(x))
        return prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Policy Fusion Architecture - PyTorch')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=10, metavar='N', help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--hybrid', action='store_true', help='use hybrid reward')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='no cuda usage')
    parser.add_argument('--train', action='store_true', default=False, help='Train')
    parser.add_argument('--test', action='store_true', default=False, help='Test')
    parser.add_argument('--train_episodes', type=int, default=10000, help='Train Episode count')
    parser.add_argument('--test_episodes', type=int, default=100, help='Test Episode count')
    parser.add_argument('--lr', type=float, default=0.01, help='Test')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and (not args.no_cuda)

    env_fn = lambda: FruitCollection()
    _env = env_fn()
    total_actions = _env.total_actions
    obs = _env.reset()
    policy_net = PolicyNet(obs.shape[0], total_actions)

    # create directories to store results
    result_dir = ensure_directory_exits(os.path.join(os.getcwd(), 'results'))
    env_dir = ensure_directory_exits(os.path.join(result_dir, _env.name, 'vanilla_Pg'))
    plots_dir = ensure_directory_exits(os.path.join(env_dir, 'Plots'))
    policy_net_path = os.path.join(env_dir, 'model.p')

    # Let the game begin !! Broom.. Broom..
    if args.cuda:
        policy_net = policy_net.cuda()
    if os.path.exists(policy_net_path):
        policy_net.load_state_dict(torch.load(policy_net_path))

    pg = PolicyGraident()

    if args.train:
        policy_net.train()
        policy_net = pg.train(policy_net, env_fn, policy_net_path, plots_dir, args)
        policy_net.load_state_dict(torch.load(policy_net_path))
    if args.test:
        policy_net.eval()
        print('Average Performance:', pg.test(policy_net, env_fn, args.test_episodes, log=True))
