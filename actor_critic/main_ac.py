import argparse
import visdom
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fruit_collection_2d import FruitCollection2D
from .tools import ensure_directory_exits, weights_init, normalized_columns_initializer
from .ac import ActorCritic


class ActorCriticNet(nn.Module):
    def __init__(self, input_size, actions):
        super(ActorCriticNet, self).__init__()
        self.fc0 = nn.Linear(input_size,  )
        self.fc1 = nn.Linear(256, 64)
        self.actor_linear = nn.Linear(64, actions)
        self.critic_linear = nn.Linear(64, 1)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, input):
        x = F.relu(self.fc0(input))
        x = F.relu(self.fc1(x))
        prob = F.softmax(self.actor_linear(x))
        critic = self.critic_linear(x)
        return prob, critic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Actor Critic')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=10, metavar='N', help='random seed (default: 10)')
    parser.add_argument('--render', action='store_true', default=False, help='render the environment')
    parser.add_argument('--hybrid', action='store_true', default=False, help='use hybrid reward')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch Size(No. of Episodes) for Training')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='interval between training status logs (default: 5)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables Cuda Usage')
    parser.add_argument('--train', action='store_true', default=False, help='Train the network')
    parser.add_argument('--test', action='store_true', default=False, help='Test the network')
    parser.add_argument('--train_episodes', type=int, default=500, help='Episode count for training')
    parser.add_argument('--test_episodes', type=int, default=100, help='Episode count for testing')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate for Training (Adam Optimizer)')
    parser.add_argument('--scratch', action='store_true', default=False,
                        help='Train the network from scratch ( or Does not load pre-trained model)')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and (not args.no_cuda)

    vis = visdom.Visdom() if args.render else None

    env_fn = lambda: FruitCollection2D(vis=vis)
    _env = env_fn()
    total_actions = _env.total_actions
    obs = _env.reset()
    _env.close()
    ac_net = ActorCriticNet(obs.shape[0], total_actions)

    # create directories to store results
    result_dir = ensure_directory_exits(os.path.join(os.getcwd(), 'results'))
    env_dir = ensure_directory_exits(os.path.join(result_dir, _env.name, 'Vanilla_Actor_Critic'))
    plots_dir = ensure_directory_exits(os.path.join(env_dir, 'Plots'))
    net_path = os.path.join(env_dir, 'model.p')

    # Let the game begin !! Broom.. Broom..
    if args.cuda:
        ac_net = ac_net.cuda()
    if os.path.exists(net_path) and not args.scratch:
        ac_net.load_state_dict(torch.load(net_path))

    ac_algo = ActorCritic(vis)

    if args.train:
        ac_net.train()
        ac_net = ac_algo.train(ac_net, env_fn, net_path, plots_dir, args)
        ac_net.load_state_dict(torch.load(net_path))
    if args.test:
        ac_net.eval()
        print('Average Performance:',
              ac_algo.test(ac_net, env_fn, args.test_episodes, log=True, render=args.render, sleep=1))
