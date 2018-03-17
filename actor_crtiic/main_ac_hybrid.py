import argparse
import os
import torch
import visdom
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from actor_crtiic.env import FruitCollection
from actor_crtiic.tools import ensure_directory_exits, weights_init, normalized_columns_initializer
from actor_crtiic.ac_hybrid import HybridActorCritic as HAC


class HybridActorCriticNet(nn.Module):
    def __init__(self, input_size, actions, reward_types):
        super(HybridActorCriticNet, self).__init__()
        self.reward_types = reward_types
        self.actions = actions
        for network_i in range(reward_types):
            layer = nn.Sequential(nn.Linear(input_size, 50),
                                  nn.ReLU())

            setattr(self, 'state_{}'.format(network_i), layer)
            setattr(self, 'actor_{}'.format(network_i), nn.Linear(50, actions))
            setattr(self, 'critic_{}'.format(network_i), nn.Linear(50, 1))

        for actor_i in range(actions):
            setattr(self, 'overall_action_{}'.format(actor_i), nn.Linear(reward_types, 1))

        self.apply(weights_init)
        for network_i in range(reward_types):
            getattr(self, 'critic_{}'.format(network_i)).weight.data.fill_(0)
            getattr(self, 'critic_{}'.format(network_i)).bias.data.fill_(0)
            getattr(self, 'actor_{}'.format(network_i)).bias.data.fill_(0)
            actor_weight = getattr(self, 'actor_{}'.format(network_i)).weight.data
            getattr(self, 'actor_{}'.format(network_i)).weight.data = normalized_columns_initializer(actor_weight, 0.01)

        for actor_i in range(actions):
            getattr(self, 'overall_action_{}'.format(actor_i)).bias.data.fill_(0)
            getattr(self, 'overall_action_{}'.format(actor_i)).weight.data.fill_(1)
            for param in getattr(self, 'overall_action_{}'.format(actor_i)).parameters():
                param.required_grad = False

    def forward(self, input):
        comb_policies = None
        comb_critic = None
        for network_i in range(self.reward_types):
            state_input = getattr(self, 'state_{}'.format(network_i))(input)
            actor = getattr(self, 'actor_{}'.format(network_i))(state_input)
            critic = getattr(self, 'critic_{}'.format(network_i))(state_input)

            if network_i == 0:
                comb_policies = [actor]
                comb_critic = [critic]
            else:
                comb_policies.append(actor)
                comb_critic.append(critic)
        comb_policies = torch.stack(comb_policies)
        comb_critic = torch.stack(comb_critic)

        actor = []
        for act_i in range(self.actions):
            action_input = comb_policies[:, :, act_i]
            action_input = Variable(action_input.data).resize(1, self.reward_types)
            actor.append(getattr(self, 'overall_action_{}'.format(act_i))(action_input))
        actor = torch.stack(actor)
        actor = actor.view(self.actions)
        prob = F.softmax(actor, dim=0)

        return prob, comb_policies, comb_critic


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
    parser.add_argument('--train_episodes', type=int, default=1000, help='Test')
    parser.add_argument('--test_episodes', type=int, default=100, help='Test')
    parser.add_argument('--lr', type=float, default=0.01, help='Test')
    parser.add_argument('--scratch', action='store_true', help='scratch')
    parser.add_argument('--decompose', action='store_true', help='Decompose reward type')
    parser.add_argument('--sleep', type=int, help='Sleep time for render', default=1)
    parser.add_argument('--batch_size', type=int, help='Sleep time for render', default=10)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and (not args.no_cuda)

    vis = visdom.Visdom() if args.render else None

    env_fn = lambda: FruitCollection(hybrid=True, vis=vis)

    _env = env_fn()
    total_actions = _env.total_actions
    net = HybridActorCriticNet(_env.reset().shape[0], total_actions, _env.total_fruits)

    # create directories to store results
    result_dir = ensure_directory_exits(os.path.join(os.getcwd(), 'results'))
    env_dir = ensure_directory_exits(os.path.join(result_dir, _env.name,
                                                  'HAC_Decompose' if args.decompose else "HAC_No_Decompose"))
    plots_dir = ensure_directory_exits(os.path.join(env_dir, 'Plots'))
    policy_net_path = os.path.join(env_dir, 'model.p')

    # Let the game begin !! Broom.. Broom..
    if args.cuda:
        net = net.cuda()
    if os.path.exists(policy_net_path) and not args.scratch:
        net.load_state_dict(torch.load(policy_net_path))

    hac = HAC(_env.total_fruits, args.decompose, vis)

    if args.train:
        net.train()
        net = hac.train(net, env_fn, policy_net_path, plots_dir, args)
        net.load_state_dict(torch.load(policy_net_path))
    if args.test:
        net.eval()
        print('Average Performance:',
              hac.test(net, env_fn, args.test_episodes, log=True, sleep=args.sleep, render=True))
