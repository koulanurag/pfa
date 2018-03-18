import numpy as np
import random
import visdom


class FruitCollection2D:
    """
    A simple Fruit Collection environment
    """

    def __init__(self, vis=None, hybrid=False):
        self.total_fruits = 10
        self.visible_fruits = 5
        self.total_actions = 4
        self._fruit_consumed = None
        self._agent_position = None
        self.name = 'FruitCollection2D'
        self.hybrid = hybrid
        self.grid_size = (10, 10)
        self.max_steps = 200
        self.curr_step_count = 0
        self._fruit_positions = [(1, 0), (3, 1), (8, 2), (2, 3), (5, 4), (1, 5), (6, 6), (9, 7), (5, 8), (1, 9)]
        # self._fruit_positions = [(0, 0), (0, 5), (5, 0), (5, 5)]
        # self._fruit_position_color = []
        self._agent_position = [2, 2]
        self.__vis = vis
        self.__image_window = None
        self.reward_threshold = 10  # optimal reward possible
        self.game_score = 0
        self.__linear_grid_window = None
        self.step_reward = 0
        self.fruit_collected = 0
        self.get_action_meanings = ['Up', 'Right', 'Down', 'Left']

    def __move(self, action):
        agent_pos = None
        if action == 0:
            agent_pos = [self._agent_position[0] - 1, self._agent_position[1]]
        elif action == 1:
            agent_pos = [self._agent_position[0], self._agent_position[1] + 1]
        elif action == 2:
            agent_pos = [self._agent_position[0] + 1, self._agent_position[1]]
        elif action == 3:
            agent_pos = [self._agent_position[0], self._agent_position[1] - 1]

        if 0 <= agent_pos[0] < self.grid_size[0] and 0 <= agent_pos[1] < self.grid_size[1]:
            self._agent_position = agent_pos
            return True
        else:
            return False

    def step(self, action):
        if action >= self.total_actions:
            raise ValueError("action must be one of %r" % range(self.total_actions))
        if self.hybrid:
            # reward = [0 if consumed else 1 for consumed in self._fruit_consumed]
            reward = [0 for _ in range(self.total_fruits)]
        else:
            reward = 0
        self.curr_step_count += 1
        if self.__move(action):
            if tuple(self._agent_position) in self._fruit_positions:
                idx = self._fruit_positions.index(tuple(self._agent_position))
                if not self._fruit_consumed[idx]:
                    self._fruit_consumed[idx] = True
                    if self.hybrid:
                        reward[idx] = 1
                    else:
                        reward = 1
                    self.fruit_collected += 1

        done = (False not in self._fruit_consumed) or (self.curr_step_count > self.max_steps)
        next_obs = self._get_observation()
        info = {}
        self.step_reward = reward
        self.game_score += sum(self.step_reward) if self.hybrid else self.step_reward
        return next_obs, reward, done, info

    def _get_observation(self):
        grid = np.zeros((self.grid_size[0], self.grid_size[1]))
        grid[self._agent_position[0], self._agent_position[1]] = 1
        fruit_vector = np.zeros(self.total_fruits)
        fruit_vector[[not x for x in self._fruit_consumed]] = 1
        return np.concatenate((grid.reshape(self.grid_size[0] * self.grid_size[1]), fruit_vector))

    def reset(self):
        self.game_score = 0
        available_fruits_loc = random.sample(range(self.total_fruits), self.visible_fruits)
        self._fruit_consumed = [(False if (i in available_fruits_loc) else True) for i in range(self.total_fruits)]
        # while True:
        #     self._agent_position = [random.randint(0, 9), random.randint(0, 9)]
        #     if tuple(self._agent_position) not in self._fruit_positions:
        #         break
        obs = self._get_observation()
        return obs

    def close(self):
        if self.__vis is not None:
            self.__vis.close(win=self.__image_window)
            self.__vis.close(win=self.__linear_grid_window)
        pass

    def seed(self, x):
        pass

    def __get_obs_image(self):
        img = np.ones((3, 10, 10))
        img[:] = 255

        # color agent
        img[0, self._agent_position[0], self._agent_position[1]] = 224
        img[1, self._agent_position[0], self._agent_position[1]] = 80
        img[2, self._agent_position[0], self._agent_position[1]] = 20

        # fruits
        for i, consumed in enumerate(self._fruit_consumed):
            if not consumed:
                img[0, self._fruit_positions[i][0], self._fruit_positions[i][1]] = 91
                img[1, self._fruit_positions[i][0], self._fruit_positions[i][1]] = 226
                img[2, self._fruit_positions[i][0], self._fruit_positions[i][1]] = 116
            else:
                img[0, self._fruit_positions[i][0], self._fruit_positions[i][1]] = 0
                img[1, self._fruit_positions[i][0], self._fruit_positions[i][1]] = 0
                img[2, self._fruit_positions[i][0], self._fruit_positions[i][1]] = 0
        return img

    def render(self):
        _obs_image = self.__get_obs_image()
        if self.__vis is not None:
            opts = dict(title='{}    \tFruit Collected:{} Overall_Reward:{} Step Reward:{} Steps:{}'
                        .format(self.name, self.fruit_collected, round(self.game_score, 3), self.step_reward,
                                self.curr_step_count),
                        width=400, height=400)
            if self.__image_window is None:
                self.__image_window = self.__vis.image(_obs_image, opts=opts)
            else:
                self.__vis.image(_obs_image, opts=opts, win=self.__image_window)
            # if self.__linear_grid_window is None:
            #     self.__linear_grid_window = self.__vis.text(self._get_observation().__str__())
            # else:
            #     self.__vis.text(self._get_observation().__str__(), win=self.__linear_grid_window)
        return _obs_image


if __name__ == '__main__':
    """ User interaction with the Environment"""
    vis = visdom.Visdom()
    env_fn = lambda: FruitCollection2D(vis=vis)
    for ep in range(5):
        random.seed(ep)
        env = env_fn()
        done = False
        obs = env.reset()
        total_reward = 0
        while not done:
            env.render()
            print(obs)
            action = int(input("action:"))
            obs, reward, done, info = env.step(action)
            total_reward += reward
        env.close()
        print("Episode Reward:", total_reward)
