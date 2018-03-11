import numpy as np
import random
import visdom


class FruitCollection:
    """
    A simple Fruit Collection environment
    """

    def __init__(self, vis=None, hybrid=False):
        self.total_fruits = 10
        self.visible_fruits = 5
        self.total_actions = 4
        self._fruit_consumed = None
        self._agent_position = None
        self.name = 'FruitCollection'
        self.hybrid = hybrid
        self.grid_size = (10, 10)
        self.max_steps = 100
        self.curr_step_count = 0
        self._fruit_positions = [(0, 0), (9, 9), (9, 0), (0, 9), (1, 7), (4, 8), (8, 4), (3, 6), (5, 8), (2, 2)]
        self.__vis = vis
        self.__image_window = None
        self.reward_threshold = 5  # optimal reward possible

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
        done = (False not in self._fruit_consumed) or (self.curr_step_count > self.max_steps)
        next_obs = self._get_observation()
        info = {}

        return next_obs, reward, done, info

    def _get_observation(self):
        grid = np.zeros((self.grid_size[0], self.grid_size[1]))
        grid[self._agent_position[0], self._agent_position[1]] = 1
        fruit_vector = np.zeros(self.total_fruits)
        fruit_vector[[ not x for x in self._fruit_consumed]] = 1
        return np.concatenate((grid.reshape(self.grid_size[0] * self.grid_size[1]), fruit_vector))

    def reset(self):
        available_fruits_loc = random.sample(range(self.total_fruits), self.visible_fruits)
        self._fruit_consumed = [(False if (i in available_fruits_loc) else True) for i in range(self.total_fruits)]
        while True:
            self._agent_position = [random.randint(0, 9), random.randint(0, 9)]
            if tuple(self._agent_position) not in self._fruit_positions:
                break
        obs = self._get_observation()
        return obs

    def close(self):
        self.__vis.close(win=self.__image_window)
        # self.__vis.close(win=None)
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
        return img

    def render(self):
        _obs_image = self.__get_obs_image()
        if self.__vis is not None:
            opts = dict(title=self.name, width=400, height=400)
            if self.__image_window is None:
                self.__image_window = self.__vis.image(_obs_image, opts=opts)
            else:
                self.__vis.image(_obs_image, opts=opts, win=self.__image_window)
        return _obs_image


if __name__ == '__main__':
    vis = visdom.Visdom()
    env_fn = lambda: FruitCollection(vis=vis)
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
