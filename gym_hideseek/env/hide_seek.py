import gym
from gym import spaces, utils
from gym.utils import seeding
from importlib_metadata import metadata

import numpy as np
from torch import layer_norm

from gym_hideseek.env.labyrinth import Labyrinth


class Hide_Seek(gym.Env):
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frames_per_second': 24
    }
        
    direction = [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]] # U D L R NOP
    
    def __init__(self, h, w, connectivity, window_ratio) -> None:
        super(Hide_Seek, self).__init__()
        self.observation_space = spaces.Box(low=0., high=1., shape=[h, w, 10], dtype=np.float32)
        
        self.h = h
        self.w = w
        self.conn = connectivity
        self.window_ratio = window_ratio
        self.field = Labyrinth()
        self.hider = self.seeker = None
        
        self.steps = 0
        
        self.viewer = None
        
        self.seed()
        self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    
    def reset(self):
        self.field.generate(self.h, self.w, self.conn, self.window_ratio, self.np_random)
        self.hider = [self.np_random.randint(0, self.h), self.np_random.randint(0, self.w)]
        self.seeker = [
            [self.np_random.randint(0, self.h), self.np_random.randint(0, self.w)],
            [self.np_random.randint(0, self.h), self.np_random.randint(0, self.w)]
        ]
        self.steps = 0
        
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
        return self.state
    
    def seeker_ai(self):
        from queue import Queue
        q = Queue()
        q.put(self.hider)
        dist = np.full([self.h, self.w], self.h*self.w+1, dtype=np.int32)
        dist[self.hider[0], self.hider[1]] = 0
        while not q.empty() > 0:
            p = q.get()
            d = dist[p[0], p[1]]
            for i in range(4):
                if self.field.field[i, p[0], p[1]] == 0:
                    newp = [p[0]+self.direction[i][0], p[1]+self.direction[i][1]]
                    if dist[newp[0], newp[1]] > d+1:
                        dist[newp[0], newp[1]] = d+1
                        q.put(newp)
                        
        for s in self.seeker:
            mdist = dist[s[0], s[1]]
            mdir = 4
            for i in range(4):
                if self.field.field[i, s[0], s[1]] == 0:
                    newp = [s[0]+self.direction[i][0], s[1]+self.direction[i][1]]
                    if dist[newp[0], newp[1]] < mdist:
                        mdist = dist[newp[0], newp[1]]
                        mdir = i
            s[0] += self.direction[mdir][0]
            s[1] += self.direction[mdir][1]
        if self.steps % 2:
            for s in self.seeker:
                mdist = dist[s[0], s[1]]
                mdir = 4
                for i in range(4):
                    if self.field.field[i, s[0], s[1]] == 0:
                        newp = [s[0]+self.direction[i][0], s[1]+self.direction[i][1]]
                        if dist[newp[0], newp[1]] < mdist:
                            mdist = dist[newp[0], newp[1]]
                            mdir = i
                s[0] += self.direction[mdir][0]
                s[1] += self.direction[mdir][1]
    
    def hider_ai(self):
        from queue import Queue
        dist = np.full([self.h, self.w], self.h*self.w+1, dtype=np.int32)
        for s in self.seeker:
            q = Queue()
            q.put(s)
            dist[s[0], s[1]] = 0
            while not q.empty() > 0:
                p = q.get()
                d = dist[p[0], p[1]]
                for i in range(4):
                    if self.field.field[i, p[0], p[1]] == 0:
                        newp = [p[0]+self.direction[i][0], p[1]+self.direction[i][1]]
                        if dist[newp[0], newp[1]] > d+1:
                            dist[newp[0], newp[1]] = d+1
                            q.put(newp)
        
        mdist = dist[self.hider[0], self.hider[1]]
        mdir = 4
        for i in range(4):
            if self.field.field[i, self.hider[0], self.hider[1]] == 0 or self.field.field[i+4, self.hider[0], self.hider[1]] == 1:
                newp = [self.hider[0]+self.direction[i][0], self.hider[1]+self.direction[i][1]]
                if dist[newp[0], newp[1]] > mdist:
                    mdist = dist[newp[0], newp[1]]
                    mdir = i
        self.hider[0] += self.direction[mdir][0]
        self.hider[1] += self.direction[mdir][1]
    
    @property
    def state(self):
        s = np.zeros([10, self.h, self.w], dtype=np.float32)
        s[0:8] = self.field.field.astype(np.float32)
        s[8, self.hider[0], self.hider[1]] = 1
        for seeker in self.seeker:
            s[9, seeker[0], seeker[1]] = 1
        return s.transpose([1, 2, 0])
    
    def render(self, mode="human"):
        block_width = 30
        wall_width = 3
        width = block_width * self.w
        height = block_width * self.h
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(width, height)
            for i in range(self.h):
                for j in range(self.w):
                    if self.field.field[0, i, j]:
                        edge = rendering.FilledPolygon([
                            (block_width*j, block_width*(self.h-i)),
                            (block_width*j, block_width*(self.h-i)-wall_width),
                            (block_width*(j+1), block_width*(self.h-i)-wall_width),
                            (block_width*(j+1), block_width*(self.h-i))
                        ])
                        edge.set_color(0, 0, 0)
                        self.viewer.add_geom(edge)
                    if self.field.field[1, i, j]:
                        edge = rendering.FilledPolygon([
                            (block_width*j, block_width*(self.h-1-i)),
                            (block_width*j, block_width*(self.h-1-i)+wall_width),
                            (block_width*(j+1), block_width*(self.h-1-i)+wall_width),
                            (block_width*(j+1), block_width*(self.h-1-i))
                        ])
                        edge.set_color(0, 0, 0)
                        self.viewer.add_geom(edge)
                    if self.field.field[2, i, j]:
                        edge = rendering.FilledPolygon([
                            (block_width*j, block_width*(self.h-i)),
                            (block_width*j, block_width*(self.h-1-i)),
                            (block_width*j+wall_width, block_width*(self.h-1-i)),
                            (block_width*j+wall_width, block_width*(self.h-i))
                        ])
                        edge.set_color(0, 0, 0)
                        self.viewer.add_geom(edge)
                    if self.field.field[3, i, j]:
                        edge = rendering.FilledPolygon([
                            (block_width*(j+1), block_width*(self.h-i)),
                            (block_width*(j+1), block_width*(self.h-1-i)),
                            (block_width*(j+1)-wall_width, block_width*(self.h-1-i)),
                            (block_width*(j+1)-wall_width, block_width*(self.h-i))
                        ])
                        edge.set_color(0, 0, 0)
                        self.viewer.add_geom(edge)
            for i in range(self.h):
                for j in range(self.w):
                    if self.field.field[4, i, j]:
                        edge = rendering.FilledPolygon([
                            (block_width*j, block_width*(self.h-i)),
                            (block_width*j, block_width*(self.h-i)-wall_width),
                            (block_width*(j+1), block_width*(self.h-i)-wall_width),
                            (block_width*(j+1), block_width*(self.h-i))
                        ])
                        edge.set_color(0, 1, 0)
                        self.viewer.add_geom(edge)
                    if self.field.field[5, i, j]:
                        edge = rendering.FilledPolygon([
                            (block_width*j, block_width*(self.h-1-i)),
                            (block_width*j, block_width*(self.h-1-i)+wall_width),
                            (block_width*(j+1), block_width*(self.h-1-i)+wall_width),
                            (block_width*(j+1), block_width*(self.h-1-i))
                        ])
                        edge.set_color(0, 1, 0)
                        self.viewer.add_geom(edge)
                    if self.field.field[6, i, j]:
                        edge = rendering.FilledPolygon([
                            (block_width*j, block_width*(self.h-i)),
                            (block_width*j, block_width*(self.h-1-i)),
                            (block_width*j+wall_width, block_width*(self.h-1-i)),
                            (block_width*j+wall_width, block_width*(self.h-i))
                        ])
                        edge.set_color(0, 1, 0)
                        self.viewer.add_geom(edge)
                    if self.field.field[7, i, j]:
                        edge = rendering.FilledPolygon([
                            (block_width*(j+1), block_width*(self.h-i)),
                            (block_width*(j+1), block_width*(self.h-1-i)),
                            (block_width*(j+1)-wall_width, block_width*(self.h-1-i)),
                            (block_width*(j+1)-wall_width, block_width*(self.h-i))
                        ])
                        edge.set_color(0, 1, 0)
                        self.viewer.add_geom(edge)
        self.viewer.draw_polygon(
            v=[
                (block_width*self.hider[1]+10, block_width*(self.h-1-self.hider[0])+10),
                (block_width*self.hider[1]+10, block_width*(self.h-1-self.hider[0])+20),
                (block_width*self.hider[1]+20, block_width*(self.h-1-self.hider[0])+20),
                (block_width*self.hider[1]+20, block_width*(self.h-1-self.hider[0])+10),
            ],
            color = (0, 0, 1)
        )
        for seeker in self.seeker:
            self.viewer.draw_polygon(
                v=[
                    (block_width*seeker[1]+10, block_width*(self.h-1-seeker[0])+10),
                    (block_width*seeker[1]+10, block_width*(self.h-1-seeker[0])+20),
                    (block_width*seeker[1]+20, block_width*(self.h-1-seeker[0])+20),
                    (block_width*seeker[1]+20, block_width*(self.h-1-seeker[0])+10),
                ],
                color = (1, 0, 0)
            )
        return self.viewer.render(return_rgb_array=mode=="rgb_array")


class Hider(Hide_Seek):
    def __init__(self, h = 10, w = 15, connectivity = 0, window_ratio = .1) -> None:
        super().__init__(h, w, connectivity, window_ratio)
        self.action_space = spaces.Discrete(5) # U D L R NOP

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        if action != 4 and \
            (self.field.field[action, self.hider[0], self.hider[1]] == 0 or \
                self.field.field[action+4, self.hider[0], self.hider[1]] == 1
            ):
            self.hider[0] += self.direction[action][0]
            self.hider[1] += self.direction[action][1]
        
        self.seeker_ai()
        
        reward = 1
        self.steps += 1
        done = self.steps > 1000
        
        for s in self.seeker:
            if s == self.hider:
                reward = -1
                done = True
        
        info = {}
        
        return self.state, reward, done, info


class Seeker(Hide_Seek):
    def __init__(self, h = 10, w = 15, connectivity = 0, window_ratio = .1) -> None:
        super().__init__(h, w, connectivity, window_ratio)
        self.action_space = spaces.MultiDiscrete([5, 5]) # U D L R NOP

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        for i, act in enumerate(action):
            if act != 4 and self.field.field[act, self.seeker[i][0], self.seeker[i][1]] == 0:
                self.seeker[i][0] += self.direction[act][0]
                self.seeker[i][1] += self.direction[act][1]
            
        if self.steps % 2 == 0:
            self.hider_ai()
        
        reward = -1
        self.steps += 1
        done = self.steps > 2000
        
        for s in self.seeker:
            if s == self.hider:
                reward = 1
                done = True
        
        info = {}
        
        return self.state, reward, done, info
            