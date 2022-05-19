import numpy as np
import random
import typing

from gym_hideseek.env.dsu import DSU

class Labyrinth:
    __wallu = 0
    __walld = 1
    __walll = 2
    __wallr = 3
        
    def generate(self, h, w, connectivity = .3, window_ratio = .1, rand = random) -> None:
        assert h >= 2 and w >= 2
        assert connectivity >= 0. and window_ratio >= 0. and connectivity + window_ratio <= 1.
        self.h = h
        self.w = w
        self.field = np.ones([8, h, w], dtype=np.int32)
        self.field[4:] = 0
        
        walls_h = []
        walls_s = []
        rest_walls_s = []
        rest_walls_h = []
        for i in range(h):
            for j in range(w-1):
                walls_h.append([i*w + j, i*w + j+1])
                walls_s.append([i*w + j, i*w + j+1])
        for i in range(w):
            for j in range(h-1):
                walls_h.append([j*w + i, (j+1)*w + i])
                walls_s.append([j*w + i, (j+1)*w + i])
        
        s_s = DSU(h*w)
        s_h = DSU(h*w)
        rand.shuffle(walls_s)
        rand.shuffle(walls_h)
        
        for wall in walls_h:
            w1 = wall[0]
            w2 = wall[1]
            p1 = s_h.find(w1)
            p2 = s_h.find(w2)
            if p1 != p2:
                if rand.random() < window_ratio:
                    if w2 - w1 == 1:
                        self.field[self.__wallr+4, w1 // w, w1 % w] = 1
                        self.field[self.__walll+4, w2 // w, w2 % w] = 1
                    else:
                        self.field[self.__walld+4, w1 // w, w1 % w] = 1
                        self.field[self.__wallu+4, w2 // w, w2 % w] = 1
                else:
                    if w2 - w1 == 1:
                        self.field[self.__wallr, w1 // w, w1 % w] = 0
                        self.field[self.__walll, w2 // w, w2 % w] = 0
                    else:
                        self.field[self.__walld, w1 // w, w1 % w] = 0
                        self.field[self.__wallu, w2 // w, w2 % w] = 0
                s_h.add(p1, p2)
            else:
                rest_walls_h.append(wall)
                
        for wall in walls_s:
            w1 = wall[0]
            w2 = wall[1]
            p1 = s_s.find(w1)
            p2 = s_s.find(w2)
            if p1 != p2:
                if w2 - w1 == 1:
                    if self.field[self.__wallr+4, w1 // w, w1 % w] == 1:
                        continue
                    self.field[self.__wallr, w1 // w, w1 % w] = 0
                    self.field[self.__walll, w2 // w, w2 % w] = 0
                else:
                    if self.field[self.__walld+4, w1 // w, w1 % w] == 1:
                        continue
                    self.field[self.__walld, w1 // w, w1 % w] = 0
                    self.field[self.__wallu, w2 // w, w2 % w] = 0
                s_s.add(p1, p2)
            else:
                rest_walls_s.append(wall)
        
        # n_del_wall = int(connectivity * len(rest_walls))
        # n_window = int(window_ratio * len(rest_walls))
        # rand.shuffle(rest_walls)
        # for i in range(n_del_wall):
        #     wall = rest_walls[i]
        #     w1 = wall[0]
        #     w2 = wall[1]
        #     if w2 - w1 == 1:
        #         self.field[self.__wallr, w1 // w, w1 % w] = 0
        #         self.field[self.__walll, w2 // w, w2 % w] = 0
        #     else:
        #         self.field[self.__walld, w1 // w, w1 % w] = 0
        #         self.field[self.__wallu, w2 // w, w2 % w] = 0
        
        # for i in range(n_del_wall, n_del_wall + n_window):
        #     wall = rest_walls[i]
        #     w1 = wall[0]
        #     w2 = wall[1]
        #     if w2 - w1 == 1:
        #         self.field[self.__wallr+4, w1 // w, w1 % w] = 1
        #         self.field[self.__walll+4, w2 // w, w2 % w] = 1
        #     else:
        #         self.field[self.__walld+4, w1 // w, w1 % w] = 1
        #         self.field[self.__wallu+4, w2 // w, w2 % w] = 1
    
    def render(self):
        for j in range(self.w):
            print('--', end='')
        print('-')
        for i in range(self.h):
            print('|', end='')
            for j in range(self.w-1):
                print(' ', end='')
                if self.field[self.__wallr, i, j]:
                    print('|', end='')
                elif self.field[4+self.__wallr, i, j]:
                    print('/', end='')
                else:
                    print(' ', end='')
            print(' |')
            if i != self.h-1:
                print('+', end='')
                for j in range(self.w):
                    if self.field[self.__walld, i, j]:
                        print('-+', end='')
                    elif self.field[4+self.__walld, i, j]:
                        print('=+', end='')
                    else:
                        print(' +', end='')
                print('')
            
        for j in range(self.w):
            print('--', end='')
        print('-')
