import gym_hideseek.env.labyrinth as labyrinth
from gym_hideseek.env.hide_seek import *
import random
from time import sleep

# lbr = labyrinth.Labyrinth()

# random.seed(0x1234567)
# lbr.generate(4, 6, 0.5)
# lbr.render()

hs = Hider(10, 15, 0.4, 0.1)
hs.reset()
hs.render()
sleep(0.1)
hs.render()
# input()
sleep(0.1)
done = False
while not done:
    hs.hider_ai()
    _, _, done, _ = hs.step(4)
    hs.render()
    # input()
    sleep(0.1)
