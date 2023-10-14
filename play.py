import gym
import pygame
from gym.utils.play import play

key_map = {
    (pygame.K_w,): 2,
    (pygame.K_s,): 5,
    (pygame.K_a,): 4,
    (pygame.K_d,): 3,
    (pygame.K_q,): 12,
    (pygame.K_e,): 11,
    (pygame.K_SPACE,): 1,
    (pygame.K_f,): 0
}
play(gym.make(id='MontezumaRevengeNoFrameskip-v4', render_mode="rgb_array"), keys_to_action=key_map, zoom=4, fps=30)
