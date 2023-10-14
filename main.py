from arguments import get_args
from Dagger import DaggerAgent, ExampleAgent, MyDaggerAgent
import numpy as np
import time
import gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import keyboard
import time
import os
from tqdm import trange
import random
from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision.transforms import ToTensor


def plot(record):
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(record['steps'], record['mean'],
            color='blue', label='reward')
    ax.fill_between(record['steps'], record['min'], record['max'],
                    color='blue', alpha=0.2)
    ax.set_xlabel('number of steps')
    ax.set_ylabel('Average score per episode')
    ax1 = ax.twinx()
    ax1.plot(record['steps'], record['query'],
            color='red', label='query')
    ax1.set_ylabel('queries')
    reward_patch = mpatches.Patch(lw=1, linestyle='-', color='blue', label='score')
    query_patch = mpatches.Patch(lw=1, linestyle='-', color='red', label='query')
    patch_set = [reward_patch, query_patch]
    ax.legend(handles=patch_set)
    fig.savefig('performance.png')


# the wrap is mainly for speed up the game
# the agent will act every num_stacks frames instead of one frame
class Env(object):
    def __init__(self, env_name, num_stacks, render=None):
        self.env = gym.make(env_name, render_mode=render)
        # num_stacks: the agent acts every num_stacks frames
        # it could be any positive integer
        self.num_stacks = num_stacks
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        reward_sum = 0
        for stack in range(self.num_stacks):
            # ret = self.env.step(action)
            # print(type(ret), len(ret))
            # obs_next, reward, terminated, truncated, info = ret
            obs_next, reward, terminated, truncated, info = self.env.step(action) ### MOD
            reward_sum += reward
            done = terminated or truncated ### ???
            if done:
                self.env.reset()
                return obs_next, reward_sum, done, info
        return obs_next, reward_sum, done, info

    def reset(self):
        return self.env.reset()[0]


def main():
    # load hyper parameters
    args = get_args()
    num_updates = int(args.num_frames // args.num_steps)
    start = time.time()
    record = {
        'steps': [0],
        'max': [0],
        'mean': [0],
        'min': [0],
        'query': [0]
    }
    # query_cnt counts queries to the expert
    query_cnt = 0
    
    
    # agent initial
    # you should finish your agent with DaggerAgent
    # e.g. agent = MyDaggerAgent()
    # agent = ExampleAgent()
    save_id = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    writer = SummaryWriter(f"./logs/{save_id}")
    agent = MyDaggerAgent(writer, args.time_try)
    data_set = {'data': [], 'label': []}
    
    if args.load_model is not None:
        agent.load_model(args.load_model)
    if args.load_data is not None:
        savez = np.load(f'./saves/{args.load_data}.npz')
        # data_set['data'] = savez['data'].tolist()
        data_set['data'] = savez['data']
        data_set['label'] = savez['label'].tolist()
    
    # long_factor = 1 if not args.long_job else 2
    long_factor = 1
    # environment initial
    envs = Env(args.env_name, args.num_stacks//long_factor, 'human')
    # action_shape is the size of the discrete action set, here is 18
    # Most of the 18 actions are useless, find important actions
    # in the tips of the homework introduction document
    action_shape = envs.action_space.n
    # observation_shape is the shape of the observation
    # here is (210,160,3)=(height, weight, channels)
    observation_shape = envs.observation_space.shape
    print(action_shape, observation_shape)
    
    # You can play this game yourself for fun
    key_maps = {
        'w': 2,
        's': 5,
        'a': 4,
        'd': 3,
        'q': 12,
        'e': 11,
        'space': 1,
        'f': 0,
    }
    actions = [2, 5, 4, 3, 12, 11, 1, 0]
    if args.play_game:
        obs = envs.reset()
        # my_key_listening(obs, envs) ###
        # keyboard.wait('esc') ###
        while True:
            # im = Image.fromarray(obs)
            # im.save('imgs/' + str('screen') + '.jpeg')
            # action = int(input('input action'))
            # while action < 0 or action >= action_shape:
            #     action = int(input('re-input action'))
            k = keyboard.read_key()
            time.sleep(0.1)
            action = key_maps[k] if k in key_maps.keys() else 0
            obs_next, reward, done, _ = envs.step(action)
            obs = obs_next
            if done:
                obs = envs.reset()
    
    if args.test_only:
        obs = envs.reset()
        obs_last = np.zeros_like(obs)
        while True:
            obs_to = obs if not args.time_try else np.concatenate((obs_last, obs), axis=2)
            action = agent.select_action(obs_to)
            obs_next, reward, done, _ = envs.step(action)
            obs_last = obs
            obs = obs_next
            if done:
                obs = envs.reset()
    
    # start train your agent
    # epsilons = [0.75, 0.5, 0.25, 0.125]
    epsilons = [0.67]
    # epsilons = [0.2]
    for round in trange(num_updates):
        # an example of interacting with the environment
        # we init the environment and receive the initial observation
        if not args.long_job or round == 0:
            obs = envs.reset() ##
            obs_last = np.zeros_like(obs)
        # we get a trajectory with the length of args.num_steps
        # 可选是否标注
        # anno = input("Annotate this epoch or not? Y/[N]: ").upper() == 'Y'
        for step in trange(long_factor*args.num_steps):
            # Sample actions
            time.sleep(0.2)
            k = keyboard.read_key()
            act = key_maps[k] if k in key_maps.keys() else 0
            
            # 如何混合Agent和Expert策略？
            epsilon = epsilons[round] if (round < len(epsilons)) else epsilons[-1]
            obs_to = obs if not args.time_try else np.concatenate((obs_last, obs), axis=2)
            print(obs_last.shape, obs.shape, obs_to.shape)
            if not args.soft_act:
                if np.random.rand() < epsilon:
                    # We choose Expert's action
                    action = act
                    query_cnt += 1
                else:
                    # we choose a special action according to our model
                    action = agent.select_action(obs_to)
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                action_expert = (torch.Tensor(actions) == act*torch.ones(len(actions))).to(torch.float32).to(device)
                action_agent = agent.infer(ToTensor()(obs_to).unsqueeze(0).to(device))
                action_prop = epsilon*action_expert + (1-epsilon)*action_agent
                action = random.choices(actions, action_prop.squeeze(0))[0]
                print(action_prop, action) ###########

            # interact with the environment
            # we input the action to the environments and it returns some information
            # obs_next: the next observation after we do the action
            # reward: (float) the reward achieved by the action
            # down: (boolean)  whether it’s time to reset the environment again.
            #           done being True indicates the episode has terminated.
            obs_next, reward, done, _ = envs.step(action)
            # we view the new observation as current observation
            obs_now = obs
            obs = obs_next
            # if the episode has terminated, we need to reset the environment.
            if done:
                envs.reset()
            
            # an example of saving observations
            if args.save_img:
                im = Image.fromarray(obs_now)
                im.save('imgs/' + str(step) + '.jpeg')
                # data_set['data'].append(obs_now.tolist())
                obs_to = np.expand_dims(obs_now, 0)
                if args.time_try:
                    obs_last = np.expand_dims(obs_last, 0)
                    obs_to = np.concatenate((obs_last, obs_to), axis=3)
                if type(data_set['data']) == list:
                    data_set['data'] = obs_to
                else:
                    data_set['data'] = np.concatenate((data_set['data'], obs_to), axis=0)
                data_set['label'].append(act)
                # print(obs.dtype, obs0.shape, data_set['data'].shape)
            obs_last = obs_now
        
        #region
        # # You need to label the images in 'imgs/' by recording the right actions in label.txt
        # # 此时手动打开图片盯帧标注
        # if anno:
        #     print("Time to Label!")
        #     with open('./imgs/label.txt', 'w') as f:
        #         for i in range(args.num_steps):
        #             k = keyboard.read_key()
        #             act = key_maps[k] if k in key_maps.keys() else 0
        #             f.write(f'{act}\n')
        #             print(f'Step: {i:>03}, Action: {k}')
        #             time.sleep(0.2)
        # # After you have labeled all the images, you can load the labels
        # # for training a model
        # if anno:
        #     with open('./imgs/label.txt', 'r') as f:
        #         for label_tmp in f.readlines():
        #             label_tmp = int(label_tmp.strip())
        #             data_set['label'].append(label_tmp)
        #endregion
        
        # design how to train your model with labeled data
        agent.update(data_set['data'], data_set['label'], round, args.long_job)
        
        if (round + 1) % args.log_interval == 0:
            q = input("Test Visiable? Y/[N]: ")
            render = 'human' if q.upper()=='Y' else None
            
            test_envs = Env(args.env_name, args.num_stacks//long_factor, render)
            
            total_num_steps = (round + 1) * args.num_steps
            obs = test_envs.reset()
            obs_last = np.zeros_like(obs)
            reward_episode_set = []
            reward_episode = 0
            # evaluate your model by testing in the environment
            for step in trange(args.test_steps):
                obs_to = obs if not args.time_try else np.concatenate((obs_last, obs), axis=2)
                action = agent.select_action(obs_to)
                # you can render to get visual results
                # test_envs.render()
                obs_next, reward, done, _ = test_envs.step(action)
                reward_episode += reward
                obs_last = obs
                obs = obs_next
                if done:
                    reward_episode_set.append(reward_episode)
                    reward_episode = 0
                    test_envs.reset()
            if len(reward_episode_set) == 0:
                reward_episode_set.append(reward_episode)
                reward_episode = 0
            
            end = time.time()
            print(
                "TIME {} Updates {}, num timesteps {}, FPS {} \n query {}, avrage/min/max reward {:.1f}/{:.1f}/{:.1f}"
                    .format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)),
                    round, total_num_steps,
                    int(total_num_steps / (end - start)),
                    query_cnt,
                    np.mean(reward_episode_set),
                    np.min(reward_episode_set),
                    np.max(reward_episode_set)
                ))
            record['steps'].append(total_num_steps)
            record['mean'].append(np.mean(reward_episode_set))
            record['max'].append(np.max(reward_episode_set))
            record['min'].append(np.min(reward_episode_set))
            record['query'].append(query_cnt)
            plot(record)
        
        # 可选是否继续
        q = input("Next Epoch? [Y]/N: ")
        if q.upper() == 'N':
            break
    
    # 可选是否保存
    q = input("Save Models? Y/[N]: ")
    if q.upper() == 'Y':
        if not os.path.exists('./saves'):
            os.mkdir('./saves')
        agent.save_model(save_id)
        np.savez(f'./saves/{save_id}.npz', data=data_set['data'], label=data_set['label'])
    
    writer.close()


if __name__ == "__main__":
    main()
