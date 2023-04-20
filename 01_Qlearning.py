"""
依赖包版本:
gym                           0.25.2
tensorflow                    2.11.0
"""

import argparse
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true',
                    default=True)
parser.add_argument('--test', dest='test', action='store_true',
                    default=True)

parser.add_argument(
    '--save_path', default=None,
    help='folder to save if mode == train else model path,'
         'qnet will be saved once target net update'
)
parser.add_argument('--seed', help='random seed', type=int, default=0)
# 环境默认为 Gym 的 FrozenLake-v1
"""
FrozenLake-v1 是一个 4*4 的网络格子，每个格子可以是起始块，目标块、冻结块或者危险块
我们的目标是让智能体学习如何从开始块如何行动到目标块上，而不是移动到危险块上
智能体可以选择向上、向下、向左或向右移动，同时游戏中还有可能吹来一阵风，将智能体吹到任意的方块上
"""
parser.add_argument('--env_id', default='FrozenLake-v1')
args = parser.parse_args()

## Load the environment
alg_name = 'Qlearning'
env_id = args.env_id
env = gym.make(env_id)

# display the game environment (设置是否渲染，展示游戏画面)
render = False  
# jy: 设置为 True 时需安装 pygame, 安装方式: pip install gym[toy_text]
#render = True


##================= Implement Q-Table learning algorithm =====================##
## Initialize table with all zeros
#  np.zeros() 函数生成一个全 0 矩阵
#  env.observation_space.n : 环境中状态的数量
#  env.action_space.n : 动作空间的数量
#  建立 Q 表格, 并初始化为全 0 数组, 形状为 [状态空间, 动作空间]
Q = np.zeros([env.observation_space.n, env.action_space.n])

## Set learning parameters
# alpha, if use value function approximation, we can ignore it
lr = .85 
# decay factor (折扣率 decay factor)
lambd = .99  
# 迭代次数, 也就是开始 10000 次游戏
num_episodes = 10000
t0 = time.time()

if args.train:
    # 用于记录每次迭代的总奖励，这样就可以知道智能体是否有进步了
    all_episode_reward = []
    # num_episodes：进行游戏的次数; 每次游戏开始都会用 env.reset() 函数把智能体设置在任意状态出发, 一直走到最终状态
    for i in range(num_episodes):
        ## Reset environment and get first new observation
        s = env.reset()
        # 用于记录这次游戏的总奖励, 这里先初始化为 0
        rAll = 0
        ## The Q-Table learning algorithm
        # 设定进行一次游戏最多进行 99 步; 这和环境状态多少有关, 当前环境状态 16 个, 
        # 在 99 步前一般都能到达最终状态
        for j in range(99):
            # jy: 判断是否渲染环境
            if render: env.render()
            ## Choose an action by greedily (with noise) picking from Q table
            # jy: noisy-greedy 的实现: 从 Q 表格中找到当前状态 S 最大 Q 值, 并在 Q 值上加上噪音,
            #     然后找到最大的 Q + 噪音的动作
            #     Q[s, :]:  Q-table 表的 s 行，即当前状态对应各个动作的 Q 值
            #     np.random.randn(1, env.action_space.n) 即噪音，我们希望噪音随着迭代的进行将会
            #         越来越小, 因此乘以 (1. / (i + 1)), 当 i 越来越大时噪音就越来越小
            #     np.argmax(): 获得最大 Q 值对应的列号, 也就是对应的动作; 注意: argmax 找出最大值后,
            #         并不是返回最大值, 而是返回最大值的列号（即动作 A）, 因为我们需要的是动作 A, 而不是 Q 值
            a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
            ## Get new state and reward from environment
            # jy: 与环境互动, 把动作传入到环境中, 得到 4 个返回值
            #     s1 : 表示执行动作后, 新的状态（newstate）
            #     r: 执行动作 a 后，获得的收获
            #     d: 一个标志位, 表示这个是否最终状态
            #     _: 其实是 info, 但一般用不到这个值, 因此把它先忽略
            s1, r, d, _ = env.step(a)
            ## Update Q-Table with new knowledge（更新 Q 表格）
            # jy: 用 newstate 的 Q 值, 更新现在状态的 Q 值（参考更新公式进行理解）
            #     注意比较 np.max 和 np.argmax 函数的区别
            #         np.max 返回最大值
            #	      np.argmax 返回时最大的行数或列数
            Q[s, a] = Q[s, a] + lr * (r + lambd * np.max(Q[s1, :]) - Q[s, a])
            # jy: 累加当前的收获
            rAll += r
            # jy: 把下一个状态 s1 赋值给 s, 准备开始下一步; 但在开始之前检查
            #     下一个状态是否就是终止状态了, 如果是, 这一次游戏就算是完成;
            s = s1
            # jy: 如果已经到达最终状态, 就跳出 for 循环 (开始下一次迭代)
            if d is True:
                break
        print('Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
              i + 1, num_episodes, rAll,
              time.time() - t0))
        # jy: 每次的总收获都放到 all_episode_reward, 可以通过打印看看算法效率
        #     每一次迭代获得的总收获 rAll, 会以 0.1 的份额加入到 all_episode_reward (原代码这里 rAll 用了 r, 个人认为是 rAll 更合适)
        if i == 0:
            all_episode_reward.append(rAll)
        else:
            all_episode_reward.append(all_episode_reward[-1] * 0.9 + rAll * 0.1)

    # save
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, 'Q_table.npy'), Q)

    plt.plot(all_episode_reward)
    if not os.path.exists('image'):
        os.makedirs('image')
    plt.savefig(os.path.join('image', '_'.join([alg_name, env_id])))
    # 把 Qtable 打印出来
    print("Final Q-Table Values: \n %s" % Q)

if args.test:
    path = os.path.join('model', '_'.join([alg_name, env_id]))
    Q = np.load(os.path.join(path, 'Q_table.npy'))
    for i in range(num_episodes):
        ## Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        ## The Q-Table learning algorithm
        for j in range(99):
            ## Choose an action by greedily (with noise) picking from Q table
            a = np.argmax(Q[s, :])
            ## Get new state and reward from environment
            s1, r, d, _ = env.step(a)
            ## Update Q-Table with new knowledge
            rAll += r
            s = s1
            if d is True:
                break
        print(
            'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                i + 1, num_episodes, rAll,
                time.time() - t0
            )
        )
