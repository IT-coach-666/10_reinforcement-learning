"""
依赖包(python 3.9 环境, gym 包的版本不对可能会导致代码运行报错):
tensorflow==2.11.0
tensorflow-probability==0.19.0
tensorlayer==2.2.5
gym==0.25.2
    pip install gym[classic_control]

    使用 "PongNoFrameskip-v4" 时需安装:
    pip install "gym[atari, accept-rom-license]"
opencv-python==4.7.0.72
    yum install mesa-libGL -y
"""
import argparse
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl


class DDPG(object):
    def __init__(self, a_dim, s_dim, action_range, lr_a=0.001,
                 lr_c=0.002, gamma=0.9, tau=0.01, var=2,
                 memory_capacity=10000, batch_size=32):
        """
        action_range: 
        lr_a: learning rate for actor
        lr_c: learning rate for critic
        gamma: reward discount
        tau: soft replacement
        memory_capacity: size of replay buffer
        batch_size: update batchsize
        """
        self.a_dim, self.s_dim, self.action_range = a_dim, s_dim, action_range

        self.gamma = gamma
        self.tau = tau
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.var = var

        # jy: memory 用于储存跑的数据的数组 (保存 memory_capacity 个)
        #     s_dim * 2 + a_dim + 1: 两个 state + 一个 action + 一个 reward
        self.memory = np.zeros((memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        self.W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        self.b_init = tf.constant_initializer(0.1)

        # jy: 建立 actor 网络
        self.actor = self.get_actor([None, s_dim])
        # jy: 建立 critic 网络
        self.critic = self.get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()

        # jy: 建立 actor_target 网络, 参数和 actor 一致 (将 actor 网络
        #     的参数复制到 actor_target 网络), 不能训练
        self.actor_target = self.get_actor([None, s_dim], name='_target')
        self.copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        # jy: 建立 critic_target 网络, 参数和 critic 一致 (将 critic 网
        #     络的参数复制到 critic_target 网络), 不能训练
        self.critic_target = self.get_critic([None, s_dim], [None, a_dim],
                                             name='_target')
        self.copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        # jy: 建立 ema, 滑动平均值 (soft replacement)
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau) 

        self.actor_opt = tf.optimizers.Adam(lr_a)
        self.critic_opt = tf.optimizers.Adam(lr_c)


    def get_actor(self, input_state_shape, name=''):
        """
        Build actor network (建立 actor 网络, 输入 s, 输出 a)

        input_state_shape: state
        name: name
        """
        # jy: 定义输入层
        inputs = tl.layers.Input(input_state_shape, name='A_input')
        # jy: 定义输出层 (含隐层)
        layer = tl.layers.Dense(n_units=64, act=tf.nn.relu,
                                W_init=self.W_init, b_init=self.b_init,
                                name='A_l1')(inputs)
        layer = tl.layers.Dense(n_units=64, act=tf.nn.relu,
                                W_init=self.W_init, b_init=self.b_init,
                                name='A_l2')(layer)
        # jy: 先用 tanh 把范围限定在 [-1, 1] 之间, 再进行映射
        layer = tl.layers.Dense(n_units=self.a_dim, act=tf.nn.tanh, 
                                W_init=self.W_init, b_init=self.b_init,
                                name='A_a')(layer)
        layer = tl.layers.Lambda(lambda x: np.array(self.action_range) * x)(layer)  
        return tl.models.Model(inputs=inputs, outputs=layer, name='Actor' + name)

    def get_critic(self, input_state_shape, input_action_shape, name=''):
        """
        Build critic network (建立 Critic 网络): 输入 s 和 a, 输出 Q 值, 即 Q(s,a)

        input_state_shape: state
        input_action_shape: act
        name: name
        """
        # jy: 定义模型输入层
        s = tl.layers.Input(input_state_shape, name='C_s_input')
        a = tl.layers.Input(input_action_shape, name='C_a_input')
        # jy: 定义模型输出层 (含隐层)
        layer = tl.layers.Concat(1)([s, a])
        layer = tl.layers.Dense(n_units=64, act=tf.nn.relu,
                                W_init=self.W_init, b_init=self.b_init,
                                name='C_l1')(layer)
        layer = tl.layers.Dense(n_units=64, act=tf.nn.relu,
                                W_init=self.W_init, b_init=self.b_init,
                                name='C_l2')(layer)
        layer = tl.layers.Dense(n_units=1, W_init=self.W_init,
                                b_init=self.b_init, name='C_out')(layer)
        return tl.models.Model(inputs=[s, a], outputs=layer, name='Critic' + name)


    def copy_para(self, from_model, to_model):
        """
        Copy parameters for soft updating (更新参数, 只用于首次赋值)
        from_model: latest model
        to_model: target model
        """
        for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
            j.assign(i)


    def ema_update(self):
        """
        滑动平均更新: 更新 actor_target 和 critic_target 网络的参数;
        """
        # jy: 和之前的硬更新类似, 只是在更新赋值之前用一个 ema.average
        # jy: 获取要更新的参数 (包括 actor 和 critic 的相关参数)
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        # jy: 建立影子参数
        self.ema.apply(paras)   
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            # jy: 用滑动平均赋值
            i.assign(self.ema.average(j))  

    def choose_action(self, s, greedy=False):
        """
        选择动作: 输入 state, 通过 actor 网络, 输出 action
        """
        # jy: 将 state 传入 actor 网络得到输出, 
        a = self.actor(np.array([s], dtype=np.float32))[0]
        if greedy:
            return a
        # jy: add randomness to action selection for exploration; 为了能
        #     保持开发, 此处用另外一种方式增加探索: 以 a 为均值、var 为标准差建立
        #     正态分布, 再从正态分布采样出 a; 因为 a 是均值, 所以 a 的概率是最大
        #     的, 但 a 相对其他概率有多大是靠 var 调整; 这里其实可以增加更新 var,
        #     动态调整 a 的确定性, 然后进行裁剪
        return np.clip(np.random.normal(a, self.var),
                       -self.action_range, self.action_range)  

    def learn(self):
        """
        更新 critic 和 actor 网络;
        更新 actor_target 和 critic_target 网络 (采用滑动平均更新的方式)
        """
        # jy: 随机 batch_size 个随机数 (即 self.memory 中的数据的下标位置)
        indices = np.random.choice(self.memory_capacity, size=self.batch_size) 
        # jy: 根据随机得到的 indices 选取数据 bt
        bt = self.memory[indices, :] 
        # jy: 从 bt 获得数据 s, a, r, s_
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  
        br = bt[:, -self.s_dim - 1:-self.s_dim]       
        bs_ = bt[:, -self.s_dim:]                       

        # jy: Critic 网络的更新: 和 DQN 很像, 只是 target 不再是 argmax, 而
        #     是基于 critic_target 计算得到
        with tf.GradientTape() as tape:
            # jy: 基于 batch_size 个 state, 通过 actor_target 网络得到相应的 action
            a_ = self.actor_target(bs_)
            # jy: 基于 batch_size 个 state 和新 action, 通过 critic_target 网络得
            #     到新 action 对应的 Q 值
            q_ = self.critic_target([bs_, a_])
            y = br + self.gamma * q_

            # jy: 基于 batch_size 个 state 和原 action, 通过 critic 网络得到
            #     原 action 对应的 Q 值
            q = self.critic([bs, ba])
            # jy: 计算 td-error
            td_error = tf.losses.mean_squared_error(y, q)
        # jy: 基于 td-error 求梯度, 随后更新 critic 网络
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            # jy: 基于 batch_size 个 state, 通过 actor 网络得到相应
            #     的 action (目标就是获取最大 Q 值的 actor)
            a = self.actor(bs)
            # jy: 基于 batch_size 个 state 和新 a, 得到 batch_size 个 Q 值
            q = self.critic([bs, a])
            # jy: 对多个 Q 值取平均 (注意: 负号表示目标是梯度上升, 即离目标会越
            #     来越远, 即越来越大)
            a_loss = -tf.reduce_mean(q) 
        # jy: 基于 loss 更新 actor 网络
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))
        # jy: 滑动平均更新: 更新 actor_target 和 critic_target 网络的参数
        self.ema_update()


    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer (保存 s, a, r, s_)
        s: state
        a: act
        r: reward
        s_: next state
        """
        # jy: 整理 s, s_, 方便直接输入网络计算
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)

        # jy: 把 s, a, [r], s_ 横向堆叠
        transition = np.hstack((s, a, [r], s_))

        # jy: self.pointer 记录已经保存的数据总量, index 即当前数据
        #     在 self.memory 的位置; 一个循环中, 当 memory_capacity
        #     满了以后, index 就重新在最底开始, 即会将老数据覆盖 (replace
        #     the old memory with new memory)
        index = self.pointer % self.memory_capacity 
        # jy: 把 transition (即 s, a, [r], s_) 存进去
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        tl.files.save_npz(self.actor.trainable_weights,
                          'model/ddpg_actor.npz')
        tl.files.save_npz(self.actor_target.trainable_weights,
                          'model/ddpg_actor_target.npz')
        tl.files.save_npz(self.critic.trainable_weights,
                          'model/ddpg_critic.npz')
        tl.files.save_npz(self.critic_target.trainable_weights,
                          'model/ddpg_critic_target.npz')

    def load_ckpt(self):
        """
        load trained weights
        """
        tl.files.load_and_assign_npz('model/ddpg_actor.npz',
                                     network=self.actor)
        tl.files.load_and_assign_npz('model/ddpg_actor_target.npz',
                                     network=self.actor_target)
        tl.files.load_and_assign_npz('model/ddpg_critic.npz',
                                     network=self.critic)
        tl.files.load_and_assign_npz('model/ddpg_critic_target.npz',
                                     network=self.critic_target)


def run(mode="train"):
    ENV_NAME = 'Pendulum-v1'
    RANDOMSEED = 1
    # jy: total number of episodes for training
    MAX_EPISODES = 200
    # jy: total number of steps for each episode
    MAX_EP_STEPS = 200
    # jy: test the model per episodes
    TEST_PER_EPISODES = 10
    # jy: control exploration
    var = 3
    memory_capacity = 10000

    # jy: 初始化环境
    env = gym.make(ENV_NAME)
    #env = env.unwrapped

    # jy: 设置随机种子 (使得能够重现)
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    # jy: 定义状态空间, 动作空间, 动作幅度范围
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    action_range = env.action_space.high

    print('s_dim == %s' % s_dim)
    print('a_dim == %s' % a_dim)

    # jy: 初始化 DDPG 类
    ddpg = DDPG(a_dim, s_dim, action_range)

    # jy: 训练部分
    if mode == "train": 
        # jy: 记录每个 EP 的 reward, 统计变化 (用于后续画图)
        reward_buffer = []
        # jy: 统计时间
        t0 = time.time() 
        for i in range(MAX_EPISODES):
            t1 = time.time()
            s = env.reset()
            # jy: 记录当前 EP 的 reward
            ep_reward = 0     
            for j in range(MAX_EP_STEPS):
                # jy: Add exploration noise (直接用 actor 估算出 a 动作)
                a = ddpg.choose_action(s)       

                # jy: 与环境进行互动
                s_, r, done, info = env.step(a)

                # jy: 保存 (s, a, r, s_)
                ddpg.store_transition(s, a, r / 10, s_)

                # jy: 数据满了就可以开始学习
                if ddpg.pointer > memory_capacity:
                    # jy: 学习并更新 4 个网络的参数
                    ddpg.learn()

                # jy: 输出数据记录
                s = s_  
                # jy: 记录当前 EP 的总 reward
                ep_reward += r  
                if j == MAX_EP_STEPS - 1:
                    print('\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                              i, MAX_EPISODES, ep_reward, time.time() - t1), end='')
            # test
            if i and not i % TEST_PER_EPISODES:
                t1 = time.time()
                s = env.reset()
                # jy: ep_reward 统计 reward, 主要用于打印输出
                ep_reward = 0
                for j in range(MAX_EP_STEPS):

                    # jy: 注意: 测试时不再需要用正态分布获取 a, 直接一个 a 就可以
                    a = ddpg.choose_action(s)  
                    s_, r, done, info = env.step(a)

                    s = s_
                    ep_reward += r
                    if j == MAX_EP_STEPS - 1:
                        print('\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                              i, MAX_EPISODES, ep_reward, time.time() - t1))
                        reward_buffer.append(ep_reward)

            if reward_buffer:
                plt.ion()
                plt.cla()
                plt.title('DDPG')
                plt.plot(np.array(range(len(reward_buffer))) * TEST_PER_EPISODES, reward_buffer) 
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.ylim(-2000, 0)
                plt.show()
                plt.pause(0.1)
        plt.ioff()
        plt.show()
        print('\nRunning time: ', time.time() - t0)
        ddpg.save_ckpt()
    # test
    else:
        ddpg.load_ckpt()
        for episode in range(MAX_EPISODES):
            state = env.reset()
            episode_reward = 0
            t0 = time.time()
            for step in range(MAX_EP_STEPS):
                #env.render()
                state, reward, done, info = env.step(ddpg.choose_action(state, greedy=True))
                episode_reward += reward
                if done:
                    break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                  episode + 1, MAX_EPISODES, episode_reward,time.time() - t0))


if __name__ == '__main__':
    mode = "train"
    run(mode)
