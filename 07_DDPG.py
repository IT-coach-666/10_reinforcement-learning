"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.

Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016


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
    def __init__(self, a_dim, s_dim, action_range, lr_a=0.001, lr_c=0.002, gamma=0.9,
                 tau=0.01, var=2, memory_capacity=10000, batch_size=32):
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

        # memory 用于储存跑的数据的数组(保存 memory_capacity 个);
        # s_dim * 2 + a_dim + 1 表示两个 state, 一个 action 和一个 reward
        self.memory = np.zeros((memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        self.W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        self.b_init = tf.constant_initializer(0.1)

        self.actor = self.get_actor([None, s_dim])
        self.critic = self.get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()

        #建立actor_target网络，并和actor参数一致，不能训练
        self.actor_target = self.get_actor([None, s_dim], name='_target')
        self.copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        #建立critic_target网络，并和actor参数一致，不能训练
        self.critic_target = self.get_critic([None, s_dim], [None, a_dim], name='_target')
        self.copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        #建立ema，滑动平均值
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(lr_a)
        self.critic_opt = tf.optimizers.Adam(lr_c)


    def get_actor(self, input_state_shape, name=''):
        """
        Build actor network (建立 actor 网络, 输入 s, 输出 a)

        input_state_shape: state
        name: name
        """
        inputs = tl.layers.Input(input_state_shape, name='A_input')
        layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=self.W_init, b_init=self.b_init, name='A_l1')(inputs)
        layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=self.W_init, b_init=self.b_init, name='A_l2')(layer)
        layer = tl.layers.Dense(n_units=self.a_dim, act=tf.nn.tanh, W_init=self.W_init, b_init=self.b_init, name='A_a')(layer)
        # 注意: 先用 tanh 把范围限定在 [-1,1] 之间, 再进行映射
        layer = tl.layers.Lambda(lambda x: np.array(self.action_range) * x)(layer)  
        return tl.models.Model(inputs=inputs, outputs=layer, name='Actor' + name)

    def get_critic(self, input_state_shape, input_action_shape, name=''):
        """
        Build critic network (建立 Critic 网络, 输入 s 和 a, 输出 Q 值)

        input_state_shape: state
        input_action_shape: act
        name: name

        返回 Q value, 即 Q(s,a)
        """
        s = tl.layers.Input(input_state_shape, name='C_s_input')
        a = tl.layers.Input(input_action_shape, name='C_a_input')
        layer = tl.layers.Concat(1)([s, a])
        layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=self.W_init, b_init=self.b_init, name='C_l1')(layer)
        layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=self.W_init, b_init=self.b_init, name='C_l2')(layer)
        layer = tl.layers.Dense(n_units=1, W_init=self.W_init, b_init=self.b_init, name='C_out')(layer)
        return tl.models.Model(inputs=[s, a], outputs=layer, name='Critic' + name)


    def copy_para(self, from_model, to_model):
        """
        Copy parameters for soft updating (更新参数, 只用于首次赋值, 之后就没用了)
        from_model: latest model
        to_model: target model
        """
        for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
            j.assign(i)


    def ema_update(self):
        """
        滑动平均更新
        """
        # 和之前的硬更新类似, 只是在更新赋值之前用一个 ema.average
        # 获取要更新的参数(包括 actor 和 critic 的)
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        # 主要是建立影子参数
        self.ema.apply(paras)   
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            # 用滑动平均赋值
            i.assign(self.ema.average(j))  

    def choose_action(self, s, greedy=False):
        """
        Choose action (选择动作, 输入 s, 输出 a)
        """
        a = self.actor(np.array([s], dtype=np.float32))[0]
        if greedy:
            return a
        # add randomness to action selection for exploration
        return np.clip(np.random.normal(a, self.var), -self.action_range, self.action_range)  


    def learn(self):
        """
        Update parameters
        """
        # 随机 batch_size 个随机数
        indices = np.random.choice(self.memory_capacity, size=self.batch_size) 
        # 根据 indices, 选取数据 bt, 相当于随机
        bt = self.memory[indices, :] 
        # 从 bt 获得数据 s, a, r, s_
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  
        br = bt[:, -self.s_dim - 1:-self.s_dim]       
        bs_ = bt[:, -self.s_dim:]                       

        # Critic: 更新和 DQN 很像, 只是 target 不是 argmax 了, 而是用 critic_target 计算出来的
        # br + self.gamma * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + self.gamma * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Actor: 目标就是获取最多 Q 值的 actor
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            # 注意: 此处是用负号, 是梯度上升(即离目标会越来越远, 即越来越大)
            a_loss = -tf.reduce_mean(q) 
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        self.ema_update()


    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer (保存 s, a, r, s_)
        s: state
        a: act
        r: reward
        s_: next state
        """
        # 整理 s, s_, 方便直接输入网络计算
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)

        # 把 s, a, [r], s_ 横向堆叠
        transition = np.hstack((s, a, [r], s_))

        # pointer 记录了曾经有多少数据进来
        # index 记录当前最新进来的数据位置
        # 所以是一个循环, 当 memory_capacity 满了以后, index 就重新在最底开始了
        # replace the old memory with new memory
        index = self.pointer % self.memory_capacity 
        # 把 transition (即 s, a, [r], s_) 存进去
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        tl.files.save_npz(self.actor.trainable_weights, 'model/ddpg_actor.npz')
        tl.files.save_npz(self.actor_target.trainable_weights, 'model/ddpg_actor_target.npz')
        tl.files.save_npz(self.critic.trainable_weights, 'model/ddpg_critic.npz')
        tl.files.save_npz(self.critic_target.trainable_weights, 'model/ddpg_critic_target.npz')

    def load_ckpt(self):
        """
        load trained weights
        """
        tl.files.load_and_assign_npz('model/ddpg_actor.npz', network=self.actor)
        tl.files.load_and_assign_npz('model/ddpg_actor_target.npz', network=self.actor_target)
        tl.files.load_and_assign_npz('model/ddpg_critic.npz', network=self.critic)
        tl.files.load_and_assign_npz('model/ddpg_critic_target.npz', network=self.critic_target)



def run(args):
    ENV_NAME = 'Pendulum-v1'
    RANDOMSEED = 1
    # total number of episodes for training
    MAX_EPISODES = 20 #200
    # total number of steps for each episode
    MAX_EP_STEPS = 200
    # test the model per episodes
    TEST_PER_EPISODES = 10
    # control exploration
    VAR = 3
    memory_capacity = 10000

    # 初始化环境
    env = gym.make(ENV_NAME)
    #env = env.unwrapped

    # reproducible, 设置随机种子 (使得能够重现)
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    # 定义状态空间, 动作空间, 动作幅度范围
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    action_range = env.action_space.high

    print('s_dim', s_dim)
    print('a_dim', a_dim)

    # 用 DDPG 算法
    ddpg = DDPG(a_dim, s_dim, action_range)

    # 训练部分
    if args.train: 
     
        # 用于记录每个 EP 的 reward, 统计变化
        reward_buffer = []
        # 统计时间
        t0 = time.time() 
        for i in range(MAX_EPISODES):
            t1 = time.time()
            s = env.reset()
            # 记录当前 EP 的 reward
            ep_reward = 0     
            for j in range(MAX_EP_STEPS):
                # Add exploration noise (直接用 actor 估算出 a 动作)
                a = ddpg.choose_action(s)       

                # 为了能保持开发，这里用了另外一种方式增加探索。
                # 因此需要需要以a为均值，VAR为标准差，建立正态分布，再从正态分布采样出a
                # 因为a是均值，所以a的概率是最大的。但a相对其他概率由多大，是靠VAR调整。这里我们其实可以增加更新VAR，动态调整a的确定性
                # 然后进行裁剪
                a = np.clip(np.random.normal(a, VAR), -2, 2)  
                # 与环境进行互动
                s_, r, done, info = env.step(a)

                # 保存s，a，r，s_
                ddpg.store_transition(s, a, r / 10, s_)

                # 第一次数据满了，就可以开始学习
                if ddpg.pointer > memory_capacity:
                    ddpg.learn()

                # 输出数据记录
                s = s_  
                # 记录当前 EP 的总 reward
                ep_reward += r  
                if j == MAX_EP_STEPS - 1:
                    print('\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                              i, MAX_EPISODES, ep_reward, time.time() - t1), end='')
                plt.show()
            # test
            if i and not i % TEST_PER_EPISODES:
                t1 = time.time()
                s = env.reset()
                ep_reward = 0
                for j in range(MAX_EP_STEPS):

                    # 注意: 测试时不再需要用正态分布, 直接一个 a 就可以
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
    if args.test:
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


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_false', default=True)
args = parser.parse_args()
if __name__ == '__main__':
    run(args)

