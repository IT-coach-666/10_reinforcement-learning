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
import tensorflow_probability as tfp

import tensorlayer as tl


class PPO(object):
    def __init__(self, state_dim, action_dim, action_bound, method_name="clip",
                 gamma=0.9, a_lr=0.0001, c_lr=0.0002, epsilon=1e-8,
                 a_update_steps=10, c_update_steps=10):
        """
        state_dim: state dimension
        action_dim: action dimension

        gamma: reward discount
        a_lr: learning rate for actor
        c_lr: learning rate for critic
        epsilon: epsilon
        a_update_steps: actor update steps
        c_update_steps: critic update steps
        """
        assert method_name in ["clip", "kl_pen"], "``method_name`` must in ['clip', 'kl_pen']"
        self.method_name = method_name

        self.epsilon = epsilon
        self.a_update_steps = a_update_steps
        self.c_update_steps = c_update_steps
        self.gamma = gamma

        # jy: 模型超参
        # jy: PPO-1 (对应 method_name 为 "kl_pen") 需要使用到
        self.lam = 0.5
        self.kl_target = 0.01
        # jy: PPO-2 (对应 method_name 为 "clip"), Clipped surrogate objective (find this is better)
        self.epsilon = 0.2

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound


        # 构建 critic 网络: 输入 state, 输出 V 值 ============================
        # jy: 定义网络的输入层;
        inputs = tl.layers.Input([None, state_dim], tf.float32, 'state')
        
        # jy: 定义网络的输出层(含隐层)
        layer = tl.layers.Dense(100, tf.nn.relu)(inputs)
        # jy: 再多加一层;
        #layer = tl.layers.Dense(64, tf.nn.relu)(layer)
        v = tl.layers.Dense(1)(layer)
        # jy: 定义网络的输入与输出;
        self.critic = tl.models.Model(inputs, v)
        self.critic.train()

        # 构建 actor 网络; 输入 state, 输出均值和方差 ========================
        # 有两个: actor 和 actor_old (用于记录行为策略的版本, 不可训练)
        # 输入 state, 输出描述动作分布的 mu 和 sigma
        self.actor = self._build_anet('pi', trainable=True)
        self.actor_old = self._build_anet('oldpi', trainable=False)

        # jy; 分别设置 actor 和 critic 的学习率;
        self.actor_opt = tf.optimizers.Adam(a_lr)
        self.critic_opt = tf.optimizers.Adam(c_lr)

    def _build_anet(self, name, trainable):
        '''
        构建策略网络(policy network)

        name: name
        trainable: trainable flag
        '''
        # 连续动作型问题, 输出 mu 和 sigma
        inputs = tl.layers.Input([None, self.state_dim], tf.float32, name + '_state')
        layer = tl.layers.Dense(100, tf.nn.relu, name=name + '_l1')(inputs)
        # jy: 多加一层
        #layer = tl.layers.Dense(64, tf.nn.relu, name=name + '_l2')(layer)

        a = tl.layers.Dense(self.action_dim, tf.nn.tanh, name=name + '_a')(layer)
        mu = tl.layers.Lambda(lambda x: x * self.action_bound, name=name + '_lambda')(a)

        sigma = tl.layers.Dense(self.action_dim, tf.nn.softplus, name=name + '_sigma')(layer)
        # jy: 构造模型(确定模型的输入、输出和模型名称)
        model = tl.models.Model(inputs, [mu, sigma], name)

        if trainable:
            model.train()
        else:
            model.eval()
        return model

    def a_train(self, tfs, tfa, tfadv):
        '''
        更新 actor 网络 (策略网络, policy network)
        输入 state, action, td-error
        如果使用的是 "kl_pen" 方法, 则会输出 kl 值;
        '''
        # 输入时 s, a, td-error (和 AC 类似)
        tfs = np.array(tfs, np.float32)
        tfa = np.array(tfa, np.float32) 
        tfadv = np.array(tfadv, np.float32) 

        with tf.GradientTape() as tape:

            # jy: 需从两个不同网络构建两个正态分布(pi, oldpi)
            # jy: 新网络(actor 网络), 基于网络产生的均值与方差生成策略 pi
            mu, sigma = self.actor(tfs)
            pi = tfp.distributions.Normal(mu, sigma)
            # jy: 老网络(actor_old 网络), 基于网络产生的均值与方差生成策略 oldpi
            mu_old, sigma_old = self.actor_old(tfs)
            oldpi = tfp.distributions.Normal(mu_old, sigma_old)

            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            # 在新旧两个分布下, 同样输出 a 的概率的比值: 除以 (oldpi.prob(tfa) + self.epsilon) 
            # 其实就是做 import-sampling; 本可以直接用 pi.prob(tfa) 去更新, 但为了能够更新多次,
            # 需要除以 (oldpi.prob(tfa) + self.epsilon)
            # 在 AC 或 PG 中是以 1, 0 作为更新目标, 缩小动作概率到 1 or 0 的差距
            # 而 PPO 可以看作是以 oldpi.prob(tfa) 出发, 不断远离(增大 or 缩小)的过程
            ratio = pi.prob(tfa) / (oldpi.prob(tfa) + self.epsilon)
            #ratio = tf.exp(pi.prob(tfa) - oldpi.prob(tfa))
            # 这个的意义和带参数更新是一样的
            surr = ratio * tfadv

            # jy: 不能让两个分布差异太大
            if self.method_name == 'kl_pen':
                # PPO1
                # jy: 计算两个策略(正太分布)的 KL 散度;
                kl = tfp.distributions.kl_divergence(oldpi, pi)
                #print("kl=== %s" % kl)
                # jy: 取均值, 用于后续返回;
                kl_mean = tf.reduce_mean(kl)
                # jy: 计算 loss 值;
                aloss = -(tf.reduce_mean(surr - self.lam * kl))
            else:
                # PPO2: clipping method (find this is better); 很直接, 就是直接进行截断
                aloss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * tfadv))
        # jy: 基于 loss 更新 actor 网络参数;
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        if self.method_name == 'kl_pen':
            return kl_mean

    def update_old_pi(self):
        '''
        更新 actor_old 网络的参数, 使得与 actor 网络保持一致;
        '''
        for p, oldp in zip(self.actor.trainable_weights, self.actor_old.trainable_weights):
            oldp.assign(p)

    def c_train(self, tfdc_r, s):
        '''
        更新 Critic 网络
        '''
        # tfdc_r 可以理解为 PG 中就是 G, 通过回溯计算得到; 只不过这 PPO 用 TD 而已
        tfdc_r = np.array(tfdc_r, dtype=np.float32)

        with tf.GradientTape() as tape:
            # jy: 基于 state 通过 critic 网络得到 v 值;
            v = self.critic(s)
            # 即 td-error
            advantage = tfdc_r - v
            # jy: 计算损失;
            closs = tf.reduce_mean(tf.square(advantage))

        # jy: 基于 loss 更新网络参数;
        grad = tape.gradient(closs, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def cal_adv(self, tfs, tfdc_r):
        """
        计算 advantage (即 td-error): 即 reward 与 critic 网络对 state 评分之间的差值;
        tfs: 即 state, 维度为 (32, 3), 其中 32 为 batch_size
        tfdc_r: 即 reward, 维度为 (32, 1); 为 N 步(即一个 batch_size )更新并通过回溯计算得到的 reward 结果
        """
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        # advantage = r - self.gamma * V(s_)
        # jy: reward 与 critic 网络对 state 的评分的差值即 td-error;
        advantage = tfdc_r - self.critic(tfs)
        return advantage.numpy()

    def update(self, s, a, r):
        """
        Update parameter with the constraint of KL divergent
        s: state, 维度为 (32, 3), 其中 32 为 batch_size
        a: action, 维度为 (32, 1)
        r: reward, 维度为 (32, 1); 为 N 步(即一个 batch_size )更新并通过回溯计算得到的 reward 结果
        """
        s, a, r = s.astype(np.float32), a.astype(np.float32), r.astype(np.float32)
        # jy: 更新 actor_old 网络的参数, 使得与 actor 网络保持一致
        self.update_old_pi()
        
        # jy: 计算 advantage (即 td-error)
        adv = self.cal_adv(s, r)
        # 归一化处理(sometimes helpful)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)

        # 更新 actor
        if self.method_name == 'kl_pen':
            # PPO1: 比较复杂, 会动态调整参数(adaptive KL penalty)
            for _ in range(self.a_update_steps):
                # jy: 基于 state, action, 和 td-error 更新 actor 网络;
                kl = self.a_train(s, a, adv)
                # jy: google 论文中采用该方式; self.kl_target 默认为 0.01
                if kl > 4 * self.kl_target:
                    break

            # adaptive lambda (OpenAI 的论文中采用该方式)
            if kl < self.kl_target / 1.5:
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
            # sometimes explode, this clipping is MorvanZhou's solution
            self.lam = np.clip(self.lam, 1e-4, 10)
        else:
            # PPO2: 比较简单, 直接就进行 a_train 更新:
            # clipping method, find this is better (OpenAI's paper)
            for _ in range(self.a_update_steps):
                # jy: 基于 state, action, 和 td-error 更新 actor 网络
                self.a_train(s, a, adv)

        # 更新 critic
        for _ in range(self.c_update_steps):
            # jy: 基于 reward 和 state 更新 critic 网络;
            self.c_train(r, s)

    def choose_action(self, s, greedy=False):
        '''
        Choose action: 基于 state 通过 actor 预估策略分布, 并基于该策略分布得
        到 action (会经过 clip 处理, 确保 action 在有效范围之内)
        s: state
        '''
        s = s[np.newaxis, :].astype(np.float32)
        # 通过 actor 网络得到与 action 的策略分布相关的两个参数: mu 和 sigma
        mu, sigma = self.actor(s)
        if greedy:
            a = mu[0]
        else:
            # 用 mu 和 sigma 构建正态分布
            pi = tfp.distributions.Normal(mu, sigma)
            # 根据概率分布随机出动作
            a = tf.squeeze(pi.sample(1), axis=0)[0]

        # 最后 sample 动作, 并进行裁剪
        return np.clip(a, -self.action_bound, self.action_bound)

    def get_v(self, s):
        '''
        计算 V (value) 值 (通过 critic 网络计算)
        传入的 s 如: [0.9987801  0.04937944 1.965758  ]
        '''
        s = s.astype(np.float32)
        # 要和输入的形状对应
        if s.ndim < 2:
            # jy: 将 s 从 [0.9987801  0.04937944 1.965758  ] 转换为: [[0.9987801  0.04937944 1.965758  ]]
            #     主要是为了适用 critic 网络的输入;
            s = s[np.newaxis, :]
        # jy: 通过 critic 网络计算得到;
        #     self.critic 得到的结果如: tf.Tensor([[-0.03649434]], shape=(1, 1), dtype=float32)
        #     self.critic(s)[0, 0] 即为: <tf.Tensor: shape=(), dtype=float32, numpy=-0.036494337>
        return self.critic(s)[0, 0]

    def save_ckpt(self):
        """
        save trained weights
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        tl.files.save_npz(self.actor.trainable_weights, 'model/ppo_actor.npz')
        tl.files.save_npz(self.actor_old.trainable_weights, 'model/ppo_actor_old.npz')
        tl.files.save_npz(self.critic.trainable_weights, 'model/ppo_critic.npz')

    def load_ckpt(self):
        """
        load trained weights
        """
        tl.files.load_and_assign_npz('model/ppo_actor.npz', network=self.actor)
        tl.files.load_and_assign_npz('model/ppo_actor_old.npz', network=self.actor_old)
        tl.files.load_and_assign_npz('model/ppo_critic.npz', network=self.critic)



def run(mode):
    # environment name
    ENV_NAME = 'Pendulum-v1'
    # random seed
    RANDOMSEED = 1
    # total number of episodes for training
    EP_MAX = 1000
    # total number of steps for each episode
    EP_LEN = 200
    # update batchsize
    batch_size = 32
    # gamma: reward discount
    gamma = 0.9

    #env = gym.make(ENV_NAME).unwrapped
    env = gym.make(ENV_NAME)

    # reproducible
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high


    method_name = "kl_pen"
    # jy: 初始化 PPO 类;
    ppo = PPO(state_dim, action_dim, action_bound, method_name)

    if mode == "train":
        # jy: 统计 reward, 主要用于后续绘图
        all_ep_r = []

        # 更新流程
        for ep in range(EP_MAX):
            # s 如: [0.9972427  0.07420918 0.90092736]
            s = env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            # jy: 统计 reward, 主要用于后续打印输出;
            ep_r = 0
            t0 = time.time()
            # in one episode
            for t in range(EP_LEN):
                # env.render()
                # jy: 基于 state 选择 action; 得到的 a 如: [0.6857695]
                a = ppo.choose_action(s)
                # jy: 得到的 (s_, r, done, _) 如: ([0.99191463 0.12690681 1.0594497 ], -0.08715443734750178, False, {})
                s_, r, done, _ = env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                # 对奖励进行归一化; 有时挺有用的(奖励是个主观的东西)
                buffer_r.append((r + 8) / 8)
                # jy: 更新 state
                s = s_
                ep_r += r

                # N 步(即一个 batch_size )更新的方法, 每 batch 步就进行一次更新
                if (t + 1) % batch_size == 0 or t == EP_LEN - 1:
                    # 计算 N 步中最后一个 state 的 V 值(v_s_)
                    v_s_ = ppo.get_v(s_)

                    # 和 PG 一样, 回溯计算(基于 buffer_r 中的 N 步得到的 reward 从后往前计算)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + gamma * v_s_
                        discounted_r.append(v_s_)
                    # jy: 由于 discounted_r 存储的是从后往前的结果, 因此需要反转恢复;
                    discounted_r.reverse()

                    # br 并不是每个状态的 reward, 而是通过回溯计算的 V 值
                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    # N 步更新后清空缓存中的数据;
                    buffer_s, buffer_a, buffer_r = [], [], []
                    # jy: 【IMP】基于 N 步得到的 state, action, reward 进行更新网络
                    ppo.update(bs, ba, br)

            # jy: 以下为日志打印和绘图部分 (主要用于查看训练过程的训练效果与趋势)
            if ep == 0:
                all_ep_r.append(ep_r)
            else:
                all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                  ep, EP_MAX, ep_r, time.time() - t0))

            # 画图
            plt.ion()
            plt.cla()
            plt.title('PPO')
            plt.plot(np.arange(len(all_ep_r)), all_ep_r)
            plt.ylim(-2000, 0)
            plt.xlabel('Episode')
            plt.ylabel('Moving averaged episode reward')
            plt.show()
            plt.pause(0.1)
        ppo.save_ckpt()
        plt.ioff()
        plt.show()
    # test
    else:
        ppo.load_ckpt()
        for episode in range(EP_MAX):
            state = env.reset()
            episode_reward = 0
            t0 = time.time()
            for step in range(EP_LEN):
                #env.render()
                state, reward, done, info = env.step(ppo.choose_action(state, greedy=True))
                episode_reward += reward
                if done:
                    break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                      episode + 1, EP_MAX, episode_reward, time.time() - t0))



if __name__ == '__main__':
    mode = "train"
    #mode = "test"
    run(mode)


