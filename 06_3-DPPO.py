"""
Distributed Proximal Policy Optimization (DPPO)

A distributed version of OpenAI's Proximal Policy Optimization (PPO).
Workers in parallel to collect data, then stop worker's roll-out and 
train PPO on collected data.

Restart workers once PPO is updated.


参考文献:
Emergence of Locomotion Behaviours in Rich Environments, Heess et al. 2017
Proximal Policy Optimization Algorithms, Schulman et al. 2017
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016


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
import queue
import threading
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import tensorlayer as tl



class PPO(object):
    def __init__(self, s_dim, a_dim, ep_max, method_name="clip", a_lr=0.0001,
                 c_lr=0.0002, a_update_steps=10, c_update_steps=10, eps=1e-8):
        """
        a_lr: learning rate for actor
        c_lr: learning rate for critic
        a_update_steps: actor update steps
        c_update_steps: critic update steps
        eps: epsilon
        s_dim: state dimension, 
        a_dim: action dimension
        ep_max: total number of episodes for training (最大更新次数)
        """
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.method_name = method_name
        self.ep_max = ep_max
        self.a_update_steps = a_update_steps
        self.c_update_steps = c_update_steps
        self.eps = eps

        # KL penalty
        self.kl_target = 0.01
        self.lam = 0.5

        # Clipped surrogate objective, find this is better
        self.epsilon = 0.2

        # 创建 critic 网络
        inputs = tl.layers.Input([None, self.s_dim], tf.float32, 'state')
        layer = tl.layers.Dense(64, tf.nn.relu)(inputs)
        # jy: 添加多层;
        #layer = tl.layers.Dense(64, tf.nn.relu)(layer)
        v = tl.layers.Dense(1)(layer)
        self.critic = tl.models.Model(inputs, v)
        self.critic.train()

        # jy: 创建 actor 网络
        self.actor = self._build_anet('pi', trainable=True)
        # jy: 创建 actor_old 网络(不可训练)
        self.actor_old = self._build_anet('oldpi', trainable=False)
        self.actor_opt = tf.optimizers.Adam(a_lr)
        self.critic_opt = tf.optimizers.Adam(c_lr)


    def _build_anet(self, name, trainable):
        """
        Build policy network (用于构建 actor 网络)
        name: name
        trainable: trainable flag
        返回: policy network
        """
        # jy: 定义 actor 网络的输入层
        inputs = tl.layers.Input([None, self.s_dim], tf.float32, name + '_state')
        # jy: 定义 actor 网络的隐藏和输出层
        layer = tl.layers.Dense(64, tf.nn.relu, name=name + '_l1')(inputs)
        # jy: 添加多层
        #layer = tl.layers.Dense(64, tf.nn.relu, name=name + '_l2')(layer)
        a = tl.layers.Dense(self.a_dim, tf.nn.tanh, name=name + '_a')(layer)
        mu = tl.layers.Lambda(lambda x: x * 2, name=name + '_lambda')(a)
        sigma = tl.layers.Dense(self.a_dim, tf.nn.softplus, name=name + '_sigma')(layer)
        # jy: 基于输入输出层定义网络
        model = tl.models.Model(inputs, [mu, sigma], name)

        if trainable:
            model.train()
        else:
            model.eval()
        return model


    def a_train(self, tfs, tfa, tfadv):
        '''
        Update policy network (更新 actor 网络)
        tfs: state
        tfa: act
        tfadv: advantage (即 td-error)
        '''
        tfs = np.array(tfs, np.float32)
        tfa = np.array(tfa, np.float32)
        # td-error
        tfadv = np.array(tfadv, np.float32)
        with tf.GradientTape() as tape:
            # jy: 获取 actor 网络的策略分布
            mu, sigma = self.actor(tfs)
            pi = tfp.distributions.Normal(mu, sigma)

            # jy: 获取 actor_old 网络的策略分布
            mu_old, sigma_old = self.actor_old(tfs)
            oldpi = tfp.distributions.Normal(mu_old, sigma_old)

            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            ratio = pi.prob(tfa) / (oldpi.prob(tfa) + self.eps)
            surr = ratio * tfadv

            # jy: 求 actor 网络的 loss
            ## PPO1
            if self.method_name == 'kl_pen':
                # jy: 基于两个策略分布, 计算 kl 散度
                kl = tfp.distributions.kl_divergence(oldpi, pi)
                # jy: 对 kl 散度取均值
                kl_mean = tf.reduce_mean(kl)
                # jy: 注意, 求得的 loss 前带负号
                aloss = -(tf.reduce_mean(surr - self.lam * kl))
            ## PPO2
            else:  
                # clipping method, find this is better
                aloss = -tf.reduce_mean(
                    tf.minimum(surr,
                               tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * tfadv)
                )
        # jy: 基于 loss 求 actor 网络的梯度, 并更新网络参数; 
        a_gard = tape.gradient(aloss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        if self.method_name == 'kl_pen':
            return kl_mean

    def update_old_pi(self):
        """
        Update old policy parameter (更新 old_pi)
        """
        for p, oldp in zip(self.actor.trainable_weights, self.actor_old.trainable_weights):
            oldp.assign(p)

    def c_train(self, tfdc_r, s):
        """
        Update actor network (更新 critic 网络)
        tfdc_r: cumulative reward
        s: state
        """
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        with tf.GradientTape() as tape:
            # 计算 advantage: V(s') * gamma + r - V(s)
            advantage = tfdc_r - self.critic(s)
            closs = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(closs, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def cal_adv(self, tfs, tfdc_r):
        """
        Calculate advantage: V(s') * gamma + r - V(s)
        tfs: state
        tfdc_r: cumulative reward
        返回: advantage
        """
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        advantage = tfdc_r - self.critic(tfs)
        return advantage.numpy()

    def update(self):
        """
        Update parameter with the constraint of KL divergent
        """
        global GLOBAL_UPDATE_COUNTER, GLOBAL_EP
        # 如果协调器没有停止
        while not COORD.should_stop():
            if GLOBAL_EP < self.ep_max:
                # PPO 进程的等待位置
                UPDATE_EVENT.wait() 
                # copy pi to old pi
                self.update_old_pi()
                # collect data from all workers
                data = [QUEUE.get() for _ in range(QUEUE.qsize())] 
                data = np.vstack(data)

                s, a, r = data[:, :self.s_dim].astype(np.float32), \
                          data[:, self.s_dim: self.s_dim + self.a_dim].astype(np.float32), \
                          data[:, -1:].astype(np.float32)

                adv = self.cal_adv(s, r)
                # 归一化处理, sometimes helpful
                # adv = (adv - adv.mean())/(adv.std()+1e-6)

                # update actor
                ## PPO1
                if self.method_name == 'kl_pen':
                    for _ in range(self.a_update_steps):
                        kl = self.a_train(s, a, adv)
                        # this in in google's paper
                        if kl > 4 * self.kl_target:  
                            break
                    # adaptive lambda, this is in OpenAI's paper
                    if kl < self.kl_target / 1.5: 
                        self.lam /= 2
                    elif kl > self.kl_target * 1.5:
                        self.lam *= 2
                    # sometimes explode, this clipping is MorvanZhou's solution
                    self.lam = np.clip(self.lam, 1e-4, 10)

                ## PPO2
                else:  
                    # clipping method, find this is better (OpenAI's paper)
                    for _ in range(self.a_update_steps):
                        self.a_train(s, a, adv)

                # update critic
                for _ in range(self.c_update_steps):
                    self.c_train(r, s)

                # updating finished
                UPDATE_EVENT.clear()
                # reset counter
                GLOBAL_UPDATE_COUNTER = 0
                # set roll-out available
                ROLLING_EVENT.set()     


    def choose_action(self, s):
        """
        Choose action (基于 state 使用 actor 网络选择动作)
        s: state
        返回: clipped act
        """
        s = s[np.newaxis, :].astype(np.float32)
        # jy: actor 网络生成策略分布参数
        mu, sigma = self.actor(s)
        # jy: 构造策略分布, 并基于该分布选择 action
        pi = tfp.distributions.Normal(mu, sigma)
        # choosing action
        a = tf.squeeze(pi.sample(1), axis=0)[0]  
        return np.clip(a, -2, 2)

    def get_v(self, s):
        """
        Compute value (计算 V)
        s: state
        返回: value
        """
        s = s.astype(np.float32)
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.critic(s)[0, 0]

    def save_ckpt(self):
        """
        save trained weights
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        tl.files.save_npz(self.actor.trainable_weights, 'model/dppo_actor.npz')
        tl.files.save_npz(self.actor_old.trainable_weights, 'model/dppo_actor_old.npz')
        tl.files.save_npz(self.critic.trainable_weights, 'model/dppo_critic.npz')

    def load_ckpt(self):
        """
        load trained weights
        """
        tl.files.load_and_assign_npz('model/dppo_actor.npz', network=self.actor)
        tl.files.load_and_assign_npz('model/dppo_actor_old.npz', network=self.actor_old)
        tl.files.load_and_assign_npz('model/dppo_critic.npz', network=self.critic)



class Worker(object):
    '''
    Worker class for distributional running
    '''
    def __init__(self, env_name, wid, ep_max, ep_len,
                 min_batch_size=64, gamma=0.9):
        """
        env_name: environment name
        gamma: reward discount
        ep_max: total number of episodes for training (最大更新次数)
        ep_len: total number of steps for each episode
        min_batch_size: minimum batch size for updating PPO
        """
        # 工号
        self.wid = wid 
        # 创建环境
        self.env = gym.make(env_name)
        # 设置不同的随机种子, 使得每个 worker 的都不一致
        self.env.seed(wid * 100) 
        self.ep_max = ep_max
        self.ep_len = ep_len
        self.min_batch_size = min_batch_size
        self.gamma = gamma

    def work(self):
        '''
        Define a worker
        '''
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER, GLOBAL_PPO
        # 从 COORD 接受消息, 看看是否应该 should_stop
        while not COORD.should_stop():
            s = self.env.reset()                    
            ep_r = 0
            # 记录 data
            buffer_s, buffer_a, buffer_r = [], [], []
            # 记录 data
            t0 = time.time()
            for t in range(self.ep_len):
                
                # 查询进程是否被阻塞, 如果在阻塞状态, 就证明如果 global PPO 正在更
                # 新, 则在这里等待, 否则就可以继续
                if not ROLLING_EVENT.is_set():  
                    # worker 进程的等待位置; wait until PPO is updated
                    ROLLING_EVENT.wait()
                    # clear history buffer, use new policy to collect data
                    buffer_s, buffer_a, buffer_r = [], [], []  
                
                # 正常跑游戏, 并搜集数据
                a = GLOBAL_PPO.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                # normalize reward, find to be useful
                buffer_r.append((r + 8) / 8)  
                s = s_
                ep_r += r

                # 每个 work 在游戏中进行一步(产生一条数据) GLOBAL_UPDATE_COUNTER 就会 +1
                # count to minimum batch size, no need to wait other workers
                # 当 GLOBAL_UPDATE_COUNTER 大于 batch(64) 时, 就可以进行更新
                GLOBAL_UPDATE_COUNTER += 1  
                # jy: 当 GLOBAL_UPDATE_COUNTER 大于 batch(64) 时, 就可以进行更新
                #     self.ep_len - 1 表示最后一步
                if t == self.ep_len - 1 or GLOBAL_UPDATE_COUNTER >= self.min_batch_size:
                    # 计算每个状态对应的 V(s')
                    v_s_ = GLOBAL_PPO.get_v(s_)
                    # compute discounted reward
                    discounted_r = []  
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    # 堆叠成数据, 并保存到公共队列中
                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    # put data in the queue
                    QUEUE.put(np.hstack((bs, ba, br)))  

                    # 如果数据足够, 就开始更新
                    if GLOBAL_UPDATE_COUNTER >= self.min_batch_size:
                        # stop collecting data
                        ROLLING_EVENT.clear() 
                        # global PPO update
                        UPDATE_EVENT.set() 

                    # stop training
                    if GLOBAL_EP >= self.ep_max:
                        # 停止更新
                        COORD.request_stop()  
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1

            print('Episode: {}/{}  | Worker: {} | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                  GLOBAL_EP, self.ep_max, self.wid, ep_r, time.time() - t0))



def run(mode="train"):
    ENV_NAME = "Pendulum-v1"
    RANDOMSEED = 1
    # total number of episodes for training
    EP_MAX = 200 #1000
    # total number of steps for each episode
    EP_LEN = 200
    # parallel workers
    N_WORKER = 4  

    env = gym.make(ENV_NAME)

    # reproducible
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print("state_dim: %s" % state_dim)
    print("action_dim: %s" % action_dim)

    global GLOBAL_PPO
    GLOBAL_PPO = PPO(state_dim, action_dim, EP_MAX, method_name="kl_pen")
    if mode == "train":
        # 创建 workers
        workers = [Worker(ENV_NAME, i, EP_MAX, EP_LEN) for i in range(N_WORKER)]

        threads = []

        # 为每个 worker 创建线程 (worker threads)
        for worker in workers: 
            # 创建线程, 线程将执行 Worker 类的 work 方法;
            t = threading.Thread(target=worker.work, args=())
            # 开始线程
            t.start() 
            # 把线程放到进程列表中, 方便管理
            threads.append(t) 
        
        print("============= add a PPO updating thread")
        # 把一个全局的 PPO 更新放到线程列表最后
        threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
        threads[-1].start()
        print("============= 把线程列表交给协调器管理")
        # 把线程列表交给协调器管理
        COORD.join(threads) 
        print("============= 准备保存全局参数")

        # 保存全局参数
        GLOBAL_PPO.save_ckpt()  

        # plot reward change and test
        plt.title('DPPO')
        plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
        plt.xlabel('Episode')
        plt.ylabel('Moving reward')
        plt.ylim(-2000, 0)
        plt.show()
    else:
        GLOBAL_PPO.load_ckpt()
        s = env.reset()
        for t in range(EP_LEN):
            #env.render()
            s, r, done, info = env.step(GLOBAL_PPO.choose_action(s))
            if done:
                break



# jy: 以下定义 PPO 类和 Worker 类中均需要使用到的全局变量 (故需定义在函数外部)
# 全局更新次数计数器, 全局 EP 计数器
GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
# 记录动态的 reward, 看成绩 (用于最终绘图)
GLOBAL_RUNNING_R = []
# 定义两组不同的事件, update 和 rolling
UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
# not update now, 相当于把标志位设置为 False
UPDATE_EVENT.clear()
# start to roll out, 相当于把标志位设置为 True, 并通知所有处于等待阻
# 塞状态的线程恢复运行状态
ROLLING_EVENT.set()
# workers putting data in this queue
QUEUE = queue.Queue()
# 创建 tensorflow 的协调器
COORD = tf.train.Coordinator()
# jy: run 函数中会创建该全局变量
GLOBAL_PPO = None


if __name__ == '__main__':
    mode = "train"
    run(mode)


