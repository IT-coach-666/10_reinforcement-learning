"""
依赖包(python 3.9 环境, gym 包的版本不对可能会导致代码运行报错):
tensorflow==2.11.0
tensorflow-probability==0.19.0
tensorlayer==2.2.5
gym==0.25.2
    pip install gym[classic_control]

    使用 "BipedalWalker-v3" 时需安装:
    pip install gym[box2d]
opencv-python==4.7.0.72
    yum install mesa-libGL -y
"""

import argparse
import multiprocessing
import threading
import time

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import tensorlayer as tl
from tensorlayer.layers import DenseLayer, InputLayer


class ACNet(object):

    def __init__(self, scope, dim_s, dim_a, A_BOUND, lr_a=0.00005,
                 lr_c=0.0001, entropy_beta=0.005):
        """
        lr_a: learning rate for actor
        lr_c: learning rate for critic
        entropy_beta: factor for entropy boosted exploration
        """
        self.scope = scope
        self.save_path = './model'
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.entropy_beta = entropy_beta
        self.A_BOUND = A_BOUND
        # jy: 创建 Actor 的优化器
        self.OPT_A = tf.optimizers.RMSprop(lr_a, name='RMSPropA')   
        # jy: 创建 Critic 的优化器
        self.OPT_C = tf.optimizers.RMSprop(lr_c, name='RMSPropC') 
        # jy: initializer, glorot=xavier
        w_init = tf.keras.initializers.glorot_normal(seed=None)  

        def get_actor(input_shape):  
            """
            policy network
            输入 state, 输出 action 分布 mu 和 sigma
            """
            with tf.name_scope(self.scope):
                ni = tl.layers.Input(input_shape, name='in')
                nn = tl.layers.Dense(n_units=500, act=tf.nn.relu6,
                                     W_init=w_init, name='la')(ni)
                nn = tl.layers.Dense(n_units=300, act=tf.nn.relu6,
                                     W_init=w_init, name='la2')(nn)
                mu = tl.layers.Dense(n_units=self.dim_a, act=tf.nn.tanh,
                                     W_init=w_init, name='mu')(nn)
                sigma = tl.layers.Dense(n_units=self.dim_a,
                                        act=tf.nn.softplus,
                                        W_init=w_init, name='sigma')(nn)
            return tl.models.Model(inputs=ni, outputs=[mu, sigma],
                                   name=scope + '/Actor')
        # jy: 创建 actor 网络;
        self.actor = get_actor([None, self.dim_s])
        # jy: train mode for Dropout, BatchNorm
        self.actor.train()  

        def get_critic(input_shape):  
            """
            we use Value-function here, but not Q-function.
            输入 state, 输出 V 值
            """
            with tf.name_scope(self.scope):
                ni = tl.layers.Input(input_shape, name='in')
                nn = tl.layers.Dense(n_units=500, act=tf.nn.relu6,
                                     W_init=w_init, name='lc')(ni)
                nn = tl.layers.Dense(n_units=300, act=tf.nn.relu6,
                                     W_init=w_init, name='lc2')(nn)
                v = tl.layers.Dense(n_units=1, W_init=w_init,
                                    name='v')(nn)
            return tl.models.Model(inputs=ni, outputs=v,
                                   name=scope + '/Critic')
        # jy: 创建 critic 网络;
        self.critic = get_critic([None, self.dim_s])
        # train mode for Dropout, BatchNorm
        self.critic.train()  

    # jy: convert numpy functions to tf.Operations in the TFgraph,
    #     return tensor
    @tf.function  
    def update_global(self, buffer_s, buffer_a, buffer_v_target): 
        """
        update the global critic (更新全局网络参数)
        updating global Actor-Crtic network with samples

        critic 会最小化 td-error，而 actor 用 td-error 作为权重更新
        """
        global GLOBAL_AC

        # jy: 更新全局网络中的 critic 网络 (也是基于 AC 算法)
        with tf.GradientTape() as tape:
            # jy: 基于局部的 critic 网络, 传入 state, 得到 V(s)
            self.v = self.critic(buffer_s)
            # V(s') * gamma + r
            self.v_target = buffer_v_target
            # td = V(s') * gamma + r - V(s')
            td = tf.subtract(self.v_target, self.v, name='TD_error')
            # jy: 多个 td-error 的均值作为 critic 网络的 loss
            self.c_loss = tf.reduce_mean(tf.square(td))

        # jy: 更新全局网络的参数 (从 buffer 拿出数据, 计算 td-error、求 
        #     a_loss, c_loss、求梯度; 都是对该 worker 的自身网络参数进行计算)
        # jy: Critic 网络的更新 (注意: 求梯度是基于局部 critic 网络求得的, 但
        #     更新的是全局的 critic 网络)
        self.c_grads = tape.gradient(self.c_loss, 
                                     self.critic.trainable_weights)
        # jy: local grads applies to global net (注意: 更新的是全局网络)
        self.OPT_C.apply_gradients(zip(self.c_grads, 
                                       GLOBAL_AC.critic.trainable_weights))
        # Drop the reference to the tape
        #del tape 

        # jy: 更新全局网络中的 actor 网络
        with tf.GradientTape() as tape:
            # jy: 基于局部的 actor 网络, 输入 state, 输出 mu 和 sigma
            self.mu, self.sigma = self.actor(buffer_s)
            # jy: 此处只是为了测试用
            self.test = self.sigma[0] 
            # jy: mu 需要映射到行动空间的范围
            self.mu, self.sigma = self.mu * self.A_BOUND[1], self.sigma + 1e-5
            # jy: 根据 mu 和 sigma 创建正态分布
            normal_dist = tfp.distributions.Normal(self.mu, self.sigma) 

            # jy: 求 action 在分布下的概率 (float32 类型)
            self.a_his = buffer_a                    
            log_prob = normal_dist.log_prob(self.a_his)

            # jy: 带权重更新; td is from the critic part, no gradients for it
            exp_v = log_prob * td 
            # jy: A3C 比 AC 的 loss 要多一项最大熵 (信息熵); 用来衡量不确定性 
            #     (当一件事情的不确定性越大, 熵越大) 
            # jy: actor 求出 mu 和 std 后建立一个服从正态分布的概率分布 
            #     normal_dist, entropy 就是这个概率分布的熵; 把熵和原来加权
            #     loss 加起来, 并最大化他们, 目的是: 我们不但要输出的动作能够
            #     按加权调整概率分布, 同时希望这个分布的熵更大 (即随机性尽量大,
            #     从而使得探索能够最大化, 即 encourage exploration)
            entropy = normal_dist.entropy()  
            
            self.exp_v = self.entropy_beta * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)

        # jy: Actor 网络的更新 (注意: 求梯度是基于局部 actor 网络求得的, 但
        #     更新的是全局的 actor 网络)
        self.a_grads = tape.gradient(self.a_loss,
                                     self.actor.trainable_weights)
        # jy: local grads applies to global net
        self.OPT_A.apply_gradients(zip(self.a_grads,
                                       GLOBAL_AC.actor.trainable_weights))  
        # jy: for test purpose
        return self.test  

    @tf.function
    def pull_global(self):  
        """
        run by a local, pull weights from the global nets
        把全局网络的参数赋值给本地网络
        """
        global GLOBAL_AC
        for l_p, g_p in zip(self.actor.trainable_weights,
                            GLOBAL_AC.actor.trainable_weights):
            l_p.assign(g_p)
        for l_p, g_p in zip(self.critic.trainable_weights,
                            GLOBAL_AC.critic.trainable_weights):
            l_p.assign(g_p)

    def choose_action(self, s, greedy=False):  
        """
        基于 actor 网络选择动作
        输入 s (如: )
        输出 a (如: [ 0.5974586  0.1178271  1.        -1.       ])
        run by a local
        """
        s = s[np.newaxis, :]
        self.mu, self.sigma = self.actor(s)

        with tf.name_scope('wrap_a_out'):
            # jy: sigma 增大了少许
            self.mu, self.sigma = self.mu * self.A_BOUND[1], self.sigma + 1e-5
        
        if greedy:
            return self.mu.numpy()[0]
        
        # jy: 构建正态分布
        normal_dist = tfp.distributions.Normal(self.mu, self.sigma)
        a = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0),
                             *self.A_BOUND)  
        return a.numpy()[0]

    def save_ckpt(self):
        """
        save trained weights
        """
        tl.files.save_npz(self.actor.trainable_weights,
                          name='model_actor.npz')
        tl.files.save_npz(self.critic.trainable_weights,
                          name='model_critic.npz')

    def load_ckpt(self):
        """
        load trained weights
        """
        tl.files.load_and_assign_npz(name='model_actor.npz',
                                     network=self.actor)
        tl.files.load_and_assign_npz(name='model_critic.npz',
                                     network=self.critic)


class Worker(object):
    def __init__(self, name, game, dim_s, dim_a, A_BOUND, 
                 COORD, gamma=0.99):
        """
        gamma: reward discount factor
        """
        # jy: 每个 worker 都要创建一个独立的环境
        self.env = gym.make(game)       
        # jy: worker 的名字
        self.name = name
        # jy: AC 网络 (非全局)
        self.AC = ACNet(name, dim_s, dim_a, A_BOUND) 
        self.COORD = COORD
        self.gamma = gamma

    def work(self):
        """
        多线程中每个线程均会执行该方法
        """
        # jy: 所有 woker 共用的全局变量, GLOBAL_RUNNING_R 用来记录奖励获得的情况
        #     GLOBAL_EP 记录当前第几个更新的 EP (这里用了 global 去声明, 所以这
        #     个变量就可以在这个函数里修改)
        global GLOBAL_RUNNING_R, MAX_GLOBAL_EP, GLOBAL_EP, GLOBAL_AC
        total_step = 1
        # jy: buffer_xx 是局部的, 每个 worker 都各自有一个
        buffer_s, buffer_a, buffer_r = [], [], []
        # jy: 剩下部分和环境进行互动, 保存互动产生的数据 (这都是基于 AC)
        t0 = time.time()
        while not self.COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            # jy: 重置环境
            s = self.env.reset() 
            # jy: 统计 ep 的总 reward
            ep_r = 0  
            while True:
                # jy: visualize Worker_0 during training (每 30 步渲染一次)
                """
                if self.name == 'Worker_0' and total_step % 30 == 0: 
                    self.env.render()
                    print("render === %s" % total_step)
                """
                # jy: double to float
                s = s.astype('float32') 
                # jy: 基于 AC 网络 (非全局) 选择动作
                a = self.AC.choose_action(s)
                # jy: 和环境互动
                s_, r, done, _info = self.env.step(a) 

                # jy: double to float
                s_ = s_.astype('float32')                              
                # jy: 当 reward 为 -100 时, 改为 -2 (set robot falls 
                #     reward to -2 instead of -100)
                if r == -100: 
                    r = -2   

                # jy: 累计 reward 数据
                ep_r += r       
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                # jy: TD(n) 的架构 (update global and assign to local net)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  
                    
                    # jy: 计算最后一步的 V(s')
                    if done:
                        # terminal
                        v_s_ = 0  
                    else:
                        # reduce dim from 2 to 0
                        # jy: 调用 critic 网络, 输入 state, 输出 V 值
                        v_s_ = self.AC.critic(s_[np.newaxis, :])[0, 0]  

                    buffer_v_target = []

                    # reverse buffer r; 计算每个 state 的 V(s')
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        buffer_v_target.append(v_s_)

                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = (
                        np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    )

                    # 更新全局网络的参数
                    # update gradients on global network
                    self.AC.update_global(buffer_s, buffer_a, buffer_v_target.astype('float32'))
                    buffer_s, buffer_a, buffer_r = [], [], []

                    # update local network from global network (把全局网络的参数赋值给本地网络)
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    # record running episode reward
                    if len(GLOBAL_RUNNING_R) == 0: 
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:  
                        # moving average
                        GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)
                    print('{}, Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        self.name, GLOBAL_EP, MAX_GLOBAL_EP, ep_r, time.time()-t0 ))
                    GLOBAL_EP += 1
                    break


def run(mode="train"):
    # jy: 以下三个环境均可使用
    game = 'BipedalWalker-v3'
    #game = "BipedalWalkerHardcore-v3"
    #game = "LunarLanderContinuous-v2"

    env = gym.make(game)

    np.random.seed(2)
    tf.random.set_seed(2)

    # jy: 状态空间
    dim_s = env.observation_space.shape[0]
    # jy: 动作空间
    dim_a = env.action_space.shape[0]

    # jy: 动作范围
    A_BOUND = [env.action_space.low, env.action_space.high]
    # jy: 动作范围形状修改
    A_BOUND[0] = A_BOUND[0].reshape(1, dim_a)
    A_BOUND[1] = A_BOUND[1].reshape(1, dim_a)
    # jy: dim_s == 24
    print("dim_s == %s" % dim_s)
    # jy: dim_a == 4
    print("dim_a == %s" % dim_a)
    # jy: A_BOUND == [array([[-1., -1., -1., -1.]], dtype=float32), array([[1., 1., 1., 1.]], dtype=float32)]
    print("A_BOUND == %s" % A_BOUND)

    # jy: 设置线程数 (等于 cpu 的数量); 数量最好和 cpu 的核一致, 因为每个线程都
    #     是在一个单独的 cpu 进行; 如果超过 CPU 数量, 就失去了多线程的意义)
    num_worker = multiprocessing.cpu_count()
    #num_worker = 2
    print("n_workers:", num_worker)

    test_episode = 5

    global GLOBAL_AC
    # jy: 创建全局 AC 网络 (we only need its params); 这个网络并不直接学习 (相当于班长的角色)
    GLOBAL_AC = ACNet("Global_Net", dim_s, dim_a, A_BOUND)

    if mode == "train":
        # jy: 创建 tensorflow 中协调器
        COORD = tf.train.Coordinator()  
        t0 = time.time()  
        # jy: 指定在 cpu:0 进行以下代码 (如果安装的是 GPU 版本, 也可以使
        #     用 "/gpu:0"); 不同的 GPU 使用 "/gpu:0" 和 "/gpu:1" 区分,
        #     而 CPU 不区分设备号, 统一使用 "/cpu:0"
        with tf.device("/cpu:0"):
            # jy: workers 列表
            workers = []
            # jy: 创建 worker, 并放在 workers 列表中, 方便统一管理
            for i in range(num_worker):
                i_name = 'Worker_%i' % i 
                workers.append(Worker(i_name, game, dim_s, dim_a,
                                      A_BOUND, COORD)) 

        # jy: start TF threading (执行每一个 worker)
        worker_threads = []
        for worker in workers:
            # jy: worker 要执行的工作 (定义每个线程执行一系列动作流程; 一般是一
            #     个函数, 这个函数包含了整个流程)
            job = lambda: worker.work() 
            t = threading.Thread(target=job)   
            # t = threading.Thread(target=worker.work)
            # jy: 开始线程, 并执行
            t.start()         
            # jy: 把线程加入 worker_threads 列表中
            worker_threads.append(t)    
        # jy: 线程由 COORD 统一管理
        COORD.join(worker_threads) 

        # jy: 画图 
        import matplotlib.pyplot as plt
        plt.plot(GLOBAL_RUNNING_R)
        plt.xlabel('episode')
        plt.ylabel('global running reward')
        plt.savefig('a3c.png')
        plt.show()

        GLOBAL_AC.save_ckpt()
    else:
        ac_net = GLOBAL_AC
        ac_net.load_ckpt()
        T0 = time.time()
        for episode in range(test_episode):
            s = env.reset()
            episode_reward = 0
            while True:
                #env.render()
                # double to float
                s = s.astype('float32')  
                a = ac_net.choose_action(s, greedy=True)
                s, r, d, _ = env.step(a)
                episode_reward += r
                if d:
                    break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                  episode + 1, test_episode, episode_reward, time.time() - T0))



tl.logging.set_verbosity(tl.logging.DEBUG)

# jy: update global policy after several episodes
UPDATE_GLOBAL_ITER = 10 
GLOBAL_RUNNING_R = []
# jy: will increase during training, stop training when 
#     it >= self.max_global_ep
GLOBAL_EP = 0
# jy: 最大训练 EP (number of training episodes)
MAX_GLOBAL_EP = 800

# jy: 全局网络 GLOBAL_AC (run 函数中会进行创建)
GLOBAL_AC = None


if __name__ == "__main__":
    mode = "train"
    run(mode)
