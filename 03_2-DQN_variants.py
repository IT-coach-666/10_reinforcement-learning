"""
依赖包(python 3.9 环境, gym 包的版本不对可能会导致代码运行报错):
tensorflow==2.11.0
tensorlayer==2.2.5
gym==0.25.2
    pip install gym[classic_control]

    使用 "PongNoFrameskip-v4" 时需安装:
    pip install "gym[atari, accept-rom-license]"
opencv-python==4.7.0.72
    pip install opencv-python
    yum install mesa-libGL -y
"""
import argparse
import os
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl



# ##############################  Network  ####################################
class MLP(tl.models.Model):
    def __init__(self, name, net_type="double"):
        assert net_type in ["double", "dueling"]
        self.net_type = net_type
        super(MLP, self).__init__(name=name)
        self.h1 = tl.layers.Dense(64, tf.nn.tanh, in_channels=in_dim[0])
        self.qvalue = tl.layers.Dense(out_dim, in_channels=64, name='q', 
                                      W_init=tf.initializers.GlorotUniform())
        # jy: self.svalue 只有在 dueling DQN 下才会使用到;
        self.svalue = tl.layers.Dense(1, in_channels=64, name='s',
                                      W_init=tf.initializers.GlorotUniform())
        self.noise_scale = 0


    def forward(self, ni):
        feature = self.h1(ni)
        # apply noise to all linear layer
        if self.noise_scale != 0:
            noises = []
            for layer in [self.qvalue, self.svalue]:
                for var in layer.trainable_weights:
                    noise = tf.random.normal(tf.shape(var), 0, self.noise_scale)
                    noises.append(noise)
                    var.assign_add(noise)

        qvalue = self.qvalue(feature)
        svalue = self.svalue(feature)

        if self.noise_scale != 0:
            idx = 0
            for layer in [self.qvalue, self.svalue]:
                for var in layer.trainable_weights:
                    var.assign_sub(noises[idx])
                    idx += 1

        # duelingDQN network
        if self.net_type == "dueling":
            return svalue + qvalue - tf.reduce_mean(qvalue, 1, keepdims=True)
        # doubleDQN network
        else:
            return qvalue


class CNN(tl.models.Model):
    def __init__(self, name, net_type="double"):
        assert net_type in ["double", "dueling"]
        self.net_type = net_type
        super(CNN, self).__init__(name=name)
        h, w, in_channels = in_dim
        dense_in_channels = 64 * ((h - 28) // 8) * ((w - 28) // 8)
        self.conv1 = tl.layers.Conv2d(
            32, (8, 8), (4, 4), tf.nn.relu, 'VALID', in_channels=in_channels, name='conv2d_1',
            W_init=tf.initializers.GlorotUniform()
        )
        self.conv2 = tl.layers.Conv2d(
            64, (4, 4), (2, 2), tf.nn.relu, 'VALID', in_channels=32, name='conv2d_2',
            W_init=tf.initializers.GlorotUniform()
        )
        self.conv3 = tl.layers.Conv2d(
            64, (3, 3), (1, 1), tf.nn.relu, 'VALID', in_channels=64, name='conv2d_3',
            W_init=tf.initializers.GlorotUniform()
        )
        self.flatten = tl.layers.Flatten(name='flatten')
        self.preq = tl.layers.Dense(
            256, tf.nn.relu, in_channels=dense_in_channels, name='pre_q', W_init=tf.initializers.GlorotUniform()
        )
        self.qvalue = tl.layers.Dense(out_dim, in_channels=256, name='q', W_init=tf.initializers.GlorotUniform())
        self.pres = tl.layers.Dense(
            256, tf.nn.relu, in_channels=dense_in_channels, name='pre_s', W_init=tf.initializers.GlorotUniform()
        )
        # jy: self.svalue 只有在 dueling DQN 下才会使用到;
        self.svalue = tl.layers.Dense(1, in_channels=256, name='state', W_init=tf.initializers.GlorotUniform())
        self.noise_scale = 0

    def forward(self, ni):
        feature = self.flatten(self.conv3(self.conv2(self.conv1(ni))))

        # apply noise to all linear layer
        if self.noise_scale != 0:
            noises = []
            for layer in [self.preq, self.qvalue, self.pres, self.svalue]:
                for var in layer.trainable_weights:
                    noise = tf.random.normal(tf.shape(var), 0, self.noise_scale)
                    noises.append(noise)
                    var.assign_add(noise)

        qvalue = self.qvalue(self.preq(feature))
        svalue = self.svalue(self.pres(feature))

        if self.noise_scale != 0:
            idx = 0
            for layer in [self.preq, self.qvalue, self.pres, self.svalue]:
                for var in layer.trainable_weights:
                    var.assign_sub(noises[idx])
                    idx += 1

        if self.net_type == "dueling":
            # dueling network
            return svalue + qvalue - tf.reduce_mean(qvalue, 1, keepdims=True)
        else:
            return qvalue


# ##############################  Replay  ####################################
class ReplayBuffer(object):
    def __init__(self, size):
        # 保存的容器
        self._storage = []
        # 容器最大的 size
        self._maxsize = size
        # 指针，表示当前新增位置
        self._next_idx = 0

    def __len__(self):
        """
        len(ReplayBuffer_obj): 查询容器的大小
        """
        return len(self._storage)

    def add(self, *args):
        """
        把信息放入 buffer
        """
        # 如果当前指针大于容器目前大小, 则扩展容器, append 数据
        if self._next_idx >= len(self._storage):
            self._storage.append(args)
        # 如果不是, 直接写进去就可以
        else:
            self._storage[self._next_idx] = args
        # 一个循环指针
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        """
        基于 index 获取容器中的数据, 并转换为 np 格式;
        """
        b_o, b_a, b_r, b_o_, b_d = [], [], [], [], []
        for i in idxes:
            o, a, r, o_, d = self._storage[i]
            b_o.append(o)
            b_a.append(a)
            b_r.append(r)
            b_o_.append(o_)
            b_d.append(d)
        return (
            np.stack(b_o).astype('float32') * ob_scale,
            np.stack(b_a).astype('int32'),
            np.stack(b_r).astype('float32'),
            np.stack(b_o_).astype('float32') * ob_scale,
            np.stack(b_d).astype('float32'),
        )

    def sample(self, batch_size):
        """
        随机抽取容器中的数据
        """
        indexes = range(len(self._storage))
        idxes = [random.choice(indexes) for _ in range(batch_size)]
        return self._encode_sample(idxes)


# #############################  Functions  ###################################
def huber_loss(x):
    """
    Loss function for value
    """
    return tf.where(tf.abs(x) < 1, tf.square(x) * 0.5, tf.abs(x) - 0.5)


def sync(net, net_tar):
    """
    Copy q network to target q network
    """
    for var, var_tar in zip(net.trainable_weights, net_tar.trainable_weights):
        var_tar.assign(var)


def log_softmax(x, dim):
    temp = x - np.max(x, dim, keepdims=True)
    return temp - np.log(np.exp(temp).sum(dim, keepdims=True))


def softmax(x, dim):
    temp = np.exp(x - np.max(x, dim, keepdims=True))
    return temp / temp.sum(dim, keepdims=True)


# ###############################  DQN  #####################################
class DQN(object):
    def __init__(self, net_type="double", noise_scale=1e-2, mode="train", reward_gamma=0.99):
        self.noise_scale = noise_scale
        self.niter = 0
        assert net_type in ["double", "dueling"]
        self.net_type = net_type
        self.alg_name = "%s_DQN" % net_type
        # jy: 确定网络类型;
        model = MLP if qnet_type == 'MLP' else CNN
        # jy: 初始化网络;
        self.qnet = model('q', self.net_type)
        self.mode = mode
        # reward discount
        self.reward_gamma = reward_gamma
        # how frequency param noise net update
        self.noise_update_freq = 50

        if self.mode == "train":
            self.qnet.train()
            self.targetqnet = model('targetq', self.net_type)
            self.targetqnet.infer()
            # jy: 调用函数: Copy q network to target q network
            sync(self.qnet, self.targetqnet)
        else:
            self.qnet.infer()
            self.load(path=None)

        if clipnorm is not None:
            self.optimizer = tf.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
        else:
            self.optimizer = tf.optimizers.Adam(learning_rate=lr)

    def get_action(self, obv):
        """
        基于 observation 获取 action;
        """
        eps = epsilon(self.niter)
        if self.mode == "train":
            if random.random() < eps:
                return int(random.random() * out_dim)
            obv = np.expand_dims(obv, 0).astype('float32') * ob_scale
            if self.niter < explore_timesteps:
                self.qnet.noise_scale = self.noise_scale
                q_ptb = self._qvalues_func(obv).numpy()
                self.qnet.noise_scale = 0
                if i % self.noise_update_freq == 0:
                    q = self._qvalues_func(obv).numpy()
                    kl_ptb = (log_softmax(q, 1) - log_softmax(q_ptb, 1))
                    kl_ptb = np.sum(kl_ptb * softmax(q, 1), 1).mean()
                    kl_explore = -np.log(1 - eps + eps / out_dim)
                    if kl_ptb < kl_explore:
                        self.noise_scale *= 1.01
                    else:
                        self.noise_scale /= 1.01
                return q_ptb.argmax(1)[0]
            else:
                return self._qvalues_func(obv).numpy().argmax(1)[0]
        else:
            obv = np.expand_dims(obv, 0).astype('float32') * ob_scale
            return self._qvalues_func(obv).numpy().argmax(1)[0]

    @tf.function
    def _qvalues_func(self, obv):
        return self.qnet(obv)

    def train(self, b_o, b_a, b_r, b_o_, b_d):
        """
        训练网络;
        """
        # jy: 训练网络(返回 td_errors); 
        self._train_func(b_o, b_a, b_r, b_o_, b_d)

        self.niter += 1
        # jy: 定期保存模型, 并同步 qnet 与 target qnet 参数值;
        if self.niter % target_q_update_freq == 0:
            # jy: 调用函数: Copy q network to target q network
            sync(self.qnet, self.targetqnet)
            self.save(path=None)

    @tf.function
    def _train_func(self, b_o, b_a, b_r, b_o_, b_d):
        with tf.GradientTape() as tape:
            # jy: 调用函数, 计算 td-error
            td_errors = self._tderror_func(b_o, b_a, b_r, b_o_, b_d)
            loss = tf.reduce_mean(huber_loss(td_errors))

        grad = tape.gradient(loss, self.qnet.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.qnet.trainable_weights))

        return td_errors

    @tf.function
    def _tderror_func(self, b_o, b_a, b_r, b_o_, b_d):
        """
        计算 td-error
        """
        # jy: doubleDQN
        if self.net_type == "double":
            b_a_ = tf.one_hot(tf.argmax(self.qnet(b_o_), 1), out_dim)
            b_q_ = (1 - b_d) * tf.reduce_sum(self.targetqnet(b_o_) * b_a_, 1)
        # jy: duelingDQN
        else:
            b_q_ = (1 - b_d) * tf.reduce_max(self.targetqnet(b_o_), 1)

        b_q = tf.reduce_sum(self.qnet(b_o) * tf.one_hot(b_a, out_dim), 1)

        return b_q - (b_r + self.reward_gamma * b_q_)

    def save(self, path):
        if path is None:
            path = os.path.join('model', '_'.join([self.alg_name, env_id]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz(self.qnet.trainable_weights, name=os.path.join(path, 'q_net.npz'))

    def load(self, path):
        """
        加载模型
        """
        if path is None:
            path = os.path.join('model', '_'.join([self.alg_name, env_id]))
        tl.files.load_and_assign_npz(name=os.path.join(path, 'q_net.npz'), network=self.qnet)


def run(mode, net_type):
    """
    mode: "train" (训练) 或 "test" (测试)
    """
    # batch size for sampling from replay buffer
    batch_size = 32
    # sample times befor learning
    warm_start = buffer_size / 10
    # jy: 初始化 DQN 类;
    dqn = DQN(net_type, mode=mode)
    t0 = time.time()
    if mode == "train":
        buffer = ReplayBuffer(buffer_size)
        nepisode = 0
        # jy: 用于最终绘图;
        all_episode_reward = []
        for i in range(1, number_timesteps + 1):
            # jy: 初始化 observation, 即 state
            o = env.reset()
            # jy: 用于打印输出;
            episode_reward = 0
            while True:
                # jy: 基于 o 获取 action;
                a = dqn.get_action(o)

                # execute action and feed to replay buffer; 
                # note that `_` tail in var name means next
                o_, r, done, info = env.step(a)
                buffer.add(o, a, r, o_, done)
                episode_reward += r

                # jy: 在特定条件下开始训练网络;
                if i >= warm_start:
                    # jy: 从容器中随机抽取数据;
                    transitions = buffer.sample(batch_size)
                    # jy: 基于随机抽取的数据进行训练;
                    dqn.train(*transitions)

                if done:
                    break
                else:
                    o = o_

            if nepisode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            # episode num starts from 1 in print
            nepisode += 1
            print('Training  | Episode: {}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                  nepisode, episode_reward, time.time() - t0))

        dqn.save(path=None)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([dqn.alg_name, env_id])))
    else:
        nepisode = 0
        for i in range(1, number_timesteps + 1):
            o = env.reset()
            episode_reward = 0
            while True:
                #env.render()
                a = dqn.get_action(o)
                o_, r, done, info = env.step(a)
                episode_reward += r
                if done:
                    break
                else:
                    o = o_
            nepisode += 1
            print('Testing  | Episode: {}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                  nepisode, episode_reward, time.time() - t0))



seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
# reproducible
tf.random.set_seed(seed_num)

# jy: env_id 有两种可选: "CartPole-v1" 和 "PongNoFrameskip-v4" (训练效果不好)
#env_id = "CartPole-v1"
env_id = "PongNoFrameskip-v4"
env = gym.make(env_id)
env.seed(seed_num)

in_dim = env.observation_space.shape
out_dim = env.action_space.n


# ####################  hyper parameters  ####################
if env_id == 'CartPole-v1':
    qnet_type = 'MLP'
    # total number of time steps to train on
    number_timesteps = 200 #10000 #200
    explore_timesteps = 100
    # epsilon-greedy schedule, final exploit prob is 0.99
    epsilon = lambda i_iter: 1 - 0.99 * min(1, i_iter / explore_timesteps)
    # learning rate
    lr = 5e-3
    # replay buffer size
    buffer_size = 1000
    # how frequency target q net update
    target_q_update_freq = 50
    # scale observations
    ob_scale = 1.0
    clipnorm = None
else:
    # reward will increase obviously after 1e5 time steps
    qnet_type = 'CNN'
    # total number of time steps to train on
    number_timesteps = int(1e6)
    explore_timesteps = 1e5
    # epsilon-greedy schedule, final exploit prob is 0.99
    epsilon = lambda i_iter: 1 - 0.99 * min(1, i_iter / explore_timesteps)
    # learning rate
    lr = 1e-4
    # replay buffer size
    buffer_size = 10000
    # how frequency target q net update
    target_q_update_freq = 200
    # scale observations
    ob_scale = 1.0 / 255
    clipnorm = 10


if __name__ == '__main__':
    mode = "train"
    #mode = "test"
    #net_type = "dueling"
    net_type = "double"
    run(mode, net_type)

