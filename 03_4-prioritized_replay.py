"""
Prioritized experience replay is an efficient replay method that replay
important transitions more frequently. Segment tree data structure is used to
speed up indexing.

参考论文: 
Schaul T, Quan J, Antonoglou I, et al. Prioritized experience replay[J]. arXiv
preprint arXiv:1511.05952, 2015.
Dhariwal P, Hesse C, Klimov O, et al. Openai baselines (2017)[J]. URL
https://github.com/openai/baselines.


依赖包(gym 包的版本不对可能会导致代码运行报错):
tensorflow==2.11.0
tensorlayer==2.2.5
gym==0.25.2
opencv-python==4.7.0.72
        pip install opencv-python
    yum install mesa-libGL -y
"""
import argparse
import operator
import os
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl


seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
# reproducible
tf.random.set_seed(seed_num) 
env_id = "CartPole-v1"
#env_id = "PongNoFrameskip-v4"
env = gym.make(env_id)
env.seed(seed_num)
alg_name = "prioritized_replay_%s" % env_id


if env_id == 'CartPole-v1':
    qnet_type = 'MLP'
    # total number of time steps to train on
    number_timesteps = 10000  
    explore_timesteps = 100
    # learning rate
    lr = 5e-3  
    # replay buffer size
    #buffer_size = 1000  
    buffer_size = 100
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
    # learning rate
    lr = 1e-4 
    # replay buffer size
    buffer_size = 10000 
    # how frequency target q net update
    target_q_update_freq = 200  
    # scale observations
    ob_scale = 1.0 / 255 
    clipnorm = 10

in_dim = env.observation_space.shape
out_dim = env.action_space.n
# batch size for sampling from replay buffer
batch_size = 32  
# sample times befor learning
warm_start = buffer_size / 10 


class MLP(tl.models.Model):
    def __init__(self, name):
        super(MLP, self).__init__(name=name)
        self.h1 = tl.layers.Dense(64, tf.nn.tanh, in_channels=in_dim[0])
        self.qvalue = tl.layers.Dense(out_dim, in_channels=64, name='q', W_init=tf.initializers.GlorotUniform())

    def forward(self, ni):
        return self.qvalue(self.h1(ni))


class CNN(tl.models.Model):
    def __init__(self, name):
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

    def forward(self, ni):
        """
        对以上初始化的 conv1-conv3 等模块进行应用, 并计算得到结果;
        传入参数为 state (observation)
        """
        feature = self.flatten(self.conv3(self.conv2(self.conv1(ni))))
        return self.qvalue(self.preq(feature))


class SegmentTree(object):

    def __init__(self, capacity, operation, neutral_element):
        """
        Build a Segment Tree data structure: https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, \
            "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """
        Returns result of applying `self.operation` to a contiguous subsequence of the array.
        
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        
        返回:
        reduced: obj
            result of reducing self.operation over the specified range of array.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx], self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity=capacity, operation=operator.add, neutral_element=0.0)

    def sum(self, start=0, end=None):
        """
        Returns arr[start] + ... + arr[end]
        """
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """
        Find the highest index `i` in the array such that:
        sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        
        if array values are probabilities, this function allows to sample
        indexes according to the discrete probability efficiently.
        
        perfixsum: float, upperbound on the sum of array prefix
        Returns: idx(int), highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):

    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(capacity=capacity, operation=min,
                                             neutral_element=float('inf'))

    def min(self, start=0, end=None):
        """
        Returns min(arr[start], ...,  arr[end])
        """
        return super(MinSegmentTree, self).reduce(start, end)


class ReplayBuffer(object):

    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, *args):
        if self._next_idx >= len(self._storage):
            self._storage.append(args)
        else:
            self._storage[self._next_idx] = args
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
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
        indexes = range(len(self._storage))
        idxes = [random.choice(indexes) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, size, alpha, beta):
        """
        Create Prioritized Replay buffer.
        
        size: int, Max number of transitions to store in the buffer. When the buffer
              overflows the old memories are dropped.
        alpha: float, how much prioritization is used
               (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self.beta = beta

    def add(self, *args):
        """
        See ReplayBuffer.store_effect
        """
        idx = self._next_idx
        super().add(*args)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        """
        抽样取 batch_size 个 index
        """
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size):
        """
        Sample a batch of experiences
        """
        # jy: 从 buffer 中获取 batch_size 个 index
        idxes = self._sample_proportional(batch_size)
        # jy: self._it_sum 为初始化的 SumSegmentTree 类, 此处调用其 sum 方法;
        it_sum = self._it_sum.sum()
        # jy: self._it_min 为已初始化的 MinSegmentTree 类, 此处调用其 min 方法;
        p_min = self._it_min.min() / it_sum
        # jy: 
        max_weight = (p_min * len(self._storage))**(-self.beta)

        p_samples = np.asarray([self._it_sum[idx] for idx in idxes]) / it_sum
        weights = (p_samples * len(self._storage))**(-self.beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample + (weights.astype('float32'), idxes)

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


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


class DQN(object):
    def __init__(self, mode="train"):
        self.mode = mode
        model = MLP if qnet_type == 'MLP' else CNN
        self.qnet = model('q')
        if mode == "train":
            # jy: self.qnet 网络设置为训练模式; 
            self.qnet.train()
            # jy: self.targetqnet 网络只有在训练的时候有, 且设置为不非训练模式(参数不更新)
            self.targetqnet = model('targetq')
            self.targetqnet.infer()
            # jy: 初始化时, 同步两个网络的参数;
            sync(self.qnet, self.targetqnet)
        else:
            # jy: 将 qnet 设置为推理状态;
            self.qnet.infer()
            self.load()

        self.niter = 0
        self.optimizer = tf.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)

    def get_action(self, obv):
        # epsilon-greedy schedule, final exploit prob is 0.99
        epsilon = lambda i_iter: 1 - 0.99 * min(1, i_iter / explore_timesteps)
        eps = epsilon(self.niter)
        if self.mode == "train" and random.random() < eps:
            return int(random.random() * out_dim)
        else:
            obv = np.expand_dims(obv, 0).astype('float32') * ob_scale
            return self._qvalues_func(obv).numpy().argmax(1)[0]

    @tf.function
    def _qvalues_func(self, obv):
        return self.qnet(obv)

    def train(self, b_o, b_a, b_r, b_o_, b_d, weights=None):
        """
        训练网络: 计算 td-error, 并求 loss, 随后基于 loss 求梯度并更新网络参数
        返回 td-error
        """
        if weights is None:
            weights = np.ones_like(b_r)
        # jy: 计算 td-error
        td_errors = self._train_func(b_o, b_a, b_r, b_o_, b_d, weights)

        self.niter += 1
        # jy: 定期同步 self.qnet 和 self.targetqnet 的网络参数;
        if self.niter % target_q_update_freq == 0:
            sync(self.qnet, self.targetqnet)
            self.save()
        return td_errors.numpy()

    @tf.function
    def _train_func(self, b_o, b_a, b_r, b_o_, b_d, weights):
        """
        计算 td-error, 并求 loss, 随后基于 loss 求梯度, 更新网络参数;
        """
        with tf.GradientTape() as tape:
            td_errors = self._tderror_func(b_o, b_a, b_r, b_o_, b_d)
            loss = tf.reduce_mean(huber_loss(td_errors) * weights)

        grad = tape.gradient(loss, self.qnet.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.qnet.trainable_weights))

        return td_errors

    @tf.function
    def _tderror_func(self, b_o, b_a, b_r, b_o_, b_d):
        """
        计算 td-error
        """
        b_q_ = (1 - b_d) * tf.reduce_max(self.targetqnet(b_o_), 1)
        b_q = tf.reduce_sum(self.qnet(b_o) * tf.one_hot(b_a, out_dim), 1)
        # reward discount
        reward_gamma = 0.99
        return b_q - (b_r + reward_gamma * b_q_)

    def save(self, path=None):
        if path is None:
            path = os.path.join('model', alg_name)
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz(self.qnet.trainable_weights, name=os.path.join(path, 'q_net.npz'))

    def load(self, path=None):
        if path is None:
            path = os.path.join('model', alg_name)
        tl.files.load_and_assign_npz(name=os.path.join(path, 'q_net.npz'), network=self.qnet)

def run(mode):
    dqn = DQN(mode)
    t0 = time.time()
    # alpha in PER
    prioritized_replay_alpha = 0.6
    # initial beta in PER
    prioritized_replay_beta0 = 0.4

    if mode == "train":
        # jy: 初始化 PrioritizedReplayBuffer 类;
        buffer = PrioritizedReplayBuffer(buffer_size, prioritized_replay_alpha, prioritized_replay_beta0)
        nepisode = 0
        all_episode_reward = []
        for i in range(1, number_timesteps + 1):
            o = env.reset()
            episode_reward = 0
            while True:
                # jy: beta 会越来越大;
                buffer.beta += (1 - prioritized_replay_beta0) / number_timesteps
                # jy: 调用 DQN 类的 get_action 方法, 获取 action;
                a = dqn.get_action(o)

                # execute action and feed to replay buffer
                # note that `_` tail in var name means next
                o_, r, done, info = env.step(a)
                # jy: 将数据加入到 buffer 缓存;
                buffer.add(o, a, r, o_, done)
                episode_reward += r

                if i >= warm_start:
                    # jy: 从 buffer 缓存中取 batch_size 个数据
                    #import pdb; pdb.set_trace()
                    # jy: transitions 为长度为 6 的列表, idxs 为长度为 batch_size 的记录 index 的列表
                    *transitions, idxs = buffer.sample(batch_size)
                    # jy: 训练网络, 得到的结果为 td-error
                    priorities = dqn.train(*transitions)
                    # jy: 对 priorities 取绝对值并进行 clip
                    priorities = np.clip(np.abs(priorities), 1e-6, None)
                    buffer.update_priorities(idxs, priorities)

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

        dqn.save()
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', alg_name))
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


if __name__ == '__main__':
    #mode = "test"
    mode = "train"
    run(mode)


