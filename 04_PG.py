"""
Here is an example on discrete action space game CartPole-v1.

To apply it on continuous action space, you need to change the last softmax layer 
and the get_action function.


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
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorlayer as tl


class PolicyGradient(tl.models.Model):
    """
    PG class
    """
    def __init__(self, state_dim, action_num, name="pg", learning_rate=0.02, gamma=0.99):
        """
        state_dim: 环境特征数量
        action_num: 动作
        learning_rate: 学习率
        gamma: 折扣
        """
        super(PolicyGradient, self).__init__(name=name)
        self.gamma = gamma

        # 用于保存每个 ep 的数据
        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []

        # jy: state_dim 为环境特征数量
        self.input_layer = tl.layers.Input([None, state_dim], tf.float32)
        self.layer = tl.layers.Dense(
            n_units=30, act=tf.nn.tanh, 
            W_init=tf.random_normal_initializer(mean=0, stddev=0.3),
            b_init=tf.constant_initializer(0.1),
            name='fc1'
        )(self.input_layer)

        self.all_act = tl.layers.Dense(
            n_units=action_num, act=None, 
            W_init=tf.random_normal_initializer(mean=0, stddev=0.3),
            b_init=tf.constant_initializer(0.1),
            name='all_act'
        )(self.layer)

        self.model = tl.models.Model(inputs=self.input_layer, outputs=self.all_act)
        self.model.train()
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def get_action(self, s, greedy=False):
        """
        choose action with probabilities. (用神经网络输出的策略选择动作)
        
        s: state
        greedy: choose action greedy or not
        返回: act
        """
        _logits = self.model(np.array([s], np.float32))
        _probs = tf.nn.softmax(_logits).numpy()
        if greedy:
            return np.argmax(_probs.ravel())
        # 根据策略 PI 选择动作
        return tl.rein.choice_action_by_probs(_probs.ravel())

    def store_transition(self, s, a, r):
        """
        store data in memory buffer (保存数据到 buffer 中)
        s: state
        a: act
        r: reward
        """
        self.state_buffer.append(np.array([s], np.float32))
        self.action_buffer.append(a)
        self.reward_buffer.append(r)

    def learn(self):
        """
        update policy parameters via stochastic gradient ascent (this process will empty episode data)
        通过带权重的策略梯度方法更新神经网络, 返回 G 值;

        一次游戏到结束状态时, 会调用该方法
        """
        # jy: _discount_and_norm_rewards 中存储当前 ep 中每个状态的 G 值
        #     计算所有经过的 state 的 G 值(会进行归一化处理, 使得有正有负)
        #import pdb; pdb.set_trace()
        discounted_reward_buffer_norm = self._discount_and_norm_rewards()

        with tf.GradientTape() as tape:
            # 把 s 放入神经网络, 计算 _logits（通过网络, 求出预测值的分布）
            _logits = self.model(np.vstack(self.state_buffer))
            # 求 loss: _logits 和真正的动作的差距（和真实值 action 进行比较, 求得 neg_log_prob）
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=_logits, labels=np.array(self.action_buffer)
            )
            # reward guided loss; 在原来的差距乘以 G 值(即以 G 值作为更新)
            # neg_log_prob 乘以 G 值, 求得 loss
            loss = tf.reduce_mean(neg_log_prob * discounted_reward_buffer_norm)

        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

        # empty episode data
        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []
        return discounted_reward_buffer_norm

    def _discount_and_norm_rewards(self):
        """
        通过回溯计算 G 值(即 discounted_reward_buffer, 会进行归一化处理)并返回
        """
        # discount episode rewards
        # 先创建一个全零数组, 大小和 self.reward_buffer 相同 (self.reward_buffer 记录每个状态的 reward)
        discounted_reward_buffer = np.zeros_like(self.reward_buffer)
        running_add = 0
        # 从 self.reward_buffer 的最后往前（反向循环的方式实现）逐个计算 G
        # 每次循环都把上一个 G 值乘以折扣（gamma）然后加上这个 state 获得的 reward, 并把每一轮计算的这个值记录
        # 在 discounted_reward_buffer, 便于后续归一化处理;
        for t in reversed(range(0, len(self.reward_buffer))):
            running_add = running_add * self.gamma + self.reward_buffer[t]
            discounted_reward_buffer[t] = running_add

        # normalize episode rewards 
        # 可以用 G 值直接进行学习, 但一般对数据进行归一化处理(减去平均数, 除以方差) 使得 G 值有正有负, 这
        # 样比较容易学习(训练效果更好)
        discounted_reward_buffer -= np.mean(discounted_reward_buffer)
        discounted_reward_buffer /= np.std(discounted_reward_buffer)
        return discounted_reward_buffer

    def save(self):
        """
        save trained weights
        """
        path = os.path.join('model', ALG_NAME)
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz(self.model.trainable_weights, name=os.path.join(path, 'pg_policy.npz'))

    def load(self):
        """
        load trained weights
        """
        path = os.path.join('model', ALG_NAME)
        #tl.files.load_hdf5_to_weights(os.path.join(path, 'pg_policy.hdf5'), self.model)
        tl.files.load_and_assign_npz(name=os.path.join(path, 'pg_policy.npz'), network=self.model)


def run(mode, random_seed=1):
    """
    random_seed: can be either an int number or None
    """
    # jy: 加 .unwrapped 会出现报错;
    #env = gym.make(ENV_ID).unwrapped
    env = gym.make(ENV_ID)

    TRAIN_EPISODES = 10
    TEST_EPISODES = 10
    MAX_STEPS = 500

    # reproducible
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    env.seed(random_seed)

    # jy: 初始化 PolicyGradient 类;
    agent = PolicyGradient(action_num=env.action_space.n,
                           state_dim=env.observation_space.shape[0])

    t0 = time.time()

    if mode == "train":
        # jy: 统计训练过程中所有 episode 的 reward;
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            state = env.reset()

            # train-version-1 ================================================
            """
            episode_reward = 0

            # in one episode
            for step in range(MAX_STEPS):
                #env.render()

                # jy: choose action with probabilities. 
                #     根据 state, 选择 action (默认不使用贪婪算法, 而是根据 pi 随机动作, 以保证一定的探索性)
                action = agent.get_action(state)
                # jy: 把 action 代入环境, 获得 next_state, reward, done, info
                next_state, reward, done, info = env.step(action)
                # jy: store data in memory buffer (保存数据)
                agent.store_transition(state, action, reward)
                # jy: 开始新一步
                state = next_state
                # jy: 汇总当前 episode 的 reward
                episode_reward += reward
                if done:
                    break

            # jy: update policy parameters via stochastic gradient ascent (this process will empty episode data)
            #     计算 G 值, 并开始学习策略
            agent.learn()
            print('Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
                  episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0))

            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            """
            # train-version-2: 采用了 MC 的方式 ============================================
            #"""
            episode_time = time.time()
            while True:
                #env.render()

                # jy: agent 即 PolicyGradient 类, 此处调用 get_action 方法; 传入的 state 为维
                #     度为 (4,) 的 numpy 数组; 得到的 action 如: 0
                #     注意：这里没有用贪婪算法，而是根据 pi 随机动作，以保证一定的探索性。
                action = agent.get_action(state)

                next_state, reward, done, info = env.step(action)

                # 将数据临时缓存起来;
                agent.store_transition(state, action, reward)

                # PG 用的是 MC, 走到最终状态时进行学习、更新网络;
                if done:
                    ep_sum_reward = sum(agent.reward_buffer)

                    if 'running_reward' not in globals():
                        running_reward = ep_sum_reward
                    else:
                        running_reward = running_reward * 0.99 + ep_sum_reward * 0.01

                    print("Episode [%d/%d] \tsum reward: %d  \trunning reward: %f \ttook: %.5fs " %
                          (episode, TRAIN_EPISODES, ep_sum_reward, running_reward, time.time() - episode_time))
                    all_episode_reward.append(running_reward)

                    # 学习、更新网络;
                    agent.learn()
                    # 画图
                    plt.ion()
                    plt.cla()
                    plt.title('PG')
                    plt.plot(all_episode_reward, )
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                    plt.pause(0.1)

                    break

                # 开始新一步
                state = next_state
            #"""

        agent.save()
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', ALG_NAME))
    else:
        # test
        agent.load()
        for episode in range(TEST_EPISODES):
            state = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                #env.render()
                state, reward, done, info = env.step(agent.get_action(state, True))
                episode_reward += reward
                if done:
                    break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
                  episode + 1, TEST_EPISODES, episode_reward, time.time() - t0))
     

if __name__ == '__main__':
    # environment id
    ENV_ID = 'CartPole-v1'

    ALG_NAME = 'PG_%s' % ENV_ID
    mode = "train"
    #mode = "test"
    run(mode)


