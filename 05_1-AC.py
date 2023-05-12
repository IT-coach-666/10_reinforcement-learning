
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
import time
import matplotlib.pyplot as plt
import os

import gym
import numpy as np
import tensorflow as tf

import tensorlayer as tl

tl.logging.set_verbosity(tl.logging.DEBUG)


class Actor(object):

    def __init__(self, state_dim, action_num, lr=0.001):
        """
        初始化网络结构
        """
        # jy: 定义输入层;
        input_layer = tl.layers.Input([None, state_dim], name='state')
        # jy: 定义输出层(含隐层)
        layer = tl.layers.Dense(
            n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), 
            name='hidden'
        )(input_layer)
        # jy: hidden layer-2: 可有可无
        """
        layer = tl.layers.Dense(n_units=10, act=tf.nn.relu6, 
                                W_init=tf.random_uniform_initializer(0, 0.01), name='hidden2')(layer)
        """
        # jy: "CartPole-v1" 的 action_num 为 2; 
        layer = tl.layers.Dense(n_units=action_num, name='actions')(layer)
        # jy: 基于输入输出层构造模型;
        self.model = tl.models.Model(inputs=input_layer, outputs=layer, name="Actor")

        self.model.train()
        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, state, action, td_error):
        """
        Actor 学习: 基于 td_error 计算策略分布
        """
        with tf.GradientTape() as tape:
            # jy: 基于 state 调用模型, 得到 action 分布的 logits 值(维度同 action_num)
            _logits = self.model(np.array([state]))
            # cross-entropy loss weighted by td-error (advantage), the cross-entropy mearsures the
            # difference of two probability distributions: the predicted logits and sampled action
            # distribution, then weighted by the td-error: small difference of real and predict 
            # actions for large td-error (advantage); and vice versa.
            # 带权重更新; Actor 的学习本质上是 PG 的更新（即加权的学习）; cross_entropy_reward_loss 函数就
            # 是在 PG 中的带参数交叉熵损失函数; tensorlayer 实现了 cross_entropy_reward_loss 函数的功能
            """
            def cross_entropy_reward_loss(logits, actions, rewards, name=None):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=actions, logits=logits, name=name)
                return tf.reduce_sum(tf.multiply(cross_entropy, rewards))
            """
            _exp_v = tl.rein.cross_entropy_reward_loss(logits=_logits, actions=[action],
                                                       rewards=td_error[0])
        grad = tape.gradient(_exp_v, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        return _exp_v

    def get_action(self, state, greedy=False):
        """
        按照分布随机选择动作
        """
        #import pdb; pdb.set_trace()
        # jy: 基于 state 调用模型, 得到 action 分布的 logits 值(维度同 action_num)
        _logits = self.model(np.array([state]))
        # jy: 将 _logits 转换为概率分布;
        _probs = tf.nn.softmax(_logits).numpy()
        # 贪婪算法
        if greedy:
            return np.argmax(_probs.ravel())
        # sample according to probability distribution (基于概率分布选择动作)
        return tl.rein.choice_action_by_probs(_probs.ravel())  

    def save(self):
        """
        save trained weights
        """
        path = os.path.join('model', ALG_NAME)
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz(self.model.trainable_weights, name=os.path.join(path, 'model_actor.npz'))

    def load(self):
        """
        load trained weights
        """
        path = os.path.join('model', ALG_NAME)
        tl.files.load_and_assign_npz(name=os.path.join(path, 'model_actor.npz'), network=self.model)


class Critic(object):

    def __init__(self, state_dim, lr=0.01):
        # jy: 输入层;
        input_layer = tl.layers.Input([1, state_dim], name='state')
        # jy: 输出层
        layer = tl.layers.Dense(
            n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01),
            name='hidden'
        )(input_layer)
        # jy: hidden layer-2: 可有可无
        """
        layer = tl.layers.Dense(
            n_units=5, act=tf.nn.relu, W_init=tf.random_uniform_initializer(0, 0.01),
            name='hidden2'
        )(layer)
        """
        layer = tl.layers.Dense(n_units=1, act=None, name='value')(layer)
        # jy: 基于输入输出层构造模型;
        self.model = tl.models.Model(inputs=input_layer, outputs=layer, name="Critic")
        # jy: 设置模型为可训练状态;
        self.model.train()
        # jy: 定义模型优化器(设置对应的学习率)
        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, state, reward, state_, done):
        """
        Critic 学习: 计算并缩小 td-error (基于梯度更新模型参数)
        返回 td-error
        """
        # jy: 计算新状态(state_)下的 V 值;
        v_ = self.model(np.array([state_]))
        with tf.GradientTape() as tape:
            # jy: 计算原先状态下的 V 值;
            v = self.model(np.array([state]))
            # td-error 的计算
            # TD_error = r + d * lambda * V(newS) - V(S)
            # reward discount in TD error
            lam = 0.9
            # jy: done 为 True 时, td-error 直接使用 reward - v 计算;
            d = 0 if done else 1
            td_error = reward + d * lam * v_ - v
            loss = tf.square(td_error)
        # jy: 基于 loss 计算模型参数的梯度;
        grad = tape.gradient(loss, self.model.trainable_weights)
        # jy: 基于梯度更新模型参数;
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        return td_error

    def save(self):
        """
        save trained weights
        """
        path = os.path.join('model', ALG_NAME)
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz(self.model.trainable_weights, name=os.path.join(path, 'model_critic.npz'))

    def load(self): 
        """
        load trained weights
        """
        path = os.path.join('model', ALG_NAME)
        tl.files.load_and_assign_npz(name=os.path.join(path, 'model_critic.npz'), network=self.model)



def run(mode):
    # jy: 加上 .unwrapped 后续运行会报错;
    #env = gym.make(ENV_ID).unwrapped
    env = gym.make(ENV_ID)

    LR_A = 0.001          # learning rate for actor
    LR_C = 0.01           # learning rate for critic
    
    # number of overall episodes for training
    TRAIN_EPISODES = 100  
    # number of overall episodes for testing
    TEST_EPISODES = 10    
    # maximum time step in one episode
    MAX_STEPS = 500   

    # random seed, can be either an int number or None
    RANDOM_SEED = 2
    
    env.seed(RANDOM_SEED)            # reproducible
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)  # reproducible

    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n

    # 4
    print("observation dimension: %d" % N_F)  
    # [ 2.4 , inf , 0.41887902 , inf]
    print("observation high: %s" % env.observation_space.high)  
    # [-2.4 , -inf , -0.41887902 , -inf]
    print("observation low : %s" % env.observation_space.low)   
    # 2 : left or right
    print("num of actions: %d" % N_A)  

    # 创建两个 network: Actor 和 Critic
    actor = Actor(state_dim=N_F, action_num=N_A, lr=LR_A)
    # we need a good teacher, so the teacher should learn faster than the actor
    critic = Critic(state_dim=N_F, lr=LR_C)

    t0 = time.time()
    if mode == "train":
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            state = env.reset().astype(np.float32)
            # number of step in this episode
            step = 0  
            # rewards of all steps
            episode_reward = 0  
            while True:
                #env.render()

                # actor 从状态 state 开始, 执行动作 action, 得到奖励 reward, 进入状态 state_new
                action = actor.get_action(state)
                state_new, reward, done, info = env.step(action)
                state_new = state_new.astype(np.float32)

                # reward shaping trick
                # 如果已经到达最终状态, 则奖励直接扣 20 点, 为什么？
                # CartPole 游戏最终目的是希望坚持越久越好, 可以想象这么一个过程: 在某个濒死状态 s 下, 选
                # 择动作 a, 进入结束状态 s, 收获 r, 在 CartPole 中这个 reward 为 1.0; 但我们并不希望游戏结
                # 束, 而是希望智能体能在濒死状态下 “力挽狂澜”, 于是把 reward 减去 20, 相当于是对濒死状态下选
                # 择动作 a 的强烈不认同; 通过 -20 大幅减少动作 a 出现的概率; 再进一步, reward 会向前传播, 让
                # 智能体濒死状态之前时, 不选择会进入濒死状态的动作, 努力避免进入濒死状态; 所以说 reward 是
                # 一个主观因素很强的数值, 当环境返回的 reward 不能满足要求时, 完全可以修改 reward, 让
                # 智能体更快学习
                if done: 
                    reward = -20   
                # these may helpful in some tasks
                if abs(state_new[0]) >= env.observation_space.high[0]:
                     # cart moves more than 2.4 units from the center
                     reward = -20
                     # reward for the distance between cart to the center
                     #reward -= abs(state_new[0])  * .1

                episode_reward += reward

                try:
                    # Critic 学习: 计算并缩小 td-error (基于梯度更新模型参数); 返回 td-error
                    # learn Value-function : gradient = grad[r + lambda * V(s_new) - V(s)]
                    td_error = critic.learn(state, reward, state_new, done)  
                    # actor 学习: 基于 td_error 计算策略分布
                    # learn Policy : true_gradient = grad[logPi(s, a) * td_error]
                    actor.learn(state, action, td_error)  
                # if Ctrl+C at running actor.learn(), then save model, or exit if not at actor.learn()
                except KeyboardInterrupt:  
                    actor.save()
                    critic.save()

                state = state_new
                step += 1

                if done or step >= MAX_STEPS:
                    break

            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)

            print('Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}' \
                  .format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0))

            # Early Stopping for quick check（it is important for this task）
            if step >= MAX_STEPS:
                print("Early Stopping")
                break
        actor.save()
        critic.save()

        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', ALG_NAME))
    else:
        actor.load()
        critic.load()

        for episode in range(TEST_EPISODES):
            episode_time = time.time()
            state = env.reset().astype(np.float32)
            # number of step in this episode
            t = 0  
            episode_reward = 0
            while True:
                #env.render()
                action = actor.get_action(state, greedy=True)
                state_new, reward, done, info = env.step(action)
                state_new = state_new.astype(np.float32)
                if done: reward = -20

                episode_reward += reward
                state = state_new
                t += 1

                if done or t >= MAX_STEPS:
                    print('Testing  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}' \
                          .format(episode + 1, TEST_EPISODES, episode_reward, time.time() - t0))
                    break



# environment id
ENV_ID = 'CartPole-v1'

ALG_NAME = 'AC_%s' % ENV_ID

if __name__ == '__main__':
    mode = "train"
    run(mode)


