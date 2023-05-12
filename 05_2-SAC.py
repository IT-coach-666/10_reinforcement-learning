""" 
Soft Actor-Critic (SAC)
Actor policy in SAC is stochastic, with off-policy training. 
And 'soft' in SAC indicates the trade-off between the entropy and expected return. 
The additional consideration of entropy term helps with more explorative policy.
And this implementation contains an automatic update for the entropy factor.

This version of Soft Actor-Critic (SAC) implementation contains 5 networks: 
2 Q net, 2 target Q net, 1 policy net.
It uses alpha loss.

paper: https://arxiv.org/pdf/1812.05905.pdf
"""

import argparse
import os
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model



class ReplayBuffer:
    """
    a ring buffer for storing transitions and sampling for training
    """
    def __init__(self, capacity=5e5):
        """
        capacity: size of the replay buffer
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        # as a ring buffer
        self.position = int((self.position + 1) % self.capacity) 

    def sample(self, BATCH_SIZE):
        # jy: 从 self.buffer 中随机抽样 BATCH_SIZE 组数据;
        batch = random.sample(self.buffer, BATCH_SIZE)
        # stack for each element
        # the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch)
        # zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)]
        # the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9]
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) 
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class SoftQNetwork(Model):
    """ 
    the network for evaluate values of state-action pairs: Q(s,a) 
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        # jy: 输入维度;
        input_dim = num_inputs + num_actions
        # jy: 权重初始化;
        # glorot initialization is better than uniform in practice
        w_init = tf.keras.initializers.glorot_normal(seed=None)
        # w_init = tf.random_uniform_initializer(-init_w, init_w)
        # jy: 定义模型层;
        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init,
                             in_channels=input_dim, name='q1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init,
                             in_channels=hidden_dim, name='q2')
        self.linear3 = Dense(n_units=1, W_init=w_init, 
                             in_channels=hidden_dim, name='q3')

    def forward(self, input):
        # jy: 使用初始化的模型层构造网络架构;
        x = self.linear1(input)
        x = self.linear2(x)
        # jy: 基于 self.linear3 定义可知, 最终输出的结果为单个数值;
        x = self.linear3(x)
        return x


class PolicyNetwork(Model):
    """ 
    the network for generating non-deterministic (Gaussian distributed) action from the state input 
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., 
                 init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        # jy: 权重初始化;
        w_init = tf.keras.initializers.glorot_normal(seed=None)
        # w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, 
                             in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, 
                             in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, 
                             in_channels=hidden_dim, name='policy3')

        # jy: 以下两层的输入的维度均为隐层维度(即 self.linear3 的输出, 输出均为 action 的数量)
        self.mean_linear = Dense(n_units=num_actions, W_init=w_init,
                                 b_init=tf.random_uniform_initializer(-init_w, init_w),
                                 in_channels=hidden_dim, name='policy_mean')
        self.log_std_linear = Dense(n_units=num_actions, W_init=w_init,
                                    b_init=tf.random_uniform_initializer(-init_w, init_w),
                                    in_channels=hidden_dim, name='policy_logstd')

        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        """
        输入 state, 输出平均数和标准差(均代表 action 的平均数和标准差取 log 的结果, 对应 action 的策略分布)
        """
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)

        # jy: 通过网络层计算 mean
        mean = self.mean_linear(x)
        # jy: 通过另一网络层计算 log_std
        log_std = self.log_std_linear(x)
        # jy: 对 log_std 进行剪枝操作;
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        """ 
        generate action with state for calculating gradients 
        传入 state, 通过神经网络获取均值和方差, 作为 action 策略分布的两个参数, 从而确定策略分布, 获取 action;
        """
        state = state.astype(np.float32)
        # jy: 通过当前 PolicyNetwork 类(神经网络)估算 mean 和 log_std (用于代表 action 的策略分布的两个参数值);
        mean, log_std = self.forward(state)
        # no clip in evaluation, clip affects gradients flow
        std = tf.math.exp(log_std)  

        # jy: 正太分布;
        normal = tfp.distributions.Normal(0, 1)
        z = normal.sample(mean.shape)
        # TanhNormal distribution as actions; reparameterization trick
        action_0 = tf.math.tanh(mean + std * z)  
        action = self.action_range * action_0
        # according to original paper, with an extra last term for normalizing different action range
        log_prob = tfp.distributions.Normal(mean, std).log_prob(mean + std * z) - tf.math.log(1. - action_0 ** 2 +
                                                                            epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the dim of actions to get 1 dim probability; or else use Multivariate Normal.
        # expand dim as reduce_sum causes 1 dim reduced
        log_prob = tf.reduce_sum(log_prob, axis=1)[:, np.newaxis]  

        return action, log_prob, z, mean, log_std

    def get_action(self, state, greedy=False):
        """ 
        generate action with state for interaction with envronment 
        基于 state 通过神经网络预估策略的均值与方差, 从而获得 action 的策略分布, 进而获取 action; 
        """
        # jy: 调用当前类的 forward 方法(注意, 不要直接调用, 否则会出现 fix-bug-1);
        #mean, log_std = self.forward([state])
        mean, log_std = self([state])
        # jy: 基于 log_std 求 std;
        std = tf.math.exp(log_std)

        # jy: 正太分布;
        normal = tfp.distributions.Normal(0, 1)
        # jy: 基于正太分布, 随机生成一个与 mean 同维度的数据;
        z = normal.sample(mean.shape)
        # TanhNormal distribution as actions; reparameterization trick
        # jy: self.action_range 值为 2, 乘以 tanh 值(取值范围为 -1 到 1), 以确保不会
        #     超出 self.action_range 范围; 
        action = self.action_range * tf.math.tanh(mean + std * z) 
        action = self.action_range * tf.math.tanh(mean) if greedy else action
        # jy: 返回 action 的值(由于调用该方法时传入的 state 对应的 batch_size 为 1, 因
        #     此直接返回第 1 个结果即可);
        return action.numpy()[0]

    def sample_action(self, ):
        """ 
        generate random actions for exploration 
        在 self.action_range 范围内随机生成一个 action
        """
        a = tf.random.uniform([self.num_actions], -1, 1)
        return self.action_range * a.numpy()


class SAC:
    def __init__(self, alg_name, state_dim, action_dim, action_range, replay_buffer,
                 hidden_dim=32, SOFT_Q_LR=3e-4, POLICY_LR=3e-4, ALPHA_LR=3e-4):
        """
        SOFT_Q_LR: q_net learning rate
        POLICY_LR: policy_net learning rate
        ALPHA_LR: alpha learning rate
        hidden_dim: size of hidden layers for networks
        """
        self.replay_buffer = replay_buffer
        self.alg_name = alg_name

        # initialize all networks (共初始化 5 个网络: 4 个 SoftQNetwork 和 1 个 PolicyNetwork)
        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        
        # set mode
        self.soft_q_net1.train()
        self.soft_q_net2.train()
        # jy: target 网络设置为非训练模式;
        self.target_soft_q_net1.eval()
        self.target_soft_q_net2.eval()
        self.policy_net.train()

        self.log_alpha = tf.Variable(0, dtype=np.float32, name='log_alpha')
        self.alpha = tf.math.exp(self.log_alpha)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        # initialize weights of target networks
        # jy: 将 target_soft_q_net 的参数初始化为与 soft_q_net 相同;
        self.target_soft_q_net1 = self.target_ini(self.soft_q_net1, self.target_soft_q_net1)
        self.target_soft_q_net2 = self.target_ini(self.soft_q_net2, self.target_soft_q_net2)

        self.soft_q_optimizer1 = tf.optimizers.Adam(SOFT_Q_LR)
        self.soft_q_optimizer2 = tf.optimizers.Adam(SOFT_Q_LR)
        self.policy_optimizer = tf.optimizers.Adam(POLICY_LR)
        self.alpha_optimizer = tf.optimizers.Adam(ALPHA_LR)

    def target_ini(self, net, target_net):
        """ 
        hard-copy update for initializing target networks 
        """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        """ 
        soft update the target net with Polyak averaging 
        """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            # copy weight value into target parameters
            target_param.assign(target_param * (1.0 - soft_tau) + param * soft_tau)
        return target_net

    def update(self, batch_size, reward_scale=10., auto_entropy=True,
               target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        """ 
        update all networks in SAC 
        reward_scale: value range of reward
        auto_entropy: automatically updating variable alpha for entropy
        """
        # jy: 从 buffer 中抽样取 batch_size 组数;
        # jy: state 维度为: (256, 3), 其中 256 为 batch_size 大小;
        #     action 维度为: (256, 1)
        #     reward 维度为: (256,), 以下会将维度扩充为 (256, 1), 方便后续操作;
        #     next_state 维度为: (256, 3)
        #     done 维度为: (256,), 以下会将维度扩充为 (256, 1), 方便后续操作;
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # jy: 扩充维度, 将 (256,) 扩充为 (256, 1), 方便后续统一操作; 
        reward = reward[:, np.newaxis]  
        done = done[:, np.newaxis]

        # jy: 归一化处理: 减均值, 除以标准差;
        # normalize with batch mean and std; plus a small number to prevent numerical problem
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (np.std(reward, axis=0) + 1e-6)

        # Training Q Function ==========================================================
        # 调用 PolicyNetwork 类的 evaluate 方法: generate action with state for calculating gradients
        # 传入 state, 通过神经网络获取均值和方差, 作为 action 策略分布的两个参数, 从而确定
        # 策略分布, 获取 action; 得到的 new_next_action 和 next_log_prob 维度均为: [256, 1]
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        
        # the dim 0 is number of samples
        target_q_input = tf.concat([next_state, new_next_action], 1)  
        # jy: 获取目标 q 值; 选取两个 target_soft_q_net 中值较小的那一个;
        target_q_min = tf.minimum(self.target_soft_q_net1(target_q_input), 
                                  self.target_soft_q_net2(target_q_input)) - self.alpha * next_log_prob
        # jy: 如果 done==1, 则 reward 即为 target_q_value
        target_q_value = reward + (1 - done) * gamma * target_q_min  
        
        # jy: 获取 q 值;
        # the dim 0 is number of samples
        q_input = tf.concat([state, action], 1) 

        # jy: 使用第一个 soft_q_net 网络预测 q 值并与目标 q 值对比计算 loss, 基于梯度更新网络;
        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.soft_q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.losses.mean_squared_error(predicted_q_value1, target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.soft_q_net1.trainable_weights)
        self.soft_q_optimizer1.apply_gradients(zip(q1_grad, self.soft_q_net1.trainable_weights))

        # jy: 使用第二个 soft_q_net 网络预测 q 值并与目标 q 值对比计算 loss, 基于梯度更新网络;
        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.soft_q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.losses.mean_squared_error(predicted_q_value2, target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.soft_q_net2.trainable_weights)
        self.soft_q_optimizer2.apply_gradients(zip(q2_grad, self.soft_q_net2.trainable_weights))


        # Training Policy Function =======================================================
        # jy: 基于当前状态 state 通过 PolicyNetwork 预测新 action, 并结合 2 个 soft_q_net 的
        #     预测结果计算损失, 计算梯度, 更新 PolicyNetwork
        with tf.GradientTape() as p_tape:
            # 传入 state, 通过神经网络获取均值和方差, 作为 action 策略分布的两个参数, 从而确
            # 定策略分布, 获取 action
            new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
            # the dim 0 is number of samples
            new_q_input = tf.concat([state, new_action], 1) 
            # jy: 选取两个网络中的较小值作为新的预测 q 值;
            predicted_new_q_value = tf.minimum(self.soft_q_net1(new_q_input),
                                               self.soft_q_net2(new_q_input))
            #predicted_new_q_value = self.soft_q_net1(new_q_input)
            policy_loss = tf.reduce_mean(self.alpha * log_prob - predicted_new_q_value)
        p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
        self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))

        # Updating alpha w.r.t entropy
        # alpha: trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            with tf.GradientTape() as alpha_tape:
                alpha_loss = -tf.reduce_mean((self.log_alpha * (log_prob + target_entropy)))
            alpha_grad = alpha_tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
            self.alpha = tf.math.exp(self.log_alpha)
        else:  
            # fixed alpha
            self.alpha = 1.
            alpha_loss = 0

        # Soft update the target value nets
        self.target_soft_q_net1 = self.target_soft_update(self.soft_q_net1, self.target_soft_q_net1, soft_tau)
        self.target_soft_q_net2 = self.target_soft_update(self.soft_q_net2, self.target_soft_q_net2, soft_tau)


    def save(self): 
        """
        save trained weights
        """
        path = os.path.join('model', self.alg_name)
        if not os.path.exists(path):
            os.makedirs(path)
        extend_path = lambda s: os.path.join(path, s)
        tl.files.save_npz(self.soft_q_net1.trainable_weights, extend_path('model_q_net1.npz'))
        tl.files.save_npz(self.soft_q_net2.trainable_weights, extend_path('model_q_net2.npz'))
        tl.files.save_npz(self.target_soft_q_net1.trainable_weights, extend_path('model_target_q_net1.npz'))
        tl.files.save_npz(self.target_soft_q_net2.trainable_weights, extend_path('model_target_q_net2.npz'))
        tl.files.save_npz(self.policy_net.trainable_weights, extend_path('model_policy_net.npz'))
        np.save(extend_path('log_alpha.npy'), self.log_alpha.numpy())  # save log_alpha variable

    def load_weights(self):  
        """
        load trained weights
        """
        path = os.path.join('model', self.alg_name)
        extend_path = lambda s: os.path.join(path, s)
        tl.files.load_and_assign_npz(extend_path('model_q_net1.npz'), self.soft_q_net1)
        tl.files.load_and_assign_npz(extend_path('model_q_net2.npz'), self.soft_q_net2)
        tl.files.load_and_assign_npz(extend_path('model_target_q_net1.npz'), self.target_soft_q_net1)
        tl.files.load_and_assign_npz(extend_path('model_target_q_net2.npz'), self.target_soft_q_net2)
        tl.files.load_and_assign_npz(extend_path('model_policy_net.npz'), self.policy_net)
        # load log_alpha variable
        self.log_alpha.assign(np.load(extend_path('log_alpha.npy'))) 



def run(mode):
    # random seed
    RANDOM_SEED = 2
    ENV_ID = 'Pendulum-v1'
    alg_name = "SAC_%s" % ENV_ID
    # total number of episodes for training
    TRAIN_EPISODES = 100
    # total number of episodes for training
    TEST_EPISODES = 10
    # total number of steps for each episode
    MAX_STEPS = 200
    # 100 for random action sampling in the beginning of training
    EXPLORE_STEPS = 100
    # update batch size
    BATCH_SIZE = 256


    # initialization of env
    #env = gym.make(ENV_ID).unwrapped
    env = gym.make(ENV_ID)
    # jy: 3
    state_dim = env.observation_space.shape[0]
    # jy: 1
    action_dim = env.action_space.shape[0]
    # jy: 2, scale action, [-action_range, action_range]
    action_range = env.action_space.high  

    # reproducible
    env.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # initialization of buffer
    replay_buffer = ReplayBuffer()
    # initialization of trainer
    agent = SAC(alg_name, state_dim, action_dim, action_range, replay_buffer)

    t0 = time.time()
    # training loop
    if mode == "train":
        frame_idx = 0
        all_episode_reward = []

        # jy: fix-bug-1: 与 PolicyNetwork 类的 get_action 方法相关;
        """
        # need an extra call here to make inside functions be able to use model.forward
        state = env.reset().astype(np.float32)
        # jy: 调用 SAC 类的 policy_net 属性为已初始化的 PolicyNetwork 类, 此处传入 [state], 即调用
        #     该类的 forward 方法;
        agent.policy_net([state])
        """

        for episode in range(TRAIN_EPISODES):
            # jy: 初始化环境;
            state = env.reset().astype(np.float32)
            episode_reward = 0
            for step in range(MAX_STEPS):
                #env.render()
                # jy: EXPLORE_STEPS 定义了 random action sampling 的 steps; 小于该值时都进行
                #     随机 action 抽样(PolicyNetwork 类的 sample_action 方法), 保证尽可能多的探索;
                if frame_idx > EXPLORE_STEPS:
                    # jy: 基于 state 通过神经网络预估策略的均值与方差, 从而获得 action 的策略分
                    #     布, 进而获取 action;
                    action = agent.policy_net.get_action(state)
                else:
                    # jy: 在 action_range 范围内随机生成一个 action
                    action = agent.policy_net.sample_action()

                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                done = 1 if done is True else 0

                # jy: 缓存数据;
                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                frame_idx += 1
                # jy: 每到一个 batch_size 大小的数据时定期更新;
                if len(replay_buffer) > BATCH_SIZE:
                    # repeated updates for single step
                    UPDATE_ITR = 3
                    for i in range(UPDATE_ITR):
                        # jy: update all networks in SAC (更新 SAC 中的所有网络)
                        agent.update(BATCH_SIZE, reward_scale=1., target_entropy= -1. * action_dim)

                if done:
                    break
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                  episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0))
        agent.save()
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', self.alg_name))
    else:
        agent.load_weights()

        # need an extra call here to make inside functions be able to use model.forward
        state = env.reset().astype(np.float32)
        agent.policy_net([state])

        for episode in range(TEST_EPISODES):
            state = env.reset().astype(np.float32)
            episode_reward = 0
            for step in range(MAX_STEPS):
                env.render()
                state, reward, done, info = env.step(agent.policy_net.get_action(state, greedy=True))
                state = state.astype(np.float32)
                episode_reward += reward
                if done:
                    break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                  episode + 1, TEST_EPISODES, episode_reward, time.time() - t0))



tl.logging.set_verbosity(tl.logging.DEBUG)

if __name__ == '__main__':
    mode = "train"
    #mode = "test"
    run(mode)


