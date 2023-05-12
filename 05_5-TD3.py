"""
Twin Delayed DDPG (TD3)

DDPG suffers from problems like overestimate of Q-values and sensitivity to hyper-parameters.
Twin Delayed DDPG (TD3) is a variant of DDPG with several tricks:
1) Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”), 
              and uses the smaller of the two Q-values to form the targets in the Bellman error
              loss functions.

2) Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less
              frequently than the Q-function. 

3) Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder
                for the policy to exploit Q-function errors by smoothing out Q along changes in action.

The implementation of TD3 includes 6 networks: 2 Q-net, 2 target Q-net, 1 policy net, 1 target policy net
Actor policy in TD3 is deterministic, with Gaussian exploration noise.


参考论文:
original paper: https://arxiv.org/pdf/1802.09477.pdf


依赖环境包:
pip install IPython==8.12.0
"""
import argparse
import math
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from IPython.display import clear_output

import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model



class ReplayBuffer:
    """
    a ring buffer for storing transitions and sampling for training
    """
    def __init__(self, capacity):
        self.capacity = capacity        # buffer 的最大值
        self.buffer = []                # buffer 列表
        self.position = 0               # 当前输入的位置，相当于指针

    def push(self, state, action, reward, next_state, done):
        """
        如果 buffer 的长度小于最大值, 即第一环时, 需要先初始化一个“空间”, 
        这个空间值为 None, 再给这个空间赋值

        state: (state_dim,)
        action: (action_dim,)
        reward: (,), scalar
        next_state: (state_dim,)
        done: (,), scalar (0 and 1) or bool (True and False)
        """
        if len(self.buffer) < self.capacity:        
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        # as a ring buffer
        self.position = int((self.position + 1) % self.capacity) 

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # stack for each element
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) 
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):
    """ 
    normalize the actions to be in reasonable range 
    修改 gym 环境的动作输出: 把输出归一化
    """
    def _action(self, action):
        # 动作空间最小值
        low = self.action_space.low
        # 动作空间最大值
        high = self.action_space.high  

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

    def step(self, ac):
        """
        必须重定义该类, 否则后续执行 step 操作会报错
        """
        return self.env.step(ac)


class QNetwork(Model):
    """
    the network for evaluate values of state-action pairs: Q(s,a) 
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        # jy: 网络输入层;
        input_dim = num_inputs + num_actions
        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=input_dim, name='q1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='q2')
        # jy: 网络输出层(输出一个值)
        self.linear3 = Dense(n_units=1, W_init=w_init, in_channels=hidden_dim, name='q3')

    def forward(self, input):
        """
        定义网络各层的拼接方式;
        输入为 state + action, 输出一个值(Q 值)
        """
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class PolicyNetwork(Model):
    """ 
    the network for generating non-determinstic (Gaussian distributed) action from the state input 
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name='policy3')

        self.output_linear = Dense(n_units=num_actions, W_init=w_init, 
                                   b_init=tf.random_uniform_initializer(-init_w, init_w),
                                   in_channels=hidden_dim, name='policy_output')

        self.action_range = action_range
        self.num_actions = num_actions


    def forward(self, state):
        """
        输入 state, 输出 action;
        """
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)
        #import pdb; pdb.set_trace()
        # unit range output [-1, 1]
        output = tf.nn.tanh(self.output_linear(x))  

        return output

    def evaluate(self, state, eval_noise_scale):
        """
        基于神经网络使用策略分布, 采样生成 noisy actions
        generate action with state for calculating gradients;
        eval_noise_scale: as the trick of target policy smoothing, for generating noisy actions.
        """
        state = state.astype(np.float32)
        # 通过 state 计算 action (action 范围是 [-1, 1])
        # jy-bug-1 相关: 尽量不要调用 self.forward(state), 直接调用 self(state)
        #action = self.forward(state)           
        action = self(state)
        # 映射到游戏的 action 取值范围
        action = self.action_range * action

        # add noise (建立一个正态分布)
        normal = tfp.distributions.Normal(0, 1)
        # 对噪声进行上下限裁剪; eval_noise_scale
        eval_noise_clip = 2 * eval_noise_scale
        # 弄个一个 noisy 和 action 的 shape 一致, 然后乘以 scale
        noise = normal.sample(action.shape) * eval_noise_scale
        # 对 noisy 进行剪切, 不要太大也不要太小
        noise = tf.clip_by_value(noise, -eval_noise_clip, eval_noise_clip) 
        # action 加上噪音
        action = action + noise

        return action

    def get_action(self, state, explore_noise_scale):
        """ 
        generate action with state for interaction with environment
        输入 state, 维度如: [64, 3]
        输出 action, 维度如: [64, 1]
        """
        # forward 函数接受输入 state, 输出 action (激活函数为 tanh, 所以范围是 [-1, 1])
        # jy-bug-1 相关: 尽量不要调用 self.forward(state), 直接调用 self(state)
        #action = self.forward([state])  
        action = self([state])
        # 将 action 变成矩阵
        action = action.numpy()[0] 

        # add noise 
        # 生成 normal 这样一个正态分布
        normal = tfp.distributions.Normal(0, 1)
        # 在正态分布中抽样一个和 action 一样 shape 的数据, 然后乘以 scale
        noise = normal.sample(action.shape) * explore_noise_scale 
        # action 乘以动作的范围, 加上 noise
        action = self.action_range * action + noise

        return action.numpy()

    def sample_action(self, ):
        """
        generate random actions for exploration 
        """
        a = tf.random.uniform([self.num_actions], -1, 1)

        return self.action_range * a.numpy()


class TD3_Trainer():

    def __init__(self, replay_buffer, hidden_dim, action_range, state_dim, action_dim,
                 policy_target_update_interval=1, q_lr=3e-4, policy_lr=3e-4):
        """
        q_lr: q_net learning rate
        policy_lr: policy_net learning rate
        """
        self.replay_buffer = replay_buffer

        # initialize all networks
        # 用两个 Qnet 来估算 (doubleDQN 的想法), 同时也有两个对应的 target_q_net
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range)
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        # initialize weights of target networks
        # jy: 把 q_net 网络参数赋值给 target_net 网络(在网络初始化时进行硬更新)
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        # jy: 把 policy_net 网络参数赋值给 target_policy_net 网络
        self.target_policy_net = self.target_ini(self.policy_net, self.target_policy_net)

        # jy: 设置 6 个网络均为可训练状态;
        self.q_net1.train()
        self.q_net2.train()
        self.target_q_net1.train()
        self.target_q_net2.train()
        self.policy_net.train()
        self.target_policy_net.train()

        # 更新次数
        self.update_cnt = 0 
        # 策略网络更新频率
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = tf.optimizers.Adam(q_lr)
        self.q_optimizer2 = tf.optimizers.Adam(q_lr)
        self.policy_optimizer = tf.optimizers.Adam(policy_lr)

    def target_ini(self, net, target_net):
        """
        hard-copy update for initializing target networks 
        """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        ''' 
        soft update the target net with Polyak averaging 
        将 net 网络参数软更新(net 网络参数和 target_net 网络参数各占一定比例)至 target_net 
        '''
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            # copy weight value into target parameters (原来参数占比 + 目前参数占比)
            target_param.assign(target_param * (1.0 - soft_tau) + param * soft_tau)
        return target_net

    def update(self, batch_size, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2):
        """
        update all networks in TD3 
        """
        # 计算更新次数
        self.update_cnt += 1
        # 从 buffer sample 数据
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # expand dim, 调整形状, 方便输入网络
        reward = reward[:, np.newaxis]

        # 输入 s', 从 target_policy_net 计算 a' (注意此处有加 noisy)
        # 基于神经网络使用策略分布, 采样生成 noisy (clipped normal noise) actions
        new_next_action = self.target_policy_net.evaluate(next_state, eval_noise_scale=eval_noise_scale) 

        # 归一化 reward (有正有负)
        # normalize with batch mean and std; plus a small number to prevent numerical problem
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (np.std(reward, axis=0) + 1e-6)  

        # Training Q Function
        # 把 next_state 和 new_action 堆叠在一起输入到 target_q_net, 从两个 qnet 中取最小值
        target_q_input = tf.concat([next_state, new_next_action], 1) 
        target_q_min = tf.minimum(self.target_q_net1(target_q_input), self.target_q_net2(target_q_input))

        # 基于 target_q_min 计算 target_q, 用于作为基准参照更新 q_net 网络;
        # 把 done 从布尔变量改为 int (if done==1, only reward)
        done = done[:, np.newaxis]
        target_q_value = reward + (1 - done) * gamma * target_q_min  
        # 基于当前的 state 和 action 构造 q_net 网络的输入;
        q_input = tf.concat([state, action], 1)  
        # 更新 q_net1 网络(此处和 DQN 一样)
        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(tf.square(predicted_q_value1 - target_q_value))
        q1_grad = q1_tape.gradient(q_value_loss1, self.q_net1.trainable_weights)
        self.q_optimizer1.apply_gradients(zip(q1_grad, self.q_net1.trainable_weights))

        # 更新 q_net2 网络(与 q_net1 网络同理)
        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(tf.square(predicted_q_value2 - target_q_value))
        q2_grad = q2_tape.gradient(q_value_loss2, self.q_net2.trainable_weights)
        self.q_optimizer2.apply_gradients(zip(q2_grad, self.q_net2.trainable_weights))

        # Training Policy Function
        # policy 不经常 update (qnet 更新一定次数才 update 一次)
        if self.update_cnt % self.policy_target_update_interval == 0:
            # 更新 policy_net
            with tf.GradientTape() as p_tape:
                # 计算 action = Policy(s) (注意: 此处没有 noise)
                # no noise, deterministic policy gradients
                new_action = self.policy_net.evaluate(state, eval_noise_scale=0.0)

                # jy: 基于当前 state 和 action
                new_q_input = tf.concat([state, new_action], 1)
                # 方式-1: 取两个 q_net 网络中的较小值;
                # predicted_new_q_value = tf.minimum(self.q_net1(new_q_input), self.q_net2(new_q_input))
                # 方式-2:
                predicted_new_q_value = self.q_net1(new_q_input)
                # 对一个 batch_size 的 Q 值取平均, 并取负值(即梯度上升)
                policy_loss = -tf.reduce_mean(predicted_new_q_value)
            # jy: 基于梯度上升, 更新 policy_net
            p_grad = p_tape.gradient(policy_loss, self.policy_net.trainable_weights)
            self.policy_optimizer.apply_gradients(zip(p_grad, self.policy_net.trainable_weights))

            # Soft update the target nets
            # 软更新 target_network 三个
            # 将 q_net 网络参数软更新(qnet 网络参数和 target_q_net 网络参数各占一定比例)至 target_q_net
            self.target_q_net1 = self.target_soft_update(self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(self.q_net2, self.target_q_net2, soft_tau)
            # 将 policy_net 网络参数软更新至 target_policy_net
            self.target_policy_net = self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)

    def save_weights(self):  
        """
        save trained weights
        """
        tl.files.save_npz(self.q_net1.trainable_weights, name='model_q_net1.npz')
        tl.files.save_npz(self.q_net2.trainable_weights, name='model_q_net2.npz')
        tl.files.save_npz(self.target_q_net1.trainable_weights, name='model_target_q_net1.npz')
        tl.files.save_npz(self.target_q_net2.trainable_weights, name='model_target_q_net2.npz')
        tl.files.save_npz(self.policy_net.trainable_weights, name='model_policy_net.npz')
        tl.files.save_npz(self.target_policy_net.trainable_weights, name='model_target_policy_net.npz')

    def load_weights(self):  
        """
        load trained weights
        """
        tl.files.load_and_assign_npz(name='model_q_net1.npz', network=self.q_net1)
        tl.files.load_and_assign_npz(name='model_q_net2.npz', network=self.q_net2)
        tl.files.load_and_assign_npz(name='model_target_q_net1.npz', network=self.target_q_net1)
        tl.files.load_and_assign_npz(name='model_target_q_net2.npz', network=self.target_q_net2)
        tl.files.load_and_assign_npz(name='model_policy_net.npz', network=self.policy_net)
        tl.files.load_and_assign_npz(name='model_target_policy_net.npz', network=self.target_policy_net)


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.savefig('td3.png')
    # plt.show()



def run(mode):

    max_frames = 40000
    # total number of steps for testing
    test_frames = 300
    # maximum number of steps for one episode
    max_steps = 150
    # udpate batchsize
    batch_size = 64
    # 500 for random action sampling in the beginning of training
    explore_steps = 500
    # repeated updates for single step
    update_itr = 3
    # size of hidden layers for networks
    hidden_dim = 32
    # delayed steps for updating the policy network and target networks
    policy_target_update_interval = 3
    # range of action noise for exploration
    explore_noise_scale = 1.0
    # range of action noise for evaluation of action value
    eval_noise_scale = 0.5
    # value range of reward
    reward_scale = 1.
    # size of replay buffer
    replay_buffer_size = 5e5

    # reproducible
    random.seed(2)
    np.random.seed(2)
    tf.random.set_seed(2)  


    ENV = 'Pendulum-v1'
    env = NormalizedActions(gym.make(ENV))
    #env = gym.make(ENV)

    # scale action, [-action_range, action_range]
    #action_range = env.action_space.high
    action_range = 1.
    # 动作空间
    action_dim = env.action_space.shape[0] 
    # 状态空间
    state_dim = env.observation_space.shape[0] 
    print("action_dim: %s" % action_dim)
    print("state_dim: %s" % state_dim)

    # initialization of buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)

    # initialization of trainer
    td3_trainer=TD3_Trainer(replay_buffer, hidden_dim, action_range, state_dim, action_dim,
                            policy_target_update_interval=policy_target_update_interval)

    # training loop
    if mode == "train":
        # 记录步数
        frame_idx = 0
        # 记录每个 EP 的总 reward
        rewards = []       
        t0 = time.time()
        # 如果小于最大步数, 就继续训练
        while frame_idx < max_frames:
            # 初始化 state, 并整理 state 的类型
            state = env.reset()  
            state = state.astype(np.float32)
            episode_reward = 0
            # 第一次需要进行初始化 trainer
            if frame_idx < 1:      
                print('intialize')
                # need an extra call here to make inside functions be able to use model.forward
                # jy-bug-1 相关: 尽量不要调用 self.forward(state), 直接调用 self(state); 如果相关
                #                模型调用 forward 方法时是明确调用, 则此处需补充一下两处代码的执行
                #_ = td3_trainer.policy_net([state])
                #_ = td3_trainer.target_policy_net([state])

            # 开始训练
            for step in range(max_steps):
                # 如果小于 500 步, 就随机; 如果大于就用 get_action 方法获取 action;
                if frame_idx > explore_steps: 
                    # 带有 noisy 的 action
                    action = td3_trainer.policy_net.get_action(state, explore_noise_scale=1.0)  
                else:
                    action = td3_trainer.policy_net.sample_action()

                # 与环境进行交互
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                #env.render()
                done = 1 if done ==True else 0

                # 记录数据在 replay_buffer
                replay_buffer.push(state, action, reward, next_state, done)

                # 赋值 state, 累计总 reward 和步数
                state = next_state
                episode_reward += reward
                frame_idx += 1

                # 如果数据超过一个 batch_size 的大小, 则开始更新
                if len(replay_buffer) > batch_size:
                    # 注意: 这里可以更新多次
                    for i in range(update_itr):
                        # jy: 
                        td3_trainer.update(batch_size, eval_noise_scale=0.5, reward_scale=1.)

                if frame_idx % 500 == 0:
                    plot(frame_idx, rewards)

                if done:
                    break
            # current episode
            episode = int(frame_idx / max_steps) 
            # total episodes
            all_episodes = int(max_frames / max_steps) 
            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                episode, all_episodes, episode_reward, time.time()-t0 ))
            rewards.append(episode_reward)
        td3_trainer.save_weights()
    else:
        frame_idx = 0
        rewards = []
        t0 = time.time()

        td3_trainer.load_weights()

        while frame_idx < test_frames:
            state = env.reset()
            state = state.astype(np.float32)
            episode_reward = 0
            if frame_idx < 1:
                print('intialize')
                # need an extra call to make inside functions be able to use forward
                _ = td3_trainer.policy_net([state])
                _ = td3_trainer.target_policy_net([state])

            for step in range(max_steps):
                action = td3_trainer.policy_net.get_action(state, explore_noise_scale=1.0)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                env.render()
                done = 1 if done ==True else 0

                state = next_state
                episode_reward += reward
                frame_idx += 1

                # if frame_idx % 50 == 0:
                #     plot(frame_idx, rewards)

                if done:
                    break
            episode = int(frame_idx / max_steps)
            all_episodes = int(test_frames / max_steps)
            print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                episode, all_episodes, episode_reward, time.time()-t0 ) )
            rewards.append(episode_reward)


tl.logging.set_verbosity(tl.logging.DEBUG)

if __name__ == '__main__':
    mode = "train"
    run(mode)


