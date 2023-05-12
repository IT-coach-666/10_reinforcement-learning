"""
Trust Region Policy Optimization (TRPO)

PG method with a large step can collapse the policy performance, even with a
small step can lead a large differences in policy.

TRPO constraint the step in policy space using KL divergence (rather than in 
parameter space), which can monotonically improve performance and avoid a 
collapsed update.


Reference
Trust Region Policy Optimization, Schulman et al. 2015
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
Approximately Optimal Approximate Reinforcement Learning, Kakade and Langford 2002
openai/spinningup : http://spinningup.openai.com/en/latest/algorithms/trpo.html
"""
import argparse
import copy
import os
import threading
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import tensorflow as tf

import tensorflow_probability as tfp
import tensorlayer as tl



class GAE_Buffer:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        # jy: 存储 state
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        # jy: 存储 action
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        # jy: 存储 GAE-lambda advantage 值 (类似于 td-error)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        # jy: 存储 reward
        self.rew_buf = np.zeros(size, dtype=np.float32)
        # jy: rewards-to-go (to be targets for the value function)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        # jy: 存储 value (结合 state 通过 critic 网络得到)
        self.val_buf = np.zeros(size, dtype=np.float32)
        # jy: 存储 logp (action 的概率的 log 值)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        # jy: 存储 mean (action 对应的策略分布的 mean 参数值)
        self.mean_buf = np.zeros(size, dtype=np.float32)
        # jy: 存储 log_std (action 对应的策略分布的 std 参数值的 log 结果)
        self.log_std_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        # jy: self.ptr 为数据存储于 buffer 中的 index 记录
        #     self.max_size 记录 buffer 的最大存储量
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, mean, log_std):
        """
        Append one timestep of agent-environment interaction to the buffer.
        
        存储的 7 个数据分别为:
        obs: state
        act: action
        rew: reward
        val: value(结合 state 通过 critic 网络得到)
        logp (action 的概率的 log 值)
        mean (action 对应的策略分布的 mean 参数值)
        log_std (action 对应的策略分布的 std 参数值的 log 结果)
        """
        # jy: 基于位置指针 self.ptr 存储数值;
        # buffer has to have room so you can store
        assert self.ptr < self.max_size  
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.mean_buf[self.ptr] = mean
        self.log_std_buf[self.ptr] = log_std
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        会补充计算 GAE-lambda advantage 和 rewards-to-go, 分别存储在 self.adv_buf
        和 self.ret_buf

        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        # jy: 设定 buffer 中数据的下标范围
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # jy: GAE-lambda advantage calculation, 类似于 td-error 的计算 (只是进行批量化处理)
        #     rews[:-1] 即最后一个 reward 不要; vals[:-1] 同理;
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        # jy: computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def _discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.

        input: vector x, 如: [x0, x1, x2, ... xn]

        output:
            [x0 + discount * x1 + discount^2 * x2 + ... + discount^n * xn,
             x1 + discount * x2 + ... + discount^(n-1) * xn,
             ...
             xn]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def is_full(self):
        """
        判断 buffer 中的数据是否已满
        """
        return self.ptr == self.max_size

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from the buffer, with
        advantages appropriately normalized (shifted to have mean zero and std one).
        
        Also, resets some pointers in the buffer.

        返回一批数据(包含 7 类), 分别为: 
        [state, action, GAE-lambda advantage,
         rewards-to-go(to be targets for the value function), 
         logp (action 的概率的 log 值), 
         mean (action 对应的策略分布的 mean 参数值),
         log_std (action 对应的策略分布的 std 参数值的 log 结果)]
        """
        # buffer has to be full before you can get
        assert self.ptr == self.max_size 
        # jy: 重置 buffer 中的位置指针
        self.ptr, self.path_start_idx = 0, 0

        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf, self.mean_buf, self.log_std_buf]


class TRPO:
    """
    Trust Region Policy Optimization
    """
    def __init__(self, state_dim, action_dim, action_bound, hidden_sizes=[64, 64],
                 gamma=0.99, delta=0.01, vf_lr=1e-3, train_vf_iters=100, damping_coeff=0.1,
                 cg_iters=10, backtrack_iters=10, backtrack_coeff=0.8, eps=1e-8,
                 lam=0.97, batch_size=512):
        """
        hidden_sizes: hidden layer size
        gamma: reward discount
        delta: KL-divergence limit for TRPO update
        vf_lr: Learning rate for value function optimizer
        train_vf_iters: Number of gradient descent steps to take on value function per epoch
        damping_coeff: Artifact for numerical stability
        cg_iters: Number of iterations of conjugate gradient to perform
        backtrack_iters: Maximum number of steps allowed in the backtracking line search
        backtrack_coeff: How far back to step during backtracking line search
        eps: epsilon
        lam: lambda for GAE-lambda
        """
        self.delta = delta
        self.train_vf_iters = train_vf_iters
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.eps = eps
        self.lam = lam

        # jy: 初始化 critic 网络
        with tf.name_scope('critic'):
            # jy: 网络输入层与隐层
            layer = input_layer = tl.layers.Input([None, state_dim], tf.float32)
            # jy: 基于隐层数量批量构建隐层
            for d in hidden_sizes:
                layer = tl.layers.Dense(d, tf.nn.relu)(layer)
            # jy: 网络输出层
            v = tl.layers.Dense(1)(layer)
        # jy: 基于输入输出层创建网络, 并设置为参数可训练模式
        self.critic = tl.models.Model(input_layer, v)
        self.critic.train()

        # jy: 初始化 actor 网络
        with tf.name_scope('actor'):
            # jy: 网络输入层和隐层;
            layer = input_layer = tl.layers.Input([None, state_dim], tf.float32)
            # jy: 基于隐层数量批量构建隐层;
            for d in hidden_sizes:
                layer = tl.layers.Dense(d, tf.nn.relu)(layer)
            mean = tl.layers.Dense(action_dim, tf.nn.tanh)(layer)
            mean = tl.layers.Lambda(lambda x: x * action_bound)(mean)
            log_std = tf.Variable(np.zeros(action_dim, dtype=np.float32))
        # jy: 基于输入输出层(模型的输出为 mean 值)创建网络, 补充可训练参数(log_std 值设置为可训练状态),
        #     并设置为参数可训练模式
        self.actor = tl.models.Model(input_layer, mean)
        self.actor.trainable_weights.append(log_std)
        self.actor.log_std = log_std
        self.actor.train()

        # jy: 初始化 GAE_Buffer 类;
        self.buf = GAE_Buffer(state_dim, action_dim, batch_size, gamma, self.lam)
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=vf_lr)
        self.action_bound = action_bound

    def get_action(self, state, greedy=False):
        """
        基于 state 获取 action
        state: state input
        greedy: get action greedy or not
        返回: 
            action
            v 值 (结合 state 通过 critic 网络得到)
            logp_pi (action 的概率的 log 值)
            mean (action 对应的策略分布的 mean 参数值)
            log_std (action 对应的策略分布的 std 参数值的 log 结果)
        """
        state = np.array([state], np.float32)
        # jy: 基于 state 通过 actor 网络获取得到 action 的策略分布的 mean 值 
        mean = self.actor(state)
        # jy: action 策略分布的 log_std 值在 actor 网络中是一个可训练的参数;
        log_std = tf.convert_to_tensor(self.actor.log_std)
        std = tf.exp(log_std)
        # jy: 将 std 值设置为与 mean 相同的维度
        std = tf.ones_like(mean) * std
        # jy: 基于 mean 和 std 构建 action 的策略分布(正太分布)
        pi = tfp.distributions.Normal(mean, std)

        if greedy:
            action = mean
        else:
            # jy: action 基于策略分布抽样得到
            action = pi.sample()
        # jy: 对 action 进行 clip, 确保其在有效范围内
        action = np.clip(action, -self.action_bound, self.action_bound)
        logp_pi = pi.log_prob(action)

        # jy: 基于 state 通过 critic 网络得到 V 值
        value = self.critic(state)
        return action[0], value, logp_pi, mean, log_std

    def pi_loss(self, states, actions, adv, old_log_prob):
        """
        calculate pi loss (计算策略分布的 loss)

        states: state batch
        actions: action batch
        adv: advantage (GAE-lambda advantage) batch
        old_log_prob: old log probability (action 的概率的 log 值)

        返回: pi loss
        """
        # jy: 基于 states 通过 actor 网络得到策略分布的 mean 值;
        mean = self.actor(states)
        # jy: 基于均值和方差构造策略分布(正太分布)
        pi = tfp.distributions.Normal(mean, tf.exp(self.actor.log_std))
        # jy: 基于策略分布判断 actions 对应的概率的 log 值;
        log_prob = pi.log_prob(actions)[:, 0]
        # jy: 基于两个概率算比率, 随后乘以 advantage 并取平均, 返回取反的结果;
        ratio = tf.exp(log_prob - old_log_prob)
        surr = tf.reduce_mean(ratio * adv)
        return -surr

    def gradient(self, states, actions, adv, old_log_prob):
        """
        pi gradients

        states: state batch
        actions: actions batch
        adv: advantage (GAE-lambda advantage) batch
        old_log_prob: old log probability batch (action 的概率的 log 值)

        返回: (gradient, loss)
        """
        with tf.GradientTape() as tape:
            # jy: 调用 pi_loss 方法, 计算策略分布的 loss
            loss = self.pi_loss(states, actions, adv, old_log_prob)
        # jy: 基于策略分布的 loss 计算 actor 网络的梯度
        grad = tape.gradient(loss, self.actor.trainable_weights)
        gradient = self._flat_concat(grad)
        return gradient, loss

    def train_vf(self, states, rewards_to_go):
        """
        train v function (即训练更新 critic 网络)

        states: state batch
        rewards_to_go: rewards-to-go (to be targets for the value function) batch
        """
        with tf.GradientTape() as tape:
            # jy: 基于 state 通过 critic 网络得到 V 值
            value = self.critic(states)
            # jy: 基于 V 值和 rewards_to_go 计算 loss
            loss = tf.reduce_mean((rewards_to_go - value[:, 0])**2)
        # jy: 基于 loss 计算梯度, 并更新 critic 网络参数
        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(grad, self.critic.trainable_weights))

    def kl(self, states, old_mean, old_log_std):
        """
        calculate kl-divergence
        states: state batch
        old_mean: mean batch of the old pi
        old_log_std: log std batch of the old pi
        返回: kl_mean or None
        """
        old_mean = old_mean[:, np.newaxis]
        old_log_std = old_log_std[:, np.newaxis]
        old_std = tf.exp(old_log_std)
        old_pi = tfp.distributions.Normal(old_mean, old_std)

        mean = self.actor(states)
        std = tf.exp(self.actor.log_std) * tf.ones_like(mean)
        pi = tfp.distributions.Normal(mean, std)

        kl = tfp.distributions.kl_divergence(pi, old_pi)
        all_kls = tf.reduce_sum(kl, axis=1)
        return tf.reduce_mean(all_kls)

    def _flat_concat(self, xs):
        """
        flat concat input

        xs: a list of tensor
        返回: flat tensor
        """
        return tf.concat([tf.reshape(x, (-1, )) for x in xs], axis=0)

    def get_pi_params(self):
        """
        get actor trainable parameters
        返回: flat actor trainable parameters
        """
        pi_params = self.actor.trainable_weights
        return self._flat_concat(pi_params)

    def set_pi_params(self, flat_params):
        """
        set actor trainable parameters

        flat_params: inputs
        """
        pi_params = self.actor.trainable_weights
        flat_size = lambda p: int(np.prod(p.shape.as_list()))  # the 'int' is important for scalars
        splits = tf.split(flat_params, [flat_size(p) for p in pi_params])
        new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(pi_params, splits)]
        return tf.group([p.assign(p_new) for p, p_new in zip(pi_params, new_params)])

    def save(self, alg_name):
        """
        save trained weights
        """
        path = os.path.join('model', alg_name)
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz(self.actor.trainable_weights, os.path.join(path, 'actor.npz'))
        tl.files.save_npz(self.critic.trainable_weights, os.path.join(path, 'critic.npz'))

    def load(self, alg_name):
        """
        load trained weights
        """
        path = os.path.join('model', alg_name)
        tl.files.load_and_assign_npz(os.path.join(path, 'actor.npz'), network=self.actor)
        tl.files.load_and_assign_npz(os.path.join(path, 'critic.npz'), network=self.critic)

    def cg(self, Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        
        传入参数:
        Ax: 一个函数句柄(对应以下的 hvp 方法)
        b:  策略分布的梯度(g)
        """
        x = np.zeros_like(b)
        # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start
        r = copy.deepcopy(b) 
        p = copy.deepcopy(r)
        r_dot_old = np.dot(r, r)
        for _ in range(self.cg_iters):
            # jy: Ax 即对应 hvp 方法(会固定传入的前 3 个参数), p 对应传入的策略分布的梯度
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + self.eps)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def hvp(self, states, old_mean, old_log_std, x):
        """
        calculate Hessian-vector product

        states: state batch
        old_mean: mean batch of the old pi
        old_log_std: log std batch of the old pi
        传入的 x 为策略分布的梯度值

        返回: hvp
        """
        pi_params = self.actor.trainable_weights
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape0:
                # jy: 基于 state, 策略分布的两个参数(mean 和 log_std)计算 kl 散度;
                d_kl = self.kl(states, old_mean, old_log_std)
            # jy: 基于 kl 散度求策略分布的梯度并摊平;
            g = self._flat_concat(tape0.gradient(d_kl, pi_params))
            # jy: 计算 loss
            l = tf.reduce_sum(g * x)
        # jy: 基于 loss (l) 计算策略分布的梯度并摊平作为 hvp 值;
        hvp = self._flat_concat(tape1.gradient(l, pi_params))

        if self.damping_coeff > 0:
            hvp += self.damping_coeff * x
        return hvp

    def update(self):
        """
        update TRPO
        """
        # jy: 从 buffer 中获取数据; 返回一批数据(包含 7 类), 分别为:
        """
        [state, action, GAE-lambda advantage,
         rewards-to-go(to be targets for the value function),
         logp (action 的概率的 log 值), 
         mean (action 对应的策略分布的 mean 参数值),
         log_std (action 对应的策略分布的 std 参数值的 log 结果)]
        """
        states, actions, adv, rewards_to_go, logp_old_ph, old_mu, old_log_std = self.buf.get()
        # jy: 计算策略分布的梯度(g) 和 loss (pi_l_old)
        g, pi_l_old = self.gradient(states, actions, adv, logp_old_ph)

        Hx = lambda x: self.hvp(states, old_mu, old_log_std, x)
        # jy: 调用 cg 方法 (Conjugate gradient algorithm)
        x = self.cg(Hx, g)

        alpha = np.sqrt(2 * self.delta / (np.dot(x, Hx(x)) + self.eps))
        # jy: 获取 actor 网络的可训练的参数;
        old_params = self.get_pi_params()

        def set_and_eval(step):
            params = old_params - alpha * x * step
            # jy: 设置 actor 网络的训练参数;
            self.set_pi_params(params)
            # jy: 求 kl 散度
            d_kl = self.kl(states, old_mu, old_log_std)
            # jy: 计算策略分布的 loss
            loss = self.pi_loss(states, actions, adv, logp_old_ph)
            return [d_kl, loss]

        # trpo with backtracking line search, hard kl
        for j in range(self.backtrack_iters):
            kl, pi_l_new = set_and_eval(step = self.backtrack_coeff ** j)
            if kl <= self.delta and pi_l_new <= pi_l_old:
                # Accepting new params at step of line search
                break
        else:
            # Line search failed! Keeping old params.
            set_and_eval(step=0.)

        # Value function updates
        for _ in range(self.train_vf_iters):
            # jy:
            self.train_vf(states, rewards_to_go)

    def finish_path(self, done, next_state):
        """
        finish a trajectory
        done: whether the epoch is done
        next_state: next state
        """
        if not done:
            next_state = np.array([next_state], np.float32)
            # jy: 结合 next_state, 基于 critic 网络生成 last_val;
            last_val = self.critic(next_state)
        else:
            last_val = 0
        # jy: 调用 GAE_Buffer 类的 finish_path 方法;
        self.buf.finish_path(last_val)


def run(mode="train"):
    ENV_ID = 'Pendulum-v1'
    RANDOM_SEED = 2
    # total number of episodes for training
    TRAIN_EPISODES = 1000
    # total number of episodes for testing
    TEST_EPISODES = 100
    # total number of steps for each episode
    MAX_STEPS = 200
    # How often (in terms of gap between epochs) to save the current policy and value function
    SAVE_FREQ = 10

    alg_name = "TRPO_%s" % ENV_ID

    env = gym.make(ENV_ID)

    # reproducible
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    env.seed(RANDOM_SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    print("state_dim: %s" % state_dim)
    print("action_dim: %s" % action_dim)
    print("action_bound: %s" % action_bound)

    # jy: 初始化 TRPO 类;
    agent = TRPO(state_dim, action_dim, action_bound)

    t0 = time.time()
    if mode == "train": 
        # jy: reward 累计, 主要用于画图;
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            # jy: 初始化环境 state
            state = env.reset()
            state = np.array(state, np.float32)
            # jy: reward 累计, 主要用于打印输出;
            episode_reward = 0
            for step in range(MAX_STEPS):
                # 渲染环境;
                #env.render()
                # jy: 基于 state 获取得到如下值:
                """
                action
                value 值 (结合 state 通过 critic 网络得到)
                logp (action 的概率的 log 值)
                mean (action 对应的策略分布的 mean 参数值)
                log_std (action 对应的策略分布的 std 参数值的 log 结果)
                """
                action, value, logp, mean, log_std = agent.get_action(state)
                # jy: 基于 action 与环境交互得到 next_state
                next_state, reward, done, _ = env.step(action)
                next_state = np.array(next_state, np.float32)
                # jy: 将相应值存储起来, 存储的 7 个数据分别为:
                """
                state, action, reward, value(结合 state 通过 critic 网络得到), logp(action 的概率的 log 值)
                mean(action 对应的策略分布的 mean 参数值), log_std(action 对应的策略分布的 std 参数值的 log 结果)
                """
                agent.buf.store(state, action, reward, value, logp, mean, log_std)
                episode_reward += reward
                state = next_state
                # jy: 判断 buffer 是否已满
                if agent.buf.is_full():
                    # jy: 调用 TRPO 类的 finish_path 方法;
                    agent.finish_path(done, next_state)
                    # jy: 调用 TRPO 类的 update 方法
                    agent.update()
                if done:
                    break
            agent.finish_path(done, next_state)
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                  episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0))
            if episode % SAVE_FREQ == 0:
                agent.save(alg_name)
        agent.save(alg_name)
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', alg_name))
    else:
        agent.load(alg_name)
        for episode in range(TEST_EPISODES):
            state = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                #env.render()
                action, *_ = agent.get_action(state, greedy=True)
                state, reward, done, info = env.step(action)
                episode_reward += reward
                if done:
                    break
            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                  episode + 1, TEST_EPISODES, episode_reward, time.time() - t0))


if __name__ == '__main__':
    mode = "train"
    run(mode)



