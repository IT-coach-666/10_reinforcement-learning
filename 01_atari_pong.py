# -*- coding: utf-8 -*-
"""Monte-Carlo Policy Network π(a|s)  (REINFORCE).
To understand Reinforcement Learning, we let computer to learn how to play
Pong game from the original screen inputs. 

Before we start, we highly recommend you to go through a famous blog called 
“Deep Reinforcement Learning: Pong from Pixels”: http://karpathy.github.io/2016/05/31/rl/
which is a minimalistic implementation of deep reinforcement learning by using
python-numpy and OpenAI gym environment.

The code here is the reimplementation of Karpathy's Blog by using TensorLayer.
Compare with Karpathy's code, we store observation for a batch, but he store
observation for only one episode and gradients. (so we will use more memory if 
the observation is very large.)

TODO: update grads every step rather than storing all observation!
"""
import time

import gym
import numpy as np
import tensorflow as tf

import tensorlayer as tl



def prepro(I):
    """
    输入的 I 是一个维度为 (210, 160, 3) 的 numpy 数组;
    返回维度为 (6400,) 的 numpy 数组;
    Prepro 210 x 160 x 3 uint8 frame into 6400 (80 x 80) 1D float vector.
    """
    # jy: 从第一维(210)中截取 160, 得到的 I 的维度为 160 x 160 x 3
    I = I[35:195]
    # jy: 得到 I 的维度为 (80, 80)
    I = I[::2, ::2, 0]
    # jy: 将矩阵中的指定值置为 0 和 1;
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    # jy: 将 I 转换为维度为 (6400,) 的 numpy 数组;
    return I.astype(np.float32).ravel()


def get_model(inputs_shape):
    """
    policy network (创建模型)
    """
    ni = tl.layers.Input(inputs_shape)
    nn = tl.layers.Dense(n_units=200, act=tf.nn.relu, name='hidden')(ni)
    nn = tl.layers.Dense(n_units=3, name='output')(nn)
    M = tl.models.Model(inputs=ni, outputs=nn, name="mlp")
    return M


tl.logging.set_verbosity(tl.logging.DEBUG)

batch_size = 10
gamma = 0.99
np.set_printoptions(threshold=np.inf)

env = gym.make("Pong-v4")

# jy: 维度为 (210, 160, 3) 的 numpy 数组;
observation = env.reset()
prev_x = None
running_reward = None
reward_sum = 0
episode_number = 0

xs, ys, rs = [], [], []


image_size = 80
model = get_model([None, image_size * image_size])
train_weights = model.trainable_weights
optimizer = tf.optimizers.RMSprop(lr=1e-4, weight_decay=0.99)
# set model to train mode (in case you add dropout into the model)
model.train()  

start_time = time.time()
game_number = 0


# jy: 不断训练:
while True:
    # display the game environment
    #env.render()
    # jy: 得到的 cur_x 是一个维度为 (6400,) 的 numpy 数组
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(image_size * image_size, dtype=np.float32)
    # jy: 将 x 重构为维度为 (1, 6400) 的数组;
    x = x.reshape(1, image_size * image_size)
    prev_x = cur_x
    # jy: 得到的 _prob 为: <tf.Tensor: shape=(1, 3), dtype=float32, numpy=array([[0., 0., 0.]], dtype=float32)>
    _prob = model(x)
    # jy: tf.Tensor([[0.33333334 0.33333334 0.33333334]], shape=(1, 3), dtype=float32)
    prob = tf.nn.softmax(_prob)

    # 得到的 action 如 3, 其中各数值的含义为: 1: STOP  2: UP  3: DOWN
    # action = np.random.choice([1,2,3], p=prob.flatten())
    # action = tl.rein.choice_action_by_probs(prob.flatten(), [1, 2, 3])
    action = tl.rein.choice_action_by_probs(prob[0].numpy(), [1, 2, 3])

    # jy: observation 是一个维度为 (210, 160, 3) 的 numpy 数组
    #     reward 如 0.0
    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    # all observations in an episode
    xs.append(x)  
    # all fake labels in an episode (action begins from 1, so minus 1)
    ys.append(action - 1)  
    # all rewards in an episode
    rs.append(reward) 

    if done:
        episode_number += 1
        game_number = 0

        if episode_number % batch_size == 0:
            print('batch over...... updating parameters......')
            epx = np.vstack(xs)
            epy = np.asarray(ys)
            epr = np.asarray(rs)
            disR = tl.rein.discount_episode_rewards(epr, gamma)
            disR -= np.mean(disR)
            disR /= np.std(disR)

            xs, ys, rs = [], [], []

            with tf.GradientTape() as tape:
                _prob = model(epx)
                _loss = tl.rein.cross_entropy_reward_loss(_prob, epy, disR)
            grad = tape.gradient(_loss, train_weights)
            optimizer.apply_gradients(zip(grad, train_weights))

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was {}. running mean: {}'.format(reward_sum, running_reward))
        reward_sum = 0
        # reset env
        observation = env.reset() 
        prev_x = None

        # jy: 输出 reward 结果信息;
        print('episode %d: game %d took %.5fs, reward: %f' % (episode_number,
              game_number, time.time() - start_time, reward))
        start_time = time.time()
        game_number += 1

