"""
依赖包(gym 包的版本不对可能会导致代码运行报错):
tensorflow==2.11.0
tensorlayer==2.2.5
gym==0.25.2
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


##################### DQN ##########################

def to_one_hot(i, n_classes=None):
    """
    把分类的数字表示，变成 onehot 表示, 例如有 4 类，那么第三类变为：[0,0,1,0] 的表示
    
    构造一个矩阵，只有该分类的标志位为 1，其他全部为 0，矩阵的大小就是类别的个数。所以热独热独，热就是 1 表示的位置。例如：
    类别 1：[1 0 0 0 0 0 0 0 0....0] 
    类别 2：[0 1 0 0 0 0 0 0 0....0] 
    类别 16：[0 0 0 0 0 0 0 0 0....1]
    """
    # 先构造一个全零的矩阵, 大小是[1, n_classes]（n_classes 类别个数），然后在对应的标志位上把 0 变成 1, 并返回
    a = np.zeros(n_classes, 'uint8')
    a[i] = 1
    return a


def get_model(inputs_shape):
    """
    定义 Q 网络模型
    Define Q-network q(a,s) that ouput the rewards of 4 actions by given state, 
    i.e. Action-Value Function.
	encoding for state: 4x4 grid can be represented by one-hot vector with 16 integers.
    
    注意输入的 shape 和输出的 shape
    inputs_shape: 状态的形状 (强化学习中需要把状态输入到网络中, 所以 inputs_shape 就是状态的形状)

    代码中用了 tensorlayer, 一般来说用 tensorflow 也很方便
    注意: tensorlayer 和 tensorflow 同样的层、同样功能的参数，名字可能不一样！还有 keras 里面的 layer, 各种坑
         但后面的代码都用 tensorlayer
    """
    ni = tl.layers.Input(inputs_shape, name='observation')
    # jy: 和之前 Fasion MNIST 一样，我们用 Dense 作为输出; 这里没有多加一个 Dense 层, 主要是
    #     因为这里的状态空间只有 16 个, 这样的模型拟合能力已经足够
    #     W_init: 用正态分布 (均值 0, 方差 0.01) 的方式随机初始化参数
    nn = tl.layers.Dense(4, act=None, W_init=tf.random_uniform_initializer(0, 0.01), 
                         b_init=None, name='q_a_s')(ni)
    return tl.models.Model(inputs=ni, outputs=nn, name="Q-Network")


def save_ckpt(model, alg_name):
    """
    save trained weights (保存参数)
    """
    path = os.path.join('model', alg_name)
    if not os.path.exists(path):
        os.makedirs(path)
    tl.files.save_npz(model.trainable_weights, name=os.path.join(path, 'dqn_model.npz'))


def load_ckpt(model, alg_name):  
    """
    load trained weights (加载参数)
    """
    path = os.path.join('model', alg_name)
    tl.files.load_and_assign_npz(name=os.path.join(path, 'dqn_model.npz'), network=model)




def run(args):
    # jy: 离散
    env_id = 'FrozenLake-v1'
    # jy: 迭代次数
    num_episodes = 2000
    alg_name = 'DQN_%s' % env_id

    # e-Greedy Exploration, the larger the more random
    # jy: Epsilon-greedy 算法参数: 如果随机出来的值小于 Epsilon 这个门槛, 就用 greedy 算法
    #     越大随机性越大，越倾向于探索行为
    epsilon = 0.1
    # decay factor (折扣率)
    lambd = .99

    # 调用 get_model 函数时输入的是 [None, 16]; 
    # 注意: 前面是状态的数量, 会输入一个 batch 大小的数, 16 是 state 数量
    qnetwork = get_model([None, 16])
    qnetwork.train()
    # 调用 tensorlayer 时, 需标注这个模型是否可训练 (再次吐槽 tenorlayers...)
    train_weights = qnetwork.trainable_weights

    # 定义优化器
    optimizer = tf.optimizers.SGD(learning_rate=0.1)
    # 定义环境
    env = gym.make(env_id)

    t0 = time.time()

    # ====== 开始训练 =======
    if args.train:
        # jy: all_episode_reward 的记录结果主要用于绘图;
        all_episode_reward = []
        for i in range(num_episodes):
            # Reset environment and get first new observation (重置环境初始状态)
            # observation is state, integer 0 ~ 15
            s = env.reset()  
            rAll = 0
            # display the game environment (渲染游戏)
            #env.render()
            # step index, maximum step is 99 (最多探索 99 步; 因为环境状态比较少, 99 步一般也够探索到最终状态了)
            for j in range(99):  
                ## Choose an action by greedily (with e chance of random action) from the Q-network
                # jy: 现在有 16 个不同的 state, 如果按照一般的分类, 可能会把这 16 个 state 变成 1,2,3,...,15,16 类
                #     在机器学习里面, 0,1,2,3.. 这些数字包含了一个大小的关系, 但在分类问题里每一类都是平等的, 并没有大小
                #     关系, 因此, 我们需要把他变成热独编码（One hot）的形式; 热独编码这个名字其实很形象，就是只有一点是热的
                """
                输入为: np.asarray([to_one_hot(s, 16)], dtype=np.float32)
                神经网络其实可以看做是一个数据的工厂，而在实际写代码的过程中，这个数据工厂有时并不那么透明（并不那么可读）。
                可以说，数据工厂就像一个黑盒子一样，在 debug 中出来问题还是比较棘手的。所以要在输入之前保证这个工厂的数据
                是标准的。最好做到以下 3 点：
                    神经网络输入的形状需要和input_shape对应
                    输入格式必须是一个 array
                    数据的格式最好都变成 float32 的形式
                在输入之前能先处理好这 3 点，能够减少很多不必要的麻烦
                
				输出：numpy array
                    数据在输入神经网络这个工厂后会变成另外一种格式：tensor（张量）, 最终产生的数据格式也是 tensor
                    tensor 可以理解为工厂专用的 array, 但有时 numpy 并不能直接对 tensor 操作, 所以在后面加上 
                    ``.numpy()``, 把 tensor 转为 array
                """
                # 把 state 放入 network, 计算 Q 值
                # 输出这个状态下所有动作的 Q 值 (即一个 [None,4] 大小的矩阵)
                # 得到的 allQ 如: array([[0.00831422, 0.00901373, 0.0019527 , 0.00257436]], dtype=float32)
                allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                # 在矩阵中找最大的 Q 值的动作
                # 得到的 a 如: [1]
                a = np.argmax(allQ, 1)

                ## e-Greedy Exploration !!! sample random action
                # e-Greedy: 如果小于 epsilon, 就让智能体随机探索, 否则就用最大 Q 值的动作
                if np.random.rand(1) < epsilon:
                    # jy: env.action_space.sample() 得到的结果为一个随机抽样的 0-3 的值;
                    a[0] = env.action_space.sample()
                    
                ## Get new state and reward from environment (将 a 输入环境, 获得下一步的 state, reward, done)
                s1, r, d, _ = env.step(a[0])
                ## Obtain the Q' values by feeding the new state through our network
                #  预测下一个 state (new-state) 的所有动作的 Q 值
                Q1 = qnetwork(np.asarray([to_one_hot(s1, 16)], dtype=np.float32)).numpy()

                
                ##======= 计算 target =======
                ## Obtain maxQ' and set our target value for chosen action.
                ## 构建更新 target: 
                #    Q'(s,a) <- Q(s,a) + alpha(r + lambd * maxQ(s',a') - Q(s, a))
                # in Q-Learning, policy is greedy, so we use "max" to select the next action.
                # jy: 获取下一个状态中最大 Q 值
                maxQ1 = np.max(Q1)  
                # jy: 用 allQ (当前状态下所有动作的 Q 值) 构建更新的 target: 只有被选择那个动作(Q 值最大
                #     的动作)才会被更新到
                targetQ = allQ
                # jy: allQ 值为 batchsize 为 1 对应的 4 个状态的 Q 值结果, 因此更新时 targetQ 第一个维度选择 0
                targetQ[0, a[0]] = r + lambd * maxQ1
                
                ## Train network using target and predicted Q values
                # it is not real target Q value, it is just an estimation,
                # but check the Q-Learning update formula:
                #    Q'(s,a) <- Q(s,a) + alpha(r + lambd * maxQ(s',a') - Q(s, a))
                # 注意: minimizing |r + lambd * maxQ(s',a') - Q(s, a)|^2 equals to force Q'(s,a) ≈ Q(s,a)
                # 利用自动求导进行参数更新
                with tf.GradientTape() as tape:
                    # 把 s 放入到 Q 网络, 计算 _qvalues
                    _qvalues = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32))
                    # _qvalues 和 targetQ 的差距就是 loss, 这里衡量的尺子是 mse
                    _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)
                # 类似梯度求导对网络参数求导
                grad = tape.gradient(_loss, train_weights)
                # 应用梯度到网络参数求导
                optimizer.apply_gradients(zip(grad, train_weights))

                # 记录累计 reward (主要用于打印查看), 并且把 s 更新为 newstate
                rAll += r
                s = s1
                
                ## Reduce chance of random action if an episode is done.
                # 更新 epsilon, 让 epsilon 随着迭代次数增加而减少, 目的就是智能体越来越少进行 “探索”
                if d ==True:
                    # reduce e, GLIE: Greey in the limit with infinite Exploration
                    # jy: 每一次迭代（每一次游戏）都会调整一次 e, 随着 i 越来越大, e 将会越来越小（门槛会随着迭代
                    #     次数越来越小）, 执行 greedy 算法的机会也将会越来越多, 让智能体逐渐从探索变为开发
                    e = 1. / ((i / 50) + 10)  
                    break

            ## Note that, the rewards here with random action
            print('Training  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
                  .format(i, num_episodes, rAll, time.time() - t0))

            # all_episode_reward 记载每一次更新的总和, 为了看清变化, 大部分是前面的, 只有一部分是后面的
            if i == 0:
                all_episode_reward.append(rAll)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + rAll * 0.1)

        # save model
        save_ckpt(qnetwork, alg_name) 
        
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', alg_name))

    ## ============ 正式游戏 (和训练一样, 只是少了 epsilon-greedy) ========
    if args.test:
        # load model
        load_ckpt(qnetwork, alg_name)  
        for i in range(num_episodes):
            ## Reset environment and get first new observation
            # observation is state, integer 0 ~ 15
            s = env.reset()  
            rAll = 0
            #env.render()
            for j in range(99):  # step index, maximum step is 99
                ## Choose an action by greedily (with e chance of random action) from the Q-network
                allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                a = np.argmax(allQ, 1)  # no epsilon, only greedy for testing

                ## Get new state and reward from environment
                s1, r, d, _ = env.step(a[0])
                rAll += r
                s = s1
                ## Reduce chance of random action if an episode is done.
                if d: break

            print('Testing  | Episode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}' \
                  .format(i, num_episodes, rAll, time.time() - t0))


# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

tl.logging.set_verbosity(tl.logging.DEBUG)
if __name__ == '__main__':
    run(args)


