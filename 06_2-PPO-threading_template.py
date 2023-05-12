import time
import tensorflow as tf
import threading
import queue

"""
tf 结合 threading 的多线程示例: 多线程应用于 PPO 的框架代码
"""

class Worker():
    def __init__(self, wid, coord, max_ep, queue, min_batch_size=10):
        # worker 对象的 id; 该程序只是模拟, 所以在填入数据时会直接把 wid 放
        # 入队列表示该 worker 产生的数据
        self.wid = wid
        self.coord = coord
        self.max_ep = max_ep
        self.queue = queue
        self.min_batch_size = min_batch_size

    def work(self):
        global GLOBAL_UPDATE_COUNTER, GLOBAL_EP
     
        print("jy ---------- ")
        # 判断是否所有线程都应该停止了
        while not self.coord.should_stop():
            # 开始新的 EP, 每个 EP 最大 200 步
            for _ in range(200): 
                print("jy ------222---- ")
                # 如果有其他 worker 线程已经被阻塞, 则其他线程也需要在这等待
                if not ROLLING_EVENT.is_set():   
                    ROLLING_EVENT.wait()
                #ROLLING_EVENT.wait()

                # 这里做了简化, 直接把 worker 的 id 当做和环境互动产生的数据放入队列中
                # 实际上, 这里会用 buffer 记录智能体和环境互动产生的数据, 当数据大于 
                # min_batch_size 才开始整理数据
                self.queue.put(self.wid)

                # GLOBAL_UPDATE_COUNTER+1: 表示有智能体走了一步了
                GLOBAL_UPDATE_COUNTER += 1      


                if GLOBAL_UPDATE_COUNTER >= self.min_batch_size:
                    """
                    这里可以插入整理数据部分
                    """
                    ROLLING_EVENT.clear()
                    UPDATE_EVENT.set()
                        
                if GLOBAL_EP >= self.max_ep:          
                    self.coord.request_stop()
                    break

            # jy-add: 必须补充此处, 否则 GLOBAL_EP 一直为 0, 程序不为停止;
            GLOBAL_EP += 1
            print("jy-add: GLOBAL_EP += 1")


class PPO(object):
    def __init__(self, coord, max_ep, queue):
        self.coord = coord
        self.max_ep = max_ep
        self.queue = queue

    def update(self):
        global GLOBAL_UPDATE_COUNTER, GLOBAL_EP
        
        # 判断是否所有线程都应该停止了
        while not self.coord.should_stop():
            if GLOBAL_EP <= self.max_ep:
                UPDATE_EVENT.wait()

                # 此处用输出表示更新
                print("====update====")
                print("GLOBAL_EP", GLOBAL_EP)
                print("GLOBAL_UPDATE_COUNTER:", GLOBAL_UPDATE_COUNTER)
                print("update_old_pi")
                print("Queuesize:", self.queue.qsize())
                print([self.queue.get() for _ in range(self.queue.qsize())])
                print("update Critic")
                print("update Actor")
                print("=====END======")
                # jy: 设置暂停时间, 方便查看
                #time.sleep(0.2)

                GLOBAL_UPDATE_COUNTER = 0

                UPDATE_EVENT.clear()
                ROLLING_EVENT.set()

        # jy: 能正常执行结束的时候才会执行到此处;
        import pdb; pdb.set_trace()
        print("jy===============")


def run(coord, max_ep, queue):
    """
    coord: 协调器
    max_ep: 最大 ep 数;
    """
    # 创建 Worker 对象
    workers = [Worker(i, coord, max_ep, queue) for i in range(4)]

    # 创建 PPO 对象
    ppo = PPO(coord, max_ep, queue)

    threads = []

    # 开启 rolling 线程
    for worker in workers:                          
        # rolling 线程, 功能是执行 Worker 类的 work 方法;
        t = threading.Thread(target=worker.work)    
        # jy: 启动线程
        t.start()                                   
        threads.append(t)
    
    #import pdb; pdb.set_trace()
    # 开启 update 线程, 执行 PPO 的 update 函数
    threads.append(threading.Thread(target=ppo.update,)) 
    # 启动最后加入的线程(即 PPO 中的 update 方法对应的线程)
    threads[-1].start()                 
    
    # 将所有线程加入协调器
    coord.join(threads)



# 用于储存数据的队列
queue = queue.Queue()
# 创建协调器
coord = tf.train.Coordinator()
max_ep = 10
GLOBAL_EP = 0 
# GLOBAL_UPDATE_COUNTER(每次更新 + 1)
GLOBAL_UPDATE_COUNTER = 0

# 新建两个 event: UPDATE_EVENT, ROLLING_EVENT
UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
# 把 UPDATE_EVENT 的信号设置为阻塞
UPDATE_EVENT.clear()
# 把 ROLLING_EVENT 的信号设置为就绪
ROLLING_EVENT.set()

if __name__ == "__main__":
    run(coord, max_ep, queue)





