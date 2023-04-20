import time
import tensorflow as tf
import threading
import queue

N_WORKER = 4            # worker 的数量
QUEUE = queue.Queue()   # 队列, 用于储存数据
EP_MAX = 10             # 执行 EP
EP_LEN = 200            # 每个 EP 的最大步数
MIN_BATCH_SIZE = 10     # 每个 batch 的大小


class Worker():
    def __init__(self, wid):
        # worker 对象的 id; 该程序只是模拟, 所以在填入数据时会直接把 wid 放
        # 入队列表示该 worker 产生的数据
        self.wid = wid              

    def work(self):
        global GLOBAL_EP, GLOBAL_UPDATE_COUNTER
        
        # 判断是否所有线程都应该停止了
        while not COORD.should_stop():
            # 开始新的 EP
            for _ in range(EP_LEN): 

                # 如果有其他 worker 线程已经被阻塞, 则其他线程也需要在这等待
                #if not ROLLING_EVENT.is_set():   
                ROLLING_EVENT.wait()

                # 这里做了简化, 直接把 worker 的 id 当做和环境互动产生的数据放入队列中
                # 实际上, 这里会用 buffer 记录智能体和环境互动产生的数据, 当数据大于 
                # MIN_BATCH_SIZE 才开始整理数据
                QUEUE.put(self.wid)

                # GLOBAL_UPDATE_COUNTER+1: 表示有智能体走了一步了
                GLOBAL_UPDATE_COUNTER += 1      


                if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE: 
                    '''
                    这里可以插入整理数据部分
                    '''
                    ROLLING_EVENT.clear()
                    UPDATE_EVENT.set()
                        
                if GLOBAL_EP >= EP_MAX:          
                    COORD.request_stop()
                    break

            


class PPO(object):
    
    def update(self):
        global GLOBAL_UPDATE_COUNTER
        
        # 判断是否所有线程都应该停止了
        while not COORD.should_stop():
            if GLOBAL_EP <= EP_MAX:
                UPDATE_EVENT.wait()

                # 此处用输出表示更新
                print("====update====")
                print("GLOBAL_EP", GLOBAL_EP)
                print("GLOBAL_UPDATE_COUNTER:", GLOBAL_UPDATE_COUNTER)
                print("update_old_pi")
                print("Queuesize:", QUEUE.qsize())
                print([QUEUE.get() for _ in range(QUEUE.qsize())])
                print("update Critic")
                print("update Actor")
                print("=====END======")
                time.sleep(1)

                GLOBAL_UPDATE_COUNTER = 0

                UPDATE_EVENT.clear()
                ROLLING_EVENT.set()       


if __name__ == "__main__":
    # 创建 worker 对象
    workers = [Worker(wid=i) for i in range(N_WORKER)]

    # 创建 PPO 对象
    GLOBAL_PPO = PPO()

    # 新建两个 event: UPDATE_EVENT, ROLLING_EVENT
    UPDATE_EVENT,ROLLING_EVENT = threading.Event(), threading.Event()
    # 把 UPDATE_EVENT 的信号设置为阻塞
    UPDATE_EVENT.clear()
    # 把 ROLLING_EVENT 的信号设置为就绪
    ROLLING_EVENT.set()

    # 定义两个全局变量: GLOBAL_UPDATE_COUNTER(每次更新 + 1)、GLOBAL_STEP
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    threads = []

    # 创建协调器
    COORD = tf.train.Coordinator()

    # 开启 rolling 线程
    for worker in workers:                          
        # 三个 rolling 线程, 线程的功能就是执行 work 函数
        t = threading.Thread(target=worker.work)    
        t.start()                                   
        threads.append(t)
    
    # 开启 update 线程, 执行 PPO 的 update 函数
    threads.append(threading.Thread(target=GLOBAL_PPO.update,)) 
    # 启动最后加入的线程(即 update 线程)
    threads[-1].start()                 
    
    # 加入协调器
    COORD.join(threads)


