from multiprocessing import Process, Queue, Pool, RLock, Lock
import os, time, random


def worker(le, ri):
    s = 0
    for i in range(le, ri):
        for j in range(100):
            s += 1
    # print(s)


if __name__ == '__main__':
    st = time.time()
    num = 8
    # p = Pool(num)
    q = []
    for i in range(num):
        q.append(Process(target=worker, args=(350000//num*i, 350000//num*(i+1))))
        # p.apply_async(worker, args=(10*i, 10*(i+1)))
    for i in range(num):
        q[i].start()
    for i in range(num):
        q[i].join()
    # p.close()
    # p.join()
    print(time.time()-st)
    # st = time.time()
    # p1 = Process(target=worker, args=(0, 175000))
    # p2 = Process(target=worker, args=(175000, 350000))
    # p1.start()
    # p2.start()
    # p1.join()
    # p2.join()
    # print(time.time()-st)
    # st = time.time()
    # p3 = Process(target=worker, args=(0, 350000))
    # p3.start()
    # p3.join()
    # print(time.time()-st)
