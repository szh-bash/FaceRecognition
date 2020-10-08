from multiprocessing import Process, Queue, Pool, RLock, Lock
import os, time, random


def worker(le, ri):
    s = 0
    for i in range(le, ri):
        for j in range(100):
            s += 1
    # print(s)


if __name__ == '__main__':
    num = 8
    print('process-num ==', num)
    st = time.time()
    q = []
    for i in range(num):
        q.append(Process(target=worker, args=(350000//num*i, 350000//num*(i+1))))
    for i in range(num):
        q[i].start()
    for i in range(num):
        q[i].join()
    print('process-list: %.3fs' % (time.time()-st))
    st = time.time()
    p = Pool(num)
    for i in range(num):
        p.apply_async(worker, args=(350000//num*i, 350000//num*(i+1)))
    p.close()
    p.join()
    print('process-pool: %.3fs' % (time.time()-st))
    st = time.time()
    p3 = Process(target=worker, args=(0, 350000))
    p3.start()
    p3.join()
    print('single-process: %.3fs' % (time.time()-st))
