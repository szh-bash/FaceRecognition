from multiprocessing import Process, Queue, Pool, RLock, Lock
import os, time, random


def worker(le, ri):
    s = 0
    for i in range(ri-le):
        for j in range(100):
            s += 1
    print(s)


if __name__ == '__main__':
    st = time.time()
    # num = 12
    # p = Pool(num)
    # for i in range(35000):
        # Process(target=worker, args=(i,)).start()
    #     p.apply_async(worker, args=(i,))
    # p.close()
    # p.join()
    p1 = Process(target=worker, args=(0, 175000))
    p2 = Process(target=worker, args=(175000, 350000))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print(time.time()-st)
    st = time.time()
    p3 = Process(target=worker, args=(0, 350000))
    p3.start()
    p3.join()
    print(time.time()-st)