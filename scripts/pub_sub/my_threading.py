import threading, time

def worker(num):
    """thread worker function"""
    print ('Worker: %s' % num)
    return

def worker2(num):
    """thread worker function"""
    print ('Worker2: %s' % num)
    return
    
threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    s = threading.Thread(target=worker2, args=(i,))
    threads.append(t)
    threads.append(s)
    t.start()
    s.start()

# from multiprocessing import Process

# def func1():
#     print('start1')
#     for i in range(1000000):pass
#     print('end1')

# def func2():
#     print('start2')
#     for i in range(1000000):pass
#     print('end2')

# if __name__ == "__main__":
#     p1 = Process(target = func1)
#     p2 = Process(target = func2)
#     p1.start()
#     p2.start()
#     p1.join()
#     p2.join()