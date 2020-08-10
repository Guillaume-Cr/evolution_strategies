import threading
import concurrent.futures
import time
import multiprocessing as mp
import gym

def sleep(thread_index):
    print("Start sleeping for thread: %i", thread_index)
    time.sleep(2)
    print("Woke up for thread: %i", thread_index)
    return "success"

def count(thread_index):
    print("Start counting for thread: %i", thread_index)
    count = 0
    for i in range(int(10e6)):
        count += 1
    print("Finished counting for thread: %i", thread_index)
    return "success"

def callback(to_print):
    print(to_print)

def main(num_threads = 15):

    # start_seq = time.time()
    # print("####      Sequential        ####")
    # for i in range(num_threads):
    #     sleep(i)
    # print("Time needed for sequential approach: %f", time.time() - start_seq)

    print("####      Threading        ####")    
    start_thread = time.time()
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=count, args=(i,))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    print("Time needed for threading approach: %f", time.time() - start_thread)

    print("####      Thread Pooling        ####")  
    start_thread_pool = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = list()
            for i in range(num_threads):
                futures.append(executor.submit(count, i))
            for future in futures:
                return_value = future.result()
                print(return_value)
    print("Time needed for thread pool approach: %f", time.time() - start_thread)

    print("####      Multi-processing        ####")  
    start_mp = time.time()
    pool = mp.Pool()
    if(mp.cpu_count() < num_threads):
        print("Warning, trying to create more threads than available cores")
    for i in range(num_threads):
        pool.apply_async(count, args = (i, ), callback = callback)
    pool.close()
    pool.join()
    print("Time needed for multi processing approach: %f", time.time() - start_mp)

main()