#include <iostream>
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

// ThreadPool class to manage a fixed number of threads executing tasks
class ThreadPool {
public:
    // Constructor initializes the thread pool with a specified number of threads
    ThreadPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            // Add a worker thread to the pool. Each thread runs the lambda function.
            // [this] captures the ThreadPool instance to allow access to its members.
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task; // Task placeholder
                    {
                        // Lock the task queue so only one thread can modify it at a time
                        std::unique_lock<std::mutex> lock(queueMutex);

                        // Wait for a condition: either stop is true or the task queue is not empty
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });

                        // If stopping and no tasks are left, exit the thread
                        if (stop && tasks.empty()) return;

                        // Retrieve the next task from the queue
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    // Execute the retrieved task outside the lock
                    task();
                }
            });
        }
    }

    // Destructor ensures threads are stopped gracefully
    ~ThreadPool() {
        {
            // Lock the queue and signal that the pool is stopping
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }

        // Notify all worker threads to wake up and exit
        condition.notify_all();

        // Wait for all threads to finish their execution
        for (auto &worker : workers) worker.join();
    }

    // Adds a task to the task queue for execution by the thread pool
    void enqueueTask(std::function<void()> task) {
        {
            // Lock the queue and add the new task
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.push(std::move(task));
        }
        // Notify one worker thread that a task is available
        condition.notify_one();
    }

private:
    std::vector<std::thread> workers;             // Vector of worker threads
    std::queue<std::function<void()>> tasks;      // Queue of tasks to execute
    std::mutex queueMutex;                        // Mutex to protect the task queue
    std::condition_variable condition;            // Condition variable for task availability
    bool stop = false;                            // Flag to signal stopping of the pool
};

// MiniOMP class provides a simplified API for parallel execution
class MiniOMP {
public:
    // Constructor initializes the thread pool with the given number of threads
    MiniOMP(size_t numThreads) : pool(numThreads) {}

    // Executes a function `func` in parallel for `numIterations` iterations
    void parallel(std::function<void(int)> func, int numIterations) {
        for (int i = 0; i < numIterations; ++i) {
            // Enqueue each iteration as a separate task
            // [=] captures all variables from the surrounding scope by value
            pool.enqueueTask([=] { func(i); });
        }
    }

private:
    ThreadPool pool; // ThreadPool instance to manage execution
};

// Global mutex for protecting output from simultaneous access by multiple threads
std::mutex outputMutex;

int main() {

    int numThreads = 2;

    // Create a MiniOMP instance with the specified number of threads
    MiniOMP omp(numThreads);

    // Parallelly execute a function for 10 iterations
    omp.parallel([](int i) {
        // Lock the output stream to prevent mixed output from multiple threads
        std::lock_guard<std::mutex> lock(outputMutex);
        std::cout << "Running iteration " << i << " on thread " << std::this_thread::get_id() << "\n";
    }, 10);

    return EXIT_SUCCESS;
}
