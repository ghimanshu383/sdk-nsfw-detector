//
// Created by ghima on 23-11-2025.
//
#include <vector>
#include <thread>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <queue>

#ifndef OSFEATURENDKDEMO_THREADPOOL_H
#define OSFEATURENDKDEMO_THREADPOOL_H
namespace ml {
    class ThreadPool {
    public:
        explicit ThreadPool(size_t poolSize);

        template<typename T>
        void enqueue_task(T &&task) {
            {
                std::lock_guard<std::mutex> lock_guard{mutex_};
                m_tasks.emplace(std::forward<T>(task));
            }
            m_cv.notify_one();
        }

        void init_threads();

        void join_all();
        void detach_all();

        ~ThreadPool();

    private:
        size_t m_poolSize;
        std::queue<std::function<void()>> m_tasks;
        std::vector<std::thread> m_threads;
        std::condition_variable m_cv;
        std::mutex mutex_;
        bool m_stop = false;
    };
}
#endif //OSFEATURENDKDEMO_THREADPOOL_H
