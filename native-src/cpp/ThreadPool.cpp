//
// Created by ghima on 23-11-2025.
//
#include "ThreadPool.h"

namespace ml {
    ThreadPool::ThreadPool(size_t poolSize) : m_poolSize{poolSize} {
        init_threads();
    }

    void ThreadPool::init_threads() {
        for (int i = 0; i < m_poolSize; i++) {
            m_threads.emplace_back([this]() -> void {
                std::function<void()> task;
                while (true) {

                    {
                        std::unique_lock<std::mutex> u_lock{mutex_};
                        m_cv.wait(u_lock, [this]() -> bool {
                            return m_stop || !this->m_tasks.empty();
                        });

                        if (m_tasks.empty() && m_stop) return;
                        task = std::move(m_tasks.front());
                        m_tasks.pop();
                    }
                    task();

                }
            });
        }
    }

    void ThreadPool::join_all() {

        {
            std::lock_guard<std::mutex> lock_{mutex_};
            m_stop = true;
        }
        m_cv.notify_all();

        for (std::thread &thread: m_threads) {
            if (thread.joinable()) thread.join();
        }
    }

    ThreadPool::~ThreadPool() {
        join_all();
    }

    void ThreadPool::detach_all() {
        {
            std::lock_guard<std::mutex> lock_{mutex_};
            m_stop = true;
        }
        m_cv.notify_all();

        for (std::thread &thread: m_threads) {
            if (thread.joinable()) thread.detach();
        }
    }
}