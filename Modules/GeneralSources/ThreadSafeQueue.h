/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace ThreadPool
{
	/// A wrapper around the standard queue to provide thread safety.
	template <typename T>
	class CThreadSafeQueue
	{
		std::atomic_bool m_valid{ true };		/// If the queue is valid.
		mutable std::mutex m_mutex;				/// Mutex to lock the queue.
		std::queue<T> m_queue;					/// The queue itself.
		std::condition_variable m_condition;	///

	public:
		CThreadSafeQueue() = default;
		CThreadSafeQueue(const CThreadSafeQueue& _other) = delete;
		CThreadSafeQueue& operator=(const CThreadSafeQueue& _other) = delete;
		CThreadSafeQueue(CThreadSafeQueue&& _other) = default;
		CThreadSafeQueue& operator=(CThreadSafeQueue&& _other) = default;

		~CThreadSafeQueue()
		{
			Invalidate();
		}

		/// Returns the first value from the queue. Returns true if a value was successfully written to the _out parameter, false otherwise.
		bool WaitPop(T& _out)
		{
			// lock the mutex
			std::unique_lock<std::mutex> lock{ m_mutex };
			// block until a value is available, unless the instance is destroyed
			m_condition.wait(lock, [this]() { return !m_queue.empty() || !m_valid; });
			// using the condition in the predicate ensures that spurious wakeups with a valid but empty queue will not proceed, so only need to check for validity before proceeding
			if (!m_valid) return false;
			// get the first value
			_out = std::move(m_queue.front());
			m_queue.pop();
			return true;
		}

		/// Pushes a new _value onto the queue.	*/
		void Push(T _value)
		{
			// lock the mutex
			std::lock_guard<std::mutex> lock{ m_mutex };
			// put the value
			m_queue.push(std::move(_value));
			// wake one thread currently waiting for this condition
			m_condition.notify_one();
		}

		/// Invalidates the queue. Ensures no conditions are being waited on in WaitPop when a thread or the application is trying to exit. The queue is invalid after calling this.
		void Invalidate()
		{
			// lock the mutex
			std::lock_guard<std::mutex> lock{ m_mutex };
			// invalidate to unblock all threads
			m_valid = false;
			// wake up all threads currently waiting for this condition
			m_condition.notify_all();
		}
	};
}