/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ThreadSafeQueue.h"
#include "ThreadTask.h"
#include <functional>
#include <thread>

namespace ThreadPool
{
	class CThreadPool
	{
		static size_t m_globalThreadsLimit;							/// Maximum number of threads available for this instance of program.
		CThreadSafeQueue<std::unique_ptr<IThreadTask>> m_workQueue;	/// Queue of submitted works.
		std::vector<std::thread> m_threads;							/// List of available threads.

		static bool m_isSystemAffinities;			/// Master process successfully set affinities taken from system. Takes precedence over user settings.
		static bool m_isUserAffinities;				/// User has specified cores that system is supposed to run on.
		static std::vector<int> m_systemCPUList;	/// Core IDs on which threads are allowed to run, set by system. Takes precedence over user settings.
		static std::vector<int> m_userCPUList;		/// Core IDs on which threads are allowed to run, set by user.
		std::vector<int> m_overallCPUList;			/// Final overall core IDs on which threads are allowed to run.

		bool m_isDedicatedMaster{ false };			/// Whether to pin master thread to an empty dedicated core.
		std::vector<int> m_pinnedCPUs;				/// List of CPUs to which threads were actually pinned.
		int m_masterCPU{ -1 };						/// CPU to which the master thread was pinned.

	public:
		explicit CThreadPool(size_t _threads = 0);

		~CThreadPool();
		CThreadPool(const CThreadPool& _other) = delete;
		CThreadPool& operator=(const CThreadPool& _other) = delete;
		CThreadPool(CThreadPool&& _other) = delete;
		CThreadPool& operator=(CThreadPool&& _other) = delete;

		/// Specifies the maximum number of threads that can be generated for this instance of program.
		static void SetMaxThreadsNumber(size_t _threads);
		/// Returns number of threads allowed on this hardware for this instance of program.
		static size_t GetAllowedThreadsNumber();
		/// Returns number of currently defined threads, for this instance of thread pool.
		size_t GetCurrentThreadsNumber() const;

		/// Creates the list of cores to run on. Information is taken from the user.
		static void SetUserCPUList(const std::vector<int>& _cores);
		/// Returns the supposed list of cores to run on. Information is taken from the user.
		static std::vector<int> GetUserCPUList();
		/// Returns the supposed list of cores to run on. Information is taken from the system or starting parameters.
		static std::vector<int> GetSystemCPUList();

		/// Submits _count of identical jobs, running _fun(i) _count times with i = [0; count).
		void SubmitParallelJobs(size_t _count, const std::function<void(size_t)>& _fun);

	private:
		/// Constantly running function, which each thread uses to acquire work items from the queue.
		void Worker();

		/// Creates the list of cores to run on. Information is taken from the system or starting parameters.
		static void SetSystemCPUList();
		/// Compose the list of cores from system and user input, creating the final list.
		void CreateOverallCPUList();
		/// Sets thread affinities according to m_overallCPUList, if possible.
		void SetThreadAffinities();

		/// Prints information about processors chosen system and user.
		void PrintCPUListsInfo() const;
		/// Prints information about applied affinities.
		void PrintAffinityInfo() const;
		/// Print out the list of core IDs.
		static void PrintCPUList(const std::vector<int>& _cpuList, const std::string& _message);
	};
}

/// Default thread pool always ready to execute tasks.
inline ThreadPool::CThreadPool& GetThreadPool()
{
	static ThreadPool::CThreadPool pool;
	return pool;
}

/// Initialize thread pool with current parameters.
inline void InitializeThreadPool()
{
	GetThreadPool();
}

/// Restart the default thread pool with new settings.
inline void RestartThreadPool()
{
	GetThreadPool().~CThreadPool();
	new(&GetThreadPool()) ThreadPool::CThreadPool();
}

/// Submits _count of identical jobs, running function _fun(i) _count times with i = [0; _count).
inline void ParallelFor(size_t _count, const std::function<void(size_t)>& _fun)
{
	return GetThreadPool().SubmitParallelJobs(_count, _fun);
}

/// Submits identical jobs, in an amount equal to the number of available threads (N), running function _fun(i) N times with i = [0; N).
inline void ParallelFor(const std::function<void(size_t)>& _fun)
{
	return GetThreadPool().SubmitParallelJobs(GetThreadPool().GetCurrentThreadsNumber(), _fun);
}

/// Returns number of defined threads.
inline size_t GetThreadsNumber()
{
	return GetThreadPool().GetCurrentThreadsNumber();
}
