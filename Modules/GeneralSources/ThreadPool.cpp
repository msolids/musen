/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ThreadPool.h"
#include "MUSENVectorFunctions.h"
#include <limits>
#include <future>
#include <iostream>
#include <string>
#ifdef __linux__
#include <pthread.h>
#include <sys/sysinfo.h>
#endif
#ifdef _WIN64
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#endif

constexpr size_t MIN_NONDEDICATED_THREADS = 8;

size_t ThreadPool::CThreadPool::m_globalThreadsLimit = std::numeric_limits<size_t>::max();
bool ThreadPool::CThreadPool::m_isSystemAffinities = false;
bool ThreadPool::CThreadPool::m_isUserAffinities = false;
std::vector<int> ThreadPool::CThreadPool::m_systemCPUList;
std::vector<int> ThreadPool::CThreadPool::m_userCPUList;

ThreadPool::CThreadPool::CThreadPool(size_t _threads)
{
	std::cout << "+---- Start creating thread pool ----+" << std::endl;
	// if number of threads not specified, calculate it
	if (_threads == 0)
		_threads = GetAllowedThreadsNumber();
	// read list of CPUs allowed by system parameters
	SetSystemCPUList();
	// match them with user parameters to get the final list
	CreateOverallCPUList();
	// final allowed number of threads
	const size_t currLimit = m_overallCPUList.empty() ? _threads : m_overallCPUList.size();
	_threads = std::min(std::min(_threads, currLimit), GetAllowedThreadsNumber());
#ifdef __linux__
	// on a Linux cluster, reduce the number of threads to allocate an empty dedicated processor to the master thread
	m_isDedicatedMaster = m_overallCPUList.size() >= _threads && _threads > MIN_NONDEDICATED_THREADS;
	if (m_isDedicatedMaster) _threads--;
#endif
	// create threads
	std::cout << " Creating thread pool ... ";
	for (size_t i = 0; i < _threads; ++i)
		m_threads.emplace_back(&CThreadPool::Worker, this);
	std::cout << "successful" << std::endl;
	// print info
	PrintCPUListsInfo();
	// set thread affinities if possible
	SetThreadAffinities();
	// print info
	PrintAffinityInfo();
	std::cout << "+----    Thread pool created     ----+" << std::endl << std::endl;
}

ThreadPool::CThreadPool::~CThreadPool()
{
	// invalidate the queue
	m_workQueue.Invalidate();
	// join all running threads
	for (auto& thread : m_threads)
		if (thread.joinable())
			thread.join();
	m_threads.clear();
}

void ThreadPool::CThreadPool::SetMaxThreadsNumber(size_t _threads)
{
	m_globalThreadsLimit = _threads;
}

size_t ThreadPool::CThreadPool::GetAllowedThreadsNumber()
{
	// use the maximum hardware threads number
	size_t threads = std::thread::hardware_concurrency();
	// if number of threads exceeds the limit, reduce to the limit
	threads = std::min(threads, m_globalThreadsLimit);
	// always create at least one thread
	threads = std::max(threads, std::size_t{ 1 });
	// return number of available threads
	return threads;
}

size_t ThreadPool::CThreadPool::GetCurrentThreadsNumber() const
{
	return m_threads.size();
}

void ThreadPool::CThreadPool::SubmitParallelJobs(size_t _count, const std::function<void(size_t)>& _fun)
{
	using FunType = std::function<void()>;

	// number of available threads
	const size_t threadsNumber = m_threads.size();
	// number of tasks per thread
	const size_t tasksPerThread = _count / threadsNumber;
	// additional tasks, arising if the number of tasks is not evenly distributable by all threads
	const size_t additionalTasks = _count % threadsNumber;

	int result_counter{ 0 };
	std::mutex wait_mutex;
	std::unique_lock<std::mutex> lock(wait_mutex);
	std::condition_variable wait_event;

	for (size_t iThread = 0; iThread < threadsNumber; ++iThread)
	{
		size_t size = tasksPerThread;
		if (additionalTasks > iThread) ++size;
		if (size == 0) break;

		result_counter++;

		// the batch task
		FunType task = [iThread, threadsNumber, size, &_fun, &result_counter, &wait_event, &wait_mutex]()
		{
			// call the _fun tasksPerThread times increasing the parameter
			for (size_t j = 0; j < size; ++j)
				_fun(threadsNumber*j + iThread);

			std::unique_lock<std::mutex> lock_task(wait_mutex);
			--result_counter;
			if (result_counter == 0)
				wait_event.notify_all();
		};

		// submit the batch task
		m_workQueue.Push(std::make_unique<CThreadTask<const FunType>>(std::move(task)));
	}

	// wait all batch tasks to finish
	while (result_counter != 0)
		wait_event.wait(lock);
}

void ThreadPool::CThreadPool::Worker()
{
	while (true)
	{
		std::unique_ptr<IThreadTask> task{ nullptr };
		if (!m_workQueue.WaitPop(task)) break; // if the queue has been invalidated, finish queuing jobs
		task->Execute();
	}
}

void ThreadPool::CThreadPool::SetSystemCPUList()
{
#ifdef __linux__
	if (m_isSystemAffinities) return;				// already read - do not repeat
	const pthread_t masterThread = pthread_self();	// get master thread
	cpu_set_t cpuset;								// mask for cpu set
	const int ec = pthread_getaffinity_np(masterThread, sizeof(cpu_set_t), &cpuset); // get cpu mask for master thread
	m_isSystemAffinities = ec == 0;
	if (m_isSystemAffinities)
	{
		// analyze and save affinity mask of master thread
		for (size_t i = 0; i < CPU_SETSIZE; ++i)
			if (CPU_ISSET(i, &cpuset))
				m_systemCPUList.push_back(i);

		// reorder threads in cyclic fashion (per socket): first even, then uneven core ids
		std::vector<unsigned> evenIDs, unevenIDs;
		for (int cpu : m_systemCPUList)
			if (cpu % 2 == 1)
				unevenIDs.push_back(cpu);
			else
				evenIDs.push_back(cpu);
		m_systemCPUList.clear();
		m_systemCPUList.insert(m_systemCPUList.end(), evenIDs.begin(), evenIDs.end());
		m_systemCPUList.insert(m_systemCPUList.end(), unevenIDs.begin(), unevenIDs.end());
	}
#elif _WIN64
	m_systemCPUList.clear();
	const unsigned maxHardwareCores = std::thread::hardware_concurrency();
	const HANDLE masterThread = GetCurrentProcess();   // get master thread
	DWORD_PTR processAffinityMask, systemAffinityMask; // masks
	const DWORD_PTR ec = GetProcessAffinityMask(masterThread, &processAffinityMask, &systemAffinityMask);  // get cpu mask for master thread
	m_isSystemAffinities = ec != 0;
	if (m_isSystemAffinities) // analyze and save cores IDs from master thread
		for (int i = maxHardwareCores - 1; i >= 0; --i)
			if (std::size_t{ 1 } << i & processAffinityMask)
				m_systemCPUList.push_back(i);
#endif
}

void ThreadPool::CThreadPool::SetUserCPUList(const std::vector<int>& _cores)
{
	m_userCPUList.clear();
	// set flag
	m_isUserAffinities = !_cores.empty();
	if (_cores.empty())	return;
	// analyze and save cores IDs from user
	m_userCPUList = _cores;
}

std::vector<int> ThreadPool::CThreadPool::GetUserCPUList()
{
	return m_userCPUList;
}

std::vector<int> ThreadPool::CThreadPool::GetSystemCPUList()
{
	return m_systemCPUList;
}

void ThreadPool::CThreadPool::CreateOverallCPUList()
{
	if (m_isUserAffinities && m_isSystemAffinities)
	{
		// create union of both lists in order defined by master thread
		std::vector<int> matches;
		for (int iMaster : m_systemCPUList)
			for (int iUser : m_userCPUList)
				if (iMaster == iUser)
					matches.push_back(iMaster);

		if (!matches.empty()) // if there are matches
			m_overallCPUList = matches;
		else // if no matches are present use master process affinities, up to available threads number
			m_overallCPUList = m_systemCPUList;
	}
	else if (m_isUserAffinities)
		m_overallCPUList = m_userCPUList;
	else if (m_isSystemAffinities)
		m_overallCPUList = m_systemCPUList;
	else
		m_overallCPUList.clear();
}

void ThreadPool::CThreadPool::SetThreadAffinities()
{
	// clear list of previous pinned processors
	m_pinnedCPUs.clear();

	if (m_overallCPUList.size() < m_threads.size())	return; // not enough information for pinning is present

#ifdef __linux__
	cpu_set_t cpuset; // mask for cpu set
	// pin threads to cores
	for (size_t i = 0; i < m_threads.size(); ++i)
	{
		CPU_ZERO(&cpuset);
		CPU_SET(m_overallCPUList[i], &cpuset);
		const int ec = pthread_setaffinity_np(m_threads[i].native_handle(), sizeof(cpu_set_t), &cpuset);
		if (ec == 0) // success
			m_pinnedCPUs.push_back(m_overallCPUList[i]);
	}

	// pin master thread to the last core in the list
	if (m_isDedicatedMaster)	// on a Linux cluster, allocate an empty dedicated core for master thread
		m_masterCPU = m_overallCPUList[m_threads.size()];
	else						// otherwise just take the last occupied one
		m_masterCPU = m_overallCPUList[m_threads.size() - 1];
	// pin master thread
	CPU_ZERO(&cpuset);
	CPU_SET(m_masterCPU, &cpuset);
	const int ec = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
	if (ec != 0) // error
		m_masterCPU = -1;
#elif _WIN64
	// pin threads to cores
	for (size_t i = 0; i < m_threads.size(); ++i)
	{
		// fill from beginning to run master thread on same core as last thread
		//const DWORD_PTR mask = std::size_t{ 1 } << m_overallCPUList[m_threads.size() - 1 - i]; // mask for cpu set
		const DWORD_PTR mask = std::size_t{ 1 } << m_overallCPUList[i]; // mask for cpu set
		const DWORD_PTR ec = SetThreadAffinityMask(m_threads[i].native_handle(), mask);
		if (ec != 0) // success
			//m_pinnedCPUs.push_back(m_overallCPUList[m_threads.size() - 1 - i]);
			m_pinnedCPUs.push_back(m_overallCPUList[i]);
	}

	// TODO: turn on only for simulation and turn off for GUI
	// pin master thread to the first core in the list
	//m_masterCPU = m_overallCPUList.front();
	//const DWORD_PTR mask = std::size_t{ 1 } << m_masterCPU;
	//const DWORD_PTR ec = SetThreadAffinityMask(GetCurrentThread(), mask);
	//if (ec == 0) // error
	//	m_masterCPU = -1;
#endif
}

void ThreadPool::CThreadPool::PrintCPUListsInfo() const
{
	if (m_overallCPUList.size() < m_threads.size())
	{
		std::cout << " === SERIOUS PERFORMANCE WARNING ===" << std::endl;
		std::cout << " === > Cannot determine affinities for threads. Threads will not be pinned, and performance will be significantly reduced." << std::endl;
	}

	// lists
	if (m_isSystemAffinities)
		PrintCPUList(m_systemCPUList, "Cores allowed by system");
	if (m_isUserAffinities)
		PrintCPUList(m_userCPUList, "Cores allowed by user");

	// reduction
	if (m_isUserAffinities && m_isSystemAffinities)
		if (m_overallCPUList == m_systemCPUList && !IsSubset(m_userCPUList, m_systemCPUList))
			std::cout << " No matches were found between system and user parameters for affinities. Using system parameters." << std::endl;
		else
			std::cout << " Matches were found between system and user process affinities. Using union of them." << std::endl;
	else if (m_isUserAffinities)
		std::cout << " No system parameters for affinities present. Using user parameters." << std::endl;
	else if (m_isSystemAffinities)
		std::cout << " No user parameters for affinities present. Using system parameters." << std::endl;

	if (m_isUserAffinities && m_isSystemAffinities)
		PrintCPUList(m_overallCPUList, "Cores allowed by matching system and user parameters");

	const size_t available = m_isDedicatedMaster ? m_systemCPUList.size() - 1 : m_systemCPUList.size();
	if (!m_systemCPUList.empty() && m_threads.size() < available)
		std::cout << " Warning: Number of parallel threads was reduced from available " << available << " to " << m_threads.size() << "." << std::endl;

	std::cout << " Number of parallel threads created: " << m_threads.size() << std::endl;
}

void ThreadPool::CThreadPool::PrintAffinityInfo() const
{
	// TODO: remove ifdef when pinning for windows is back

	if (!m_pinnedCPUs.empty())
		PrintCPUList(m_pinnedCPUs, "Threads pinned to cores");

	if (m_masterCPU != -1)
		std::cout << " Master thread is on core " << m_masterCPU << std::endl;
	if (
#ifdef __linux__
		m_masterCPU != -1 &&
#endif
		m_pinnedCPUs.size() == m_threads.size())
		std::cout << " Affinities set successfully" << std::endl;
	else
	{
#ifdef __linux__
		if (m_masterCPU == -1)
			std::cout << " Error setting affinity for master thread." << std::endl;
#endif
		if (m_pinnedCPUs.size() != m_threads.size())
			for (size_t i = 0; i < m_threads.size(); ++i)
				if (!VectorContains(m_pinnedCPUs, m_overallCPUList[i]))
					std::cout << " Error setting affinity of thread " << i << " to core " << m_overallCPUList[i] << std::endl;
		std::cout << " Errors occured during setting affinities. Performance may be significantly reduced." << std::endl;
	}
}

void ThreadPool::CThreadPool::PrintCPUList(const std::vector<int>& _cpuList, const std::string& _message)
{
	std::cout << " " << _message << ": ";
	for (int cpu : _cpuList)
		std::cout << cpu << " ";
	std::cout << std::endl;
}
