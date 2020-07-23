/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include <utility>
#include <thread>

namespace ThreadPool
{
	/// An interface of the task to be able to operate a collection of tasks through one container.
	class IThreadTask
	{
	public:
		IThreadTask() = default;
		virtual ~IThreadTask() = default;
		IThreadTask(const IThreadTask& _other) = delete;
		IThreadTask& operator=(const IThreadTask& _other) = delete;
		IThreadTask(IThreadTask&& _other) = default;
		IThreadTask& operator=(IThreadTask&& _other) = default;

		/// Run the task.
		virtual void Execute() = 0;
	};

	/// Description of the task itself.
	template <typename Func>
	class CThreadTask : public IThreadTask
	{
		Func m_function;

	public:
		CThreadTask(Func&& _func) : m_function{ std::move(_func) } { }
		~CThreadTask() override = default;
		CThreadTask(const CThreadTask& _other) = delete;
		CThreadTask& operator=(const CThreadTask& _other) = delete;
		CThreadTask(CThreadTask&& _other) = default;
		CThreadTask& operator=(CThreadTask&& _other) = default;

		/// Run the task
		void Execute() override
		{
			m_function();
		}
	};
}