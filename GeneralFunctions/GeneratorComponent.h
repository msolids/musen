/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "MUSENDefinitions.h"
#include <vector>

template <typename T>
class IGenerator
{
protected:
	ERunningStatus m_status{ ERunningStatus::IDLE };	// Current status of the generator: IDLE, RUNNING, etc.
	std::vector<T> m_generators;						// List of generators.
	mutable std::string m_errorMessage{ "" };			// Description of the last occured error.

public:
	IGenerator()                                        = default;
	IGenerator(const IGenerator& _other)                = default;
	IGenerator(IGenerator&& _other) noexcept            = default;
	IGenerator& operator=(const IGenerator& _other)     = default;
	IGenerator& operator=(IGenerator&& _other) noexcept = default;
	virtual ~IGenerator() {}

	/// Returns current running status.
	ERunningStatus Status() const
	{
		return m_status;
	}

	/// Sets new running status.
	void SetStatus(ERunningStatus _status)
	{
		m_status = _status;
	}

	/// Returns number of defined generators.
	size_t GeneratorsNumber() const
	{
		return m_generators.size();
	}

	/// Adds new empty generator.
	T* AddGenerator()
	{
		m_generators.push_back(T{});
		return &m_generators.back();
	}

	/// Adds new generator.
	T* AddGenerator(const T& _generator)
	{
		m_generators.push_back(_generator);
		return &m_generators.back();
	}

	/// Removes the specified generator.
	void RemoveGenerator(size_t _index)
	{
		if (_index < m_generators.size())
			m_generators.erase(m_generators.begin() + _index);
	}

	/// Moves selected generator upwards in the list of generators.
	void UpGenerator(size_t _index)
	{
		if (_index < m_generators.size() && _index != 0)
			std::iter_swap(m_generators.begin() + _index, m_generators.begin() + _index - 1);
	}

	/// Moves selected generator downwards in the list of generators.
	void DownGenerator(size_t _index)
	{
		if (_index < m_generators.size() && _index != m_generators.size() - 1)
			std::iter_swap(m_generators.begin() + _index, m_generators.begin() + _index + 1);
	}

	/// Returns const pointer to selected generator.
	const T* Generator(size_t _index) const
	{
		if (_index < m_generators.size())
			return &m_generators[_index];
		return nullptr;
	}

	/// Returns pointer to selected generator.
	T* Generator(size_t _index)
	{
		return const_cast<T*>(static_cast<const IGenerator&>(*this).Generator(_index));
	}

	/// Returns const pointers to all generators.
	std::vector<const T*> Generators() const
	{
		std::vector<const T*> generators;
		for (const auto& g : m_generators)
			generators.push_back(&g);
		return generators;
	}

	/// Returns pointers to all generators.
	std::vector<T*> Generators()
	{
		std::vector<T*> generators;
		for (auto& g : m_generators)
			generators.push_back(&g);
		return generators;
	}

	/// Returns description of the last occured error.
	std::string ErrorMessage() const
	{
		return m_errorMessage;
	}
};