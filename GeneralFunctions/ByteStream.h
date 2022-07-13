/* Copyright (c) 2013-2022, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <vector>
#include <stdexcept>
#include <cstring>

/*
 * Simple in-memory byte stream. Works only with POD.
 */
class CByteStream
{
	std::vector<char> m_data; // Internal data representation.
	size_t m_iRd{ 0 };        // Current read index.

public:
	// Writes data into stream.
	template<typename T>
	void Write(const T& _val)
	{
		static_assert(std::is_trivial_v<T>, "T must be trivial");
		m_data.resize(m_data.size() + sizeof(T));
		std::memcpy(&m_data[m_data.size() - sizeof(T)], reinterpret_cast<const void*>(&_val), sizeof(T));
	}

	// Writes data into stream.
	template<typename T>
	friend CByteStream operator<<(CByteStream& _s, const T& _val)
	{
		_s.Write(_val);
		return _s;
	}

	// Reads data from stream.
	template<typename T>
	T Read()
	{
		static_assert(std::is_trivial_v<T>, "T must be trivial");
		if (m_iRd >= m_data.size() || m_iRd + sizeof(T) > m_data.size())
			throw std::runtime_error("Out of bounds read");
		T res;
		std::memcpy(reinterpret_cast<void*>(&res), &m_data[m_iRd], sizeof(T));
		m_iRd += sizeof(T);
		return res;
	}

	// Shifts read position to the right on the specified number of bytes.
	void Ignore(size_t _shift)
	{
		m_iRd += _shift;
	}

	// Reads data from stream.
	template<typename T>
	friend CByteStream& operator>>(CByteStream& _s, T& _val)
	{
		_val = _s.Read<T>();
		return _s;
	}

	// Returns internal bytes representation of the data.
	[[nodiscard]] const std::vector<char>& GetDataRef() const
	{
		return m_data;
	}

	// Returns internal bytes representation of the data.
	[[nodiscard]] std::vector<char>& GetDataRef()
	{
		return m_data;
	}

	// Changes current length in bytes.
	void Resize(size_t _size)
	{
		return m_data.resize(_size);
	}

	// Returns length in bytes of the representation of the data.
	[[nodiscard]] size_t Size() const
	{
		return m_data.size();
	}
};
