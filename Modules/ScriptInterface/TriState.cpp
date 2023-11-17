/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "TriState.h"
#include "MUSENStringFunctions.h"
#include <cassert>

CTriState::CTriState() : m_state{ EState::UNDEFINED } {}

CTriState::CTriState(EState _state) : m_state{_state} {}

CTriState::CTriState(bool _flag) : m_state{ _flag ? EState::TRUE : EState::FALSE } {}

CTriState& CTriState::operator=(bool _flag)
{
	m_state = _flag ? EState::TRUE : EState::FALSE;
	return *this;
}

bool CTriState::operator==(bool _flag) const
{
	return _flag && m_state == EState::TRUE || !_flag && m_state == EState::FALSE;
}

bool CTriState::IsDefined() const
{
	return m_state != EState::UNDEFINED;
}

bool CTriState::ToBool(bool _undefinedValue/* = false*/) const
{
	if (_undefinedValue == false)
		return m_state == EState::TRUE;
	else
		return m_state != EState::FALSE;
}

std::istream& operator>>(std::istream& _s, CTriState& _v)
{
	const std::string str = GetValueFromStream<std::string>(&_s);
	if (str == "1" || ToLowerCase(str) == "yes" || ToLowerCase(str) == "true")
		_v.m_state = CTriState::EState::TRUE;
	else if (str == "0" || ToLowerCase(str) == "no" || ToLowerCase(str) == "false")
		_v.m_state = CTriState::EState::FALSE;
	else
		_v.m_state = CTriState::EState::UNDEFINED;
	return _s;
}
