/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <istream>
#undef TRUE
#undef FALSE

/// Tri-state boolean.
class CTriState
{
public:
	enum class EState { UNDEFINED, TRUE, FALSE };

private:
	EState m_state;

public:
	CTriState();
	explicit CTriState(EState _state);
	explicit CTriState(bool _flag);

	CTriState& operator=(bool _flag);

	explicit operator bool() const = delete;
	bool operator==(bool _flag) const;

	bool IsDefined() const;
	bool ToBool(bool _undefinedValue = false) const;	// Converts to bool taking _undefinedValue as UNDEFINED value.

	friend std::istream& operator>>(std::istream& _s, CTriState& _v);
};

