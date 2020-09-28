/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "QtListSpinBox.h"
#include <QLineEdit>

CQtListSpinBox::CQtListSpinBox(const std::vector<int>& _allowed, QWidget* _parent)
	: QSpinBox{ _parent }
{
	// block direct edit
	lineEdit()->setReadOnly(true);

	SetList(_allowed);
}

void CQtListSpinBox::SetList(const std::vector<int>& _allowed)
{
	m_list = _allowed;

	// set limits
	if (!m_list.empty())
	{
		setMinimum(m_list.front());
		setMaximum(m_list.back());
	}
	else
	{
		setMinimum(std::numeric_limits<int>::min());
		setMaximum(std::numeric_limits<int>::max());
	}
}

void CQtListSpinBox::SetValue(int _value)
{
	if (m_list.empty() || std::find(m_list.begin(), m_list.end(), _value) != m_list.end())
		setValue(_value);
}

void CQtListSpinBox::SetEditable(bool _flag)
{
	setSingleStep(_flag ? 1 : 0);
	setButtonSymbols(_flag ? UpDownArrows : NoButtons);
}

void CQtListSpinBox::stepBy(int _steps)
{
	if (m_list.empty()) return;
	// find current value
	const auto it = std::find(m_list.begin(), m_list.end(), value());
	// not found - something went wrong
	if (it == m_list.end())
	{
		setValue(m_list.front());
		return;
	}
	// index of current value
	const auto index = std::distance(m_list.begin(), it);
	// check whether new value is outside the range
	if (_steps < 0 && index + _steps < 0)
	{
		setValue(m_list.front());
		return;
	}
	if (_steps > 0 && index + _steps >= static_cast<decltype(index)>(m_list.size()))
	{
		setValue(m_list.back());
		return;
	}
	// set new value
	setValue(m_list[index + _steps]);
}

QAbstractSpinBox::StepEnabled CQtListSpinBox::stepEnabled() const
{
	return StepUpEnabled | StepDownEnabled;
}
