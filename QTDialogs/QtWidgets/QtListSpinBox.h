/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <QSpinBox>

// Spin box allowing only the selected values from the list.
class CQtListSpinBox : public QSpinBox
{
	Q_OBJECT

	std::vector<int> m_list;	// List of allowed values.

public:
	CQtListSpinBox(const std::vector<int>& _allowed, QWidget* _parent);

	void SetList(const std::vector<int>& _allowed);	// Sets new list of allowed values.
	void SetValue(int _value);						// Sets new value, must be in the list.

	void SetEditable(bool _flag);					// Enables or disables possibility to change values.
	
private:
	void stepBy(int _steps) override;			// Is called whenever the user triggers a step.
	StepEnabled	stepEnabled() const override;	// Determines whether stepping up and down is legal.
};
