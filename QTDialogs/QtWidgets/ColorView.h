/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <QWidget>
#include "MixedFunctions.h"

class CColorView : public QWidget
{
	Q_OBJECT

	QColor m_actualColor;

public:
	CColorView(QWidget* parent = nullptr);

	void SetColor(const CColor& _color);
	void SetColor(const QColor& _color);
	QColor getColor() const;
	CColor getColor2() const;

protected:
	void paintEvent(QPaintEvent* _event) override;
	void mouseDoubleClickEvent(QMouseEvent* _event) override;

signals:
	void ColorChanged();	// Is emitted whenever the color is changed, either by user of programmatically.
	void ColorEdited();		// Is emitted whenever the color is changed by user.
};
