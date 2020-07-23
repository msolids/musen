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

	void setColor(const CColor& _color);
	void setColor(const QColor& _color);
	QColor getColor() const;

protected:
	void paintEvent(QPaintEvent* _event) override;
	void mouseDoubleClickEvent(QMouseEvent* _event) override;

signals:
	void ColorChanged();
};
