/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ColorView.h"
#include <QPainter>
#include <QColorDialog>

CColorView::CColorView(QWidget* parent): QWidget(parent)
{
	setMinimumSize(10, 10);
}

void CColorView::SetColor(const CColor& _color)
{
	QColor color;
	color.setRgbF(_color.r, _color.g, _color.b, _color.a);
	SetColor(color);
}

void CColorView::SetColor(const QColor& _color)
{
	m_actualColor = _color;
	update();
	if(!signalsBlocked())
		emit ColorChanged();
}

QColor CColorView::getColor() const
{
	return m_actualColor;
}

CColor CColorView::getColor2() const
{
	qreal r, g, b, f;
	m_actualColor.getRgbF(&r, &g, &b, &f);
	return {static_cast<float>(r), static_cast<float>(g), static_cast<float>(b), static_cast<float>(f)};
}

void CColorView::paintEvent(QPaintEvent* _event)
{
	QPainter p(this);
	if (isEnabled())
		p.setBrush(QBrush(m_actualColor));
	else
		p.setBrush(Qt::lightGray);
	p.drawRect(QRect(1, 1, width() - 1, height() - 1));
}

void CColorView::mouseDoubleClickEvent(QMouseEvent* _event)
{
	const QColor color = QColorDialog::getColor(m_actualColor, this);
	if (!color.isValid() || color == m_actualColor) return;
	SetColor(color);
	if (!signalsBlocked())
		emit ColorEdited();
}
