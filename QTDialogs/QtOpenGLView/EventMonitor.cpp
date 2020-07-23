/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "EventMonitor.h"
#include <QMouseEvent>

CEventMonitor::CEventMonitor(QObject* _parent) : QObject(_parent) {}

bool CEventMonitor::eventFilter(QObject* _obj, QEvent* _event)
{
	if (_event->type() == QEvent::MouseButtonPress) // catch ctrl+left or ctrl+right
	{
		const QMouseEvent* mouseEvent = dynamic_cast<QMouseEvent*>(_event);
		if (mouseEvent->modifiers() & Qt::ControlModifier && mouseEvent->buttons() & Qt::LeftButton)
			emit ParticleSelected(mouseEvent->pos());
		else if (mouseEvent->modifiers() & Qt::ControlModifier && mouseEvent->buttons() & Qt::RightButton)
			emit GroupSelected(mouseEvent->pos());
	}

	if (_event->type() == QEvent::Paint)
		emit CameraChanged();

	// standard event processing
	return QObject::eventFilter(_obj, _event);
}
