/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <QObject>

/// Is used to filter mouse events, happening in viewers, from within view manager.
class CEventMonitor : public QObject
{
	Q_OBJECT
public:
	CEventMonitor(QObject* _parent);

protected:
	bool eventFilter(QObject *_obj, QEvent *_event) override;

signals:
	void ParticleSelected(const QPoint&);
	void GroupSelected(const QPoint&);
	void CameraChanged();
};
