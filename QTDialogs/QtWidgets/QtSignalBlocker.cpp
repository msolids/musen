/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "QtSignalBlocker.h"

CQtSignalBlocker::CQtSignalBlocker(QObject* _object)
{
	m_objects.push_back(_object);
	m_flags.push_back(m_objects.front()->blockSignals(true));
}

CQtSignalBlocker::CQtSignalBlocker(std::initializer_list<QObject*> _objects)
{
	m_objects = _objects;
	for (auto& object : m_objects)
		m_flags.push_back(object->blockSignals(true));
}

CQtSignalBlocker::~CQtSignalBlocker()
{
	for (size_t i = 0; i < m_objects.size(); ++i)
		m_objects[i]->blockSignals(m_flags[i]);
}
