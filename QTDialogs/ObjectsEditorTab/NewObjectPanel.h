/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include <QWidget>
#include "ui_NewObjectPanel.h"
#include "SystemStructure.h"
#include "UnitConvertor.h"

class CNewObjectPanel : public QWidget
{
	Q_OBJECT
private:
	enum class EObjectTypes : int
	{
		PARTICLES = 0,
		SOLID_BONDS = 1,
	};

	Ui::CNewObjectPanel ui;

	CSystemStructure* m_systemStructure;
	CMaterialsDatabase* m_materialsDB;
	CUnitConvertor* m_converer;

	QString m_statusMessage;

public:
	CNewObjectPanel(QWidget *parent = Q_NULLPTR);
	~CNewObjectPanel();

	void SetPointers(CSystemStructure* _systemStructure, CMaterialsDatabase* _materialsDB, CUnitConvertor* _unitConvertor);
	void UpdateWholeView() const;

	QString StatusMessage() const;

private:
	void InitializeConnections() const;

	void UpdateLabels() const;
	void UpdateMaterials() const;
	void UpdateVisibility() const;

	void ObjectTypeChanged(bool _checked) const;
	void AddObject();

	QString CheckData() const;
	EObjectTypes ObjectType() const; /// Currently selected type of new object.

signals:
	void ObjectAdded();
};
