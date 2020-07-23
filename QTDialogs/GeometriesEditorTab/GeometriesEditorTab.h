/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GeneralMUSENDialog.h"
#include "ui_GeometriesEditorTab.h"
#include "GeometriesDatabase.h"
#include "SystemStructure.h"
#include <QSignalMapper>

class CGeometriesEditorTab : public CMusenDialog
{
	Q_OBJECT

private:
	Ui::geometriesEditorTab ui;

	QMenu* m_pSubmenuVirtStd;
	QMenu* m_pSubmenuVirtLib;
	QMenu* m_pSubmenuRealStd;
	QMenu* m_pSubmenuRealLib;
	QSignalMapper* m_pMapperRealStd;
	QSignalMapper* m_pMapperRealLib;
	QSignalMapper* m_pMapperVirtStd;
	QSignalMapper* m_pMapperVirtLib;

public:
	CGeometriesEditorTab(QWidget *parent = 0);

public slots:
	void UpdateWholeView();

protected:
	void keyPressEvent(QKeyEvent *event);

private:
	void InitializeConnections();
	void UpdateAddButton();
	void UpdateHeaders();
	void UpdateGeometriesList();
	void UpdateInfoAboutGeometry(int _index);
	void UpdateInfoAboutAnalysisVolume(int _index);
	void UpdateInfo(const EVolumeType& _volumeType, const QString& _sTypePrefix, const std::vector<double>& _vParams, size_t _nPlanesNum);
	void EnableAllFields(bool _bEnable);

	bool ConfirmRemoveAllTP();

private slots:
	void UpdateSelectedGeometryProperties();
	void ShowContextMenu(const QPoint& _pos);

	void GeometryNameChanged(QListWidgetItem* _item);
	void AddGeometryRealStd(int _type);
	void AddGeometryRealLib(int _indexDB);
	void AddGeometryVirtStd(int _type);
	void AddGeometryVirtLib(int _indexDB);
	void RemoveGeometry();
	void SetMaterial(int _iMaterial);
	void GeometryColorChanged();
	void ScaleGeometry();
	void MoveGeometry();
	void RotateGeometry();
	void ShiftGeometryUpwards();
	void ShiftGeometryDownwards();
	void GeometryParamsChanged();
	void SetFreeMotion();

	void AddTimePoint();
	void RemoveTimePoints();
	void SetTDVelocities();

signals:
	void AnalysisGeometriesChanged();
	void ObjectsChanged();
};

