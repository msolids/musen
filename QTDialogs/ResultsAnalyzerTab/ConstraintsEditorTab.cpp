/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ConstraintsEditorTab.h"
#include "qtOperations.h"

CConstraintsEditorTab::CConstraintsEditorTab(QWidget *parent)
	: CMusenDialog(parent)
{
	ui.setupUi(this);
	ui.tableDiameters->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

	m_pConstraints = nullptr;
	m_bAvoidSignal = false;

	InitializeConnections();
}

CConstraintsEditorTab::~CConstraintsEditorTab()
{
}

void CConstraintsEditorTab::UpdateSettings()
{
	m_pConstraints->UpdateSettings();

	ui.groupBoxMaterials->setChecked(m_pConstraints->IsMaterialsActive());
	ui.groupBoxVolumes->setChecked(m_pConstraints->IsVolumesActive());
	ui.groupBoxGeometries->setChecked(m_pConstraints->IsGeometriesActive());
	ui.groupBoxDiameters->setChecked(m_pConstraints->IsDiametersActive());
}

void CConstraintsEditorTab::InitializeConnections()
{
	connect(ui.listVolumes, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(NewVolumeChecked(QListWidgetItem*)));
	connect(ui.listGeometries, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(NewGeometryChecked(QListWidgetItem*)));
	connect(ui.listMaterials, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(NewMaterialChecked(QListWidgetItem*)));
	connect(ui.listMaterials, SIGNAL(currentRowChanged(int)), this, SLOT(NewMaterialSelected(int)));
	connect(ui.listMaterials2, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(NewMaterial2Checked(QListWidgetItem*)));
	connect(ui.tableDiameters, SIGNAL(cellChanged(int, int)), this, SLOT(NewDiameterEntered(int, int)));
	connect(ui.groupBoxMaterials, SIGNAL(toggled(bool)), this, SLOT(MaterialsActivated(bool)));
	connect(ui.groupBoxVolumes, SIGNAL(toggled(bool)), this, SLOT(VolumesActivated(bool)));
	connect(ui.groupBoxGeometries, SIGNAL(toggled(bool)), this, SLOT(GeometriesActivated(bool)));
	connect(ui.groupBoxDiameters, SIGNAL(toggled(bool)), this, SLOT(DiametersActivated(bool)));
}

void CConstraintsEditorTab::SetConstraintsPtr(CConstraints *_pConstraints)
{
	m_pConstraints = _pConstraints;
}

void CConstraintsEditorTab::SetMaterialsVisible(bool _bVisible)
{
	ui.groupBoxMaterials->setVisible(_bVisible);
}

void CConstraintsEditorTab::SetMaterials2Visible(bool _bVisible)
{
	ui.listMaterials2->setVisible(_bVisible);
}

void CConstraintsEditorTab::SetVolumesVisible(bool _bVisible)
{
	ui.groupBoxVolumes->setVisible(_bVisible);
}

void CConstraintsEditorTab::SetGeometriesVisible(bool _bVisible)
{
	ui.groupBoxGeometries->setVisible(_bVisible);
}

void CConstraintsEditorTab::SetDiametersVisible(bool _bVisible)
{
	ui.frameSizes->setVisible(_bVisible);
}

void CConstraintsEditorTab::SetDiameters2Visible(bool _bVisible)
{
	m_bAvoidSignal = true;
	if (_bVisible && (ui.tableDiameters->rowCount() == 1))
	{
		ui.tableDiameters->insertRow(1);
		ui.tableDiameters->setVerticalHeaderItem(1, new QTableWidgetItem("Diameter 2"));
		ui.tableDiameters->setItem(1, 0, new QTableWidgetItem(""));
		ui.tableDiameters->setItem(1, 1, new QTableWidgetItem(""));

	}
	else if (!_bVisible && (ui.tableDiameters->rowCount() == 2))
		ui.tableDiameters->removeRow(1);
	m_bAvoidSignal = false;
	if (_bVisible)
		UpdateDiameters();
}

void CConstraintsEditorTab::SetWidgetsEnabled(bool _bEnabled)
{
	ui.frameConstraintsTab->setEnabled(_bEnabled);
}

void CConstraintsEditorTab::UpdateMaterials()
{
	if (ui.listMaterials2->isVisible())
	{
		m_bAvoidSignal = true;
		int nOldRow = ui.listMaterials->currentRow();
		ui.listMaterials->clear();
		ui.listMaterials2->clear();
		// add 'all' entry to the materials lists
		if (m_pMaterialsDB->CompoundsNumber() > 0)
		{
			ui.listMaterials->insertItem(0, new QListWidgetItem("All"));
			ui.listMaterials2->insertItem(0, new QListWidgetItem("All"));
			ui.listMaterials2->item(0)->setCheckState(Qt::Unchecked);
		}
		// add new entries to the particle material lists
		for (unsigned i = 0; i < m_pMaterialsDB->CompoundsNumber(); i++)
		{
			const CCompound* pTempMaterial = m_pMaterialsDB->GetCompound(i);
			ui.listMaterials->insertItem(i + 1, new QListWidgetItem(ss2qs(pTempMaterial->GetName())));
			ui.listMaterials->item(i + 1)->setData(Qt::UserRole, ss2qs(pTempMaterial->GetKey()));
			ui.listMaterials2->insertItem(i + 1, new QListWidgetItem(ss2qs(pTempMaterial->GetName())));
			ui.listMaterials2->item(i + 1)->setData(Qt::UserRole, ss2qs(pTempMaterial->GetKey()));
			ui.listMaterials2->item(i + 1)->setCheckState(Qt::Unchecked);
		}
		m_bAvoidSignal = false;
		// select material
		if (nOldRow < ui.listMaterials->count())
			ui.listMaterials->setCurrentRow(nOldRow);
		else if (nOldRow != -1)
			ui.listMaterials->setCurrentRow(ui.listMaterials->count() - 1);
		else
			ui.listMaterials->setCurrentRow(-1);
	}
	else if (ui.groupBoxMaterials->isVisible())
	{
		m_bAvoidSignal = true;
		ui.listMaterials->clear();
		// add 'all' entry to the materials list
		if (m_pMaterialsDB->CompoundsNumber() > 0)
		{
			ui.listMaterials->insertItem(0, new QListWidgetItem("All"));
			ui.listMaterials->item(0)->setCheckState(Qt::Unchecked);
		}
		// add new entries to the particle material list
		for (unsigned i = 0; i < m_pMaterialsDB->CompoundsNumber(); i++)
		{
			const CCompound* pTempMaterial = m_pMaterialsDB->GetCompound(i);
			ui.listMaterials->insertItem(i + 1, new QListWidgetItem(ss2qs(pTempMaterial->GetName())));
			ui.listMaterials->item(i + 1)->setData(Qt::UserRole, ss2qs(pTempMaterial->GetKey()));
			ui.listMaterials->item(i + 1)->setCheckState(Qt::Unchecked);
		}
		// set check states
		if (ui.listMaterials->count() > 0)
		{
			if (m_pConstraints->IsAllMaterialsSelected())
				ui.listMaterials->item(0)->setCheckState(Qt::Checked);
			else
			{
				std::vector<std::string> vMaterialsKeys = m_pConstraints->GetMaterials();
				for (unsigned i = 0; i < vMaterialsKeys.size(); ++i)
					for (int j = 1; j < ui.listMaterials->count(); ++j)
						if (qs2ss(ui.listMaterials->item(j)->data(Qt::UserRole).toString()) == vMaterialsKeys[i])
						{
							ui.listMaterials->item(j)->setCheckState(Qt::Checked);
							break;
						}
			}
		}
		m_bAvoidSignal = false;
	}
}

void CConstraintsEditorTab::UpdateVolumes()
{
	if (ui.groupBoxVolumes->isHidden()) return;
	m_bAvoidSignal = true;
	ui.listVolumes->clear();
	// add 'all' entry to the volume list
	if (m_pSystemStructure->GetAnalysisVolumesNumber() > 0)
	{
		ui.listVolumes->insertItem(0, new QListWidgetItem("Everywhere"));
		ui.listVolumes->item(0)->setCheckState(Qt::Unchecked);
	}
	// add new entries to the volume list
	for (unsigned i = 0; i < m_pSystemStructure->GetAnalysisVolumesNumber(); ++i)
	{
		CAnalysisVolume* pVolume = m_pSystemStructure->GetAnalysisVolume(i);
		ui.listVolumes->insertItem(i + 1, new QListWidgetItem(ss2qs(pVolume->sName)));
		ui.listVolumes->item(i + 1)->setCheckState(Qt::Unchecked);
	}
	// set flags
	if (ui.listVolumes->count() > 0)
	{
		if (m_pConstraints->IsAllVolumeSelected())
			ui.listVolumes->item(0)->setCheckState(Qt::Checked);
		else
			for (unsigned i = 0; i < m_pConstraints->GetVolumes().size(); ++i)
				ui.listVolumes->item(m_pConstraints->GetVolumes()[i] + 1)->setCheckState(Qt::Checked);
	}
	m_bAvoidSignal = false;
}

void CConstraintsEditorTab::UpdateGeometries()
{
	if (ui.groupBoxGeometries->isHidden()) return;
	m_bAvoidSignal = true;
	ui.listGeometries->clear();
	// add 'all' entry to the geometries list
	if (m_pSystemStructure->GetGeometriesNumber() > 0)
	{
		ui.listGeometries->insertItem(0, new QListWidgetItem("All"));
		ui.listGeometries->item(0)->setCheckState(Qt::Unchecked);
	}
	// add new entries to the geometries list
	for (unsigned i = 0; i < m_pSystemStructure->GetGeometriesNumber(); ++i)
	{
		SGeometryObject* pGeomObj = m_pSystemStructure->GetGeometry(i);
		ui.listGeometries->insertItem(i + 1, new QListWidgetItem(ss2qs(pGeomObj->sName)));
		ui.listGeometries->item(i + 1)->setCheckState(Qt::Unchecked);
	}
	// set flags
	if (ui.listGeometries->count() > 0)
	{
		if (m_pConstraints->IsAllGeometriesSelected())
			ui.listGeometries->item(0)->setCheckState(Qt::Checked);
		else
			for (unsigned i = 0; i < m_pConstraints->GetGeometries().size(); ++i)
				ui.listGeometries->item(m_pConstraints->GetGeometries()[i] + 1)->setCheckState(Qt::Checked);
	}
	m_bAvoidSignal = false;
}

void CConstraintsEditorTab::UpdateDiameters()
{
	if (ui.groupBoxDiameters->isHidden()) return;

	ShowConvLabel(ui.tableDiameters->horizontalHeaderItem(0), "Min", EUnitType::PARTICLE_DIAMETER);
	ShowConvLabel(ui.tableDiameters->horizontalHeaderItem(1), "Max", EUnitType::PARTICLE_DIAMETER);

	m_bAvoidSignal = true;
	if (ui.tableDiameters->rowCount() == 1)
	{
		CConstraints::SInterval interval = m_pConstraints->GetDiameter();
		ShowConvValue(ui.tableDiameters->item(0, 0), interval.dMin, EUnitType::PARTICLE_DIAMETER);
		ShowConvValue(ui.tableDiameters->item(0, 1), interval.dMax, EUnitType::PARTICLE_DIAMETER);
	}
	if (ui.tableDiameters->rowCount() == 2)
	{
		CConstraints::SInterval interval2 = m_pConstraints->GetDiameter2();
		ShowConvValue(ui.tableDiameters->item(1, 0), interval2.dMin, EUnitType::PARTICLE_DIAMETER);
		ShowConvValue(ui.tableDiameters->item(1, 1), interval2.dMax, EUnitType::PARTICLE_DIAMETER);
	}
	m_bAvoidSignal = false;
}

void CConstraintsEditorTab::UpdateWholeView()
{
	UpdateMaterials();
	UpdateVolumes();
	UpdateGeometries();
	UpdateDiameters();
}


void CConstraintsEditorTab::NewVolumeChecked(QListWidgetItem* _pItem)
{
	if (m_bAvoidSignal) return;

	int nRow = ui.listVolumes->row(_pItem);
	if ((nRow == 0) && (_pItem->checkState() == Qt::Checked))
	{
		m_pConstraints->ClearVolumes();
		for (int i = 1; i < ui.listVolumes->count(); ++i)
			ui.listVolumes->item(i)->setCheckState(Qt::Unchecked);
	}
	else if (nRow != 0)
	{
		if (_pItem->checkState() == Qt::Checked)
		{
			ui.listVolumes->item(0)->setCheckState(Qt::Unchecked);
			if(m_pConstraints->GetVolumes().size() == m_pSystemStructure->GetAnalysisVolumesNumber())
				m_pConstraints->ClearVolumes();
			m_pConstraints->AddVolume(nRow - 1);
		}
		else if (_pItem->checkState() == Qt::Unchecked)
		{
			m_pConstraints->RemoveVolume(nRow - 1);
			if (m_pConstraints->IsAllVolumeSelected())
				ui.listVolumes->item(0)->setCheckState(Qt::Checked);
		}
	}
}

void CConstraintsEditorTab::NewGeometryChecked(QListWidgetItem* _pItem)
{
	if (m_bAvoidSignal) return;
	int nRow = ui.listGeometries->row(_pItem);
	if ((nRow == 0) && (_pItem->checkState() == Qt::Checked))
	{
		m_pConstraints->ClearGeometries();
		for (int i = 1; i < ui.listGeometries->count(); ++i)
			ui.listGeometries->item(i)->setCheckState(Qt::Unchecked);
	}
	else if (nRow != 0)
	{
		if (_pItem->checkState() == Qt::Checked)
		{
			ui.listGeometries->item(0)->setCheckState(Qt::Unchecked);
			if(m_pConstraints->GetGeometries().size() == m_pSystemStructure->GetGeometriesNumber())
				m_pConstraints->ClearGeometries();
			m_pConstraints->AddGeometry(nRow - 1);
		}
		else if (_pItem->checkState() == Qt::Unchecked)
		{
			m_pConstraints->RemoveGeometry(nRow - 1);
			if (m_pConstraints->IsAllGeometriesSelected())
				ui.listGeometries->item(0)->setCheckState(Qt::Checked);
		}
	}
}

void CConstraintsEditorTab::NewMaterialChecked(QListWidgetItem* _pItem)
{
	if (m_bAvoidSignal) return;
	if (ui.listMaterials2->isVisible()) return;

	m_bAvoidSignal = true;
	if (ui.groupBoxMaterials->isVisible())
	{
		int nRow = ui.listMaterials->row(_pItem);
		if ((nRow == 0) && (_pItem->checkState() == Qt::Checked))
		{
			m_pConstraints->ClearMaterials();
			for (int i = 1; i < ui.listMaterials->count(); ++i)
				ui.listMaterials->item(i)->setCheckState(Qt::Unchecked);
		}
		else if (nRow != 0)
		{
			if (_pItem->checkState() == Qt::Checked)
			{
				ui.listMaterials->item(0)->setCheckState(Qt::Unchecked);
				if (m_pConstraints->GetMaterials().size() == m_pMaterialsDB->CompoundsNumber())
					m_pConstraints->ClearMaterials();
				m_pConstraints->AddMaterial(qs2ss(_pItem->data(Qt::UserRole).toString()));
			}
			else if (_pItem->checkState() == Qt::Unchecked)
			{
				m_pConstraints->RemoveMaterial(qs2ss(_pItem->data(Qt::UserRole).toString()));
				if (m_pConstraints->IsAllMaterialsSelected())
					ui.listMaterials->item(0)->setCheckState(Qt::Checked);
			}
		}
	}
	m_bAvoidSignal = false;
}

void CConstraintsEditorTab::NewMaterialSelected(int _nRow)
{
	if (m_bAvoidSignal) return;
	if (ui.listMaterials2->isHidden()) return;

	m_bAvoidSignal = true;

	for (int i = 1; i < ui.listMaterials2->count(); ++i)
		ui.listMaterials2->item(i)->setCheckState(Qt::Unchecked);

	if (_nRow == 0) // 'all' selected
	{
		if(m_pConstraints->IsAllMaterials2Selected())
		{
			ui.listMaterials2->item(0)->setCheckState(Qt::Checked);
		}
		else
		{
			ui.listMaterials2->item(0)->setCheckState(Qt::Unchecked);
			std::vector<std::string> vCommonMaterials = m_pConstraints->GetCommonMaterials();
			for (unsigned i = 0; i < vCommonMaterials.size(); ++i)
				for (int j = 1; j < ui.listMaterials2->count(); ++j)
					if (qs2ss(ui.listMaterials->item(j)->data(Qt::UserRole).toString()) == vCommonMaterials[i])
					{
						ui.listMaterials2->item(j)->setCheckState(Qt::Checked);
						break;
					}
		}
	}
	else // material selected
	{
		if (m_pConstraints->IsAllMaterials2Selected(qs2ss(ui.listMaterials->item(_nRow)->data(Qt::UserRole).toString())))
			ui.listMaterials2->item(0)->setCheckState(Qt::Checked);
		else
		{
			ui.listMaterials2->item(0)->setCheckState(Qt::Unchecked);
			std::vector<std::string> vMaterials = m_pConstraints->GetMaterials2(qs2ss(ui.listMaterials->item(_nRow)->data(Qt::UserRole).toString()));
			for (unsigned i = 0; i < vMaterials.size(); ++i)
				for (int j = 1; j < ui.listMaterials2->count(); ++j)
					if (qs2ss(ui.listMaterials->item(j)->data(Qt::UserRole).toString()) == vMaterials[i])
					{
						ui.listMaterials2->item(j)->setCheckState(Qt::Checked);
						break;
					}
		}
	}
	m_bAvoidSignal = false;
}

void CConstraintsEditorTab::NewMaterial2Checked(QListWidgetItem* _pItem)
{
	if (m_bAvoidSignal) return;
	if (ui.listMaterials2->isHidden()) return;

	m_bAvoidSignal = true;
	int nRow = ui.listMaterials->currentRow();
	int nRow2 = ui.listMaterials2->row(_pItem);
	std::string sKey = qs2ss(ui.listMaterials->item(nRow)->data(Qt::UserRole).toString());
	std::string sKey2 = qs2ss(_pItem->data(Qt::UserRole).toString());
	if (nRow == 0)	// all with smth
	{
		if(nRow2 == 0)	// all with all
		{
			m_pConstraints->ClearMaterials2();
			for (int i = 1; i < ui.listMaterials2->count(); ++i)
				ui.listMaterials2->item(i)->setCheckState(Qt::Unchecked);
			ui.listMaterials2->item(0)->setCheckState(Qt::Checked);
		}
		else // all with material
		{
			if (_pItem->checkState() == Qt::Checked)
			{
				bool bAllWithAll = true;
				for (unsigned i = 0; i < m_pMaterialsDB->CompoundsNumber(); ++i)
					if (m_pConstraints->GetMaterials2(m_pMaterialsDB->GetCompoundKey(i)).size() != m_pMaterialsDB->CompoundsNumber())
					{
						bAllWithAll = false;
						break;
					}
				if(bAllWithAll)
					m_pConstraints->ClearMaterials2();
				m_pConstraints->AddMaterial2(sKey2);
				ui.listMaterials2->item(0)->setCheckState(Qt::Unchecked);
			}
			else
			{
				m_pConstraints->RemoveMaterial2(sKey2);
				if (m_pConstraints->IsAllMaterials2Selected())
					ui.listMaterials2->item(0)->setCheckState(Qt::Checked);
			}
		}
	}
	else // material with smth
	{
		if (nRow2 == 0)	// material with all
		{
			if (_pItem->checkState() == Qt::Checked)
			{
				m_pConstraints->AddMaterial2(sKey);
				for (int i = 1; i < ui.listMaterials2->count(); ++i)
					ui.listMaterials2->item(i)->setCheckState(Qt::Unchecked);
			}
			else
			{
				m_pConstraints->RemoveMaterial2(sKey);
			}
		}
		else // material with material
		{
			if (_pItem->checkState() == Qt::Checked)
			{
				if (m_pConstraints->GetMaterials2(sKey).size() == m_pMaterialsDB->CompoundsNumber())
					m_pConstraints->ClearMaterials2(sKey);
				m_pConstraints->AddMaterial2(sKey, sKey2);
				ui.listMaterials2->item(0)->setCheckState(Qt::Unchecked);
			}
			else
			{
				m_pConstraints->RemoveMaterial2(sKey, sKey2);
				if (m_pConstraints->IsAllMaterials2Selected(sKey))
					ui.listMaterials2->item(0)->setCheckState(Qt::Checked);
			}
		}
	}
	m_bAvoidSignal = false;
}

void CConstraintsEditorTab::NewDiameterEntered(int _nRow, int _nCol)
{
	if (m_bAvoidSignal) return;
	if (_nRow == 0)
		m_pConstraints->SetDiameter(GetConvValue(ui.tableDiameters->item(0, 0), EUnitType::PARTICLE_DIAMETER), GetConvValue(ui.tableDiameters->item(0, 1), EUnitType::PARTICLE_DIAMETER));
	else if (_nRow == 1)
		m_pConstraints->SetDiameter2(GetConvValue(ui.tableDiameters->item(1, 0), EUnitType::PARTICLE_DIAMETER), GetConvValue(ui.tableDiameters->item(1, 1), EUnitType::PARTICLE_DIAMETER));
}

void CConstraintsEditorTab::MaterialsActivated(bool _bActive)
{
	m_pConstraints->SetMateralsActive(_bActive);
	if (_bActive)
		UpdateMaterials();
}

void CConstraintsEditorTab::VolumesActivated(bool _bActive)
{
	m_pConstraints->SetVolumesActive(_bActive);
	if (_bActive)
		UpdateVolumes();
}

void CConstraintsEditorTab::GeometriesActivated(bool _bActive)
{
	m_pConstraints->SetGeometriesActive(_bActive);
	if (_bActive)
		UpdateGeometries();
}

void CConstraintsEditorTab::DiametersActivated(bool _bActive)
{
	m_pConstraints->SetDiametersActive(_bActive);
	if (_bActive)
		UpdateDiameters();
}
