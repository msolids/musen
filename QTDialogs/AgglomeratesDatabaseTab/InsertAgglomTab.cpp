/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "InsertAgglomTab.h"
CInsertAgglomTab::CInsertAgglomTab(QWidget *parent) : CMusenDialog(parent)
{
	ui.setupUi(this);
	ui.propTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	ui.tableWidgetParticles->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	ui.tableWidgetBonds->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

	connect(ui.addAgglom, SIGNAL(clicked()), this, SLOT(AddAgglomerate()));
}

void CInsertAgglomTab::UpdateWholeView()
{
	ui.statusMessage->setText("");
	ShowConvLabel( ui.propTable->verticalHeaderItem( 0 ), "Position X:Y:Z", EUnitType::LENGTH );
	ShowConvLabel( ui.propTable->verticalHeaderItem( 1 ), "Velocity Vx:Vy:Vz", EUnitType::VELOCITY );

	const SAgglomerate* pAgglo = m_pAgglomDB->GetAgglomerate(m_sAgglomKey);
	if (!pAgglo) return;

	QSet<QString> setPartAliases, setBondAliases;
	for (size_t i = 0; i < pAgglo->vParticles.size(); ++i)
		setPartAliases.insert(ss2qs(pAgglo->vParticles[i].sCompoundAlias));
	for (size_t i = 0; i < pAgglo->vBonds.size(); ++i)
		setBondAliases.insert(ss2qs(pAgglo->vBonds[i].sCompoundAlias));

	QList<QString> vPartAliases = setPartAliases.values();
	std::sort(vPartAliases.begin(), vPartAliases.end());
	QList<QString> vBondAliases = setBondAliases.values();
	std::sort(vBondAliases.begin(), vBondAliases.end());
	ui.tableWidgetParticles->setRowCount(vPartAliases.size());
	for (int i = 0; i < vPartAliases.size(); ++i)
	{
		QTableWidgetItem* pItem = new QTableWidgetItem(vPartAliases.at(i));
		pItem->setFlags(pItem->flags() ^ Qt::ItemIsEditable);
		ui.tableWidgetParticles->setItem(i, 0, pItem);
		QComboBox* pCombo = CreateMaterialCombo(ui.tableWidgetParticles);
		ui.tableWidgetParticles->setCellWidget(i, 1, pCombo);
	}

	ui.tableWidgetBonds->setRowCount(vBondAliases.size());
	for (int i = 0; i < vBondAliases.size(); ++i)
	{
		QTableWidgetItem* pItem = new QTableWidgetItem(vBondAliases.at(i));
		pItem->setFlags(pItem->flags() ^ Qt::ItemIsEditable);
		ui.tableWidgetBonds->setItem(i, 0, pItem);
		QComboBox* pCombo = CreateMaterialCombo(ui.tableWidgetBonds);
		ui.tableWidgetBonds->setCellWidget(i, 1, pCombo);
	}
}

void CInsertAgglomTab::AddAgglomerate()
{
	SAgglomerate* pAgglomerate = m_pAgglomDB->GetAgglomerate(m_sAgglomKey);
	if (!pAgglomerate) return;

	std::map<std::string, std::string> partCompounds;
	for (int i = 0; i < ui.tableWidgetParticles->rowCount(); ++i)
	{
		std::string sAlias = qs2ss(ui.tableWidgetParticles->item(i, 0)->text());
		QComboBox* pCombo = static_cast<QComboBox*>(ui.tableWidgetParticles->cellWidget(i, 1));
		std::string sCompondKey = qs2ss(pCombo->currentData().toString());
		if (!m_pSystemStructure->m_MaterialDatabase.GetCompound(sCompondKey))
		{
			ui.statusMessage->setText("Materials of particles have been incorrectly defined!");
			return;
		}
		partCompounds[sAlias] = sCompondKey;
	}

	std::map<std::string, std::string> bondCompounds;
	for (int i = 0; i < ui.tableWidgetBonds->rowCount(); ++i)
	{
		std::string sAlias = qs2ss(ui.tableWidgetBonds->item(i, 0)->text());
		QComboBox* pCombo = static_cast<QComboBox*>(ui.tableWidgetBonds->cellWidget(i, 1));
		std::string sCompondKey = qs2ss(pCombo->currentData().toString());
		if (!m_pSystemStructure->m_MaterialDatabase.GetCompound(sCompondKey))
		{
			ui.statusMessage->setText("Materials of bonds have been incorrectly defined!");
			return;
		}
		bondCompounds[sAlias] = sCompondKey;
	}

	double dScalingFactor = ui.scalingFactor->text().toDouble();
	if (dScalingFactor == 0)
	{
		ui.statusMessage->setText("Wrong scaling factor");
		return;
	}

	CVector3 vPos = GetVectorFromTableRow( ui.propTable, 0, 0, EUnitType::LENGTH );
	CVector3 vVel = GetVectorFromTableRow( ui.propTable, 1, 0, EUnitType::VELOCITY );
	CVector3 vAngleRad = GetVectorFromTableRow( ui.propTable, 2, 0 )*PI / 180.0;

	CVector3 vCenterOfMass( 0, 0, 0 );
	double dTotalVolume = 0; // total mass of a system
	for (size_t i = 0; i < pAgglomerate->vParticles.size(); i++)
	{
		double dVolume1 = PI*pow(2 * pAgglomerate->vParticles[i].dRadius, 3) / 6.0;
		vCenterOfMass = (vCenterOfMass*dTotalVolume + pAgglomerate->vParticles[i].vecCoord*dVolume1) / (dVolume1 + dTotalVolume);
		dTotalVolume += dVolume1;
	}

	// determine desired rotation quaternion
	CQuaternion rotQuat;
	rotQuat = CQuaternion(vAngleRad);

	// add objects to the system structure
	std::vector<size_t> vNewParticlesIndexes;
	for (size_t i = 0; i < pAgglomerate->vParticles.size(); i++)
	{
		CSphere* pSphere = (CSphere*)m_pSystemStructure->AddObject(SPHERE);
		pSphere->SetStartActivityTime(0.0);
		pSphere->SetEndActivityTime(DEFAULT_ACTIVITY_END);
		pSphere->SetCoordinates(0, QuatRotateVector(rotQuat, (pAgglomerate->vParticles[i].vecCoord * dScalingFactor - vCenterOfMass)) + vPos);
		CQuaternion partNewQuat = rotQuat * pAgglomerate->vParticles[i].qQuaternion;
		partNewQuat.Normalize();
		pSphere->SetOrientation(0, partNewQuat);
		pSphere->SetVelocity(0, vVel);
		pSphere->SetRadius(pAgglomerate->vParticles[i].dRadius * dScalingFactor);
		pSphere->SetContactRadius(pAgglomerate->vParticles[i].dContactRadius * dScalingFactor);
		pSphere->SetCompound(m_pSystemStructure->m_MaterialDatabase.GetCompound(partCompounds[pAgglomerate->vParticles[i].sCompoundAlias]));
		vNewParticlesIndexes.push_back(pSphere->m_lObjectID);
	}

	for (size_t i = 0; i < pAgglomerate->vBonds.size(); i++)
	{
		CSolidBond* pBond = (CSolidBond*)m_pSystemStructure->AddObject(SOLID_BOND );
		pBond->SetStartActivityTime(0.0);
		pBond->SetEndActivityTime(DEFAULT_ACTIVITY_END);
		pBond->SetDiameter(pAgglomerate->vBonds[i].dRadius * dScalingFactor * 2);
		pBond->m_nLeftObjectID = (unsigned)vNewParticlesIndexes[pAgglomerate->vBonds[i].nLeftID];
		pBond->m_nRightObjectID = (unsigned)vNewParticlesIndexes[pAgglomerate->vBonds[i].nRightID];
		pBond->SetCompound(m_pSystemStructure->m_MaterialDatabase.GetCompound(bondCompounds[pAgglomerate->vBonds[i].sCompoundAlias]));
		pBond->SetInitialLength(m_pSystemStructure->GetBond(0, pBond->m_lObjectID).Length());
	}

	if ( pAgglomerate->nType == AGGLOMERATE )
		ui.statusMessage->setText("Agglomerate has been successfully added");
	else
	{
		ui.statusMessage->setText( "Multisphere has been successfully added" );
		m_pSystemStructure->AddMultisphere( vNewParticlesIndexes );
	}
	m_pSystemStructure->UpdateAllObjectsCompoundsProperties();
	emit NewAgglomerateAdded();
}

QComboBox* CInsertAgglomTab::CreateMaterialCombo(QWidget* _pParent)
{
	QComboBox* pCombo = new QComboBox(_pParent);
	for (size_t i = 0; i < m_pSystemStructure->m_MaterialDatabase.CompoundsNumber(); ++i)
		pCombo->insertItem(static_cast<int>(i), ss2qs(m_pSystemStructure->m_MaterialDatabase.GetCompoundName(i)), ss2qs(m_pSystemStructure->m_MaterialDatabase.GetCompoundKey(i)));
	pCombo->setCurrentIndex(-1);
	return pCombo;
}

void CInsertAgglomTab::SetCurrentAgglom(const std::string& _sKey)
{
	m_sAgglomKey = _sKey;
}
