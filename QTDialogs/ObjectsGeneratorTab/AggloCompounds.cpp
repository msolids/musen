/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "AggloCompounds.h"

CAggloCompounds::CAggloCompounds(QWidget *parent /*= 0*/) : QDialog(parent)
{
	ui.setupUi(this);
	m_pSystemStructure = nullptr;
	m_pAgglomeratesDB = nullptr;
	m_pGenerator = nullptr;
	ui.tableWidgetParticles->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	ui.tableWidgetBonds->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

	connect(ui.pushButtonOk, SIGNAL(clicked()), this, SLOT(OkButtonClicked()));
}

CAggloCompounds::~CAggloCompounds()
{

}

void CAggloCompounds::UpdateWholeView()
{
	if (!m_pAgglomeratesDB || !m_pGenerator) return;
	SAgglomerate* pAgglo = m_pAgglomeratesDB->GetAgglomerate(m_pGenerator->m_sAgglomerateKey);
	if (!pAgglo) return;

	QSet<QString> setPartAliases, setBondAliases;
	for (size_t i = 0; i < pAgglo->vParticles.size(); ++i)
		setPartAliases.insert(ss2qs(pAgglo->vParticles[i].sCompoundAlias));
	for (size_t i = 0; i < pAgglo->vBonds.size(); ++i)
		setBondAliases.insert(ss2qs(pAgglo->vBonds[i].sCompoundAlias));

	QList<QString> vPartAliases = setPartAliases.toList();
	qSort(vPartAliases);
	QList<QString> vBondAliases = setBondAliases.toList();
	qSort(vBondAliases);

	ui.tableWidgetParticles->setRowCount(vPartAliases.size());
	for (int i = 0; i < vPartAliases.size(); ++i)
	{
		QTableWidgetItem* pItem = new QTableWidgetItem(vPartAliases[i]);
		pItem->setFlags(pItem->flags() ^ Qt::ItemIsEditable);
		ui.tableWidgetParticles->setItem(i, 0, pItem);
		QComboBox* pCombo = CreateMaterialCombo(ui.tableWidgetParticles, vPartAliases[i]);
		ui.tableWidgetParticles->setCellWidget(i, 1, pCombo);
	}

	ui.tableWidgetBonds->setRowCount(vBondAliases.size());
	for (int i = 0; i < vBondAliases.size(); ++i)
	{
		QTableWidgetItem* pItem = new QTableWidgetItem(vBondAliases[i]);
		pItem->setFlags(pItem->flags() ^ Qt::ItemIsEditable);
		ui.tableWidgetBonds->setItem(i, 0, pItem);
		QComboBox* pCombo = CreateMaterialCombo(ui.tableWidgetBonds, vBondAliases[i]);
		ui.tableWidgetBonds->setCellWidget(i, 1, pCombo);
	}
}

void CAggloCompounds::setVisible(bool _bVisible)
{
	QDialog::setVisible(_bVisible);
	if (_bVisible)
		UpdateWholeView();
}

QComboBox* CAggloCompounds::CreateMaterialCombo(QWidget* _pParent, const QString& _sAlias)
{
	QComboBox* pCombo = new QComboBox(_pParent);
	int nCurrIndex = -1;
	for (unsigned i = 0; i < m_pSystemStructure->m_MaterialDatabase.CompoundsNumber(); ++i)
	{
		pCombo->insertItem(i, ss2qs(m_pSystemStructure->m_MaterialDatabase.GetCompoundName(i)), ss2qs(m_pSystemStructure->m_MaterialDatabase.GetCompoundKey(i)));
		if ((_pParent == ui.tableWidgetParticles) && (m_pGenerator->m_partMaterials[qs2ss(_sAlias)] == m_pSystemStructure->m_MaterialDatabase.GetCompoundKey(i))
			|| (_pParent == ui.tableWidgetBonds) && (m_pGenerator->m_bondMaterials[qs2ss(_sAlias)] == m_pSystemStructure->m_MaterialDatabase.GetCompoundKey(i)))
			nCurrIndex = i;
	}
	pCombo->setCurrentIndex(nCurrIndex);

	return pCombo;
}

void CAggloCompounds::SetGenerator(CObjectsGenerator* _pGenerator)
{
	m_pGenerator = _pGenerator;
	UpdateWholeView();
}

void CAggloCompounds::SetPointers(CSystemStructure* _pSystemStructure, CAgglomeratesDatabase* _pAgglomeratesDB)
{
	m_pSystemStructure = _pSystemStructure;
	m_pAgglomeratesDB = _pAgglomeratesDB;
}

void CAggloCompounds::OkButtonClicked()
{
	if (!m_pSystemStructure) return;

	for (int i = 0; i < ui.tableWidgetParticles->rowCount(); ++i)
	{
		std::string sAlias = qs2ss(ui.tableWidgetParticles->item(i, 0)->text());
		QComboBox* pCombo = static_cast<QComboBox*>(ui.tableWidgetParticles->cellWidget(i, 1));
		m_pGenerator->m_partMaterials[sAlias] = qs2ss(pCombo->currentData().toString());
	}

	for (int i = 0; i < ui.tableWidgetBonds->rowCount(); ++i)
	{
		std::string sAlias = qs2ss(ui.tableWidgetBonds->item(i, 0)->text());
		QComboBox* pCombo = static_cast<QComboBox*>(ui.tableWidgetBonds->cellWidget(i, 1));
		m_pGenerator->m_bondMaterials[sAlias] = qs2ss(pCombo->currentData().toString());
	}

	accept();
}
