/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "AgglomCompounds.h"
#include <QMessageBox>

CAgglomCompounds::CAgglomCompounds(SAgglomerate* _pAgglomerate, CMaterialsDatabase* _pMaterialsDB, QList<QString> _vPartKeys, QList<QString> _vBondKeys, QWidget *parent /*= 0*/)
	: QDialog(parent), m_pAgglomerate(_pAgglomerate), m_pMaterialsDB(_pMaterialsDB), m_vPartKeys(_vPartKeys), m_vBondKeys(_vBondKeys)
{
	ui.setupUi(this);
	ui.tableWidgetParticles->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	ui.tableWidgetBonds->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	connect(ui.pushButtonOk, SIGNAL(clicked()), this, SLOT(OkButtonClicked()));
}

CAgglomCompounds::~CAgglomCompounds()
{

}

void CAgglomCompounds::UpdateWholeView()
{
	if (!m_pAgglomerate || !m_pMaterialsDB) return;

	ui.tableWidgetParticles->setRowCount(m_vPartKeys.size());
	for (int i = 0; i < m_vPartKeys.size(); ++i)
	{
		QString sName = ss2qs(m_pMaterialsDB->GetCompoundName(qs2ss(m_vPartKeys.at(i))));
		if (sName.isEmpty())
			sName = m_vPartKeys.at(i);
		QTableWidgetItem* pItemName = new QTableWidgetItem(sName);
		pItemName->setFlags(pItemName->flags() ^ Qt::ItemIsEditable);
		ui.tableWidgetParticles->setItem(i, 0, pItemName);
		QTableWidgetItem* pItemAlias = new QTableWidgetItem(sName);
		ui.tableWidgetParticles->setItem(i, 1, pItemAlias);
	}

	ui.tableWidgetBonds->setRowCount(m_vBondKeys.size());
	for (int i = 0; i < m_vBondKeys.size(); ++i)
	{
		QString sName = ss2qs(m_pMaterialsDB->GetCompoundName(qs2ss(m_vBondKeys.at(i))));
		if (sName.isEmpty())
			sName = m_vBondKeys.at(i);
		QTableWidgetItem* pItemName = new QTableWidgetItem(sName);
		pItemName->setFlags(pItemName->flags() ^ Qt::ItemIsEditable);
		ui.tableWidgetBonds->setItem(i, 0, pItemName);
		QTableWidgetItem* pItemAlias = new QTableWidgetItem(sName);
		ui.tableWidgetBonds->setItem(i, 1, pItemAlias);
	}

}

void CAgglomCompounds::setVisible(bool _bVisible)
{
	QDialog::setVisible(_bVisible);
	if (_bVisible)
		UpdateWholeView();
}

void CAgglomCompounds::OkButtonClicked()
{
	// check repetitive aliases
	QStringList sRepeatPartAliases, sRepeatBondAliases;
	QSet<QString> aliases;
	std::map<std::string, std::string> mapPatr, mapBond;
	for (int i = 0; i < ui.tableWidgetParticles->rowCount(); ++i)
	{
		QString sAlias = ui.tableWidgetParticles->item(i, 1)->text();
		if (aliases.contains(sAlias))
			sRepeatPartAliases.append(sAlias);
		aliases.insert(sAlias);
		mapPatr[qs2ss(m_vPartKeys[i])] = qs2ss(sAlias);
	}
	aliases.clear();
	for (int i = 0; i < ui.tableWidgetBonds->rowCount(); ++i)
	{
		QString sAlias = ui.tableWidgetBonds->item(i, 1)->text();
		if (aliases.contains(sAlias))
			sRepeatBondAliases.append(sAlias);
		aliases.insert(sAlias);
		mapBond[qs2ss(m_vBondKeys[i])] = qs2ss(sAlias);
	}

	if (!sRepeatPartAliases.isEmpty() || !sRepeatBondAliases.isEmpty())
	{
		int res = QMessageBox::warning(this, "Add agglomerate", "Some materials have the same aliases. Continue?", QMessageBox::Cancel, QMessageBox::Ok);
		if (res == QMessageBox::Cancel) return;
	}

	for (size_t i = 0; i < m_pAgglomerate->vParticles.size(); ++i)
		m_pAgglomerate->vParticles[i].sCompoundAlias = mapPatr[m_pAgglomerate->vParticles[i].sCompoundAlias];
	for (size_t i = 0; i < m_pAgglomerate->vBonds.size(); ++i)
		m_pAgglomerate->vBonds[i].sCompoundAlias = mapBond[m_pAgglomerate->vBonds[i].sCompoundAlias];

	if ( ui.asAgglomerate->isChecked() )
		m_pAgglomerate->nType = AGGLOMERATE;
	else
	{
		m_pAgglomerate->nType = MULTISPHERE;
		m_pAgglomerate->vBonds.clear();
	}
	accept();
}
