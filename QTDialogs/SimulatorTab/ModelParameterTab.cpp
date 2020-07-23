/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelParameterTab.h"

CModelParameterTab::CModelParameterTab(CAbstractDEMModel* _pModel, QWidget *parent) :QDialog(parent)
{
	ui.setupUi(this);
	m_pModel = _pModel;
	ui.parametersTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
	CreateInitialForms();
	UpdateParametersView();
	InitializeConnections();
}

void CModelParameterTab::setVisible(bool _bVisible)
{
	if (_bVisible)
		UpdateWholeView();
	QDialog::setVisible(_bVisible);
}

void CModelParameterTab::InitializeConnections()
{
	QObject::connect(ui.applyButton, SIGNAL(clicked()), this, SLOT(SetNewParameters()));
	QObject::connect(ui.setDefault, SIGNAL(clicked()), this, SLOT(SetDefaultParameters()));
}

void CModelParameterTab::CreateInitialForms()
{
	std::vector<SModelParameter> vParameters = m_pModel->GetAllParameters();
	ui.parametersTable->setRowCount((int)vParameters.size());
	for (int i = 0; i < (int)vParameters.size(); ++i)
	{
		ui.parametersTable->SetItemNotEditable(i, 0, ss2qs(vParameters[i].uniqueName));
		ui.parametersTable->SetItemNotEditable(i, 1, ss2qs(vParameters[i].description));
		ui.parametersTable->SetItemEditable(i, 2, QString::number(vParameters[i].value));
	}
}

void CModelParameterTab::UpdateWholeView()
{
	UpdateParametersView();
}

void CModelParameterTab::UpdateParametersView()
{
	std::vector<SModelParameter> vParameters = m_pModel->GetAllParameters();
	for (int i = 0; i < (int)vParameters.size(); ++i)
		ui.parametersTable->item(i, 2)->setText(QString::number(vParameters[i].value));
}

void CModelParameterTab::SetDefaultParameters()
{
	m_pModel->SetDefaultValues();
	UpdateParametersView();
}

void CModelParameterTab::SetNewParameters()
{
	std::vector<SModelParameter> vParameters = m_pModel->GetAllParameters();
	for (int i = 0; i < (int)vParameters.size(); ++i)
		m_pModel->SetParameterValue(vParameters[i].uniqueName, ui.parametersTable->item(i, 2)->text().toDouble());
	close();
}
