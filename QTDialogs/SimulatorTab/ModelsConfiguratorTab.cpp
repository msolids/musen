/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelsConfiguratorTab.h"
#include "ModelWidget.h"
#include "QtSignalBlocker.h"
#include <QDesktopServices>

CModelsConfiguratorTab::CModelsConfiguratorTab(CModelManager* _modelManager, bool _blockModelsChange, QWidget* _parent)
	: QDialog{ _parent }
	, m_modelManager{ _modelManager }
	, m_blockModelsChange{ _blockModelsChange }
{
	ui.setupUi(this);

	m_layouts[EMusenModelType::PP] = ui.layoutPP;
	m_layouts[EMusenModelType::PW] = ui.layoutPW;
	m_layouts[EMusenModelType::SB] = ui.layoutSB;
	m_layouts[EMusenModelType::EF] = ui.layoutEF;
	m_layouts[EMusenModelType::PPHT] = ui.layoutHT;

	InitializeConnections();
}

void CModelsConfiguratorTab::InitializeConnections()
{
	connect(ui.buttonOK, &QPushButton::clicked, this, &QDialog::accept);

	connect(ui.buttonAddPP, &QPushButton::clicked, this, [=] { AddModelClicked(EMusenModelType::PP); });
	connect(ui.buttonAddPW, &QPushButton::clicked, this, [=] { AddModelClicked(EMusenModelType::PW); });
	connect(ui.buttonAddSB, &QPushButton::clicked, this, [=] { AddModelClicked(EMusenModelType::SB); });
	connect(ui.buttonAddEF, &QPushButton::clicked, this, [=] { AddModelClicked(EMusenModelType::EF); });
	connect(ui.buttonAddHT, &QPushButton::clicked, this, [=] { AddModelClicked(EMusenModelType::PPHT); });
}

void CModelsConfiguratorTab::setVisible(bool _visible)
{
	if (_visible)
		UpdateWholeView();
	QDialog::setVisible(_visible);
}

void CModelsConfiguratorTab::UpdateWholeView()
{
	UpdateSelectedModels();
}

void CModelsConfiguratorTab::UpdateSelectedModels()
{
	const auto AddWidgets = [&](EMusenModelType _type)
	{
		[[maybe_unused]] CQtSignalBlocker blocker{ m_layouts[_type] };
		//for (const auto* widget : m_widgets[_type])
		//	delete widget;
		while (m_layouts[_type]->takeAt(0) != nullptr)
		{
			QLayoutItem* item = m_layouts[_type]->takeAt(0);
			m_layouts[_type]->removeWidget(item->widget());
			delete item->widget();
			delete item;
		}
		for (const auto& modelDescriptor : m_modelManager->GetModelsDescriptors(_type))
			AddModelWedget(_type, modelDescriptor);
	};

	AddWidgets(EMusenModelType::PP);
	AddWidgets(EMusenModelType::PW);
	AddWidgets(EMusenModelType::SB);
	AddWidgets(EMusenModelType::EF);
	AddWidgets(EMusenModelType::PPHT);
}

void CModelsConfiguratorTab::AddModelClicked(EMusenModelType _type)
{
	AddModelWedget(_type, nullptr);
}

void CModelsConfiguratorTab::RemoveModelClicked(EMusenModelType _type, CModelWidget* _widget)
{
	const auto* modelDescriptor = _widget->GetModelDescriptor();
	if (modelDescriptor && modelDescriptor->GetModel())
		m_modelManager->RemoveActiveModel(modelDescriptor->GetName());
	m_layouts[_type]->removeWidget(_widget);
	delete _widget;
}

void CModelsConfiguratorTab::AddModelWedget(EMusenModelType _type, CModelDescriptor* _modelDescriptor)
{
	auto* widget = new CModelWidget{ m_modelManager, _type, _modelDescriptor, this };
	widget->setEnabled(!m_blockModelsChange);
	connect(widget, &CModelWidget::RemoveRequested, this, [=] { RemoveModelClicked(_type, widget); });
	m_layouts[_type]->addWidget(widget);
}

int CModelsConfiguratorTab::exec()
{
	activateWindow();
	QDialog::exec();
	return Accepted;
}
