/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "CameraSettings.h"

#include "MUSENStringFunctions.h"

#include <fstream>
#include <QFileDialog>

CCameraSettings::CCameraSettings(CViewManager* _viewManager, QWidget* _parent)
	: QDialog(_parent),
	m_viewManager{ _viewManager }
{
	ui.setupUi(this);
	connect(ui.buttonExport,    &QPushButton::clicked,			this, &CCameraSettings::Export);
	connect(ui.buttonImport,    &QPushButton::clicked,			this, &CCameraSettings::Import);
	connect(ui.table,			&QTableWidget::itemChanged,		this, &CCameraSettings::SetCamera);
}

void CCameraSettings::setVisible(bool _visible)
{
	QDialog::setVisible(_visible);
	if (_visible)
		UpdateWholeView();

	// keep connected only when the window is visible to spam with CameraChanged signal
	if (_visible)
		connect(m_viewManager, &CViewManager::CameraChanged, this, &CCameraSettings::UpdateWholeView);
	else
		disconnect(m_viewManager, &CViewManager::CameraChanged, this, &CCameraSettings::UpdateWholeView);
}

void CCameraSettings::UpdateWholeView() const
{
	QSignalBlocker blocker{ ui.table };

	const SCameraSettings settings = m_viewManager->GetCameraSettings();
	ui.table->SetItemEditable(		ERows::ANGLE,  0, settings.viewport.fovy);
	ui.table->SetItemNotEditable(	ERows::ASPECT, 0, settings.viewport.aspect);
	ui.table->SetItemEditable(		ERows::ZNEAR,  0, settings.viewport.zNear);
	ui.table->SetItemEditable(		ERows::ZFAR,   0, settings.viewport.zFar);
	ui.table->SetItemEditable(		ERows::POSX,   0, settings.translation.x());
	ui.table->SetItemEditable(		ERows::POSY,   0, settings.translation.y());
	ui.table->SetItemEditable(		ERows::POSZ,   0, settings.translation.z());
	ui.table->SetItemEditable(		ERows::ROTX,   0, settings.rotation.x());
	ui.table->SetItemEditable(		ERows::ROTY,   0, settings.rotation.y());
	ui.table->SetItemEditable(		ERows::ROTZ,   0, settings.rotation.z());
}

void CCameraSettings::SetCamera() const
{
	const SCameraSettings settings{
		SViewport {
		ui.table->item(ERows::ANGLE,  0)->text().toFloat(),
		ui.table->item(ERows::ASPECT, 0)->text().toFloat(),
		ui.table->item(ERows::ZNEAR,  0)->text().toFloat(),
		ui.table->item(ERows::ZFAR,   0)->text().toFloat()},
		QVector3D {
		ui.table->item(ERows::POSX,   0)->text().toFloat(),
		ui.table->item(ERows::POSY,   0)->text().toFloat(),
		ui.table->item(ERows::POSZ,   0)->text().toFloat()},
		QVector3D {
		ui.table->item(ERows::ROTX,   0)->text().toFloat(),
		ui.table->item(ERows::ROTY,   0)->text().toFloat(),
		ui.table->item(ERows::ROTZ,   0)->text().toFloat()}
	};
	m_viewManager->SetCameraSettings(settings);
}

void CCameraSettings::Export()
{
	const QString fileName = QFileDialog::getSaveFileName(this, tr("Export camera settings"), "", tr("Text files (*.txt);;All files (*.*);;"));
	if (fileName.simplified().isEmpty()) return;
	std::ofstream file{ UnicodePath(fileName.toStdString()) };
	if (file.fail()) return;

	const SCameraSettings settings = m_viewManager->GetCameraSettings();
	file << settings.viewport.fovy << std::endl << settings.viewport.aspect << std::endl << settings.viewport.zNear << std::endl << settings.viewport.zFar << std::endl
		 << settings.translation[0] << std::endl << settings.translation[1] << std::endl << settings.translation[2] << std::endl
		 << settings.rotation[0] << std::endl << settings.rotation[1] << std::endl << settings.rotation[2];
	file.close();
}

void CCameraSettings::Import()
{
	const QString fileName = QFileDialog::getOpenFileName(this, tr("Import camera settings"), "", tr("Text files (*.txt);;All files (*.*);;"));
	if (fileName.simplified().isEmpty()) return;
	std::ifstream file{ UnicodePath(fileName.toStdString()) };
	if (file.fail()) return;

	SCameraSettings settings = m_viewManager->GetCameraSettings();
	file >> settings.viewport.fovy >> settings.viewport.aspect >> settings.viewport.zNear >> settings.viewport.zFar
		 >> settings.translation[0] >> settings.translation[1] >> settings.translation[2]
		 >> settings.rotation[0] >> settings.rotation[1] >> settings.rotation[2];
	file.close();

	m_viewManager->SetCameraSettings(settings);
}
