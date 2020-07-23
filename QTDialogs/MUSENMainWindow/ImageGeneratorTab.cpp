/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ImageGeneratorTab.h"
#include "QtSignalBlocker.h"
#include <QFileDialog>
#include <QDateTime>

CImageGeneratorTab::CImageGeneratorTab(CViewManager* _viewManager, QSettings* _settings, QWidget* parent) :
	CMusenDialog(parent),
	m_settings{ _settings },
	m_viewManager{ _viewManager }
{
	ui.setupUi(this);

	LoadConfiguration();
	InitializeConnections();
}

void CImageGeneratorTab::InitializeConnections()
{
	// time
	connect(ui.lineEditStartTime, &QLineEdit::editingFinished, this, &CImageGeneratorTab::TimeParametersChanged);
	connect(ui.lineEditEndTime,   &QLineEdit::editingFinished, this, &CImageGeneratorTab::TimeParametersChanged);
	connect(ui.toolButtonMinTime, &QToolButton::clicked,       this, &CImageGeneratorTab::SetMinTime);
	connect(ui.toolButtonMaxTime, &QToolButton::clicked,       this, &CImageGeneratorTab::SetMaxTime);

	// time step
	connect(ui.radioOnlySaved,   &QRadioButton::clicked, this, &CImageGeneratorTab::TimeModeChanged);
	connect(ui.radioTimeStep,    &QRadioButton::clicked, this, &CImageGeneratorTab::TimeModeChanged);
	connect(ui.lineEditTimeStep, &QLineEdit::textEdited, this, &CImageGeneratorTab::TimeParametersChanged);

	// update config
	connect(ui.spinBoxScaling,     QOverload<int>::of(&QSpinBox::valueChanged),         this, &CImageGeneratorTab::SaveConfiguration);
	connect(ui.spinBoxCompression, QOverload<int>::of(&QSpinBox::valueChanged),         this, &CImageGeneratorTab::SaveConfiguration);
	connect(ui.lineEditPrefix,	                      &QLineEdit::editingFinished,      this, &CImageGeneratorTab::SaveConfiguration);
	connect(ui.comboExtension,     QOverload<int>::of(&QComboBox::currentIndexChanged), this, &CImageGeneratorTab::SaveConfiguration);

	// buttons
	connect(ui.toolButtonPickFolder, &QToolButton::clicked, this, &CImageGeneratorTab::SelectOutputFolder);
	connect(ui.pushButtonStart,      &QPushButton::clicked, this, &CImageGeneratorTab::StartGeneration);
}

void CImageGeneratorTab::UpdateWholeView()
{
	ShowConvLabel(ui.startTimeLabel, "Start", EUnitType::TIME);
	ShowConvLabel(ui.endTimeLabel, "End time", EUnitType::TIME);
	ShowConvLabel(ui.radioTimeStep, "Time step", EUnitType::TIME);

	SetMinTime();
	SetMaxTime();
	SetDefaultTimeStep();
	TimeModeChanged();
	UpdateOutputPath();

	ui.labelStatus->clear();
}

void CImageGeneratorTab::TimeParametersChanged() const
{
	CQtSignalBlocker blocker({ ui.lineEditStartTime, ui.lineEditEndTime, ui.lineEditTimeStep });

	if (GetConvValue(ui.lineEditStartTime, EUnitType::TIME) < 0)	ShowConvValue(ui.lineEditStartTime, 0, EUnitType::TIME, 10);
	if (GetConvValue(ui.lineEditEndTime, EUnitType::TIME) < 0)		ShowConvValue(ui.lineEditEndTime, 0, EUnitType::TIME, 10);
	if (GetConvValue(ui.lineEditTimeStep, EUnitType::TIME) < 0)	ShowConvValue(ui.lineEditTimeStep, 0, EUnitType::TIME, 10);

	UpdateImagesNumber();
}

void CImageGeneratorTab::UpdateImagesNumber() const
{
	ui.lineEditImagesNumber->setText(QString::number(GetTimePoints().size()));
}

void CImageGeneratorTab::UpdateOutputPath(const QString& _path) const
{
	CQtSignalBlocker blocker(ui.lineEditFolder);
	if (!_path.isEmpty())
		ui.lineEditFolder->setText(_path);
	else if(ui.lineEditFolder->text().isEmpty())
		ui.lineEditFolder->setText(QFileInfo(QString::fromStdString(m_pSystemStructure->GetFileName())).absolutePath());
}

void CImageGeneratorTab::TimeModeChanged() const
{
	ui.lineEditTimeStep->setEnabled(ui.radioTimeStep->isChecked());
	UpdateImagesNumber();
}

void CImageGeneratorTab::SetMinTime() const
{
	CQtSignalBlocker blocker(ui.lineEditStartTime);
	ShowConvValue(ui.lineEditStartTime, m_pSystemStructure->GetMinTime(), EUnitType::TIME, 10);
	UpdateImagesNumber();
}

void CImageGeneratorTab::SetMaxTime() const
{
	CQtSignalBlocker blocker(ui.lineEditEndTime);
	ShowConvValue(ui.lineEditEndTime, m_pSystemStructure->GetMaxTime(), EUnitType::TIME, 10);
	UpdateImagesNumber();
}

void CImageGeneratorTab::SetDefaultTimeStep() const
{
	CQtSignalBlocker blocker(ui.lineEditTimeStep);
	ShowConvValue(ui.lineEditTimeStep, (m_pSystemStructure->GetMaxTime() - m_pSystemStructure->GetMinTime()) / 100.0, EUnitType::TIME, 10);
}

void CImageGeneratorTab::SelectOutputFolder()
{
	const QString initPath = !ui.lineEditFolder->text().isEmpty() ? ui.lineEditFolder->text() : QFileInfo(QString::fromStdString(m_pSystemStructure->GetFileName())).absolutePath();
	const QString folderPath = QFileDialog::getExistingDirectory(this, "Output folder", initPath, QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
	UpdateOutputPath(folderPath);
}

void CImageGeneratorTab::StartGeneration()
{
	const QString namePrefix = !ui.lineEditPrefix->text().isEmpty() ? ui.lineEditPrefix->text() : QDateTime::currentDateTime().toString("yy-MM-dd_HH-mm-ss");
	const QString ext = ui.comboExtension->currentText();
	const QString folderPath = !ui.lineEditFolder->text().isEmpty() ? ui.lineEditFolder->text() : QFileInfo(QString::fromStdString(m_pSystemStructure->GetFileName())).absolutePath();
	const QString pathPrefix = folderPath + "/" + namePrefix;
	const uint8_t scaling = ui.spinBoxScaling->value();
	const uint8_t quality = 100 - ui.spinBoxCompression->value();

	if (!QDir(folderPath).exists())
		const bool res = QDir().mkdir(folderPath);

	setWindowModality(Qt::ApplicationModal);

	ui.labelStatus->setText("Image generation is running");
	std::vector<double> vTimePoints = GetTimePoints();
	for (size_t i = 0; i < vTimePoints.size(); ++i)
	{
		m_viewManager->SetTime(vTimePoints[i]);
		m_viewManager->GetSnapshot(scaling).save(pathPrefix + QString("%1").arg(i, 4, 10, QChar('0')) + ext, nullptr, quality);
		ui.progressBar->setValue((i + 1.0) / vTimePoints.size() * 100.0);
	}
	ui.labelStatus->setText("Files saved successfully");

	setWindowModality(Qt::NonModal);
}

std::vector<double> CImageGeneratorTab::GetTimePoints() const
{
	std::vector<double> timePoints;
	double timeMin = GetConvValue(ui.lineEditStartTime, EUnitType::TIME);
	double timeMax = GetConvValue(ui.lineEditEndTime, EUnitType::TIME);

	if (ui.radioOnlySaved->isChecked())
	{
		timePoints = m_pSystemStructure->GetAllTimePoints();
		timePoints.erase(std::remove_if(timePoints.begin(), timePoints.end(), [&timeMin, &timeMax](double t) { return t < timeMin || t > timeMax; }), timePoints.end());
	}
	else
	{
		const double timeStep = GetConvValue(ui.lineEditTimeStep, EUnitType::TIME);
		const size_t number = timeStep != 0 ? (timeMax - timeMin) / timeStep + 1 : 0;
		for (unsigned i = 0; i < number; ++i)
			timePoints.push_back(timeMin + timeStep * i);
	}

	return timePoints;
}

void CImageGeneratorTab::SaveConfiguration() const
{
	m_settings->setValue(c_SCALING_FACTOR,    ui.spinBoxScaling->value());
	m_settings->setValue(c_IMAGE_COMPRESSION, ui.spinBoxCompression->value());
	m_settings->setValue(c_IMAGE_PREFIX,      ui.lineEditPrefix->text());
	m_settings->setValue(c_IMAGE_EXTENSION,   ui.comboExtension->currentIndex());
}

void CImageGeneratorTab::LoadConfiguration() const
{
	CQtSignalBlocker blocker({ ui.spinBoxScaling, ui.spinBoxCompression, ui.lineEditPrefix, ui.comboExtension });

	if (m_settings->value(c_SCALING_FACTOR).isValid())		ui.spinBoxScaling->setValue(m_settings->value(c_SCALING_FACTOR).toInt());
	if (m_settings->value(c_IMAGE_COMPRESSION).isValid())	ui.spinBoxCompression->setValue(m_settings->value(c_IMAGE_COMPRESSION).toInt());
	if (m_settings->value(c_IMAGE_EXTENSION).isValid())		ui.comboExtension->setCurrentIndex(m_settings->value(c_IMAGE_EXTENSION).toInt());
	if (m_settings->value(c_IMAGE_PREFIX).isValid())		ui.lineEditPrefix->setText(m_settings->value(c_IMAGE_PREFIX).toString());
}
