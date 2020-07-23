/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ui_ImageGeneratorTab.h"
#include "GeneralMUSENDialog.h"
#include "ViewManager.h"

class CImageGeneratorTab: public CMusenDialog
{
	Q_OBJECT
private:
	const QString c_SCALING_FACTOR    = "SCALING_FACTOR";
	const QString c_IMAGE_COMPRESSION = "IMAGE_COMPRESSION";
	const QString c_IMAGE_PREFIX      = "IMAGE_PREFIX";
	const QString c_IMAGE_EXTENSION   = "IMAGE_EXTENSION";

	Ui::imageGeneratorTab ui;

	QSettings* m_settings;	     // File to save application specific settings.
	CViewManager* m_viewManager; // Manager of OpenGL views.

public:
	CImageGeneratorTab(CViewManager* _viewManager, QSettings* _settings, QWidget* parent = nullptr);
	void InitializeConnections();

public slots:
	void UpdateWholeView() override;

private slots:
	void TimeParametersChanged() const;
	void UpdateImagesNumber() const;
	void UpdateOutputPath(const QString& _path = "") const;
	void TimeModeChanged() const;
	void SetMinTime() const;
	void SetMaxTime() const;
	void SetDefaultTimeStep() const;
	void SelectOutputFolder();

	void StartGeneration();

private:
	std::vector<double> GetTimePoints() const;

	void SaveConfiguration() const;
	void LoadConfiguration() const;
};
