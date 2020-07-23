/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "GeneralMUSENDialog.h"
#include "ViewManager.h"
#include "ui_ConfigurationTab.h"

class CConfigurationTab: public CMusenDialog
{
	Q_OBJECT
private:
	const QString c_ASK_TD_DATA_REMOVAL = "ASK_TD_DATA_REMOVAL";

	Ui::configurationTab ui;

	QSettings* m_settings;
	CViewManager* m_viewManager;
	CViewSettings* m_viewSettings;

public:
	CConfigurationTab(CViewManager* _viewManager, CViewSettings* _viewSettings, QSettings* _pSettings, QWidget* parent = nullptr);

public slots:
	void UpdateWholeView() override;

private:
	void InitializeConnections() const;
	void LoadConfiguration() const;

private slots:
	void RenderTypeChanged(bool _checked) const;
	void ViewQualityChanged() const;
	void SetPartTextureDefault() const;
	void LoadPartTexture();
	void PickPartTexture();
	void ShowAxesToggled() const;
	void ShowTimeToggled() const;
	void ShowLegendToggled() const;
	void OnAxisFontChange();
	void OnTimeFontChange();
	void OnLegendFontChange();
	void OnAxisColorChange() const;
	void OnTimeColorChange() const;
	void OnLegendColorChange() const;
	void OnAskTDDataChange() const;
};

