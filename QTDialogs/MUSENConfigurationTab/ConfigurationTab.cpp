/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ConfigurationTab.h"
#include "TexturePicker.h"
#include "QtSignalBlocker.h"
#include <QFontDialog>
#include <QFileDialog>

CConfigurationTab::CConfigurationTab(CViewManager* _viewManager, CViewSettings* _viewSettings, QSettings* _pSettings, QWidget* parent)
	: CMusenDialog(parent),
	m_settings{ _pSettings },
	m_viewManager{ _viewManager },
	m_viewSettings{ _viewSettings }
{
	ui.setupUi(this);

	LoadConfiguration();
	InitializeConnections();
}

void CConfigurationTab::InitializeConnections() const
{
	// rendering settings
	connect(ui.radioRenderStandart,          &QRadioButton::toggled, this, &CConfigurationTab::RenderTypeChanged);
	connect(ui.radioRenderTexture,           &QRadioButton::toggled, this, &CConfigurationTab::RenderTypeChanged);
	connect(ui.radioRenderOpenGL,            &QRadioButton::toggled, this, &CConfigurationTab::RenderTypeChanged);
	connect(ui.sliderViewQuality,            &QSlider::valueChanged, this, &CConfigurationTab::ViewQualityChanged);
	connect(ui.pushButtonPartTextureDefault, &QPushButton::clicked,  this, &CConfigurationTab::SetPartTextureDefault);
	connect(ui.pushButtonPartTextureLoad,	 &QPushButton::clicked,  this, &CConfigurationTab::LoadPartTexture);
	connect(ui.pushButtonPartTexturePick,	 &QPushButton::clicked,  this, &CConfigurationTab::PickPartTexture);

	// visibility settings
	connect(ui.checkBoxShowAxes,         &QCheckBox::stateChanged, this, &CConfigurationTab::ShowAxesToggled);
	connect(ui.checkBoxShowTime,         &QCheckBox::stateChanged, this, &CConfigurationTab::ShowTimeToggled);
	connect(ui.checkBoxShowLegend,       &QCheckBox::stateChanged, this, &CConfigurationTab::ShowLegendToggled);

	// fonts
	connect(ui.toolButtonFontAxes,   &QToolButton::clicked, this, &CConfigurationTab::OnAxisFontChange);
	connect(ui.toolButtonFontTime,   &QToolButton::clicked, this, &CConfigurationTab::OnTimeFontChange);
	connect(ui.toolButtonFontLegend, &QToolButton::clicked, this, &CConfigurationTab::OnLegendFontChange);

	// colors
	connect(ui.widgetColorAxes,   &CColorView::ColorChanged, this, &CConfigurationTab::OnAxisColorChange);
	connect(ui.widgetColorTime,   &CColorView::ColorChanged, this, &CConfigurationTab::OnTimeColorChange);
	connect(ui.widgetColorLegend, &CColorView::ColorChanged, this, &CConfigurationTab::OnLegendColorChange);

	// system
	connect(ui.checkBoxAskTDData, &QCheckBox::stateChanged, this, &CConfigurationTab::OnAskTDDataChange);

	// buttons
	connect(ui.buttonClose, &QPushButton::clicked, this, &CConfigurationTab::accept);
}

void CConfigurationTab::LoadConfiguration() const
{
	CQtSignalBlocker blocker(ui.checkBoxAskTDData);

	// TODO: it is ton the best idea to store it directly in settings and access in simulator by name
	if(m_settings->contains(c_ASK_TD_DATA_REMOVAL))
		ui.checkBoxAskTDData->setChecked(m_settings->value(c_ASK_TD_DATA_REMOVAL).toBool());
	else
		ui.checkBoxAskTDData->setChecked(true);
}

void CConfigurationTab::UpdateWholeView()
{
	CQtSignalBlocker blocker({ ui.radioRenderStandart, ui.radioRenderTexture, ui.radioRenderOpenGL, ui.sliderViewQuality,	ui.checkBoxShowAxes,
		ui.checkBoxShowTime, ui.checkBoxShowLegend, ui.checkBoxAskTDData, ui.widgetColorAxes, ui.widgetColorTime, ui.widgetColorLegend });

	ui.radioRenderStandart->setChecked(m_viewSettings->RenderType() == ERenderType::GLU);
	ui.radioRenderTexture->setChecked(m_viewSettings->RenderType() == ERenderType::MIXED);
	ui.radioRenderOpenGL->setChecked(m_viewSettings->RenderType() == ERenderType::SHADER);

	ui.sliderViewQuality->setSliderPosition(m_viewSettings->RenderQuality());

	ui.checkBoxShowAxes->setChecked(m_viewSettings->Visibility().axes);
	ui.checkBoxShowTime->setChecked(m_viewSettings->Visibility().time);
	ui.checkBoxShowLegend->setChecked(m_viewSettings->Visibility().legend);

	ui.widgetColorAxes->SetColor(m_viewSettings->FontAxes().color);
	ui.widgetColorTime->SetColor(m_viewSettings->FontTime().color);
	ui.widgetColorLegend->SetColor(m_viewSettings->FontLegend().color);
}

void CConfigurationTab::RenderTypeChanged(bool _checked) const
{
	if (!_checked) return; // don't need to handle uncheck

	if (ui.radioRenderStandart->isChecked())
		m_viewSettings->RenderType(ERenderType::GLU);
	else if (ui.radioRenderTexture->isChecked())
		m_viewSettings->RenderType(ERenderType::MIXED);
	else if (ui.radioRenderOpenGL->isChecked())
		m_viewSettings->RenderType(ERenderType::SHADER);

	m_viewManager->UpdateRenderType();
}

void CConfigurationTab::ViewQualityChanged() const
{
	m_viewSettings->RenderQuality(ui.sliderViewQuality->sliderPosition());
	m_viewManager->UpdateViewQuality();
}

void CConfigurationTab::SetPartTextureDefault() const
{
	m_viewSettings->ParticleTexture(m_viewSettings->c_defaultPartTexture);
	m_viewManager->UpdateParticleTexture();
}

void CConfigurationTab::LoadPartTexture()
{
	m_viewSettings->ParticleTexture(QFileDialog::getOpenFileName(this, "Select particle texture", m_viewSettings->ParticleTexture(), "Image Files (*.png *.jpg *.bmp);;All files (*.*);;"));
	m_viewManager->UpdateParticleTexture();
}

void CConfigurationTab::PickPartTexture()
{
	CTexturePicker picker(this);
	if (picker.exec() == Accepted)
	{
		m_viewSettings->ParticleTexture(picker.SelectedTexture());
		m_viewManager->UpdateParticleTexture();
	}
}

void CConfigurationTab::ShowAxesToggled() const
{
	CViewSettings::SVisibility vis = m_viewSettings->Visibility();
	vis.axes = ui.checkBoxShowAxes->isChecked();
	m_viewSettings->Visibility(vis);
	m_viewManager->UpdateAxes();
}

void CConfigurationTab::ShowTimeToggled() const
{
	CViewSettings::SVisibility vis = m_viewSettings->Visibility();
	vis.time = ui.checkBoxShowTime->isChecked();
	m_viewSettings->Visibility(vis);
	m_viewManager->UpdateTime();
}

void CConfigurationTab::ShowLegendToggled() const
{
	CViewSettings::SVisibility vis = m_viewSettings->Visibility();
	vis.legend = ui.checkBoxShowLegend->isChecked();
	m_viewSettings->Visibility(vis);
	m_viewManager->UpdateLegend();
}

void CConfigurationTab::OnAxisFontChange()
{
	bool accepted;
	const QFont font = QFontDialog::getFont(&accepted, m_viewSettings->FontAxes().font, this);
	if (!accepted) return;
	m_viewSettings->FontAxes(CViewSettings::SFont{ font, m_viewSettings->FontAxes().color });
	m_viewManager->UpdateFontAxes();
}

void CConfigurationTab::OnTimeFontChange()
{
	bool accepted;
	const QFont font = QFontDialog::getFont(&accepted, m_viewSettings->FontTime().font, this);
	if (!accepted) return;
	m_viewSettings->FontTime(CViewSettings::SFont{ font, m_viewSettings->FontTime().color });
	m_viewManager->UpdateFontTime();
}

void CConfigurationTab::OnLegendFontChange()
{
	bool accepted;
	const QFont font = QFontDialog::getFont(&accepted, m_viewSettings->FontLegend().font, this);
	if (!accepted) return;
	m_viewSettings->FontLegend(CViewSettings::SFont{ font, m_viewSettings->FontLegend().color });
	m_viewManager->UpdateFontLegend();
}

void CConfigurationTab::OnAxisColorChange() const
{
	m_viewSettings->FontAxes(CViewSettings::SFont{ m_viewSettings->FontAxes().font, ui.widgetColorAxes->getColor() });
	m_viewManager->UpdateFontAxes();
}

void CConfigurationTab::OnTimeColorChange() const
{
	m_viewSettings->FontTime(CViewSettings::SFont{ m_viewSettings->FontTime().font, ui.widgetColorTime->getColor() });
	m_viewManager->UpdateFontTime();
}

void CConfigurationTab::OnLegendColorChange() const
{
	m_viewSettings->FontLegend(CViewSettings::SFont{ m_viewSettings->FontLegend().font, ui.widgetColorLegend->getColor() });
	m_viewManager->UpdateFontLegend();
}

void CConfigurationTab::OnAskTDDataChange() const
{
	m_settings->setValue(c_ASK_TD_DATA_REMOVAL, ui.checkBoxAskTDData->isChecked());
}
