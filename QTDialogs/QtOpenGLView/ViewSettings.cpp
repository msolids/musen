/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ViewSettings.h"
#include "MixedFunctions.h"

CViewSettings::CViewSettings(QSettings* _settings) :
	m_settings(_settings)
{
	m_fontAxes.font.setBold(true);
	m_fontTime.color = QColor{0, 130, 200};
	LoadConfiguration();
}

void CViewSettings::RenderType(ERenderType _val)
{
	m_renderType = _val;
	// TODO: do not save it here
	m_settings->setValue(c_RENDER_TYPE, E2I(m_renderType));
}

void CViewSettings::RenderQuality(uint8_t _val)
{
	m_renderQuality = std::max(static_cast<uint8_t>(0), std::min(_val, static_cast<uint8_t>(100)));
	m_settings->setValue(c_RENDER_QUALITY, m_renderQuality);
}

void CViewSettings::ParticleTexture(const QString& _val)
{
	if (_val.isEmpty()) return;
	m_particleTexture = _val;
	m_settings->setValue(c_RENDER_PART_TEXTURE, m_particleTexture);
}

void CViewSettings::Visibility(const SVisibility& _val)
{
	m_visibility = _val;
	m_settings->setValue(c_DISPLAY_SHOW_PARTICLES,   m_visibility.particles);
	m_settings->setValue(c_DISPLAY_SHOW_BONDS,       m_visibility.bonds);
	m_settings->setValue(c_DISPLAY_SHOW_WALLS,       m_visibility.geometries);
	m_settings->setValue(c_DISPLAY_SHOW_VOLUMES,     m_visibility.volumes);
	m_settings->setValue(c_DISPLAY_SHOW_DOMAIN,      m_visibility.domain);
	m_settings->setValue(c_DISPLAY_SHOW_ORIENTATION, m_visibility.orientations);
	m_settings->setValue(c_DISPLAY_SHOW_AXES,        m_visibility.axes);
	m_settings->setValue(c_DISPLAY_SHOW_TIME,        m_visibility.time);
	m_settings->setValue(c_DISPLAY_SHOW_LEGEND,      m_visibility.legend);
}

void CViewSettings::FontAxes(const SFont& _val)
{
	m_fontAxes = _val;
	m_settings->setValue(c_DISPLAY_FONT_AXES,       m_fontAxes.font);
	m_settings->setValue(c_DISPLAY_FONT_COLOR_AXES, m_fontAxes.color);
}

void CViewSettings::FontTime(const SFont& _val)
{
	m_fontTime = _val;
	m_settings->setValue(c_DISPLAY_FONT_TIME,       m_fontTime.font);
	m_settings->setValue(c_DISPLAY_FONT_COLOR_TIME, m_fontTime.color);
}

void CViewSettings::FontLegend(const SFont& _val)
{
	m_fontLegend = _val;
	m_settings->setValue(c_DISPLAY_FONT_LEGEND,       m_fontLegend.font);
	m_settings->setValue(c_DISPLAY_FONT_COLOR_LEGEND, m_fontLegend.color);
}

void CViewSettings::LoadConfiguration()
{
	if (m_settings->value(c_RENDER_TYPE).isValid())
		m_renderType = static_cast<ERenderType>(m_settings->value(c_RENDER_TYPE).toUInt());
	if (m_settings->value(c_RENDER_QUALITY).isValid())
		m_renderQuality = m_settings->value(c_RENDER_QUALITY).toUInt();
	if (m_settings->value(c_RENDER_PART_TEXTURE).isValid() && !m_settings->value(c_RENDER_PART_TEXTURE).toString().isEmpty())
		m_particleTexture = m_settings->value(c_RENDER_PART_TEXTURE).toString();

	if (m_settings->value(c_DISPLAY_SHOW_PARTICLES).isValid())
		m_visibility.particles = m_settings->value(c_DISPLAY_SHOW_PARTICLES).toBool();
	if (m_settings->value(c_DISPLAY_SHOW_BONDS).isValid())
		m_visibility.bonds = m_settings->value(c_DISPLAY_SHOW_BONDS).toBool();
	if (m_settings->value(c_DISPLAY_SHOW_WALLS).isValid())
		m_visibility.geometries = m_settings->value(c_DISPLAY_SHOW_WALLS).toBool();
	if (m_settings->value(c_DISPLAY_SHOW_VOLUMES).isValid())
		m_visibility.volumes = m_settings->value(c_DISPLAY_SHOW_VOLUMES).toBool();
	if (m_settings->value(c_DISPLAY_SHOW_DOMAIN).isValid())
		m_visibility.domain = m_settings->value(c_DISPLAY_SHOW_DOMAIN).toBool();
	if (m_settings->value(c_DISPLAY_SHOW_ORIENTATION).isValid())
		m_visibility.orientations = m_settings->value(c_DISPLAY_SHOW_ORIENTATION).toBool();
	if (m_settings->value(c_DISPLAY_SHOW_AXES).isValid())
		m_visibility.axes = m_settings->value(c_DISPLAY_SHOW_AXES).toBool();
	if (m_settings->value(c_DISPLAY_SHOW_TIME).isValid())
		m_visibility.time = m_settings->value(c_DISPLAY_SHOW_TIME).toBool();
	if (m_settings->value(c_DISPLAY_SHOW_LEGEND).isValid())
		m_visibility.legend = m_settings->value(c_DISPLAY_SHOW_LEGEND).toBool();

	if (m_settings->value(c_DISPLAY_FONT_AXES).isValid())
		m_fontAxes.font = m_settings->value(c_DISPLAY_FONT_AXES).value<QFont>();
	if (m_settings->value(c_DISPLAY_FONT_COLOR_AXES).isValid())
		m_fontAxes.color = m_settings->value(c_DISPLAY_FONT_COLOR_AXES).value<QColor>();
	if (m_settings->value(c_DISPLAY_FONT_TIME).isValid())
		m_fontTime.font = m_settings->value(c_DISPLAY_FONT_TIME).value<QFont>();
	if (m_settings->value(c_DISPLAY_FONT_COLOR_TIME).isValid())
		m_fontTime.color = m_settings->value(c_DISPLAY_FONT_COLOR_TIME).value<QColor>();
	if (m_settings->value(c_DISPLAY_FONT_LEGEND).isValid())
		m_fontLegend.font = m_settings->value(c_DISPLAY_FONT_LEGEND).value<QFont>();
	if (m_settings->value(c_DISPLAY_FONT_COLOR_LEGEND).isValid())
		m_fontLegend.color = m_settings->value(c_DISPLAY_FONT_COLOR_LEGEND).value<QColor>();
}
