/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "BaseGLView.h"
#include <QPainter>
#include <QtMath>

void CBaseGLView::SetFontTime(const QFont& _font, const QColor& _color)
{
		m_fontTime.font = _font;
		m_fontTime.color = _color;
	}

void CBaseGLView::SetFontAxes(const QFont& _font, const QColor& _color)
{
		m_fontAxes.font = _font;
		m_fontAxes.color = _color;
	}

void CBaseGLView::SetFontLegend(const QFont& _font, const QColor& _color)
{
		m_fontLegend.font = _font;
		m_fontLegend.color = _color;
	}

void CBaseGLView::SetCameraStandardView(const SBox& _box, const QVector3D& _cameraDirection)
{
	const QVector3D maxLength{
		std::max(_box.maxCoord.y() - _box.minCoord.y(), _box.maxCoord.z() - _box.minCoord.z()),
		std::max(_box.maxCoord.x() - _box.minCoord.x(), _box.maxCoord.z() - _box.minCoord.z()),
		std::max(_box.maxCoord.x() - _box.minCoord.x(), _box.maxCoord.y() - _box.minCoord.y()) };
	const QVector3D midCoord = (_box.maxCoord + _box.minCoord) / 2;

	// update viewport and projection matrix
	m_viewport.zNear = std::max({ maxLength.x(), maxLength.y(), maxLength.z() }) * 10e-2f;
	m_viewport.zFar = m_viewport.zNear * 10e+5f;
	UpdatePerspective();

	// set camera translation
	m_cameraTranslation[0] = -midCoord.x();
	m_cameraTranslation[1] = -midCoord.z();
	m_cameraTranslation[2] = -midCoord.y();

	// setup camera position according to length and height
	if (_cameraDirection.x() != 0)	m_cameraTranslation[0] = -midCoord.y();
	if (_cameraDirection.x() == -1)	m_cameraTranslation[0] += 2.0f * midCoord.y();

	if (_cameraDirection.y() == 1)	m_cameraTranslation[0] += 2.0f * midCoord.x();

	if (_cameraDirection.z() != 0)	std::swap(m_cameraTranslation[1], m_cameraTranslation[2]);
	if (_cameraDirection.z() == -1)	m_cameraTranslation[1] += 2.0f * midCoord.y();

	// setup camera position according to depth
	if (_cameraDirection.x() != 0)	m_cameraTranslation[2] = -maxLength.x() / (2 * std::tan(m_viewport.fovy / 2 * static_cast<float>(M_PI) / 180));
	if (_cameraDirection.x() == 1)	m_cameraTranslation[2] -= _box.maxCoord.x();
	if (_cameraDirection.x() == -1)	m_cameraTranslation[2] += _box.minCoord.x();

	if (_cameraDirection.y() != 0)	m_cameraTranslation[2] = -maxLength.y() / (2 * std::tan(m_viewport.fovy / 2 * static_cast<float>(M_PI) / 180));
	if (_cameraDirection.y() == 1)	m_cameraTranslation[2] -= _box.maxCoord.y();
	if (_cameraDirection.y() == -1)	m_cameraTranslation[2] += _box.minCoord.y();

	if (_cameraDirection.z() != 0)	m_cameraTranslation[2] = -maxLength.z() / (2 * std::tan(m_viewport.fovy / 2 * static_cast<float>(M_PI) / 180));
	if (_cameraDirection.z() == 1)	m_cameraTranslation[2] -= _box.maxCoord.z();
	if (_cameraDirection.z() == -1)	m_cameraTranslation[2] += _box.minCoord.z();

	// set camera rotation
	m_cameraRotation[0] = _cameraDirection.z() * 90.0f - 90.0f;
	m_cameraRotation[1] = 0.0f;
	m_cameraRotation[2] = -_cameraDirection.x() * 90.0f + 180.0f * std::max(_cameraDirection.y(), 0.0f);
}

SCameraSettings CBaseGLView::GetCameraSettings() const
{
	return { m_viewport, m_cameraTranslation, m_cameraRotation };
}

void CBaseGLView::SetCameraSettings(const SCameraSettings& _settings)
{
	m_viewport = _settings.viewport;
	m_cameraTranslation = _settings.translation;
	m_cameraRotation = _settings.rotation;
	UpdatePerspective();
	Redraw();
}

void CBaseGLView::SetupPainter(QPainter* _painter, const SFont& _font) const
{
	QFont font(_font.font);
	font.setPointSize(_font.font.pointSize() * m_scaling);
	_painter->setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);
	_painter->setPen(_font.color);
	_painter->setFont(font);
}
