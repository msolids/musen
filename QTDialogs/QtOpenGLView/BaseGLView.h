/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "OpenGLTypes.h"
#include <QFont>
#include <QColor>
#include <QSize>

// Common interface for all viewers.
class CBaseGLView
{
protected:
	struct SFont
	{
		QFont font;
		QColor color;
	};

	QSize m_windowSize{ 100, 100 };	// Current window size.
	SViewport m_viewport;			// Current viewport.

	QVector3D m_cameraRotation{ -90.0f , 0.0f , 0.0f };		// Vector to describe rotations of camera.
	QVector3D m_cameraTranslation{ 0.0f, 0.0f, -0.18f };	// Vector to describe transitions of camera.

	QPoint m_lastMousePos;	// Last mouse position, needed to track mouse movements.

	uint8_t m_scaling{ 1 };	// Scaling factor needed to adjust quality during rendering into a file, may be in range [1:10].

	SFont m_fontTime;
	SFont m_fontAxes;
	SFont m_fontLegend;

public:
	CBaseGLView()                                         = default;
	virtual ~CBaseGLView()                                = default;
	CBaseGLView(const CBaseGLView& _other)                = default;
	CBaseGLView(CBaseGLView&& _other) noexcept            = default;
	CBaseGLView& operator=(const CBaseGLView& _other)     = default;
	CBaseGLView& operator=(CBaseGLView&& _other) noexcept = default;

	virtual void SetRenderQuality(uint8_t) = 0;
	virtual void SetParticleTexture(const QString&) = 0;
	virtual void Redraw() = 0;
	virtual QImage Snapshot(uint8_t) = 0;
	virtual SBox WinCoord2LineOfSight(const QPoint&) const = 0;	// Converts window coordinates to a scene coordinates, describing the line of sight through this point.
	virtual void UpdatePerspective() = 0;	// Sets new viewport according to given parameters and updates perspective projection matrix accordingly.

	void SetFontTime(const QFont& _font, const QColor& _color);
	void SetFontAxes(const QFont& _font, const QColor& _color);
	void SetFontLegend(const QFont& _font, const QColor& _color);

	// Sets camera to show the whole volume of the _box, from direction pointed by _cameraDirection(X,Y,Z). For X/Y/Z -1, 0 or 1 are allowed.
	// -1: the corresponding scene's coordinate will be pointed from observer; '1': the corresponding scene's coordinate will be pointed to observer.
	virtual void SetCameraStandardView(const SBox& _box, const QVector3D& _cameraDirection);
	// Returns current settings of the camera.
	SCameraSettings GetCameraSettings() const;
	// Sets new settings of the camera.
	void SetCameraSettings(const SCameraSettings& _settings);

protected:
	// Sets font type, size and color to painter. Painter must be initialized.
	void SetupPainter(QPainter* _painter, const SFont& _font) const;
};

