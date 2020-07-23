/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ui_CameraSettings.h"
#include <QDialog>
#include "ViewManager.h"

class CCameraSettings : public QDialog
{
	Q_OBJECT

	enum ERows
	{
		ANGLE, ASPECT, ZNEAR, ZFAR, POSX, POSY, POSZ, ROTX, ROTY, ROTZ
	};

	Ui::CCameraSettings ui{};

	CViewManager* m_viewManager;

public:
	CCameraSettings(CViewManager* _viewManager, QWidget* _parent = nullptr);

private:
	void setVisible(bool _visible) override;

	void UpdateWholeView() const;	// Updates all widgets.

	void SetCamera() const;			// Applies current camera settings

	void Export();					// Exports current camera settings to a text file.
	void Import();					// Imports current camera settings from a text file and applies them.
};
