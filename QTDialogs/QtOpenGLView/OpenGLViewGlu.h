/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "OpenGLView.h"

class COpenGLViewGlu : public COpenGLView
{
	Q_OBJECT

public:
	COpenGLViewGlu(CViewSettings* _viewSettings, QWidget* _parent = nullptr);
	COpenGLViewGlu(const CBaseGLView& _other, CViewSettings* _viewSettings, QWidget* _parent = nullptr);
	~COpenGLViewGlu();

protected:
	void initializeGL() override;

private:
	void DrawParticles() override;
	void DrawParticlesWithFFP(); // standard OpenGL functionality
	void DrawSelectedParticles();
};
