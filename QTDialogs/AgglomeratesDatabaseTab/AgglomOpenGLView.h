/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include <QGLWidget>
#include <QWheelEvent>
#include <GL/glu.h>
#include <QtOpenGL>
#include <QImage>
#include <QPainter>
#include <QGLFunctions>
#include "qtOperations.h"
#include "AgglomeratesDatabase.h"

class CAgglomOpenGLView : public  QGLWidget
{
	 Q_OBJECT

private:
	// camera parameters:
	float m_pCameraTranslation[3];
	float m_pCameraRotation[3];

	// window size
	int m_nWindowWidth;
	int m_nWindowHeight;

	// projection matrix camera parameters:
	float m_fFovy;
	float m_fZnear;
	float m_fZfar;

	GLUquadricObj *m_pQuadObj;
	QPoint m_LastMousePos; // last mouse position

	SAgglomerate m_CurrentAgglomerate;

public:
	CAgglomOpenGLView(QWidget *parent = 0);

	void SetCurrentAgglomerate(SAgglomerate* _pNewAgglomerate);

public slots:
	void UpdateView();

	// place the point of view into the center of the system
	void AutoCentrateView();

protected:
	// standard OpenGL functions of initialization, painting the scene and resizing
	void initializeGL();
	void SetupViewPort(int width, int height);
	void paintGL();
	void resizeGL(int width, int height);

private:
	void DrawScene();

	// increase/decrease zoom - positive=increase; negative=decrease
	void ZoomView( int _nZoomIn );

	// mouse events
	void mouseMoveEvent( QMouseEvent *event );
	void mousePressEvent( QMouseEvent *event );
	void wheelEvent ( QWheelEvent * event );

	inline void SetColorGL( CColor _color ) { glColor4f( _color.r, _color.g, _color.b, _color.a ); }
};