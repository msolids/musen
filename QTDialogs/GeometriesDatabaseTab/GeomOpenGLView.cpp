/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GeomOpenGLView.h"

CGeomOpenGLView::CGeomOpenGLView(QWidget *parent) : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
	// camera position:
	m_pCameraTranslation[0] = 0.0f;
	m_pCameraTranslation[1] = 0.0f;
	m_pCameraTranslation[2] = -0.18f;
	m_pCameraRotation[0] = -90.0f;
	m_pCameraRotation[1] = 0.0f;
	m_pCameraRotation[2] = 0.0f;

	m_fFovy = 15.0f;
	m_fZnear = 0.02f;
	m_fZfar = m_fZnear * 10000;

	m_nWindowWidth = 0;
	m_nWindowHeight = 0;

	m_sGeomName = "";
}

void CGeomOpenGLView::SetCurrentGeometry(const std::string& _sName, const std::vector<STriangleType>& _vTriangles)
{
	m_vAllTriangles = _vTriangles;

	if ( !m_vAllTriangles.empty() ) // shift geometry to the center of coordinate
	{
		CVector3 vCenterOfMass(0, 0, 0);
		for (size_t i = 0; i < m_vAllTriangles.size(); ++i)
			vCenterOfMass += m_vAllTriangles[i].p1 + m_vAllTriangles[i].p2 + m_vAllTriangles[i].p3;
		vCenterOfMass = vCenterOfMass / (m_vAllTriangles.size() * 3);
		for (size_t i = 0; i < m_vAllTriangles.size(); ++i)
		{
			m_vAllTriangles[i].p1 -= vCenterOfMass;
			m_vAllTriangles[i].p2 -= vCenterOfMass;
			m_vAllTriangles[i].p3 -= vCenterOfMass;
		}
	}
	m_sGeomName = _sName;
	AutoCentrateView();
	DrawScene();
}

void CGeomOpenGLView::initializeGL()
{
	setAutoFillBackground(false);
	glEnable(0x809D);
}

void CGeomOpenGLView::paintGL( )
{
	glClearColor( 1.0f, 1.0f, 1.0f, 0.0f ); // background color
	glEnable( GL_DEPTH_TEST );
	glDepthFunc( GL_LEQUAL );
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	glEnable( GL_LINE_SMOOTH );

	glLoadIdentity();

	glEnable( GL_LIGHTING );
	glEnable( GL_LIGHT0 );

	GLfloat Alpha = 1.0f;
	GLfloat Diffuse[4]  = { 1.0f, 1.0f, 1.0f, Alpha };
	GLfloat Ambient[4]  = { 1.0f, 1.0f, 1.0f, Alpha };
	GLfloat Specular[4] = { 0.4f, 0.4f, 0.4f, Alpha };
	GLfloat Phongsize   = 164.0f;
	glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE,   Diffuse   );
	glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT,   Ambient   );
	glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR,  Specular  );
	glMaterialf(  GL_FRONT_AND_BACK, GL_SHININESS, Phongsize );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

	//// Lighting:
	GLfloat pLightPosition[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glLightfv( GL_LIGHT0, GL_POSITION, pLightPosition );
	glLightfv( GL_LIGHT0, GL_DIFFUSE, pLightPosition );
	glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT,   Ambient   );
	glEnable( GL_COLOR_MATERIAL );

	SetupViewPort(width(), height());

	glClear( GL_COLOR_BUFFER_BIT |  GL_DEPTH_BUFFER_BIT );
	glMatrixMode( GL_MODELVIEW );

	glLoadIdentity();
	// camera transformation:
	glTranslatef( m_pCameraTranslation[0], m_pCameraTranslation[1], m_pCameraTranslation[2] );
	glRotatef( m_pCameraRotation[0], 1.0, 0.0, 0.0 );
	glRotatef( m_pCameraRotation[1], 0.0, 1.0, 0.0 );
	glRotatef( m_pCameraRotation[2], 0.0, 0.0, 1.0 );

	DrawScene();
}

void CGeomOpenGLView::SetupViewPort( int _nWidth, int _nHeight )
{
	//// saving new width and height:
	m_nWindowWidth = _nWidth;
	m_nWindowHeight = _nHeight;
	// changing viewport size to much the new size of the window:
	glViewport( 0, 0, _nWidth, _nHeight );
	// changing the projection matrix to match the new size of the window:
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluPerspective( m_fFovy, (float) _nWidth / (float) _nHeight, m_fZnear, m_fZfar );
}

void CGeomOpenGLView::resizeGL( int _nWidth, int _nHeight )
{
	SetupViewPort( _nWidth, _nHeight );
	paintGL();
}

void CGeomOpenGLView::DrawScene()
{
	// show the rest of objects
	for (size_t i = 0; i < m_vAllTriangles.size(); ++i)
	{
		SetColorGL( CColor(0.5f, 0.5f, 1.0f, 1.0f) );
		glPushMatrix();
		CVector3 Vert1 = m_vAllTriangles[i].p1;
		CVector3 Vert2 = m_vAllTriangles[i].p2 - Vert1;
		CVector3 Vert3 = m_vAllTriangles[i].p3 - Vert1;
		CVector3 vNormal = (Vert2)*(Vert3);
		vNormal = vNormal.Normalized();

		glTranslatef( (GLfloat)Vert1.x, (GLfloat)Vert1.y , (GLfloat)Vert1.z  );
		glBegin(GL_TRIANGLES);
		glNormal3f( (GLfloat)vNormal.x, (GLfloat)vNormal.y, (GLfloat)vNormal.z );    // point 1
		glVertex3f( 0.0f, -0.0f, -0.0f);    // point 1
		glNormal3f( (GLfloat)vNormal.x, (GLfloat)vNormal.y, (GLfloat)vNormal.z );    // point 1
		glVertex3f( (GLfloat)Vert2.x, (GLfloat)Vert2.y, (GLfloat)Vert2.z );    // point 2
		glNormal3f( (GLfloat)vNormal.x, (GLfloat)vNormal.y, (GLfloat)vNormal.z );    // point 1
		glVertex3f( (GLfloat)Vert3.x, (GLfloat)Vert3.y, (GLfloat)Vert3.z );   // point 3
		glEnd();
		glPopMatrix();
	}
}

void CGeomOpenGLView::AutoCentrateView()
{
	if ( m_vAllTriangles.empty() ) return;
	int nMagnitude;

	SVolumeType bb = GetBoundingBox(m_vAllTriangles);

	nMagnitude = (int)(log10( (bb.coordEnd.z - bb.coordBeg.z) ));
	m_fZfar = (float)(pow( 10.0, nMagnitude + 4.0 ));
	m_fZnear = (float)(pow( 10.0, nMagnitude - 1.0 ));

	m_pCameraTranslation[ 0 ] = (float)(-(bb.coordEnd.x+ bb.coordBeg.x)/2);
	m_pCameraTranslation[ 1 ] = (float)(-(bb.coordEnd.z+ bb.coordBeg.z)/2);
	double dMaxLength = fabs(bb.coordEnd.x - bb.coordBeg.x );
	if ( fabs(bb.coordEnd.z- bb.coordBeg.z )> dMaxLength )
		dMaxLength = fabs(bb.coordEnd.z- bb.coordBeg.z );
	m_pCameraTranslation[2] = (float)(-bb.coordBeg.y - dMaxLength / (2 * tan(m_fFovy / 2 * PI / 180))) / 0.8;	// 0.8 for scaling

	m_pCameraRotation[0] = -90;
	m_pCameraRotation[1] = 0;
	m_pCameraRotation[2] = 0;

	update();
	updateGL();	// black magic to omit shading of first frame
}

void CGeomOpenGLView::ZoomView( int _nZoomIn )
{
	//m_pCameraTranslation[2] += (float)_nZoomIn / WHEEL_DELTA * 0.05f * fabs( m_pCameraTranslation[2] );
	m_pCameraTranslation[2] += (float)_nZoomIn / 120 * 0.05f * fabs(m_pCameraTranslation[2]);
	update();
}

void CGeomOpenGLView::wheelEvent ( QWheelEvent * event )
{
   ZoomView( event->delta() );
}

void CGeomOpenGLView::mouseMoveEvent( QMouseEvent *event )
{
	float dx = (event->x() - m_LastMousePos.x()) / (float)m_nWindowHeight;
	float dy = (event->y() - m_LastMousePos.y()) / (float)m_nWindowWidth;

	if (( event->buttons() & Qt::LeftButton ) && ( event->modifiers() & Qt::ShiftModifier) )
	{
		m_pCameraRotation[2] += dx * 100;
	//	m_Axes.AnglesWereChanged();
		update();
	}
	else if ( event->buttons() & Qt::LeftButton )
	{
	  m_pCameraRotation[0] += dy * 100;
	  m_pCameraRotation[1] += dx * 100;
	//	m_Axes.AnglesWereChanged();
		update();
	} else if ( event->buttons() & Qt::RightButton )
	{
		m_pCameraTranslation[0] += dx / 5.0f * fabs( m_pCameraTranslation[2] );
		m_pCameraTranslation[1] -= dy / 5.0f * fabs( m_pCameraTranslation[2] );
		update();
	}
   m_LastMousePos = event->pos();
}

void CGeomOpenGLView::mousePressEvent( QMouseEvent *event )
{
	m_LastMousePos = event->pos();
}

void CGeomOpenGLView::UpdateView()
{
	update();
}

