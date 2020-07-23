/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "AgglomOpenGLView.h"

CAgglomOpenGLView::CAgglomOpenGLView( QWidget *parent )	: QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
	//camera position :
	m_pCameraTranslation[ 0 ] = 0.0f;
	m_pCameraTranslation[ 1 ] = 0.0f;
	m_pCameraTranslation[ 2 ] = -0.18f;
	m_pCameraRotation[ 0 ] = -90.0f;
	m_pCameraRotation[ 1 ] = 0.0f;
	m_pCameraRotation[ 2 ] = 0.0f;

	m_pQuadObj = gluNewQuadric();
	m_fFovy = 15.0f;
	m_fZnear = 0.02f;
	m_fZfar = m_fZnear*10000;

	m_nWindowWidth = 0;
	m_nWindowHeight = 0;

	m_CurrentAgglomerate = SAgglomerate{};
}

void CAgglomOpenGLView::SetCurrentAgglomerate( SAgglomerate* _pNewAgglomerate )
{
	if ( _pNewAgglomerate == NULL ) return;
	m_CurrentAgglomerate = *_pNewAgglomerate;
	AutoCentrateView();
	DrawScene();
}

void CAgglomOpenGLView::initializeGL()
{
	setAutoFillBackground(false);
	glEnable(0x809D);
}

void CAgglomOpenGLView::paintGL()
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

void CAgglomOpenGLView::SetupViewPort( int _nWidth, int _nHeight )
{
	//// saving new width and height:
	m_nWindowWidth = _nWidth;
	m_nWindowHeight = _nHeight;
	// changing viewport size to much the new size of the window:
	glViewport( 0, 0, _nWidth, _nHeight );
	// changing the projection matrix to much the new size of the window:
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluPerspective( m_fFovy, (float) _nWidth / (float) _nHeight, m_fZnear, m_fZfar );
}

void CAgglomOpenGLView::resizeGL( int _nWidth, int _nHeight )
{
	SetupViewPort( _nWidth, _nHeight );
	paintGL();
}

void CAgglomOpenGLView::DrawScene()
{
	// show particles
	for ( unsigned long i = 0; i < m_CurrentAgglomerate.vParticles.size(); i++ )
	{
		glPushMatrix();
		SetColorGL( CColor( 0.5f, 0.5f, 0.5f, 1.0f ) );
		glTranslatef( (GLfloat)m_CurrentAgglomerate.vParticles[i].vecCoord.x, (GLfloat)m_CurrentAgglomerate.vParticles[i].vecCoord.y, (GLfloat)m_CurrentAgglomerate.vParticles[i].vecCoord.z );
		gluSphere( m_pQuadObj, m_CurrentAgglomerate.vParticles[i].dRadius, 12, 12 );
		glPopMatrix();
	}
	// show bonds
	for ( unsigned i = 0; i<m_CurrentAgglomerate.vBonds.size(); i++ )
	{
		double angle, dLengthVec;
		CVector3 vecLeft = m_CurrentAgglomerate.vParticles[m_CurrentAgglomerate.vBonds[i].nLeftID].vecCoord;
		CVector3 vecRight = m_CurrentAgglomerate.vParticles[m_CurrentAgglomerate.vBonds[i].nRightID].vecCoord;

		CVector3 z( 0, 0, -1 );
		// Get difference between two points you want cylinder along
		CVector3 p = vecRight - vecLeft;
		// Get CROSS product (the axis of rotation)
		CVector3 t = z*p;
		dLengthVec = p.Length();
		// check correctness
		if ( dLengthVec <= 0 ) break;

		// Get angle. LENGTH is magnitude of the vector
		angle = _180_PI * acos(DotProduct(z, p) / dLengthVec);

		glPushMatrix();
		SetColorGL( CColor( 0.5f, 0.5f, 0.5f, 1.0f ) );

		glTranslatef( (GLfloat)vecRight.x, (GLfloat)vecRight.y, (GLfloat)vecRight.z );
		glRotatef( (GLfloat)angle, (GLfloat)t.x, (GLfloat)t.y, (GLfloat)t.z );
		gluQuadricOrientation( m_pQuadObj, GLU_OUTSIDE );
		gluCylinder(m_pQuadObj, m_CurrentAgglomerate.vBonds[i].dRadius, m_CurrentAgglomerate.vBonds[i].dRadius, dLengthVec, 20, 1);
		glPopMatrix();
	}
}

void CAgglomOpenGLView::AutoCentrateView()
{
	if ( m_CurrentAgglomerate.vParticles.empty() ) return;
	int nMagnitude;
	CVector3 vMax, vMin;

	vMax = m_CurrentAgglomerate.vParticles[ 0 ].vecCoord +  m_CurrentAgglomerate.vParticles[ 0 ].dRadius;
	vMin = m_CurrentAgglomerate.vParticles[ 0 ].vecCoord -  m_CurrentAgglomerate.vParticles[ 0 ].dRadius;
	for (size_t i = 1; i < m_CurrentAgglomerate.vParticles.size(); ++i)
	{
		const SAggloParticle& part = m_CurrentAgglomerate.vParticles[i];
		vMax.x = std::max(vMax.x, part.vecCoord.x + part.dRadius);
		vMax.y = std::max(vMax.y, part.vecCoord.y + part.dRadius);
		vMax.z = std::max(vMax.z, part.vecCoord.z + part.dRadius);
		vMin.x = std::min(vMin.x, part.vecCoord.x - part.dRadius);
		vMin.y = std::min(vMin.y, part.vecCoord.y - part.dRadius);
		vMin.z = std::min(vMin.z, part.vecCoord.z - part.dRadius);
	}
	nMagnitude = (int)(log10( (vMax.z - vMin.z)/ 100.0 ));
	m_fZfar = (float)(pow( 10.0, nMagnitude + 4.0 ));
	m_fZnear = (float)(pow( 10.0, nMagnitude - 1.0 ));

	m_pCameraTranslation[ 0 ] = (float)(-(vMax.x+vMin.x)/2);
	m_pCameraTranslation[ 1 ] = (float)(-(vMax.z+vMin.z)/2);
	double dMaxLength = fabs( vMax.x - vMin.x );
	if ( fabs( vMax.z-vMin.z )> dMaxLength )
		dMaxLength = fabs( vMax.z-vMin.z );
	m_pCameraTranslation[2] = (float)(-vMin.y - dMaxLength / (2 * tan(m_fFovy / 2 * PI / 180))) / 0.8;	// 0.8 for scaling
	m_pCameraRotation[0] = -90;
	m_pCameraRotation[1] = 0;
	m_pCameraRotation[2] = 0;

	update();
	updateGL();	// black magic to omit shading of first frame
}

void CAgglomOpenGLView::ZoomView( int _nZoomIn )
{
	//m_pCameraTranslation[2] += (float)_nZoomIn / WHEEL_DELTA * 0.05f * fabs( m_pCameraTranslation[2] );
	m_pCameraTranslation[2] += (float)_nZoomIn / 120 * 0.05f * fabs(m_pCameraTranslation[2]);
	update();
}

void CAgglomOpenGLView::wheelEvent( QWheelEvent * event )
{
   ZoomView( event->delta() );
}

void CAgglomOpenGLView::mouseMoveEvent( QMouseEvent *event )
{
	float dx = (event->x() - m_LastMousePos.x()) / (float)m_nWindowHeight;
	float dy = (event->y() - m_LastMousePos.y()) / (float)m_nWindowWidth;

	if (( event->buttons() & Qt::LeftButton ) && ( event->modifiers() & Qt::ShiftModifier) )
	{
		m_pCameraRotation[2] += dx * 100;
		update();
	}
	else if ( event->buttons() & Qt::LeftButton )
	{
	  m_pCameraRotation[0] += dy * 100;
	  m_pCameraRotation[1] += dx * 100;
		update();
	} else if ( event->buttons() & Qt::RightButton )
	{
		m_pCameraTranslation[0] += dx / 5.0f * fabs( m_pCameraTranslation[2] );
		m_pCameraTranslation[1] -= dy / 5.0f * fabs( m_pCameraTranslation[2] );
		update();
	}
   m_LastMousePos = event->pos();
}

void CAgglomOpenGLView::mousePressEvent( QMouseEvent *event )
{
	m_LastMousePos = event->pos();
}

void CAgglomOpenGLView::UpdateView()
{
	update();
}
