/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <QGLWidget>
#include <GL/glu.h>
#include <QtOpenGL>
#include <QImage>
#include "SystemStructure.h"
#include "SampleAnalyzerTab.h"
#include "BaseGLView.h"
#include "ViewSettings.h"

class COpenGLView : public QGLWidget, public CBaseGLView
{
	Q_OBJECT

	friend class CViewManager;

protected:
	bool m_bShowOrientation{ false };	// show orientation of particles
	bool m_bShowAxes{ true };			// show axes
	bool m_bShowTime{ true };			// show time
	bool m_bShowLegend{ true };			// show color legend

	// window size
	double m_dWinSize{ 0.0 };	// used for drawing axis
	double m_dWinSizePx{ 0.0 };	// used for drawing axis

	CColor m_DefaultObjectColor{ 0.7f, 0.7f, 0.7f, 1.0f };

	CSystemStructure* m_pSystemStructure{ nullptr };
	GLUquadricObj *m_pQuadObj{ gluNewQuadric() }; // used in DrawGLScene()
	double m_dCurrentTime{ 0.0 };

	std::vector<unsigned> m_vCoordNumber;	// used to precalculate coord numbers for particle coloring
	std::vector<double> m_vOverlaps;		// used to precalculate overlaps for particle coloring
	std::vector<size_t> m_vAgglSize;		// size of agglomerate in number of primary particles
	bool m_bCoordNumberReady{ false };	// coord numbers are calculated for current time
	bool m_bOverlapsReady{ false };		// overlaps are calculated for current time
	bool m_bAgglSizeReady{ false };
	bool m_bUpdating{ true };

	GLint m_sphereSlices{ 15 };		// Quality-dependent setting to draw spheres.
	GLint m_sphereStacks{ 14 };		// Quality-dependent setting to draw spheres.
	GLint m_cylinderSlices{ 20 };	// Quality-dependent setting to draw cylinders.
	GLint m_cylinderStacks{ 1 };	// Quality-dependent setting to draw cylinders.

	std::vector<bool> m_vBrokenBonds;	// vector containing indexes of bonds which was brocken in specific time interval

	CViewSettings* m_viewSettings;  // Contains all visualization settings.

public:
	CSampleAnalyzerTab* m_pSampleAnalyzerTab{ nullptr };

public:
	COpenGLView(CViewSettings* _viewSettings, QWidget* _parent = nullptr);
	COpenGLView(const CBaseGLView& _other, CViewSettings* _viewSettings, QWidget* _parent = nullptr);
	~COpenGLView();

	void SetSystemStructure(CSystemStructure* _pSystemStructure);

	QSize minimumSizeHint() const override;
	QSize sizeHint() const override;

	void Redraw() override;

	void SetCurrentTime(double _dTime);
	double GetCurrentTime() { return m_dCurrentTime; }
	QImage Snapshot(uint8_t _scaling) override;

	void SetRenderQuality(uint8_t _quality) override;

	void SetAxesVisible(bool _bVisible);
	void SetTimeVisible(bool _bVisible);
	void SetLegendVisible(bool _bVisible);
	void SetOrientationVisible(bool _bVisible);

	void SetParticleTexture(const QString&) override {};
	void RecalculateBrokenBonds(); // update list of broken bonds

	SBox WinCoord2LineOfSight(const QPoint& _pos) const override;

	void SetCameraStandardView(const SBox& _box, const QVector3D& _cameraDirection) override;

public slots:
	void UpdateView();
	void RedrawScene();

	void EnableOpenGLView();
	void DisableOpenGLView();

	void RecalculateColoringProperties(); // called when user has press F5 key

protected:
	// standard OpenGL functions of initialization, painting the scene and resizing
	void initializeGL() override;
	void resizeGL(int width, int height) override;
	void paintEvent(QPaintEvent* _event) override; // use instead of paintGL() for proper work of QPainter needed to draw text.

	// mouse events
	void mouseMoveEvent(QMouseEvent *event) override;
	void mousePressEvent(QMouseEvent *event) override;
	void wheelEvent(QWheelEvent * event) override;

	virtual void DrawParticles() {};

	void DrawBonds();
	void DrawBox(SVolumeType& _Volume, CColor& _Color);
	void DrawScene();

	void DrawOrientations();
	void DrawSampleAnalyzerVolume();
	void DrawSimulationDomain();
	void DrawPBC();
	void DrawGeometricalObjects() const;
	void DrawAnalysisVolumes() const;

	bool IsPointCuttedByPlanes(const CVector3& _vCoord) const;

	// increase/decrease zoom - positive=increase; negative=decrease
	void ZoomView(int _nZoomIn);

	void SetColorGL(const CColor& _color);

	void GetObjectColor(unsigned int nObjectID, CColor* _ResultColor); // return for specified object ID its colors according to the coloring scheme
	CVector3 WinCoord2GL(double _x, double _y, double _z) const;

private:
	void DrawAxes() const;
	void DrawAxesText(QPainter& _painter) const;
	void DrawTime() const;
	void DrawTimeText(QPainter& _painter) const;
	void DrawLegend() const;
	void DrawLegendText(QPainter& _painter) const;

	static void DrawQuad(const CVector3& _v1, const CVector3& _v2, const CVector3& _v3, const CVector3& _v4, const CColor& _color); // CCW
	void DrawTriangularPlane(const CTriangle& _triangle, const CVector3& _normal, const CColor& _color) const;

	// Sets new viewport according to given parameters and updates perspective projection matrix accordingly.
	void UpdatePerspective() override;
};
