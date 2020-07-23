/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLVertexArrayObject>
#include <QPainter>
#include "SystemStructure.h"
#include "BaseGLView.h"

class COpenGLViewShader : public QOpenGLWidget, protected QOpenGLFunctions, public CBaseGLView
{
public:
	/* Describes input info needed for rendering of particles. */
	struct SParticle
	{
		QVector3D coord;    // Coordinate on scene.
		QColor color;       // Color.
		float radius;       // Real radius.
	};

	/* Describes input info needed for rendering of bonds. */
	struct SBond
	{
		QVector3D coord1;   // Coordinates of the center of the 1st cylinder base.
		QVector3D coord2;   // Coordinates of the center of the 2nd cylinder base.
		QColor color;       // Color.
		float radius;       // Real radius.
	};

	/* Describes input info needed for rendering of walls and volumes. */
	struct STriangle
	{
		QVector3D coord1;   // Coordinates of the 1st point of the triangle.
		QVector3D coord2;   // Coordinates of the 2st point of the triangle.
		QVector3D coord3;   // Coordinates of the 3st point of the triangle.
		QColor color;       // Color.
	};

	/* Describes input info needed for rendering of discs. */
	struct SDiscs
	{
		QVector3D coord;    // Coordinate on scene.
		QColor color;       // Color.
		float radius;       // Real radius.
		Qt::Axis plane;		// Slicing plane.
	};

	/* Describes input info needed for rendering of simulation domain. */
	struct SDomain
	{
		QVector3D coordMin; // Coordinates of some point of the box.
		QVector3D coordMax; // Coordinates of opposite point of the box.
		QColor color;       // Color.
	};

	/* Describes input info needed for rendering of periodic boundaries. */
	struct SPeriodic
	{
		bool x{}, y{}, z{}; // Activity of planes.
		QVector3D coordMin; // Coordinates of some point of the box.
		QVector3D coordMax; // Coordinates of opposite point of the box.
	};

	/* Describes input info needed for rendering of orientation of particles. */
	struct SOrientation
	{
		QVector3D coord;    // Real coordinate of particle.
		QQuaternion orient; // Orientation of particle.
		float radius{};     // Real radius of particle.
	};

	/* Describes input info needed for rendering of legend. */
	struct SLegend
	{
		double minValue{}, maxValue{};       // Values to render.
		QColor minColor, midColor, maxColor; // Colors to render.
	};

private:
	/* Vertex info needed for rendering of particles. */
	struct SParticleVertex
	{
		QVector3D coord;  // Vertex coordinates.
		QVector4D color;  // Vertex color.
		GLfloat radius{}; // Vertex radius.
	};

	/* Vertex info needed for rendering of bonds and walls. */
	struct SSolidVertex
	{
		QVector3D coord;  // Vertex coordinates.
		QVector3D normal; // Vertex normal vector.
		QVector4D color;  // Vertex color.
	};

	/* Vertex info needed for rendering of volumes and simulation domain. */
	struct SFrameVertex
	{
		QVector3D coord;  // Vertex coordinates.
		QVector4D color;  // Vertex color.
		//float type{};     // Vertex number in triangle (0 - first / 1 - second / 2 - third).
	};

	/* Vertex info needed for rendering of discs. */
	struct SDiscVertex
	{
		QVector3D coord;  // Vertex coordinates.
		QVector4D color;  // Vertex color.
		QVector2D local;  // Vertex local coordinates.
	};

	/* Vertex info needed for rendering of periodic boundaries and orientations. */
	struct SSimpleVertex
	{
		QVector3D coord;  // Vertex coordinates.
		float type{};     // Axis type (X=0 / Y=1 / Z=2).
	};

	/* Vertex info needed for rendering of coordinate axes. */
	struct SAxisVertex
	{
		QVector3D coord;  // Vertex coordinates.
		QVector3D normal; // Vertex normal vector.
		float type{};     // Axis type (X=0 / Y=1 / Z=2).
	};

	/* Precalculated values for cylinders rendering. */
	struct SCylinderPreData
	{
		std::vector<float> sin; // Precalculated values of sinus for cylinders rendering; needed to calculate positions of points, laying on the edges of cylinder's bases.
		std::vector<float> cos; // Precalculated values of cosine for cylinders rendering; needed to calculate positions of points, laying on the edges of cylinder's bases.
	};

	/* Data needed to draw objects of each type. */
	struct SShaderProgram
	{
		QOpenGLShaderProgram program;                               // Shader program.
		QOpenGLTexture texture{ QOpenGLTexture::Target::Target2D }; // Selected texture.
		QOpenGLBuffer dataBuf{ QOpenGLBuffer::VertexBuffer };       // Vertex data for shader program.
		QOpenGLBuffer indexBuf{ QOpenGLBuffer::IndexBuffer };       // Index data for shader program.
		QOpenGLVertexArrayObject vao;                               // Vertex array object for shader program.
		GLsizei objects{};                                          // Number of objects to draw.
	};

	/* Data needed to setup shader program of each type. */
	struct SShaderBlock
	{
		std::vector<std::unique_ptr<SShaderProgram>> shaders; // List of shader programs with all data needed for rendering.
		bool useIndex{};                                      // Whether to use index buffers.
		bool useTexture{};                                    // Whether to use textures.
		QString fileTexture;                                  // Full path to a texture file.
		QString fileShader;                                   // Specific part of a file name with shader program.
		std::vector<QString> attributeNames;                  // List of names for all attribute buffers.
	};

	const size_t c_BYTES_PER_BLOCK = 200 * 1024 * 1024;		  // Number of bytes allowed per each shader block.
	const uint8_t c_LINES_PER_CYLINDER_MIN = 3;				  // Minimum allowed number of edges to approximate cylindrical walls of bonds.
	const uint8_t c_LINES_PER_CYLINDER_MAX = 21;			  // Maximum allowed number of edges to approximate cylindrical walls of bonds.

	QMatrix4x4 m_projection;		// Projection matrix.
	QMatrix4x4 m_world;				// Model-view matrix.

	// Shader programs.
	SShaderBlock m_partProgram;               // Shader data for drawing particles.
	SShaderBlock m_bondProgram;               // Shader data for drawing bonds.
	SShaderBlock m_wallProgram;               // Shader data for drawing walls.
	SShaderBlock m_volmProgram;               // Shader data for drawing analysis volumes.
	SShaderBlock m_discProgram;               // Shader data for drawing discs.
	SShaderBlock m_sdomProgram;               // Shader data for drawing simulation domain.
	SShaderBlock m_pbcsProgram;               // Shader data for drawing periodic boundaries.
	SShaderBlock m_orntProgram;               // Shader data for drawing orientations of particles.
	SShaderBlock m_axisProgram;               // Shader data for drawing coordinate axes.
	std::vector<SShaderBlock*> m_allPrograms; // List of pointers to all programs for unified access.

	double m_time{};                // Time value to show.
	SLegend m_legend;				// Info about legend.

	SCylinderPreData m_bondPreData; // Precalculated values for bonds rendering.
	SCylinderPreData m_arrwPreData; // Precalculated values for orientations rendering.
	float m_axisSize{ 80.0f };      // Size of coordinate axes, roughly 2 = 1px.

public:
	COpenGLViewShader(QWidget* _parent);
	COpenGLViewShader(const CBaseGLView& _other, QWidget* _parent = nullptr);
	~COpenGLViewShader();

	bool IsValid() const; // Tells whether the viewer is ready for setting data.

	void SetParticles(const std::vector<SParticle>& _particles);
	void SetBonds(const std::vector<SBond>& _bonds);
	void SetWalls(const std::vector<STriangle>& _walls);
	void SetVolumes(const std::vector<STriangle>& _walls);
	void SetDiscs(const std::vector<SDiscs>& _discs);
	void SetDomain(const SDomain& _box);
	void SetPeriodic(const SPeriodic& _pbc);
	void SetOrientations(const std::vector<SOrientation>& _orientations);
	void SetAxes(bool _visible = true);
	void SetTime(double _time, bool _visible = true);
	void SetLegend(const SLegend& _legend, bool _visible = true);

	void SetParticleTexture(const QString& _path) override;

	// Request redraw of the whole scene.
	void Redraw() override;

	// Sets the rendering quality in the range [1..100].
	void SetRenderQuality(uint8_t _quality) override;

	// Returns snapshot of the current view, scaling it with provided factor.
	QImage Snapshot(uint8_t _scaling) override;

	// Converts window coordinates to a scene coordinates, describing the line of sight through this point.
	SBox WinCoord2LineOfSight(const QPoint& _pos) const override;

private:
	void Construct();	// Common part of constructors.

	void initializeGL() override;
	void resizeGL(int _w, int _h) override;
	void paintGL() override;

	// Sets new viewport according to given parameters and updates perspective projection matrix accordingly.
	void UpdatePerspective() override;

	void mousePressEvent(QMouseEvent* _event) override;
	void mouseMoveEvent(QMouseEvent* _event) override;
	void wheelEvent(QWheelEvent* _event) override;

	// Initializes all shaders in provided _shaderBlock.
	void InitShaders(SShaderBlock* _shaderBlock) const;
	// Initializes all textures in provided _shaderBlock.
	static void InitTextures(SShaderBlock* _shaderBlock);
	// Resizes list of shader programs in provided _shaderBlock.
	void ResizeShaderesBlock(SShaderBlock* _shaderBlock, size_t _newSize);
	// De-allocates and clears all data in provided _shaderBlock.
	static void ClearShaderesBlock(SShaderBlock* _shaderBlock);

	void DrawParticles();
	void DrawBonds();
	void DrawWalls();
	void DrawVolumes();
	void DrawDiscs();
	void DrawDomain();
	void DrawPeriodic();
	void DrawOrientations();
	void DrawAxes();
	void DrawTime();
	void DrawLegend();

	// Performs precalculation of cylinder-specific data for faster rendering.
	static SCylinderPreData PrecalculateCylinderData(uint8_t _linesPerBond);

	// Returns offsets of points lying on a cylindrical surface relative to the central axis of the cylinder.
	static std::vector<QVector3D> CylinderOffsets(const QVector3D& _coord1, const QVector3D& _coord2, float _radius, const SCylinderPreData& _preData);

	// Converts QColor to [0..1]-ranged QVector4D.
	static QVector4D QColorToQVector4D(const QColor& _c);

	// Returns a normalized vector. Must be used instead of a built-in function, which has very low precision.
	static QVector3D Normalize(const QVector3D& _v);
};
