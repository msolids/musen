/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "OpenGLViewShader.h"
#include <QMouseEvent>
#include "GeometricFunctions.h"

COpenGLViewShader::COpenGLViewShader(QWidget* _parent) :
	QOpenGLWidget(_parent)
{
	Construct();
}

COpenGLViewShader::COpenGLViewShader(const CBaseGLView& _other, QWidget* _parent) :
	QOpenGLWidget(_parent),
	CBaseGLView(_other)
{
	Construct();
}

void COpenGLViewShader::Construct()
{
	// precalculate data for rendering of cylinders
	m_bondPreData = PrecalculateCylinderData((c_LINES_PER_CYLINDER_MIN + c_LINES_PER_CYLINDER_MAX) / 2); // with default quality
	m_arrwPreData = PrecalculateCylinderData(4); //

	// gather all programs
	m_allPrograms.push_back(&m_partProgram);
	m_allPrograms.push_back(&m_bondProgram);
	m_allPrograms.push_back(&m_wallProgram);
	m_allPrograms.push_back(&m_volmProgram);
	m_allPrograms.push_back(&m_discProgram);
	m_allPrograms.push_back(&m_sdomProgram);
	m_allPrograms.push_back(&m_pbcsProgram);
	m_allPrograms.push_back(&m_orntProgram);
	m_allPrograms.push_back(&m_axisProgram);

	// setup programs
	m_partProgram.useTexture = true;

	m_partProgram.fileTexture = ":/MusenGUI/Pictures/SphereTexture0.png";

	m_bondProgram.useIndex = true;
	m_discProgram.useIndex = true;
	m_sdomProgram.useIndex = true;
	m_orntProgram.useIndex = true;

	m_partProgram.fileShader = "Particle";
	m_bondProgram.fileShader = "Bond";
	m_wallProgram.fileShader = "Wall";
	m_volmProgram.fileShader = "Volume";
	m_discProgram.fileShader = "Disc";
	m_sdomProgram.fileShader = "Volume";
	m_pbcsProgram.fileShader = "Periodic";
	m_orntProgram.fileShader = "Orientation";
	m_axisProgram.fileShader = "Axis";

	m_partProgram.attributeNames = { "a_position", "a_color", "a_radius" };
	m_bondProgram.attributeNames = { "a_position", "a_normal", "a_color" };
	m_wallProgram.attributeNames = { "a_position", "a_normal", "a_color" };
	m_volmProgram.attributeNames = { "a_position", "a_color"/*, "a_type"*/ };
	m_discProgram.attributeNames = { "a_position", "a_color", "a_local" };
	m_sdomProgram.attributeNames = { "a_position", "a_color" };
	m_pbcsProgram.attributeNames = { "a_position", "a_type" };
	m_orntProgram.attributeNames = { "a_position", "a_type" };
	m_axisProgram.attributeNames = { "a_position", "a_normal", "a_type" };

	// add one program of each type
	for (size_t i = 0; i < m_allPrograms.size(); ++i)
		m_allPrograms[i]->shaders.emplace_back(new SShaderProgram);
}

COpenGLViewShader::~COpenGLViewShader()
{
	makeCurrent();

	for (auto& shaderBlock : m_allPrograms)
		ClearShaderesBlock(shaderBlock);

	doneCurrent();
}

bool COpenGLViewShader::IsValid() const
{
	return isValid();
}

void COpenGLViewShader::initializeGL()
{
	initializeOpenGLFunctions();

	//const QSurfaceFormat::OpenGLContextProfile profile = format().profile();
	//std::cout << "Current OpenGL version: " << format().majorVersion() << "." << this->format().minorVersion() << std::endl;
	//std::cout << "Current OpenGL profile: " << (profile == QSurfaceFormat::CoreProfile ? "CoreProfile" : profile == QSurfaceFormat::CompatibilityProfile ? "CompatibilityProfile" : "NoProfile") << std::endl;

	// settings for proper display of particles as sprites
	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

	// background color
	glClearColor(1.0, 1.0, 1.0, 1.0);

	// initialize all shader programs
	for (auto& program : m_allPrograms)
		InitShaders(program);

	// Enable depth buffer
	glEnable(GL_DEPTH_TEST);

	// Enable back face culling
	glEnable(GL_CULL_FACE);
}

void COpenGLViewShader::resizeGL(int _w, int _h)
{
	m_windowSize.setWidth(_w);
	m_windowSize.setHeight(_h);

	// update viewport and projection matrix
	UpdatePerspective();
}

void COpenGLViewShader::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// calculate model view transformation
	m_world.setToIdentity();
	m_world.translate(m_cameraTranslation);
	m_world.rotate(m_cameraRotation.x(), 1, 0, 0);
	m_world.rotate(m_cameraRotation.y(), 0, 1, 0);
	m_world.rotate(m_cameraRotation.z(), 0, 0, 1);

	// draw objects (order is important!)
	DrawVolumes();
	DrawParticles();
	DrawOrientations();
	DrawDiscs();
	DrawBonds();
	DrawWalls();
	DrawDomain();
	DrawPeriodic();
	DrawAxes();
	DrawTime();
	DrawLegend();
}

void COpenGLViewShader::UpdatePerspective()
{
	// calculate aspect ratio
	m_viewport.aspect = static_cast<float>(m_windowSize.width()) / static_cast<float>(m_windowSize.height() != 0 ? m_windowSize.height() : 1);
	// reset projection
	m_projection.setToIdentity();
	// set perspective projection
	m_projection.perspective(m_viewport.fovy, m_viewport.aspect, m_viewport.zNear, m_viewport.zFar);
}

void COpenGLViewShader::SetParticles(const std::vector<SParticle>& _particles)
{
	if (!isValid()) return; // check whether current context is valid

	const size_t nParticles = _particles.size();						               // total number of particles to be rendered
	const size_t nParticlesPerBlock = c_BYTES_PER_BLOCK / sizeof(SParticleVertex);     // number of particles that will be rendered together
	const size_t nBlocks = (nParticles + nParticlesPerBlock - 1) / nParticlesPerBlock; // number of needed blocks to render all particles
	ResizeShaderesBlock(&m_partProgram, nBlocks); // resize and initialize all blocks of shader program

	// for each block
	for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock)
	{
		const size_t shift = iBlock * nParticlesPerBlock;                                  // shift in _particles vector for current block
		const size_t nParticlesInBlock = std::min(nParticlesPerBlock, nParticles - shift); // number of particles in current block

		// a continuous list of vertices for all particles
		std::vector<SParticleVertex> data(nParticlesInBlock);

		// for each particle
		ParallelFor(nParticlesInBlock, [&](size_t j)
		{
			const size_t iPart = j + shift; // index of particle in _particles vector
			data[j] = { _particles[iPart].coord, QColorToQVector4D(_particles[iPart].color), _particles[iPart].radius };
		});

		makeCurrent();

		// transfer vertex data to VBO 0
		m_partProgram.shaders[iBlock]->dataBuf.bind();
		m_partProgram.shaders[iBlock]->dataBuf.allocate(&data[0], static_cast<int>(data.size() * sizeof(data[0])));
		m_partProgram.shaders[iBlock]->dataBuf.release();

		doneCurrent();

		// save number of objects to be drawn
		m_partProgram.shaders[iBlock]->objects = static_cast<GLsizei>(data.size());
	}
}

void COpenGLViewShader::SetBonds(const std::vector<SBond>& _bonds)
{
	if (!isValid()) return; // check whether current context is valid

	const size_t nBonds = _bonds.size();						                                  // total number of bonds to be rendered
	const size_t nLinesPerBond = m_bondPreData.sin.size();		                                  // number of edges to approximate cylindrical walls of each bond
	const size_t nBondsPerBlock = c_BYTES_PER_BLOCK / (2 * nLinesPerBond * sizeof(SSolidVertex)); // number of bonds that will be rendered together
	const size_t nBlocks = (nBonds + nBondsPerBlock - 1) / nBondsPerBlock;                        // number of needed blocks to render all bonds
	ResizeShaderesBlock(&m_bondProgram, nBlocks); // resize and initialize all blocks of shader program

	// for each block
	for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock)
	{
		const size_t shift = iBlock * nBondsPerBlock;                          // shift in _bonds vector for current block
		const size_t nBondsInBlock = std::min(nBondsPerBlock, nBonds - shift); // number of bonds in current block
		const size_t nVertices = 2 * nBondsInBlock * nLinesPerBond;		       // number of vertices needed to describe all bonds in block
		const size_t nIndices = nBondsInBlock * (2 * nLinesPerBond + 3);       // number of indices needed to describe all bonds in block

		// a continuous list of vertices for all cylinders in block, 2 * nLinesPerBond per bond
		std::vector<SSolidVertex> data(nVertices);
		// list of indices, pointing to vertices, which indicate the order of points drawing; draw as GL_TRIANGLE_STRIP
		// e.g., for the first cylinder, drawn with 4 walls, it will look like [0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 1]
		std::vector<GLuint> indices(nIndices);

		// for each bond
		ParallelFor(nBondsInBlock, [&](size_t j)
		{
			const size_t iBond = j + shift; // index of bond in _bonds vector
			const QVector3D& coord1 = _bonds[iBond].coord1; // center of the first cylinder's base
			const QVector3D& coord2 = _bonds[iBond].coord2; // center of the second cylinder's base
			const QVector4D color = QColorToQVector4D(_bonds[iBond].color);

			// get offsets relative to the cylinder's central axis for points lying on the cylindrical surface
			std::vector<QVector3D> points = CylinderOffsets(coord1, coord2, _bonds[iBond].radius, m_bondPreData);

			// for each line on bond
			size_t iDataOffset = 0, iIndexOffset = 0;
			for (GLuint iLine = 0; iLine < nLinesPerBond; ++iLine)
			{
				// calculate vertices
				iDataOffset = 2 * j * nLinesPerBond + 2 * iLine;
				data[iDataOffset + 0] = { { coord2 + points[iLine] }, { points[iLine] }, color };
				data[iDataOffset + 1] = { { coord1 + points[iLine] }, { points[iLine] }, color };

				// calculate indices
				iIndexOffset = iDataOffset + 3 * j;
				indices[iIndexOffset + 0] = static_cast<GLuint>(iDataOffset + 1);
				indices[iIndexOffset + 1] = static_cast<GLuint>(iDataOffset + 0);
			}
			indices[iIndexOffset + 2] = static_cast<GLuint>(iDataOffset + 2 - 2 * nLinesPerBond + 1); // repeat first two points...
			indices[iIndexOffset + 3] = static_cast<GLuint>(iDataOffset + 2 - 2 * nLinesPerBond + 0); // ...to close the cylinder
			indices[iIndexOffset + 4] = 0xFFFFFFFF;	// special index to restart the primitive drawing
		});

		makeCurrent();

		// transfer vertex data to VBO 0
		m_bondProgram.shaders[iBlock]->dataBuf.bind();
		m_bondProgram.shaders[iBlock]->dataBuf.allocate(&data[0], static_cast<int>(data.size() * sizeof(data[0])));
		m_bondProgram.shaders[iBlock]->dataBuf.release();

		// transfer index data to VBO 1
		m_bondProgram.shaders[iBlock]->indexBuf.bind();
		m_bondProgram.shaders[iBlock]->indexBuf.allocate(&indices[0], static_cast<int>(indices.size() * sizeof(indices[0])));
		m_bondProgram.shaders[iBlock]->indexBuf.release();

		doneCurrent();

		// save number of objects to be drawn
		m_bondProgram.shaders[iBlock]->objects = static_cast<GLsizei>(indices.size());
	}
}

void COpenGLViewShader::SetWalls(const std::vector<STriangle>& _walls)
{
	if (!isValid()) return; // check whether current context is valid

	const size_t nWalls = _walls.size();						                  // total number of triangles to be rendered
	const size_t nWallsPerBlock = c_BYTES_PER_BLOCK / (3 * sizeof(SSolidVertex)); // number of triangles that will be rendered together
	const size_t nBlocks = (nWalls + nWallsPerBlock - 1) / nWallsPerBlock;        // number of needed blocks to render all triangles
	ResizeShaderesBlock(&m_wallProgram, nBlocks); // resize and initialize all blocks of shader program

	// for each block
	for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock)
	{
		const size_t shift = iBlock * nWallsPerBlock;                          // shift in _walls vector for current block
		const size_t nWallsInBlock = std::min(nWallsPerBlock, nWalls - shift); // number of triangles in current block

		// a continuous list of vertices for all triangles
		std::vector<SSolidVertex> data(3 * nWallsInBlock);

		// for each triangle
		ParallelFor(nWallsInBlock, [&](size_t j)
		{
			const size_t iWall = j + shift; // index of triangle in _walls vector
			const QVector4D color = QColorToQVector4D(_walls[iWall].color);
			const QVector3D normal = Normalize(QVector3D::crossProduct(_walls[iWall].coord2 - _walls[iWall].coord1, _walls[iWall].coord3 - _walls[iWall].coord1));
			data[3 * j + 0] = { _walls[iWall].coord1, normal, color };
			data[3 * j + 1] = { _walls[iWall].coord2, normal, color };
			data[3 * j + 2] = { _walls[iWall].coord3, normal, color };
		});

		makeCurrent();

		// transfer vertex data to VBO 0
		m_wallProgram.shaders[iBlock]->dataBuf.bind();
		m_wallProgram.shaders[iBlock]->dataBuf.allocate(&data[0], static_cast<int>(data.size() * sizeof(data[0])));
		m_wallProgram.shaders[iBlock]->dataBuf.release();

		doneCurrent();

		// save number of objects to be drawn
		m_wallProgram.shaders[iBlock]->objects = static_cast<GLsizei>(data.size());
	}
}

void COpenGLViewShader::SetVolumes(const std::vector<STriangle>& _walls)
{
	if (!isValid()) return; // check whether current context is valid

	const size_t nWalls = _walls.size();						                  // total number of triangles to be rendered
	const size_t nWallsPerBlock = c_BYTES_PER_BLOCK / (3 * sizeof(SFrameVertex)); // number of triangles that will be rendered together
	const size_t nBlocks = (nWalls + nWallsPerBlock - 1) / nWallsPerBlock;        // number of needed blocks to render all triangles
	ResizeShaderesBlock(&m_volmProgram, nBlocks); // resize and initialize all blocks of shader program

	// for each block
	for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock)
	{
		const size_t shift = iBlock * nWallsPerBlock;                          // shift in _walls vector for current block
		const size_t nWallsInBlock = std::min(nWallsPerBlock, nWalls - shift); // number of triangles in current block

		// a continuous list of vertices for all triangles
		std::vector<SFrameVertex> data(3 * nWallsInBlock);

		// for each triangle
		ParallelFor(nWallsInBlock, [&](size_t j)
		{
			const size_t iWall = j + shift; // index of triangle in _walls vector
			const QVector4D color = QColorToQVector4D(_walls[iWall].color);
			data[3 * j + 0] = { _walls[iWall].coord1, color/*, 0*/ };
			data[3 * j + 1] = { _walls[iWall].coord2, color/*, 1*/ };
			data[3 * j + 2] = { _walls[iWall].coord3, color/*, 2*/ };
		});

		makeCurrent();

		// transfer vertex data to VBO 0
		m_volmProgram.shaders[iBlock]->dataBuf.bind();
		m_volmProgram.shaders[iBlock]->dataBuf.allocate(&data[0], static_cast<int>(data.size() * sizeof(data[0])));
		m_volmProgram.shaders[iBlock]->dataBuf.release();

		doneCurrent();

		// save number of objects to be drawn
		m_volmProgram.shaders[iBlock]->objects = static_cast<GLsizei>(data.size());
	}
}

void COpenGLViewShader::SetDiscs(const std::vector<SDiscs>& _discs)
{
	if (!isValid()) return; // check whether current context is valid

	// local coordinates of the square described around the disk
	static const std::vector<QVector2D> local{ { +1.0, -1.0 }, { +1.0, +1.0 }, { -1.0, +1.0 }, { -1.0, -1.0 } };

	const size_t nDiscs = _discs.size();										// total number of discs to be rendered
	const size_t nDiscsPerBlock = c_BYTES_PER_BLOCK / sizeof(SDiscVertex);		// number of discs that will be rendered together
	const size_t nBlocks = (nDiscs + nDiscsPerBlock - 1) / nDiscsPerBlock;		// number of needed blocks to render all discs
	const size_t nVerticesPerDisc = 4;											// number of vertices needed to describe one disc
	const size_t nIndicesPerDisc = 5;											// number of indices needed to describe one disc

	ResizeShaderesBlock(&m_discProgram, nBlocks);	// resize and initialize all blocks of shader program

	// for each block
	for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock)
	{
		const size_t shift = iBlock * nDiscsPerBlock;							// shift in _discs vector for current block
		const size_t nDiscsInBlock = std::min(nDiscsPerBlock, nDiscs - shift);	// number of discs in current block
		const size_t nVertices = nVerticesPerDisc * nDiscsInBlock;				// number of vertices needed to describe all discs in block
		const size_t nIndices = nIndicesPerDisc * nDiscsInBlock;				// number of indices needed to describe all discs in block

		// a continuous list of vertices for all discs, 4 per disc
		std::vector<SDiscVertex> data(nVertices);
		// list of indices, pointing to vertices, which indicate the order of points drawing; draw as GL_TRIANGLE_FAN
		// e.g., for the first disc, drawn with 2 triangles, it will look like [0, 1, 2, 3, 0xFFFFFFFF]
		std::vector<GLuint> indices(nIndices);

		// for each disc
		ParallelFor(nDiscsInBlock, [&](size_t j)
		{
			const size_t iDisc = j + shift; // index of disc in _discs vector

			// initialize indices
			const size_t iDataOffset = j * nVerticesPerDisc;
			const size_t iIndexOffset = j * nIndicesPerDisc;

			const QVector3D center = _discs[iDisc].coord;						// center of disc
			const float radius     = _discs[iDisc].radius;						// radius of disc
			const QVector4D color  = QColorToQVector4D(_discs[iDisc].color);	// color of disc
			const Qt::Axis plane   = _discs[iDisc].plane;						// projection plane

			// add 4 points of the square described around the disk
			for (size_t i = 0; i < 4; ++i)
			{
				QVector3D coord;
				switch (plane)
				{
				case Qt::XAxis: coord = { center.x(), center.y() + radius * local[i].x(), center.z() + radius * local[i].y() }; break;
				case Qt::YAxis: coord = { center.x() + radius * local[i].x(), center.y(), center.z() + radius * local[i].y() }; break;
				case Qt::ZAxis: coord = { center.x() + radius * local[i].x(), center.y() + radius * local[i].y(), center.z() }; break;
				}
				data[iDataOffset + i] = { coord, color, local[i] };
				indices[iIndexOffset + i] = static_cast<GLuint>(iDataOffset + i);
			}

			indices[iIndexOffset + 4] = 0xFFFFFFFF;	// special index to restart the primitive drawing
		});

		makeCurrent();

		// transfer vertex data to VBO 0
		m_discProgram.shaders[iBlock]->dataBuf.bind();
		m_discProgram.shaders[iBlock]->dataBuf.allocate(&data[0], static_cast<int>(data.size() * sizeof(data[0])));
		m_discProgram.shaders[iBlock]->dataBuf.release();

		// transfer index data to VBO 1
		m_discProgram.shaders[iBlock]->indexBuf.bind();
		m_discProgram.shaders[iBlock]->indexBuf.allocate(&indices[0], static_cast<int>(indices.size() * sizeof(indices[0])));
		m_discProgram.shaders[iBlock]->indexBuf.release();

		doneCurrent();

		// save number of objects to be drawn
		m_discProgram.shaders[iBlock]->objects = static_cast<GLsizei>(indices.size());
	}
}

void COpenGLViewShader::SetDomain(const SDomain& _box)
{
	if (!isValid()) return; // check whether current context is valid

	const bool visible = _box.coordMin != _box.coordMax;
	ResizeShaderesBlock(&m_sdomProgram, visible ? 1 : 0); // resize and initialize all blocks of shader program
	if (!visible) return; // exit if nothing to draw

	const size_t nVertices = 8; // total number of vertices to draw a box
	const size_t nEdges = 12; // total number of edges to draw a box
	// a continuous list of all vertices of the box
	std::vector<SFrameVertex> data(nVertices);
	// list of indices, pointing to vertices, which indicate the order of points drawing; draw as GL_LINES
	std::vector<GLuint> indices(2 * nEdges);

	// color
	const QVector4D color = QColorToQVector4D(_box.color);

	// bottom rectangle
	data[0] = { {_box.coordMin.x(), _box.coordMin.y(), _box.coordMin.z() }, color };
	data[1] = { {_box.coordMax.x(), _box.coordMin.y(), _box.coordMin.z() }, color };
	data[2] = { {_box.coordMax.x(), _box.coordMax.y(), _box.coordMin.z() }, color };
	data[3] = { {_box.coordMin.x(), _box.coordMax.y(), _box.coordMin.z() }, color };
	// top rectangle
	data[4] = { {_box.coordMin.x(), _box.coordMin.y(), _box.coordMax.z() }, color };
	data[5] = { {_box.coordMax.x(), _box.coordMin.y(), _box.coordMax.z() }, color };
	data[6] = { {_box.coordMax.x(), _box.coordMax.y(), _box.coordMax.z() }, color };
	data[7] = { {_box.coordMin.x(), _box.coordMax.y(), _box.coordMax.z() }, color };

	// bottom rectangle
	indices[ 0] = 0;	indices[ 1] = 1;
	indices[ 2] = 1;	indices[ 3] = 2;
	indices[ 4] = 2;	indices[ 5] = 3;
	indices[ 6] = 3;	indices[ 7] = 0;
	// top rectangle
	indices[ 8] = 4;	indices[ 9] = 5;
	indices[10] = 5;	indices[11] = 6;
	indices[12] = 6;	indices[13] = 7;
	indices[14] = 7;	indices[15] = 4;
	// vertical lines
	indices[16] = 0;	indices[17] = 4;
	indices[18] = 1;	indices[19] = 5;
	indices[20] = 2;	indices[21] = 6;
	indices[22] = 3;	indices[23] = 7;

	makeCurrent();

	// transfer vertex data to VBO 0
	m_sdomProgram.shaders.front()->dataBuf.bind();
	m_sdomProgram.shaders.front()->dataBuf.allocate(&data[0], static_cast<int>(data.size() * sizeof(data[0])));
	m_sdomProgram.shaders.front()->dataBuf.release();

	// transfer index data to VBO 1
	m_sdomProgram.shaders.front()->indexBuf.bind();
	m_sdomProgram.shaders.front()->indexBuf.allocate(&indices[0], static_cast<int>(indices.size() * sizeof(indices[0])));
	m_sdomProgram.shaders.front()->indexBuf.release();

	doneCurrent();

	// save number of objects to be drawn
	m_sdomProgram.shaders.front()->objects = static_cast<GLsizei>(indices.size());
}

void COpenGLViewShader::SetPeriodic(const SPeriodic& _pbc)
{
	if (!isValid()) return; // check whether current context is valid

	const bool visible = _pbc.x || _pbc.y || _pbc.z;
	ResizeShaderesBlock(&m_pbcsProgram, visible ? 1 : 0); // resize and initialize all blocks of shader program
	if (!visible) return; // exit if nothing to draw

	const size_t nVerticesPerBoundary = 12; // total number of vertices to draw a single boundary

	// a continuous list of all vertices of the periodic boundaries
	std::vector<SSimpleVertex> data;

	// all points
	const QVector3D v0(_pbc.coordMin.x(), _pbc.coordMin.y(), _pbc.coordMin.z());
	const QVector3D v1(_pbc.coordMax.x(), _pbc.coordMin.y(), _pbc.coordMin.z());
	const QVector3D v2(_pbc.coordMax.x(), _pbc.coordMax.y(), _pbc.coordMin.z());
	const QVector3D v3(_pbc.coordMin.x(), _pbc.coordMax.y(), _pbc.coordMin.z());
	const QVector3D v4(_pbc.coordMin.x(), _pbc.coordMin.y(), _pbc.coordMax.z());
	const QVector3D v5(_pbc.coordMax.x(), _pbc.coordMin.y(), _pbc.coordMax.z());
	const QVector3D v6(_pbc.coordMax.x(), _pbc.coordMax.y(), _pbc.coordMax.z());
	const QVector3D v7(_pbc.coordMin.x(), _pbc.coordMax.y(), _pbc.coordMax.z());

	if (_pbc.x)
	{
		data.reserve(data.size() + nVerticesPerBoundary);
		data.push_back(SSimpleVertex{ v0, 0.0f }); data.push_back(SSimpleVertex{ v4, 0.0f }); data.push_back(SSimpleVertex{ v3, 0.0f });
		data.push_back(SSimpleVertex{ v3, 0.0f }); data.push_back(SSimpleVertex{ v4, 0.0f }); data.push_back(SSimpleVertex{ v7, 0.0f });
		data.push_back(SSimpleVertex{ v1, 0.0f }); data.push_back(SSimpleVertex{ v2, 0.0f }); data.push_back(SSimpleVertex{ v5, 0.0f });
		data.push_back(SSimpleVertex{ v5, 0.0f }); data.push_back(SSimpleVertex{ v2, 0.0f }); data.push_back(SSimpleVertex{ v6, 0.0f });
	}
	if (_pbc.y)
	{
		data.reserve(data.size() + nVerticesPerBoundary);
		data.push_back(SSimpleVertex{ v0, 1.0f }); data.push_back(SSimpleVertex{ v1, 1.0f }); data.push_back(SSimpleVertex{ v5, 1.0f });
		data.push_back(SSimpleVertex{ v5, 1.0f }); data.push_back(SSimpleVertex{ v4, 1.0f }); data.push_back(SSimpleVertex{ v0, 1.0f });
		data.push_back(SSimpleVertex{ v2, 1.0f }); data.push_back(SSimpleVertex{ v3, 1.0f }); data.push_back(SSimpleVertex{ v7, 1.0f });
		data.push_back(SSimpleVertex{ v7, 1.0f }); data.push_back(SSimpleVertex{ v2, 1.0f }); data.push_back(SSimpleVertex{ v6, 1.0f });
	}
	if (_pbc.z)
	{
		data.reserve(data.size() + nVerticesPerBoundary);
		data.push_back(SSimpleVertex{ v0, 2.0f }); data.push_back(SSimpleVertex{ v3, 2.0f }); data.push_back(SSimpleVertex{ v1, 2.0f });
		data.push_back(SSimpleVertex{ v1, 2.0f }); data.push_back(SSimpleVertex{ v3, 2.0f }); data.push_back(SSimpleVertex{ v2, 2.0f });
		data.push_back(SSimpleVertex{ v4, 2.0f }); data.push_back(SSimpleVertex{ v5, 2.0f }); data.push_back(SSimpleVertex{ v7, 2.0f });
		data.push_back(SSimpleVertex{ v7, 2.0f }); data.push_back(SSimpleVertex{ v5, 2.0f }); data.push_back(SSimpleVertex{ v6, 2.0f });
	}

	// transfer vertex data to VBO 0
	makeCurrent();
	m_pbcsProgram.shaders.front()->dataBuf.bind();
	m_pbcsProgram.shaders.front()->dataBuf.allocate(data.data(), static_cast<int>(data.size() * sizeof(data[0])));
	m_pbcsProgram.shaders.front()->dataBuf.release();
	doneCurrent();

	// save number of objects to be drawn
	m_pbcsProgram.shaders.front()->objects = static_cast<GLsizei>(data.size());
}

void COpenGLViewShader::SetOrientations(const std::vector<SOrientation>& _orientations)
{
	if (!isValid()) return; // check whether current context is valid

	const size_t nArrows3 = _orientations.size();						                            // total number of triple arrows to be rendered
	const size_t nLinesPerArrow = m_arrwPreData.sin.size();		                                    // number of edges to approximate base of each single arrow
	const size_t nPointsPerArrow3 = 3 * (nLinesPerArrow + 1);                                       // number of points to render each triple arrow
	const size_t nArrows3PerBlock = c_BYTES_PER_BLOCK / (nPointsPerArrow3 * sizeof(SSimpleVertex)); // number of triple arrows that will be rendered together
	const size_t nBlocks = (nArrows3 + nArrows3PerBlock - 1) / nArrows3PerBlock;                    // number of needed blocks to render all triple arrows
	ResizeShaderesBlock(&m_orntProgram, nBlocks);

	// for each block
	for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock)
	{
		const size_t shift = iBlock * nArrows3PerBlock;                              // shift in _orientations vector for current block
		const size_t nArrows3InBlock = std::min(nArrows3PerBlock, nArrows3 - shift); // number of triple arrows in current block
		const size_t nIndicesPerArrow3 = nPointsPerArrow3 + 3 * 2;					 // number of indices needed to describe each triple arrow
		const size_t nVertices = nArrows3InBlock * nPointsPerArrow3;		         // number of vertices needed to describe all triple arrows in block
		const size_t nIndices = nArrows3InBlock * nIndicesPerArrow3;				 // number of indices needed to describe all triple arrows in block

		// a continuous list of vertices for all triple arrows in block
		std::vector<SSimpleVertex> data(nVertices);
		// list of indices, pointing to vertices, which indicate the order of points drawing; draw as GL_TRIANGLE_FAN
		// e.g., for the first arrow, drawn with 4 walls, it will look like [0, 1, 2, 3, 4, 1, 0xFFFFFFFF], if 0 is a spike point
		std::vector<GLuint> indices(nIndices);

		// for each triple arrow
		ParallelFor(nArrows3InBlock, [&](size_t j)
		{
			const size_t iArrow3 = j + shift; // index of triple arrow in _orientations vector
			const float lenght = 2 * _orientations[iArrow3].radius;  // arrow length
			const float radius = _orientations[iArrow3].radius / 2;  // arrow base radius
			const QVector3D& center = _orientations[iArrow3].coord;  // center of bases
			const QQuaternion& quat = _orientations[iArrow3].orient; // quaternion, describing orientation

			// initialize indices
			size_t iDataOffset = j * nPointsPerArrow3;
			size_t iIndexOffset = j * nIndicesPerArrow3;

			// function to set data and index buffer for a single arrow
			const auto SetupBuffers = [&](float _type)
			{
				// center of spike in current direction
				const QVector3D& coord = quat.rotatedVector({ _type == 0.0f ? lenght : 0.0f, _type == 1.0f ? lenght : 0.0f, _type == 2.0f ? lenght : 0.0f }) + center;
				// get offsets relative to the cylinder's central axis for points lying on the cylindrical surface
				std::vector<QVector3D> points = CylinderOffsets(center, coord, radius, m_arrwPreData);
				// set point at spike
				data[iDataOffset++] = { coord, _type };
				indices[iIndexOffset++] = static_cast<GLuint>(iDataOffset - 1); // index of the spike point
				// for each point on base
				for (size_t iLine = 0; iLine < nLinesPerArrow; ++iLine)
				{
					// calculate vertex
					data[iDataOffset++] = { { center + points[nLinesPerArrow - iLine - 1] }, _type };
					// set index
					indices[iIndexOffset++] = static_cast<GLuint>(iDataOffset - 1);
				}
				indices[iIndexOffset++] = static_cast<GLuint>(iDataOffset - nLinesPerArrow); // repeat the first point to close the figure
				indices[iIndexOffset++] = 0xFFFFFFFF;										 // special index to restart the primitive drawing
			};

			SetupBuffers(0.0f); // X axis
			SetupBuffers(1.0f); // Y axis
			SetupBuffers(2.0f); // Z axis
		});

		makeCurrent();

		// transfer vertex data to VBO 0
		m_orntProgram.shaders[iBlock]->dataBuf.bind();
		m_orntProgram.shaders[iBlock]->dataBuf.allocate(&data[0], static_cast<int>(data.size() * sizeof(data[0])));
		m_orntProgram.shaders[iBlock]->dataBuf.release();

		// transfer index data to VBO 1
		m_orntProgram.shaders[iBlock]->indexBuf.bind();
		m_orntProgram.shaders[iBlock]->indexBuf.allocate(&indices[0], static_cast<int>(indices.size() * sizeof(indices[0])));
		m_orntProgram.shaders[iBlock]->indexBuf.release();

		doneCurrent();

		// save number of objects to be drawn
		m_orntProgram.shaders[iBlock]->objects = static_cast<GLsizei>(indices.size());
	}
}

void COpenGLViewShader::SetAxes(bool _visible/* = true*/)
{
	if (!isValid()) return; // check whether current context is valid

	ResizeShaderesBlock(&m_axisProgram, _visible ? 1 : 0); // resize and initialize all blocks of shader program
	if (!_visible) return; // exit if nothing to draw

	// each axis is drawn as a cylinder, closed from bottom and a closed cone
	const float lengthBase = 0.8f;				   // coefficient of an axis' base length (0..1)
	const float lengthArrow = 1.0f - lengthBase;   // coefficient of an axis' arrow length
	const float radiusBase = lengthBase / 20.0f;   // radius of an axis' base
	const float radiusArrow = 1.75f * radiusBase;  // radius of an axis' arrow

	const size_t nLines = 20;					   // number of points to approximate each circle
	const size_t nVerticesPerAxis = nLines * 15;   // number of vertices needed to describe a single axis
	const size_t nVertices = nVerticesPerAxis * 3; // total number of vertices for 3 axes

	// a continuous list of vertices for all axes
	std::vector<SAxisVertex> data(nVertices);

	// precalculate cylinder-specific data, needed for calculation of offsets
	const SCylinderPreData preData = PrecalculateCylinderData(nLines);

	// a function to set axis data to VBO array
	const auto SetData = [&](const QVector3D& _p1, const QVector3D& _p2, const QVector3D& _p3, float _type, size_t& _i)
	{
		const QVector3D normal = Normalize(QVector3D::crossProduct(_p2 - _p1, _p3 - _p1)); // normal vector
		data[_i++] = { _p1, normal, _type };
		data[_i++] = { _p2, normal, _type };
		data[_i++] = { _p3, normal, _type };
	};

	ParallelFor(3, [&](size_t iAxis)
	{
		const QVector3D coord0{ 0.0f , 0.0f , 0.0f };														// first coordinate of all axes
		const QVector3D coordBase = {  iAxis == 0 ? coord0.x() + lengthBase : coord0.x() ,
									   iAxis == 1 ? coord0.y() + lengthBase : coord0.y() ,
									   iAxis == 2 ? coord0.z() - lengthBase : coord0.z() };					// center of the second cylinder's base
		const QVector3D coordArrow = { iAxis == 0 ? coord0.x() + lengthBase + lengthArrow : coord0.x() ,
									   iAxis == 1 ? coord0.y() + lengthBase + lengthArrow : coord0.y() ,
									   iAxis == 2 ? coord0.z() - lengthBase - lengthArrow : coord0.z() };	// arrowhead

		// get offsets relative to the cylinder's central axis for points lying on the cylindrical surface
		std::vector<QVector3D> offsBase = CylinderOffsets(coord0, coordBase, radiusBase, preData);			// for base
		offsBase.push_back(offsBase[0]);	// duplicate the first for easier iteration
		std::vector<QVector3D> offsArrow = CylinderOffsets(coordBase, coordArrow, radiusArrow, preData);	// for arrow
		offsArrow.push_back(offsArrow[0]);	// duplicate the first for easier iteration

		/// draw base circle
		size_t iDataOffset = iAxis * nVerticesPerAxis;
		for (size_t iLine = 0; iLine < nLines; ++iLine)
			SetData(coord0, coord0 + offsBase[iLine + 1], coord0 + offsBase[iLine + 0], static_cast<float>(iAxis), iDataOffset);
		/// draw cylinder
		for (size_t iLine = 0; iLine < nLines; ++iLine)
		{
			SetData(coord0 + offsBase[iLine + 0], coord0 + offsBase[iLine + 1], coordBase + offsBase[iLine + 0], static_cast<float>(iAxis), iDataOffset);
			SetData(coordBase + offsBase[iLine + 0], coord0 + offsBase[iLine + 1], coordBase + offsBase[iLine + 1], static_cast<float>(iAxis), iDataOffset);
		}
		/// draw arrow circle
		for (size_t iLine = 0; iLine < nLines; ++iLine)
			SetData(coordBase, coordBase + offsArrow[iLine + 1], coordBase + offsArrow[iLine + 0], static_cast<float>(iAxis), iDataOffset);
		/// draw arrow head
		for (size_t iLine = 0; iLine < nLines; ++iLine)
			SetData(coordArrow, coordBase + offsArrow[iLine + 0], coordBase + offsArrow[iLine + 1], static_cast<float>(iAxis), iDataOffset);
	});

	makeCurrent();

	// transfer vertex data to VBO 0
	m_axisProgram.shaders.front()->dataBuf.bind();
	m_axisProgram.shaders.front()->dataBuf.allocate(data.data(), static_cast<int>(data.size() * sizeof(data[0])));
	m_axisProgram.shaders.front()->dataBuf.release();

	doneCurrent();

	// save number of objects to be drawn
	m_axisProgram.shaders.front()->objects = static_cast<GLsizei>(data.size());
}

void COpenGLViewShader::SetTime(double _time, bool _visible/* = true*/)
{
	m_time = _visible ? _time : -1;
}

void COpenGLViewShader::SetLegend(const SLegend& _legend, bool _visible/* = true*/)
{
	m_legend = _visible ? _legend : SLegend{ 0.0, 0.0, Qt::transparent, Qt::transparent, Qt::transparent };
}

void COpenGLViewShader::SetParticleTexture(const QString& _path)
{
	// update path to texture
	if (_path.isEmpty()) return;
	m_partProgram.fileTexture = _path;
	// if the context is already created, replace texture with new one
	makeCurrent();
	if(isInitialized(this->QOpenGLFunctions::d_ptr))
		InitTextures(&m_partProgram);
	doneCurrent();
}

void COpenGLViewShader::Redraw()
{
	update();
}

void COpenGLViewShader::SetRenderQuality(uint8_t _quality)
{
	// scale to allowed values
	const uint8_t quality = c_LINES_PER_CYLINDER_MIN + (c_LINES_PER_CYLINDER_MAX - c_LINES_PER_CYLINDER_MIN) / 100.0f * static_cast<float>(_quality);
	m_bondPreData = PrecalculateCylinderData(quality);
}

QImage COpenGLViewShader::Snapshot(uint8_t _scaling)
{
	// adjust scaling factor for rendering
	m_scaling = _scaling;
	// save current sizes
	const QSize oldSize = m_windowSize;
	// resize with scaling
	resize(m_windowSize * m_scaling);
	// get current image
	QImage image = grabFramebuffer();
	// restore sizes
	m_scaling = 1;
	resize(oldSize);

	return image;
}

SBox COpenGLViewShader::WinCoord2LineOfSight(const QPoint& _pos) const
{
	// get the viewport rectangle
	const QRect viewPort(0, 0, m_windowSize.width(), m_windowSize.height());
	// get window coordinates
	const GLdouble winX = _pos.x();
	const GLdouble winY = static_cast<GLdouble>(m_windowSize.height() - _pos.y() - 1);
	const GLdouble winZMin = -1.0;	// zNear
	const GLdouble winZMax = 1.0;	// zFar
	// get scene coordinates
	const QVector3D coordMin = QVector3D(winX, winY, winZMin).unproject(m_world, m_projection, viewPort);
	const QVector3D coordMax = QVector3D(winX, winY, winZMax).unproject(m_world, m_projection, viewPort);
	// gather box
	return SBox{ coordMin, coordMax };
}

void COpenGLViewShader::mousePressEvent(QMouseEvent* _event)
{
	m_lastMousePos = _event->pos();
}

void COpenGLViewShader::mouseMoveEvent(QMouseEvent* _event)
{
	const float dx = static_cast<float>(_event->x() - m_lastMousePos.x()) / static_cast<float>(m_windowSize.height());
	const float dy = static_cast<float>(_event->y() - m_lastMousePos.y()) / static_cast<float>(m_windowSize.width());

	if (_event->buttons() & Qt::LeftButton && _event->modifiers() & Qt::ShiftModifier)
	{
		m_cameraRotation.setZ(m_cameraRotation.z() + dx * 100);
		update();
	}
	else if (_event->buttons() & Qt::LeftButton)
	{
		m_cameraRotation.setX(m_cameraRotation.x() + dy * 100);
		m_cameraRotation.setY(m_cameraRotation.y() + dx * 100);
		update();
	}
	else if (_event->buttons() & Qt::RightButton)
	{
		m_cameraTranslation.setX(m_cameraTranslation.x() + dx / 5.0f * std::fabs(m_cameraTranslation.z()));
		m_cameraTranslation.setY(m_cameraTranslation.y() - dy / 5.0f * std::fabs(m_cameraTranslation.z()));
		update();
	}
	m_lastMousePos = _event->pos();
}

void COpenGLViewShader::wheelEvent(QWheelEvent* _event)
{
	m_cameraTranslation.setZ(m_cameraTranslation.z() + static_cast<float>(_event->angleDelta().y()) / 120 * 0.05f * std::fabs(m_cameraTranslation.z()));
	update();
}

void COpenGLViewShader::InitShaders(SShaderBlock* _shaderBlock) const
{
	const QString fileNameSuffix = ":/MusenGUI/shaders/" + _shaderBlock->fileShader;

	// check if an OpenGL version without deprecated functions is used
	const bool isCore = format().profile() == QSurfaceFormat::CoreProfile;

	for (auto& shader : _shaderBlock->shaders)
	{
		// compile vertex shader
		shader->program.addShaderFromSourceFile(QOpenGLShader::Vertex, fileNameSuffix + (isCore ? "VertCore.glsl" : "VertCompatibility.glsl"));

		// compile fragment shader
		shader->program.addShaderFromSourceFile(QOpenGLShader::Fragment, fileNameSuffix + (isCore ? "FragCore.glsl" : "FragCompatibility.glsl"));

		// link shader pipeline
		shader->program.link();

		// try to create VAO
		shader->vao.create();
		/*[[maybe_unused]]*/ QOpenGLVertexArrayObject::Binder vaoBinder(&shader->vao);

		// enable attributes in vertex shader
		for (auto& name : _shaderBlock->attributeNames)
			shader->program.enableAttributeArray(name.toLocal8Bit().data());

		// generate VBOs
		shader->dataBuf.create();
		if (_shaderBlock->useIndex)
			shader->indexBuf.create();
	}

	// initialize textures if needed
	InitTextures(_shaderBlock);
}

void COpenGLViewShader::InitTextures(SShaderBlock* _shaderBlock)
{
	if (!_shaderBlock->useTexture || _shaderBlock->fileTexture.isEmpty())
		return;

	for (auto& shader : _shaderBlock->shaders)
	{
		// destroy texture if it already exists
		if(shader->texture.isCreated())
			shader->texture.destroy();

		// load texture image
		shader->texture.setData(QImage(_shaderBlock->fileTexture));

		// set filtering modes for texture minification and magnification
		shader->texture.setMinificationFilter(QOpenGLTexture::Nearest);
		shader->texture.setMagnificationFilter(QOpenGLTexture::Linear);

		// wrap texture coordinates by repeating
		shader->texture.setWrapMode(QOpenGLTexture::Repeat);
	}
}

void COpenGLViewShader::ResizeShaderesBlock(SShaderBlock* _shaderBlock, size_t _newSize)
{
	if (_shaderBlock->shaders.size() == _newSize) return;

	makeCurrent();

	ClearShaderesBlock(_shaderBlock);
	for (size_t i = 0; i < _newSize; ++i)
		_shaderBlock->shaders.emplace_back(new SShaderProgram);
	InitShaders(_shaderBlock);

	doneCurrent();
}

void COpenGLViewShader::ClearShaderesBlock(SShaderBlock* _shaderBlock)
{
	for (auto& shader : _shaderBlock->shaders)
	{
		shader->dataBuf.destroy();
		if (_shaderBlock->useIndex)
			shader->indexBuf.destroy();
		if (_shaderBlock->useTexture)
			shader->texture.destroy();
		if (shader->program.isLinked())
			shader->program.release();
	}
	_shaderBlock->shaders.clear();
}

void COpenGLViewShader::DrawParticles()
{
	// check if there is something to draw
	if (m_partProgram.shaders.empty()) return;

	// fill polygons
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	// disable blending of alpha channel
	glDisable(GL_BLEND);

	// scaling factor for particles sizes
	const GLfloat scale = (static_cast<GLfloat>(m_windowSize.height()) > 0.0f ? static_cast<GLfloat>(m_windowSize.height()) : 1.0f) / std::tan(m_viewport.fovy * 0.5f * static_cast<GLfloat>(PI_180));

	for (auto& shader : m_partProgram.shaders)
	{
		// check if there is something to draw
		if (!shader->objects) continue;

		// bind VAO
		/*[[maybe_unused]]*/ QOpenGLVertexArrayObject::Binder vaoBinder(&shader->vao);

		// bind program for drawing
		shader->program.bind();

		// bind selected texture
		shader->texture.bind();

		// set uniform parameters for vertex and fragment shaders
		shader->program.setUniformValue("u_matrix_p",   m_projection);
		shader->program.setUniformValue("u_matrix_mv",  m_world);
		shader->program.setUniformValue("u_matrix_mvp", m_projection * m_world);
		shader->program.setUniformValue("u_scale",      scale);
		shader->program.setUniformValue("u_texture",    0);

		// tell OpenGL which vertex buffers to use
		shader->dataBuf.bind();

		// tell OpenGL programmable pipeline how to locate data in selected vertex buffer
		shader->program.setAttributeBuffer("a_position", GL_FLOAT, 0,                                     3, sizeof(SParticleVertex));
		shader->program.setAttributeBuffer("a_color",    GL_FLOAT, sizeof(QVector3D),                     4, sizeof(SParticleVertex));
		shader->program.setAttributeBuffer("a_radius",   GL_FLOAT, sizeof(QVector3D) + sizeof(QVector4D), 1, sizeof(SParticleVertex));

		// use VAO if available
		if (shader->vao.isCreated())
			shader->vao.bind();

		// draw objects
		glDrawArrays(GL_POINTS, 0, shader->objects);

		// release selected resources
		shader->vao.release();
		shader->dataBuf.release();
		shader->program.release();
	}

	// turn back blending of alpha channel
	glEnable(GL_BLEND);
}

void COpenGLViewShader::DrawBonds()
{
	// check if there is something to draw
	if (m_bondProgram.shaders.empty()) return;

	// fill polygons
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	// enable splitting of primitives with a specified index in index array
	glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
	glEnable(GL_PRIMITIVE_RESTART);

	for (auto& shader : m_bondProgram.shaders)
	{
		// check if there is something to draw
		if (!shader->objects) continue;

		// bind VAO
		/*[[maybe_unused]]*/ QOpenGLVertexArrayObject::Binder vaoBinder(&shader->vao);

		// bind program for drawing
		shader->program.bind();

		// set uniform parameters for vertex and fragment shaders
		shader->program.setUniformValue("u_matrix_normal", m_world.normalMatrix());
		shader->program.setUniformValue("u_matrix_mvp",    m_projection * m_world);
		shader->program.setUniformValue("u_matrix_mv",     m_world);

		// tell OpenGL which vertex buffers and index buffers to use
		shader->dataBuf.bind();
		shader->indexBuf.bind();

		// tell OpenGL programmable pipeline how to locate data in selected vertex buffer
		shader->program.setAttributeBuffer("a_position", GL_FLOAT, 0,                     3, sizeof(SSolidVertex));
		shader->program.setAttributeBuffer("a_normal",   GL_FLOAT, sizeof(QVector3D),     3, sizeof(SSolidVertex));
		shader->program.setAttributeBuffer("a_color",    GL_FLOAT, 2 * sizeof(QVector3D), 4, sizeof(SSolidVertex));

		// use VAO if available
		if (shader->vao.isCreated())
			shader->vao.bind();

		// draw objects
		glDrawElements(GL_TRIANGLE_STRIP, shader->objects, GL_UNSIGNED_INT, nullptr);

		// release selected resources
		shader->vao.release();
		shader->dataBuf.release();
		shader->indexBuf.release();
		shader->program.release();
	}

	// turn off splitting of primitives with a specified index in index array
	glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
	glDisable(GL_PRIMITIVE_RESTART);
}

void COpenGLViewShader::DrawWalls()
{
	// check if there is something to draw
	if (m_wallProgram.shaders.empty()) return;

	// fill polygons
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	// TODO: turn off if culling enabled
	//// show back-facing polygons
	//glDisable(GL_CULL_FACE);

	for (auto& shader : m_wallProgram.shaders)
	{
		// check if there is something to draw
		if (!shader->objects) continue;

		// bind VAO
		/*[[maybe_unused]]*/ QOpenGLVertexArrayObject::Binder vaoBinder(&shader->vao);

		// bind program for drawing
		shader->program.bind();

		// set uniform parameters for vertex and fragment shaders
		shader->program.setUniformValue("u_matrix_normal", m_world.normalMatrix());
		shader->program.setUniformValue("u_matrix_mvp",    m_projection * m_world);
		shader->program.setUniformValue("u_matrix_mv",     m_world);

		// tell OpenGL which vertex buffers buffers to use
		shader->dataBuf.bind();

		// tell OpenGL programmable pipeline how to locate data in selected vertex buffer
		shader->program.setAttributeBuffer("a_position", GL_FLOAT, 0,                     3, sizeof(SSolidVertex));
		shader->program.setAttributeBuffer("a_normal",   GL_FLOAT, sizeof(QVector3D),     3, sizeof(SSolidVertex));
		shader->program.setAttributeBuffer("a_color",    GL_FLOAT, 2 * sizeof(QVector3D), 4, sizeof(SSolidVertex));

		// use VAO if available
		if (shader->vao.isCreated())
			shader->vao.bind();

		// draw objects
		glDrawArrays(GL_TRIANGLES, 0, shader->objects);

		// release selected resources
		shader->vao.release();
		shader->dataBuf.release();
		shader->program.release();
	}

	//// turn back culling of back-facing polygons
	//glEnable(GL_CULL_FACE);
}

void COpenGLViewShader::DrawVolumes()
{
	// check if there is something to draw
	if (m_volmProgram.shaders.empty()) return;

	// do not fill polygons
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	// show back-facing polygons
	glDisable(GL_CULL_FACE);

	for (auto& shader : m_volmProgram.shaders)
	{
		// check if there is something to draw
		if (!shader->objects) continue;

		// bind VAO
		/*[[maybe_unused]]*/ QOpenGLVertexArrayObject::Binder vaoBinder(&shader->vao);

		// bind program for drawing
		shader->program.bind();

		// set uniform parameters for vertex and fragment shaders
		shader->program.setUniformValue("u_matrix_mvp", m_projection * m_world);
		//shader->program.setUniformValue("u_thickness",  1.0f);

		// tell OpenGL which vertex buffers buffers to use
		shader->dataBuf.bind();

		// tell OpenGL programmable pipeline how to locate data in selected vertex buffer
		shader->program.setAttributeBuffer("a_position", GL_FLOAT, 0,                                     3, sizeof(SFrameVertex));
		shader->program.setAttributeBuffer("a_color",    GL_FLOAT, sizeof(QVector3D),                     4, sizeof(SFrameVertex));
		//shader->program.setAttributeBuffer("a_type",     GL_FLOAT, sizeof(QVector3D) + sizeof(QVector4D), 1, sizeof(SFrameVertex));

		// use VAO if available
		if (shader->vao.isCreated())
			shader->vao.bind();

		// draw objects
		glDrawArrays(GL_TRIANGLES, 0, shader->objects);

		// release selected resources
		shader->vao.release();
		shader->dataBuf.release();
		shader->program.release();
	}

	// turn back culling of back-facing polygons
	glEnable(GL_CULL_FACE);
}

void COpenGLViewShader::DrawDiscs()
{
	// check if there is something to draw
	if (m_discProgram.shaders.empty()) return;

	// fill polygons
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	// show back-facing polygons
	glDisable(GL_CULL_FACE);
	// enable splitting of primitives with a specified index in index array
	glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
	glEnable(GL_PRIMITIVE_RESTART);

	for (auto& shader : m_discProgram.shaders)
	{
		// check if there is something to draw
		if (!shader->objects) continue;

		// bind VAO
		/*[[maybe_unused]]*/ QOpenGLVertexArrayObject::Binder vaoBinder(&shader->vao);

		// tell OpenGL which vertex buffers and index buffers to use
		shader->program.bind();

		// set uniform parameters for vertex and fragment shaders
		shader->program.setUniformValue("u_matrix_mvp", m_projection * m_world);

		// tell OpenGL which vertex buffers to use
		shader->dataBuf.bind();
		shader->indexBuf.bind();

		// tell OpenGL programmable pipeline how to locate data in selected vertex buffer
		shader->program.setAttributeBuffer("a_position", GL_FLOAT, 0,                 3, sizeof(SDiscVertex));
		shader->program.setAttributeBuffer("a_color",    GL_FLOAT, 3 * sizeof(float), 4, sizeof(SDiscVertex));
		shader->program.setAttributeBuffer("a_local",    GL_FLOAT, 7 * sizeof(float), 2, sizeof(SDiscVertex));

		// use VAO if available
		if (shader->vao.isCreated())
			shader->vao.bind();

		// draw objects
		glDrawElements(GL_TRIANGLE_FAN, shader->objects, GL_UNSIGNED_INT, nullptr);

		// release selected resources
		shader->vao.release();
		shader->dataBuf.release();
		shader->indexBuf.release();
		shader->program.release();
	}

	// turn off showing of back-facing polygons
	glEnable(GL_CULL_FACE);
	// turn off splitting of primitives with a specified index in index array
	glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
	glDisable(GL_PRIMITIVE_RESTART);
}

void COpenGLViewShader::DrawDomain()
{
	// check if there is something to draw
	if (m_sdomProgram.shaders.empty()) return;

	// do not fill polygons
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	// show back-facing polygons
	glDisable(GL_CULL_FACE);

	for (auto& shader : m_sdomProgram.shaders)
	{
		// check if there is something to draw
		if (!shader->objects) continue;

		// bind VAO
		/*[[maybe_unused]]*/ QOpenGLVertexArrayObject::Binder vaoBinder(&shader->vao);

		// bind program for drawing
		shader->program.bind();

		// set uniform parameters for vertex and fragment shaders
		shader->program.setUniformValue("u_matrix_mvp", m_projection * m_world);
		//shader->program.setUniformValue("u_thickness",  1.0f);

		// tell OpenGL which vertex buffers and index buffers to use
		shader->dataBuf.bind();
		shader->indexBuf.bind();

		// tell OpenGL programmable pipeline how to locate data in selected vertex buffer
		shader->program.setAttributeBuffer("a_position", GL_FLOAT, 0,                 3, sizeof(SFrameVertex));
		shader->program.setAttributeBuffer("a_color",    GL_FLOAT, sizeof(QVector3D), 4, sizeof(SFrameVertex));
		//shader->program.setAttributeBuffer("a_type",     GL_FLOAT, sizeof(QVector3D) + sizeof(QVector4D), 1, sizeof(SFrameVertex));

		// use VAO if available
		if (shader->vao.isCreated())
			shader->vao.bind();

		// draw objects
		glDrawElements(GL_LINES, shader->objects, GL_UNSIGNED_INT, nullptr);

		// release selected resources
		shader->vao.release();
		shader->dataBuf.release();
		shader->indexBuf.release();
		shader->program.release();
	}

	// turn back culling of back-facing polygons
	glEnable(GL_CULL_FACE);
}

void COpenGLViewShader::DrawPeriodic()
{
	// check if there is something to draw
	if (m_pbcsProgram.shaders.empty()) return;

	// fill polygons
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	// show back-facing polygons
	glDisable(GL_CULL_FACE);

	for (auto& shader : m_pbcsProgram.shaders)
	{
		// check if there is something to draw
		if (!shader->objects) continue;

		// bind VAO
		/*[[maybe_unused]]*/ QOpenGLVertexArrayObject::Binder vaoBinder(&shader->vao);

		// bind program for drawing
		shader->program.bind();

		// set uniform parameters for vertex and fragment shaders
		shader->program.setUniformValue("u_matrix_mvp", m_projection * m_world);

		// tell OpenGL which vertex buffers and index buffers to use
		shader->dataBuf.bind();

		// tell OpenGL programmable pipeline how to locate data in selected vertex buffer
		shader->program.setAttributeBuffer("a_position", GL_FLOAT, 0,                 3, sizeof(SSimpleVertex));
		shader->program.setAttributeBuffer("a_type",     GL_FLOAT, sizeof(QVector3D), 1, sizeof(SSimpleVertex));

		// use VAO if available
		if (shader->vao.isCreated())
			shader->vao.bind();

		// draw objects
		glDrawArrays(GL_TRIANGLES, 0, shader->objects);

		// release selected resources
		shader->vao.release();
		shader->dataBuf.release();
		shader->program.release();
	}

	// turn back culling of back-facing polygons
	glEnable(GL_CULL_FACE);
}

void COpenGLViewShader::DrawOrientations()
{
	// check if there is something to draw
	if (m_orntProgram.shaders.empty()) return;

	// fill polygons
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	// enable splitting of primitives with a specified index in index array
	glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
	glEnable(GL_PRIMITIVE_RESTART);

	for (auto& shader : m_orntProgram.shaders)
	{
		// check if there is something to draw
		if (!shader->objects) continue;

		// bind VAO
		/*[[maybe_unused]]*/ QOpenGLVertexArrayObject::Binder vaoBinder(&shader->vao);

		// bind program for drawing
		shader->program.bind();

		// set uniform parameters for vertex and fragment shaders
		shader->program.setUniformValue("u_matrix_mvp", m_projection * m_world);

		// tell OpenGL which vertex buffers and index buffers to use
		shader->dataBuf.bind();
		shader->indexBuf.bind();

		// tell OpenGL programmable pipeline how to locate data in selected vertex buffer
		shader->program.setAttributeBuffer("a_position", GL_FLOAT, 0,                 3, sizeof(SSimpleVertex));
		shader->program.setAttributeBuffer("a_type",     GL_FLOAT, sizeof(QVector3D), 1, sizeof(SSimpleVertex));

		// use VAO if available
		if (shader->vao.isCreated())
			shader->vao.bind();

		// draw objects
		glDrawElements(GL_TRIANGLE_FAN, shader->objects, GL_UNSIGNED_INT, nullptr);

		// release selected resources
		shader->vao.release();
		shader->dataBuf.release();
		shader->indexBuf.release();
		shader->program.release();
	}

	// turn off splitting of primitives with a specified index in index array
	glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
	glDisable(GL_PRIMITIVE_RESTART);
}

void COpenGLViewShader::DrawAxes()
{
	// check if there is something to draw
	if (m_axisProgram.shaders.empty()) return;

	// fill polygons
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	// calculate rotation matrix
	static QMatrix4x4 rotation;
	rotation.setToIdentity();
	rotation.rotate(-m_cameraRotation.x(), 1, 0, 0);
	rotation.rotate(-m_cameraRotation.y(), 0, 1, 0);
	rotation.rotate( m_cameraRotation.z(), 0, 0, 1);

	// scale axis size
	const float axisSize = m_axisSize * static_cast<float>(m_scaling);

	// draw axes
	for (auto& shader : m_axisProgram.shaders)
	{
		// check if there is something to draw
		if (!shader->objects) continue;

		// bind VAO
		/*[[maybe_unused]]*/ QOpenGLVertexArrayObject::Binder vaoBinder(&shader->vao);

		// bind program for drawing
		shader->program.bind();

		// set uniform parameters for vertex and fragment  shaders
		shader->program.setUniformValue("u_matrix_rot",    rotation);
		shader->program.setUniformValue("u_matrix_normal", rotation.normalMatrix());
		shader->program.setUniformValue("u_matrix_mv",     m_world);
		shader->program.setUniformValue("u_scaling",       axisSize);
		shader->program.setUniformValue("u_win_width",     m_windowSize.width());
		shader->program.setUniformValue("u_win_height",    m_windowSize.height());

		// tell OpenGL which vertex buffers and index buffers to use
		shader->dataBuf.bind();

		// tell OpenGL programmable pipeline how to locate data in selected vertex buffer
		shader->program.setAttributeBuffer("a_position", GL_FLOAT, 0,                     3, sizeof(SAxisVertex));
		shader->program.setAttributeBuffer("a_normal",   GL_FLOAT, sizeof(QVector3D),     3, sizeof(SAxisVertex));
		shader->program.setAttributeBuffer("a_type",     GL_FLOAT, 2 * sizeof(QVector3D), 1, sizeof(SAxisVertex));

		// use VAO if available
		if (shader->vao.isCreated())
			shader->vao.bind();

		// clear previous depth information to let axes overlay other objects on scene
		glClear(GL_DEPTH_BUFFER_BIT);

		// draw objects
		glDrawArrays(GL_TRIANGLES, 0, shader->objects);

		// release selected resources
		shader->vao.release();
		shader->dataBuf.release();
		shader->program.release();
	}

	/// calculate positions for labels
	// vector for size scaling
	const QVector3D vResize{ float(m_windowSize.width()) / axisSize , float(m_windowSize.height()) / axisSize, 1.0f };
	// vector for position shifting
	const QVector3D vShift{ axisSize * 1.3f / float(m_windowSize.width()) - 1, axisSize * 1.3f / float(m_windowSize.height()) - 1, 0.0f };

	// convert given coordinate from openGL to screen
	const auto ConvertCoord = [&](const QVector3D& _coord)
	{
		// shift in the direction of axis to make indent from axis tip
		const QVector3D shift = _coord * 1.15f;
		// rotate coordinates, resize, shift and project them on screen
		const QVector3D coord = (QVector3D{ rotation * QVector4D(shift, 1.0) } / vResize + vShift).project(QMatrix4x4(), QMatrix4x4(), QRect{ QPoint() , m_windowSize });
		// translate to proper screen coordinates
		return QPoint{ int(coord.x()) - m_windowSize.width() / 2 , m_windowSize.height() - int(coord.y()) - m_windowSize.height() / 2 };
	};

	// disable depth test for overlay
	glDisable(GL_DEPTH_TEST);

	// draw labels
	QPainter painter(this);
	SetupPainter(&painter, m_fontAxes);
	painter.drawText(QRect{ ConvertCoord({ 1.0, 0.0,  0.0 }), m_windowSize }, Qt::AlignCenter, "X");
	painter.drawText(QRect{ ConvertCoord({ 0.0, 1.0,  0.0 }), m_windowSize }, Qt::AlignCenter, "Y");
	painter.drawText(QRect{ ConvertCoord({ 0.0, 0.0, -1.0 }), m_windowSize }, Qt::AlignCenter, "Z");
	painter.end();

	// return depth test to the on state
	glEnable(GL_DEPTH_TEST);
}

void COpenGLViewShader::DrawTime()
{
	// check if there is something to draw
	if (m_time == -1) return;

	// fill polygons
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	// disable depth test for overlay
	glDisable(GL_DEPTH_TEST);

	// size and position constants
	const int margin = 5;

	// draw label
	QPainter painter(this);
	SetupPainter(&painter, m_fontTime);
	painter.drawText(QRect{ {margin * m_scaling, margin * m_scaling}, m_windowSize }, Qt::AlignLeft | Qt::AlignTop, "Time [s]: " + QString::number(m_time));
	painter.end();

	// return depth test to the on state
	glEnable(GL_DEPTH_TEST);
}

void COpenGLViewShader::DrawLegend()
{
	// check if there is something to draw
	if (m_legend.minValue == 0 && m_legend.maxValue == 0 && m_legend.minColor == Qt::transparent && m_legend.midColor == Qt::transparent && m_legend.maxColor == Qt::transparent) return;

	// fill polygons
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	// disable depth test for overlay
	glDisable(GL_DEPTH_TEST);
	// show back-facing polygons; is required for painter
	glDisable(GL_CULL_FACE);

	// size and position constants
	const int barMargin  = 10 * m_scaling;
	const int textMargin = 5 * m_scaling;
	const int barWidth   = 15 * m_scaling;
	const int barHeight  = m_windowSize.height() / 2;
	const int barTop     = static_cast<int>(0.15 * m_windowSize.height());

	// create color bar
	const QRect colorbar{ barMargin, barTop, barWidth, barHeight };
	// create gradient
	QLinearGradient gradient(colorbar.left(), colorbar.top(), colorbar.left(), colorbar.bottom());
	gradient.setColorAt(1.0, m_legend.minColor);
	gradient.setColorAt(0.5, m_legend.midColor);
	gradient.setColorAt(0.0, m_legend.maxColor);
	// get position and rect for text
	const QRect textPos = colorbar.translated(barWidth + textMargin, 0).adjusted(0, 0, m_windowSize.width() / 2, 0);

	// draw labels
	QPainter painter(this);
	SetupPainter(&painter, m_fontLegend);
	painter.fillRect(colorbar, gradient);
	painter.drawText(textPos.translated(0, -barHeight / 2), Qt::AlignVCenter | Qt::AlignLeft, QString::number(m_legend.maxValue));
	painter.drawText(textPos.translated(0, -barHeight / 4), Qt::AlignVCenter | Qt::AlignLeft, QString::number((m_legend.minValue + 3 * m_legend.maxValue) / 4));
	painter.drawText(textPos.translated(0,  0),             Qt::AlignVCenter | Qt::AlignLeft, QString::number((m_legend.minValue + m_legend.maxValue) / 2));
	painter.drawText(textPos.translated(0,  barHeight / 4), Qt::AlignVCenter | Qt::AlignLeft, QString::number((3 * m_legend.minValue + m_legend.maxValue) / 4));
	painter.drawText(textPos.translated(0,  barHeight / 2), Qt::AlignVCenter | Qt::AlignLeft, QString::number(m_legend.minValue));
	painter.end();

	// return depth test to the on state
	glEnable(GL_DEPTH_TEST);
	// turn back culling of back-facing polygons
	glEnable(GL_CULL_FACE);
}

COpenGLViewShader::SCylinderPreData COpenGLViewShader::PrecalculateCylinderData(uint8_t _linesPerBond)
{
	SCylinderPreData res { std::vector<float>(_linesPerBond), std::vector<float>(_linesPerBond) };
	// needed to calculate positions of points, laying on the edges of both cylinder's circles
	for (size_t iLine = 0; iLine < _linesPerBond; ++iLine)
	{
		const float angle = 2.0f * static_cast<float>(PI) * iLine / static_cast<float>(_linesPerBond);
		res.sin[iLine] = std::sin(angle);
		res.cos[iLine] = std::cos(angle);
	}
	return res;
}

std::vector<QVector3D> COpenGLViewShader::CylinderOffsets(const QVector3D& _coord1, const QVector3D& _coord2, float _radius, const SCylinderPreData& _preData)
{
	const QVector3D len = _coord1 - _coord2;
	QVector3D p1(0.0f, 0.0f, 0.0f);
	if (len.x() == 0.0f || len.y() == 0.0f || len.z() == 0.0f)
	{
		if (len.x() == 0.0f)		p1.setX(1.0f);
		else if (len.y() == 0.0f)	p1.setY(1.0f);
		else						p1.setZ(1.0f);
	}
	else
	{
		// if X, Y, Z are all set, set the Z coordinate as first and second argument
		// as the scalar product must be zero, add the negated sum of X and Y as third argument
		p1.setX(len.z());				// scalp = z*x
		p1.setY(len.z());				// scalp = z*(x+y)
		p1.setZ(-(len.x() + len.y()));	// scalp = z*(x+y)-z*(x+y) = 0
		p1 = Normalize(p1);				// normalize vector
	}
	const QVector3D p2 = Normalize(QVector3D::crossProduct(len, p1));
	std::vector<QVector3D> offsets(_preData.sin.size()); // offsets relative to the cylinder's central axis for points lying on the surface
	for (GLuint j = 0; j < _preData.sin.size(); ++j)
	{
		offsets[j] = QVector3D{
			_radius*(_preData.cos[j] * p1.x() + _preData.sin[j] * p2.x()) ,
			_radius*(_preData.cos[j] * p1.y() + _preData.sin[j] * p2.y()) ,
			_radius*(_preData.cos[j] * p1.z() + _preData.sin[j] * p2.z()) };
	}

	return offsets;
}

QVector4D COpenGLViewShader::QColorToQVector4D(const QColor& _c)
{
	return QVector4D{ static_cast<float>(_c.redF()), static_cast<float>(_c.greenF()), static_cast<float>(_c.blueF()), static_cast<float>(_c.alphaF()) };
}

QVector3D COpenGLViewShader::Normalize(const QVector3D& _v)
{
	const float len = std::sqrt(_v.x() * _v.x() + _v.y() * _v.y() + _v.z() * _v.z());
	return len != 0 ? _v / len : QVector3D(0,0,0);
}
