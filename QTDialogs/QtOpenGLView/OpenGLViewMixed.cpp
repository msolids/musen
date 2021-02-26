/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "OpenGLViewMixed.h"

const QString COpenGLViewMixed::m_csStandardSphereTexture = QString(":/QT_GUI/Pictures/SphereTexture0.png");

COpenGLViewMixed::COpenGLViewMixed(CViewSettings* _viewSettings, QWidget *_parent)
	:COpenGLView(_viewSettings, _parent)
{
	m_sCurrTexture = m_csStandardSphereTexture;
}

COpenGLViewMixed::COpenGLViewMixed(const CBaseGLView& _other, CViewSettings* _viewSettings, QWidget* _parent) :
	COpenGLView(_other, _viewSettings, _parent)
{
	m_sCurrTexture = m_csStandardSphereTexture;
}

COpenGLViewMixed::~COpenGLViewMixed()
{
	makeCurrent();

	FreeShader();
	FreeVBO();
	FreeTexture();

	glDisable(GL_PROGRAM_POINT_SIZE);
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glDisable(GL_COLOR_MATERIAL);

	doneCurrent();
}

void COpenGLViewMixed::DrawParticles()
{
	if (!m_viewSettings->Visibility().particles) return;

	InitializeVBO();

	const GLfloat spheresScale = static_cast<GLfloat>(m_windowSize.height()) / std::tan(m_viewport.fovy * 0.5f * static_cast<GLfloat>(PI_180));

	m_MatrixProjection.setToIdentity();
	m_MatrixProjection.perspective(m_viewport.fovy, m_viewport.aspect, m_viewport.zNear, m_viewport.zFar);

	m_MatrixModelView.setToIdentity();
	m_MatrixModelView.translate(m_cameraTranslation[0], m_cameraTranslation[1], m_cameraTranslation[2]);
	m_MatrixModelView.rotate(m_cameraRotation[0], 1.0, 0.0, 0.0);
	m_MatrixModelView.rotate(m_cameraRotation[1], 0.0, 1.0, 0.0);
	m_MatrixModelView.rotate(m_cameraRotation[2], 0.0, 0.0, 1.0);

	m_pProgram->bind();
	m_pProgram->setUniformValue(m_nUniformMatrixP, m_MatrixProjection);
	m_pProgram->setUniformValue(m_nUniformMatrixMV, m_MatrixModelView);
	m_pProgram->setUniformValue(m_nUniformMatrixMVP, m_MatrixProjection*m_MatrixModelView);
	m_pProgram->setUniformValue(m_nUniformScale, spheresScale);

	m_pProgram->enableAttributeArray(m_nAttributeVertex);
	m_pProgram->enableAttributeArray(m_nAttributeRadius);
	m_pProgram->enableAttributeArray(m_nAttributeColor);

	m_VBOVertex.bind();
	m_pProgram->setAttributeBuffer(m_nAttributeVertex, GL_FLOAT, 0, 3);
	m_VBOVertex.release();

	m_VBORadius.bind();
	m_pProgram->setAttributeBuffer(m_nAttributeRadius, GL_FLOAT, 0, 1);
	m_VBORadius.release();

	m_VBOColor.bind();
	m_pProgram->setAttributeBuffer(m_nAttributeColor, GL_FLOAT, 0, 3);
	m_VBOColor.release();

	m_pTexture->bind();
	m_pProgram->setUniformValue(m_nUniformTexture, 0);

	glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(m_nSpheresToPaint));

	m_pProgram->disableAttributeArray(m_nAttributeVertex);
	m_pProgram->disableAttributeArray(m_nAttributeRadius);
	m_pProgram->disableAttributeArray(m_nAttributeColor);

	m_pProgram->release();
}

void COpenGLViewMixed::initializeGL()
{
	COpenGLView::initializeGL();

	initializeOpenGLFunctions();

	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SPRITE);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

	InitializeShader();
	InitializeTextures();

}

void COpenGLViewMixed::InitializeShader()
{
	m_pVShader = new QOpenGLShader(QOpenGLShader::Vertex);
	m_pVShader->compileSourceFile(":/QT_GUI/shaders/vSphereShader.glsl");

	m_pFShader = new QOpenGLShader(QOpenGLShader::Fragment);
	m_pFShader->compileSourceFile(":/QT_GUI/shaders/fSphereShader.glsl");

	m_pProgram = new QOpenGLShaderProgram;
	m_pProgram->addShader(m_pVShader);
	m_pProgram->addShader(m_pFShader);
	m_pProgram->link();

	m_nAttributeVertex = m_pProgram->attributeLocation("SphereCoordinate");
	m_nAttributeRadius = m_pProgram->attributeLocation("SphereRadius");
	m_nAttributeColor = m_pProgram->attributeLocation("SphereColor");
	m_nUniformMatrixP = m_pProgram->uniformLocation("MatrixP");
	m_nUniformMatrixMV = m_pProgram->uniformLocation("MatrixMV");
	m_nUniformMatrixMVP = m_pProgram->uniformLocation("MatrixMVP");
	m_nUniformTexture = m_pProgram->uniformLocation("Texture");
	m_nUniformScale = m_pProgram->uniformLocation("SphereScale");
}

void COpenGLViewMixed::InitializeVBO()
{
	size_t nSpheres = m_pSystemStructure->GetNumberOfSpecificObjects(SPHERE) + m_viewSettings->SelectedObjects().size();
	SGLvertex *pVertices = new SGLvertex[nSpheres];
	GLfloat *pRadiuses = new GLfloat[nSpheres];
	SGLvertex *pColors = new SGLvertex[nSpheres];

	//////// Spheres
	std::vector<bool> vVisibleCutPart; // contains
	if (m_viewSettings->Cutting().cutByVolumes)
	{
		vVisibleCutPart.resize(m_pSystemStructure->GetTotalObjectsCount(), false);
		for (std::string volumeKey : m_viewSettings->Cutting().volumes)
		{
			std::vector<size_t> vNewIndexes = m_pSystemStructure->AnalysisVolume(volumeKey)->GetParticleIndicesInside(m_dCurrentTime);
			for (size_t j = 0; j < vNewIndexes.size(); j++)
				vVisibleCutPart[vNewIndexes[j]] = true;
		}
	}

	CVector3 vCoord;
	CColor tempColor(1.0, 1.0, 1.0, 1.0);
	m_nSpheresToPaint = 0;

	// cutting by materials
	const auto visiblePartMaterials = m_viewSettings->VisiblePartMaterials();

	for (unsigned long i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); i++)
	{
		CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(i);
		if ((object == NULL) || (!object->IsActive(m_dCurrentTime)) || (object->GetObjectType() != SPHERE)) continue;
		vCoord = object->GetCoordinates(m_dCurrentTime);

		if (!SetContains(visiblePartMaterials, object->GetCompoundKey())) continue;

		// cut by plane
		if (IsPointCuttedByPlanes(vCoord)) continue;
		// cut by geometries
		if (m_viewSettings->Cutting().cutByVolumes && !vVisibleCutPart[i]) continue;

		pVertices[m_nSpheresToPaint].x = vCoord.x;
		pVertices[m_nSpheresToPaint].y = vCoord.y;
		pVertices[m_nSpheresToPaint].z = vCoord.z;
		pRadiuses[m_nSpheresToPaint] = static_cast<CSphere*>(object)->GetRadius();
		if (m_viewSettings->Coloring().type != EColoring::NONE)
			GetObjectColor(i, &tempColor);
		pColors[m_nSpheresToPaint].x = tempColor.r;
		pColors[m_nSpheresToPaint].y = tempColor.g;
		pColors[m_nSpheresToPaint].z = tempColor.b;
		m_nSpheresToPaint++;
	}


	///////// Selected spheres
	tempColor.SetColor(1.0, 1.0, 0.0, 1.0);
	for (unsigned long i = 0; i < m_viewSettings->SelectedObjects().size(); i++)
	{
		CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(m_viewSettings->SelectedObjects()[i]);
		if ((object != NULL) && (object->IsActive(m_dCurrentTime)) && (object->GetObjectType() == SPHERE))
		{
			vCoord = object->GetCoordinates(m_dCurrentTime);

			pVertices[m_nSpheresToPaint].x = vCoord.x;
			pVertices[m_nSpheresToPaint].y = vCoord.y;
			pVertices[m_nSpheresToPaint].z = vCoord.z;
			pRadiuses[m_nSpheresToPaint] = static_cast<CSphere*>(object)->GetRadius();
			//if (m_pViewOptionsTab->m_nColouringType != COLOURING_NONE)
			//	GetObjectColor(m_vSelectedObjects[i], &tempColor);
			pColors[m_nSpheresToPaint].x = /*1.0f - */tempColor.r;
			pColors[m_nSpheresToPaint].y = /*1.0f - */tempColor.g;
			pColors[m_nSpheresToPaint].z = /*1.0f - */tempColor.b;
			m_nSpheresToPaint++;
		}
	}

	RecreateBuffer(&m_VBOVertex, pVertices, sizeof(GLfloat)*m_nSpheresToPaint * 3);
	RecreateBuffer(&m_VBORadius, pRadiuses, sizeof(GLfloat)*m_nSpheresToPaint * 1);
	RecreateBuffer(&m_VBOColor, pColors, sizeof(GLfloat)*m_nSpheresToPaint * 3);

	delete[] pVertices;
	delete[] pRadiuses;
	delete[] pColors;
}

void COpenGLViewMixed::RecreateBuffer(QOpenGLBuffer* _pBuffer, const void* _pData, size_t _nBytes)
{
	_pBuffer->destroy();
	_pBuffer->create();
	_pBuffer->bind();
	_pBuffer->allocate(_pData, static_cast<int>(_nBytes));
	_pBuffer->release();
}

void COpenGLViewMixed::InitializeTextures(const QString& _sPath /*= ""*/)
{
	QString sActualPath = _sPath.isEmpty() ? m_sCurrTexture : _sPath;

	QImage image;
	if (image.load(sActualPath))
		m_sCurrTexture = sActualPath;
	else
	{
		image.load(m_csStandardSphereTexture);
		m_sCurrTexture = m_csStandardSphereTexture;
	}
	m_pTexture = new QOpenGLTexture(image);
	m_pTexture->setMinificationFilter(QOpenGLTexture::Nearest);
	m_pTexture->setMagnificationFilter(QOpenGLTexture::Linear);
	m_pTexture->setWrapMode(QOpenGLTexture::Repeat);
}

void COpenGLViewMixed::SetParticleTexture(const QString& _sPath)
{
	makeCurrent();
	if (m_pTexture)
		FreeTexture();
	InitializeTextures(_sPath);

	doneCurrent();
	update();
}

void COpenGLViewMixed::FreeShader()
{
	glUseProgram(0);

	delete m_pProgram;
	delete m_pVShader;
	delete m_pFShader;
}

void COpenGLViewMixed::FreeVBO()
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	m_VBOVertex.destroy();
	m_VBORadius.destroy();
	m_VBOColor.destroy();
}

void COpenGLViewMixed::FreeTexture()
{
	m_pTexture->destroy();
	delete m_pTexture;
	m_pTexture = nullptr;
}
