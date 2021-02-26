/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "OpenGLView.h"

class COpenGLViewMixed : public COpenGLView, protected QOpenGLFunctions
{
	Q_OBJECT

private:
	static const QString m_csStandardSphereTexture;

protected:
	struct SGLvertex
	{
		GLfloat x;
		GLfloat y;
		GLfloat z;
	};

	GLint  m_nAttributeVertex;
	GLint  m_nAttributeRadius;
	GLint  m_nAttributeColor;
	GLint  m_nUniformMatrixP;
	GLint  m_nUniformMatrixMV;
	GLint  m_nUniformMatrixMVP;
	GLint  m_nUniformTexture;
	GLint  m_nUniformScale;
	QMatrix4x4 m_MatrixProjection;
	QMatrix4x4 m_MatrixModelView;

	QOpenGLShaderProgram *m_pProgram{ nullptr };
	QOpenGLShader *m_pFShader{ nullptr }; //shaders for particles
	QOpenGLShader *m_pVShader{ nullptr };

	QOpenGLBuffer m_VBOVertex;
	QOpenGLBuffer m_VBORadius;
	QOpenGLBuffer m_VBOColor;

	QOpenGLTexture *m_pTexture{ nullptr };

	size_t m_nSpheresToPaint;
	QString m_sCurrTexture;

public:
	COpenGLViewMixed(CViewSettings* _viewSettings, QWidget* _parent = nullptr);
	COpenGLViewMixed(const CBaseGLView& _other, CViewSettings* _viewSettings, QWidget* _parent = nullptr);
	~COpenGLViewMixed();

	void SetParticleTexture(const QString& _sPath = "") override;

private:
	void DrawParticles() override;

	void RecreateBuffer(QOpenGLBuffer* _pBuffer, const void* _pData, size_t _nBytes);

protected:
	void initializeGL() override;

	void InitializeShader();
	void InitializeVBO();
	void InitializeTextures(const QString& _sPath = "");
	void FreeShader();
	void FreeVBO();
	void FreeTexture();
};
