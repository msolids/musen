/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "OpenGLViewGlu.h"

COpenGLViewGlu::COpenGLViewGlu(CViewSettings* _viewSettings, QWidget *_parent) :
	COpenGLView(_viewSettings, _parent)
{
}

COpenGLViewGlu::COpenGLViewGlu(const CBaseGLView& _other, CViewSettings* _viewSettings, QWidget* _parent) :
	COpenGLView(_other, _viewSettings, _parent)
{
}

COpenGLViewGlu::~COpenGLViewGlu()
{
	glDisable(GL_MULTISAMPLE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_LIGHTING);
	glDisable(GL_LIGHT0);
	glDisable(GL_COLOR_MATERIAL);
}


void COpenGLViewGlu::initializeGL()
{
	COpenGLView::initializeGL();
}

void COpenGLViewGlu::DrawParticles()
{
	if (!m_viewSettings->Visibility().particles) return;

	DrawParticlesWithFFP();
	DrawSelectedParticles();
}

void COpenGLViewGlu::DrawSelectedParticles()
{
	CVector3 tempVector;
	SetColorGL(CColor(1.0f, 1.0f, 0.2f, 0.5f));
	for (unsigned long i = 0; i < m_viewSettings->SelectedObjects().size(); i++)
	{
		CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(m_viewSettings->SelectedObjects()[i]);
		if (object != NULL)
			if (object->IsActive(m_dCurrentTime))
				if (object->GetObjectType() == SPHERE)
				{
					tempVector = object->GetCoordinates(m_dCurrentTime);
					glPushMatrix();
					glTranslatef((GLfloat)tempVector.x, (GLfloat)tempVector.y, (GLfloat)tempVector.z);
					gluSphere(m_pQuadObj, static_cast<CSphere*>(object)->GetRadius()*1.1, m_sphereSlices, m_sphereStacks);
					glPopMatrix();
				}
	}
}

void COpenGLViewGlu::DrawParticlesWithFFP()
{
	CVector3 vCoord;
	CColor tempColor;
	size_t nSpheres = m_pSystemStructure->GetNumberOfSpecificObjects(SPHERE) + m_viewSettings->SelectedObjects().size();

	// cutting by materials
	const auto visiblePartMaterials = m_viewSettings->VisiblePartMaterials();

	std::vector<bool> vVisibleCutPart; // cutting by geometries
	if (m_viewSettings->Cutting().cutByVolumes)
	{
		vVisibleCutPart.resize(m_pSystemStructure->GetTotalObjectsCount(), false);
		for (std::string volumeKey : m_viewSettings->Cutting().volumes)
		{
			std::vector<size_t> vNewIndexes = m_pSystemStructure->GetParticleIndicesInVolume(m_dCurrentTime, volumeKey);
			for (size_t j = 0; j < vNewIndexes.size(); j++)
				vVisibleCutPart[vNewIndexes[j]] = true;
		}
	}

	for (unsigned long i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); i++)
	{
		CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(i);
		if ((object == NULL) || (!object->IsActive(m_dCurrentTime)) || (object->GetObjectType() != SPHERE)) continue;
		vCoord = object->GetCoordinates(m_dCurrentTime);

		if (!SetContains(visiblePartMaterials, object->GetCompoundKey())) continue;

		if (IsPointCuttedByPlanes(vCoord)) continue; // cut by planes
		if (m_viewSettings->Cutting().cutByVolumes && !vVisibleCutPart[object->m_lObjectID]) continue; // cut by geometries

		glPushMatrix();
		GetObjectColor(i, &tempColor);
		SetColorGL(tempColor);
		glTranslatef((GLfloat)vCoord.x, (GLfloat)vCoord.y, (GLfloat)vCoord.z);
		gluSphere(m_pQuadObj, static_cast<CSphere*>(object)->GetRadius(), m_sphereSlices, m_sphereStacks);
		glPopMatrix();
	}
}

