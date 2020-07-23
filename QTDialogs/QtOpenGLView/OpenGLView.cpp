/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "OpenGLView.h"
#include "AgglomeratesAnalyzer.h"

COpenGLView::COpenGLView(CViewSettings* _viewSettings, QWidget* _parent) :
	QGLWidget(QGLFormat(QGL::SampleBuffers), _parent),
	m_viewSettings{ _viewSettings }
{
	COpenGLView::SetRenderQuality(100);
}

COpenGLView::COpenGLView(const CBaseGLView& _other, CViewSettings* _viewSettings, QWidget* _parent) :
	QGLWidget(QGLFormat(QGL::SampleBuffers), _parent),
	CBaseGLView(_other),
	m_viewSettings{ _viewSettings }
{
}

COpenGLView::~COpenGLView()
{
	gluDeleteQuadric(m_pQuadObj);
}

void COpenGLView::SetSystemStructure(CSystemStructure* _pSystemStructure)
{
	m_pSystemStructure = _pSystemStructure;
}

void COpenGLView::initializeGL()
{
	setAutoFillBackground(false);

#ifdef _DEBUG
	qDebug() << "OpenGL version used: " << this->format().majorVersion() << "." << this->format().minorVersion();
	qDebug() << "OpenGL profile used: " << this->format().profile();
#endif
}

void COpenGLView::resizeGL(int width, int height)
{
	// saving new width and height:
	m_windowSize.setWidth(width);
	m_windowSize.setHeight(height);
	// update perspective
	UpdatePerspective();

	CVector3 v1 = WinCoord2GL(0, 0, 0.01);
	CVector3 v2 = WinCoord2GL(100, 0, 0.01);
	double dPtInPx = Length(v1 - v2) / 100;
	m_dWinSizePx = std::min(m_windowSize.width(), m_windowSize.height());
	m_dWinSize = m_dWinSizePx * dPtInPx;
}

void COpenGLView::UpdatePerspective()
{
	// calculate aspect ratio
	m_viewport.aspect = static_cast<float>(m_windowSize.width()) / static_cast<float>(m_windowSize.height() != 0 ? m_windowSize.height() : 1);
	// changing viewport size to match the new size of the window:
	glViewport(0, 0, m_windowSize.width(), m_windowSize.height());
	// changing the projection matrix to match the new size of the window:
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(m_viewport.fovy, m_viewport.aspect, m_viewport.zNear, m_viewport.zFar);
}

void COpenGLView::paintEvent(QPaintEvent* _event)
{
	makeCurrent();

	glClearColor(1.0f, 1.0f, 1.0f, 0.0f); // background color
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_LINE_SMOOTH);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);

	glLoadIdentity();

	GLfloat Alpha = 1.0f;
	GLfloat Diffuse[4] = { 1.0f, 1.0f, 1.0f, Alpha };
	GLfloat Ambient[4] = { 1.0f, 1.0f, 1.0f, Alpha };
	GLfloat Specular[4] = { 0.4f, 0.4f, 0.4f, Alpha };
	GLfloat Phongsize = 128.0f;
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, Diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, Ambient);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, Specular);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, Phongsize);

	//// Lighting:
	GLfloat pLightPosition[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glLightfv(GL_LIGHT0, GL_POSITION, pLightPosition);

	// camera transformation:
	glTranslatef(m_cameraTranslation[0], m_cameraTranslation[1], m_cameraTranslation[2]);
	glRotatef(m_cameraRotation[0], 1.0, 0.0, 0.0);
	glRotatef(m_cameraRotation[1], 0.0, 1.0, 0.0);
	glRotatef(m_cameraRotation[2], 0.0, 0.0, 1.0);

	DrawScene();
	DrawAxes();
	DrawTime();
	DrawLegend();

	// text rendering must be the last to omit blinking
	glDisable(GL_DEPTH_TEST);
	QPainter painter(this);
	painter.setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);
	DrawAxesText(painter);
	DrawTimeText(painter);
	DrawLegendText(painter);
	painter.end();
	glEnable(GL_DEPTH_TEST);
}

void COpenGLView::GetObjectColor(unsigned int nObjectID, CColor* _ResultColor)
{
	CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(nObjectID);
	double dCurrentValue = 0;// m_pViewOptionsTab->m_dMinColouringValue;
	CVector3 vTempVector;
	*_ResultColor = m_DefaultObjectColor;
	if (object == NULL) return;

	if (m_viewSettings->Coloring().type == EColoring::OVERLAP && !m_bOverlapsReady
	 || m_viewSettings->Coloring().type == EColoring::COORD_NUMBER && !m_bCoordNumberReady
	 || m_viewSettings->Coloring().type == EColoring::AGGL_SIZE && !m_bAgglSizeReady)
		RecalculateColoringProperties();

	switch (m_viewSettings->Coloring().type)
	{
	case EColoring::VELOCITY:
		if ((object->GetObjectType() == SOLID_BOND) || (object->GetObjectType() == LIQUID_BOND))
			vTempVector = m_pSystemStructure->GetBondVelocity(m_dCurrentTime, nObjectID);
		else
			vTempVector = object->GetVelocity(m_dCurrentTime);
		break;
	case EColoring::COORDINATE:
		if ((object->GetObjectType() == SOLID_BOND) || (object->GetObjectType() == LIQUID_BOND))
			vTempVector = m_pSystemStructure->GetBondCoordinate(m_dCurrentTime, nObjectID);
		else
			vTempVector = object->GetCoordinates(m_dCurrentTime);
		break;
	case EColoring::ANGLE_VELOCITY:
		if (object->GetObjectType() == SPHERE)
			vTempVector = object->GetAngleVelocity(m_dCurrentTime);
		else
			return;
		break;
	case EColoring::FORCE: 	vTempVector = object->GetForce(m_dCurrentTime); break;
	case EColoring::BOND_TOTAL_FORCE:
		if (object->GetObjectType() == SOLID_BOND)
			dCurrentValue = object->GetForce(m_dCurrentTime).Length();
		if (m_pSystemStructure->GetBond(m_dCurrentTime, nObjectID).Length() > ((CBond*)object)->GetInitLength()) // pulling state
			dCurrentValue *= -1;
		break;
	case EColoring::BOND_STRAIN:
		if (object->GetObjectType() == SOLID_BOND)
			dCurrentValue = (m_pSystemStructure->GetBond(m_dCurrentTime, nObjectID).Length() - ((CBond*)object)->GetInitLength()) / ((CBond*)object)->GetInitLength();
		break;
	case EColoring::BOND_NORMAL_STRESS:
		if (object->GetObjectType() != SOLID_BOND) return;
		dCurrentValue = -1 * DotProduct(object->GetForce(m_dCurrentTime), m_pSystemStructure->GetBond(m_dCurrentTime, nObjectID).Normalized())
			/ ((CSolidBond*)object)->m_dCrossCutSurface; // projection of total force on bond
		break;
	case EColoring::STRESS:
		if (object->GetObjectType() != SPHERE) return;
		vTempVector = object->GetNormalStress(m_dCurrentTime);
		break;
	case EColoring::DIAMETER:
		if (object->GetObjectType() == SPHERE)
			dCurrentValue = ((CSphere*)object)->GetRadius() * 2;
		else if ((object->GetObjectType() == SOLID_BOND) || (object->GetObjectType() == LIQUID_BOND))
			dCurrentValue = ((CBond*)object)->GetDiameter();
		else
			dCurrentValue = 0;
		break;
	case EColoring::CONTACT_DIAMETER:
		if (object->GetObjectType() == SPHERE)
			dCurrentValue = ((CSphere*)object)->GetContactRadius() * 2;
		else
			dCurrentValue = 0;
		break;
	case EColoring::MATERIAL: *_ResultColor = object->GetColor(); return;
	case EColoring::COORD_NUMBER:
		if (object->GetObjectType() != SPHERE)
			return;
		else if (nObjectID < m_vCoordNumber.size())
			dCurrentValue = m_vCoordNumber[nObjectID];
		break;
	case EColoring::AGGL_SIZE:
		if ((object->GetObjectType() == SPHERE) || (object->GetObjectType() == SOLID_BOND))
			dCurrentValue = m_vAgglSize[nObjectID];
		else
			return;
		break;
	case EColoring::OVERLAP:
		if (object->GetObjectType() != SPHERE)
			return;
		else if (nObjectID < m_vOverlaps.size())
			dCurrentValue = m_vOverlaps[nObjectID];
		break;
	case EColoring::TEMPERATURE:
		if (object->GetObjectType() == SPHERE)
			dCurrentValue = object->GetTemperature(m_dCurrentTime);
		else
			return;
		break;
	case EColoring::DISPLACEMENT:
		if (object->GetObjectType() == SOLID_BOND || object->GetObjectType() == LIQUID_BOND)
			vTempVector = m_pSystemStructure->GetBondCoordinate(m_dCurrentTime, nObjectID) - m_pSystemStructure->GetBondCoordinate(0.0, nObjectID);
		else
			vTempVector = object->GetCoordinates(m_dCurrentTime) - object->GetCoordinates(0.0);
		break;
	case EColoring::NONE: return;
	}
	if (m_viewSettings->Coloring().type == EColoring::FORCE || m_viewSettings->Coloring().type == EColoring::VELOCITY
	 || m_viewSettings->Coloring().type == EColoring::COORDINATE || m_viewSettings->Coloring().type == EColoring::ANGLE_VELOCITY
	 || m_viewSettings->Coloring().type == EColoring::STRESS || m_viewSettings->Coloring().type == EColoring::DISPLACEMENT)
	{
		switch (m_viewSettings->Coloring().component)
		{
		case EColorComponent::TOTAL: dCurrentValue = vTempVector.Length(); break;
		case EColorComponent::X:     dCurrentValue = vTempVector.x;        break;
		case EColorComponent::Y:     dCurrentValue = vTempVector.y;        break;
		case EColorComponent::Z:     dCurrentValue = vTempVector.z;        break;
		}
	}

	double dValue = 0;
	if (m_viewSettings->Coloring().maxValue - m_viewSettings->Coloring().minValue != 0)
		dValue = (dCurrentValue - m_viewSettings->Coloring().minValue) / (m_viewSettings->Coloring().maxValue - m_viewSettings->Coloring().minValue);

	double eps = 0.00001;
	if (dValue < eps)
	{
		*_ResultColor = CColor{ (float)m_viewSettings->Coloring().minColor.redF(), (float)m_viewSettings->Coloring().minColor.greenF(), (float)m_viewSettings->Coloring().minColor.blueF(), (float)m_viewSettings->Coloring().minColor.alphaF() };
		return;
	}
	else if ((dValue > 0.5 - eps) && (dValue < 0.5 + eps))
	{
		*_ResultColor = CColor{ (float)m_viewSettings->Coloring().midColor.redF(), (float)m_viewSettings->Coloring().midColor.greenF(), (float)m_viewSettings->Coloring().midColor.blueF(), (float)m_viewSettings->Coloring().midColor.alphaF() };
		return;
	}
	else if (dValue > 1.0 - eps)
	{
		*_ResultColor = CColor{ (float)m_viewSettings->Coloring().maxColor.redF(), (float)m_viewSettings->Coloring().maxColor.greenF(), (float)m_viewSettings->Coloring().maxColor.blueF(), (float)m_viewSettings->Coloring().maxColor.alphaF() };
		return;
	}
	double x1, x2, x3, f1, f2, f3, sumX, sumF, k1, k2, k3;
	x1 = dValue;
	x2 = fabs(0.5 - dValue);
	x3 = 1 - dValue;
	sumX = x1 + x2 + x3;
	f1 = sumX / x1;
	f2 = sumX / x2;
	f3 = sumX / x3;
	sumF = f1 + f2 + f3;
	k1 = f1 / sumF;
	k2 = f2 / sumF;
	k3 = f3 / sumF;
	_ResultColor->r = (float)(k1 * m_viewSettings->Coloring().minColor.redF()   + k2 * m_viewSettings->Coloring().midColor.redF()   + k3 * m_viewSettings->Coloring().maxColor.redF());
	_ResultColor->g = (float)(k1 * m_viewSettings->Coloring().minColor.greenF() + k2 * m_viewSettings->Coloring().midColor.greenF() + k3 * m_viewSettings->Coloring().maxColor.greenF());
	_ResultColor->b = (float)(k1 * m_viewSettings->Coloring().minColor.blueF()  + k2 * m_viewSettings->Coloring().midColor.blueF()  + k3 * m_viewSettings->Coloring().maxColor.blueF());
}

CVector3 COpenGLView::WinCoord2GL(double _x, double _y, double _z) const
{
	CVector3 vRes;
	GLint viewport[4];
	GLdouble modelview[16], projection[16];
	GLdouble objX, objY, objZ;
	glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
	glGetDoublev(GL_PROJECTION_MATRIX, projection);
	glGetIntegerv(GL_VIEWPORT, viewport);
	if (gluUnProject(GLdouble(_x), GLdouble(_y), GLdouble(_z), modelview, projection, viewport, &objX, &objY, &objZ) == GLU_TRUE)
		vRes.Init(objX, objY, objZ);
	else
		vRes.Init(0);
	return vRes;
}

bool COpenGLView::IsPointCuttedByPlanes(const CVector3& _vCoord) const
{
	if (!m_viewSettings->Cutting().CutByPlanes()) return false;
	const auto& cuttingSettings = m_viewSettings->Cutting();
	if (cuttingSettings.cutByX && (_vCoord.x < cuttingSettings.minX || _vCoord.x > cuttingSettings.maxX)) return true;
	if (cuttingSettings.cutByY && (_vCoord.y < cuttingSettings.minY || _vCoord.y > cuttingSettings.maxY)) return true;
	if (cuttingSettings.cutByZ && (_vCoord.z < cuttingSettings.minZ || _vCoord.z > cuttingSettings.maxZ)) return true;
	return false;
}

void COpenGLView::DrawBox(SVolumeType& _Volume, CColor& _Color)
{
	glDisable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	SetColorGL(_Color);

	glPushMatrix();
	glTranslatef((GLfloat)(_Volume.coordBeg.x + _Volume.coordEnd.x) / 2, (GLfloat)(_Volume.coordEnd.y + _Volume.coordBeg.y) / 2,
		(GLfloat)(_Volume.coordBeg.z));

	double dRadius = (_Volume.coordEnd.x - _Volume.coordBeg.x) / M_SQRT2; // -- sqrt(2)
	//glTranslatef( 0, 0, -vVolumes[ i ].dProperty3/2 );
	glScalef(1.0f, (GLfloat)((_Volume.coordEnd.y - _Volume.coordBeg.y) / (_Volume.coordEnd.x - _Volume.coordBeg.x)), 1.0f);
	glRotatef(45.0f, 0.0f, 0.0f, 1.0f);
	gluCylinder(m_pQuadObj, dRadius, dRadius, _Volume.coordEnd.z - _Volume.coordBeg.z, 4, 1);
	glPopMatrix();

	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void COpenGLView::DrawSimulationDomain()
{
	if (!m_viewSettings->Visibility().domain) return;

	SVolumeType simDomain = m_pSystemStructure->GetSimulationDomain();
	if (simDomain.coordEnd.y == simDomain.coordBeg.y) return;
	CColor tempColor = CColor(0.2f, 0.2f, 1.0f, 0.2f);

	DrawBox(simDomain, tempColor);
}

void COpenGLView::DrawSampleAnalyzerVolume()
{
	if (m_pSampleAnalyzerTab == NULL) return;
	if (m_pSampleAnalyzerTab->isVisible() == false) return;

	glDisable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	SetColorGL(CColor(0.4f, 0.4f, 0.4f, 0.5f));

	int nSlices = 20;
	int nStacks = 15;
	glPushMatrix();
	glTranslatef((GLfloat)m_pSampleAnalyzerTab->m_vCenter.x, (GLfloat)m_pSampleAnalyzerTab->m_vCenter.y, (GLfloat)m_pSampleAnalyzerTab->m_vCenter.z);
	gluSphere(m_pQuadObj, m_pSampleAnalyzerTab->m_dRadius, nSlices, nStacks);
	glPopMatrix();
	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void COpenGLView::DrawPBC()
{
	if (!m_viewSettings->Visibility().pbc) return;
	SPBC pbc = m_pSystemStructure->GetPBC();
	pbc.UpdatePBC(m_dCurrentTime);
	if (!pbc.bEnabled) return;

	SVolumeType vol = m_pSystemStructure->GetSimulationDomain();

	if (pbc.bX)
	{
		vol.coordBeg.x = pbc.currentDomain.coordBeg.x;
		vol.coordEnd.x = pbc.currentDomain.coordEnd.x;
	}
	if (pbc.bY)
	{
		vol.coordBeg.y = pbc.currentDomain.coordBeg.y;
		vol.coordEnd.y = pbc.currentDomain.coordEnd.y;
	}
	if (pbc.bZ)
	{
		vol.coordBeg.z = pbc.currentDomain.coordBeg.z;
		vol.coordEnd.z = pbc.currentDomain.coordEnd.z;
	}

	/*
	axis: →x ↗y ↑z
	cube:
	front face:	1-5		back face:	3-7
				| |					| |
				0-4					2-6
	vol.coordBegin = 0, vol.coordEnd = 7
	*/

	const CVector3 v0(vol.coordBeg.x, vol.coordBeg.y, vol.coordBeg.z);
	const CVector3 v1(vol.coordBeg.x, vol.coordBeg.y, vol.coordEnd.z);
	const CVector3 v2(vol.coordBeg.x, vol.coordEnd.y, vol.coordBeg.z);
	const CVector3 v3(vol.coordBeg.x, vol.coordEnd.y, vol.coordEnd.z);
	const CVector3 v4(vol.coordEnd.x, vol.coordBeg.y, vol.coordBeg.z);
	const CVector3 v5(vol.coordEnd.x, vol.coordBeg.y, vol.coordEnd.z);
	const CVector3 v6(vol.coordEnd.x, vol.coordEnd.y, vol.coordBeg.z);
	const CVector3 v7(vol.coordEnd.x, vol.coordEnd.y, vol.coordEnd.z);

	glDisable(GL_LIGHTING);

	if (pbc.bX)
	{
		DrawQuad(v2, v3, v1, v0, CColor(1.0f, 0.0f, 0.0f, 0.2f));
		DrawQuad(v4, v5, v7, v6, CColor(1.0f, 0.0f, 0.0f, 0.2f));
	}
	if (pbc.bY)
	{
		DrawQuad(v0, v1, v5, v4, CColor(0.0f, 1.0f, 0.0f, 0.2f));
		DrawQuad(v6, v7, v3, v2, CColor(0.0f, 1.0f, 0.0f, 0.2f));
	}

	if (pbc.bZ)
	{
		DrawQuad(v2, v0, v4, v6, CColor(0.0f, 0.0f, 1.0f, 0.2f));
		DrawQuad(v1, v3, v7, v5, CColor(0.0f, 0.0f, 1.0f, 0.2f));
	}

	glEnable(GL_LIGHTING);
}

void COpenGLView::DrawGeometricalObjects() const
{
	if (!m_viewSettings->Visibility().geometries) return;
	for (const auto& geometryKey : m_viewSettings->VisibleGeometries())
	{
		const auto geometry = m_pSystemStructure->GetGeometry(geometryKey);
		const CColor color{ geometry->color, 1.f - m_viewSettings->GeometriesTransparency() };
		for (const auto& w : m_pSystemStructure->GetAllWallsForGeometry(m_dCurrentTime, geometryKey))
			DrawTriangularPlane(w->GetCoords(m_dCurrentTime), w->GetNormalVector(m_dCurrentTime), color);
	}
}

void COpenGLView::DrawAnalysisVolumes() const
{
	if (!m_viewSettings->Visibility().volumes) return;
	glDisable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	for (const auto& volumeKey : m_viewSettings->VisibleVolumes())
	{
		const auto volume = m_pSystemStructure->GetAnalysisVolume(volumeKey);
		const CVector3 shift = volume->GetShift(m_dCurrentTime);
		for (const auto& triangle : volume->vTriangles)
		{
			const STriangleType t = triangle.Shifted(shift);
			DrawTriangularPlane(t, t.Normal(), volume->color);
		}
	}
	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void COpenGLView::DrawScene()
{
	if (m_bUpdating == false) return;

	// if the package generator is active than draw volume where particles should be generated
	DrawSampleAnalyzerVolume();
	DrawSimulationDomain();

	DrawParticles();
	DrawOrientations();
	DrawBonds();

	DrawGeometricalObjects();
	DrawAnalysisVolumes();

	DrawPBC();
}

void COpenGLView::DrawOrientations()
{
	if (!m_bShowOrientation) return;
	CColor tempColor;
	for (unsigned long i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); i++)
	{
		CPhysicalObject* pObject = m_pSystemStructure->GetObjectByIndex(i);
		if ((pObject == NULL) || (!pObject->IsActive(m_dCurrentTime)) || (pObject->GetObjectType() != SPHERE)) continue;

		CVector3 vecCenter = pObject->GetCoordinates(m_dCurrentTime);
		if (IsPointCuttedByPlanes(vecCenter)) continue;

		CVector3 vXAxis(2 * ((CSphere*)pObject)->GetRadius(), 0, 0);
		vXAxis = QuatRotateVector(pObject->GetOrientation(m_dCurrentTime), vXAxis) + vecCenter;
		glPushMatrix();
		glLineWidth(2.5);
		glBegin(GL_LINES);
		glVertex3f(vecCenter.x, vecCenter.y, vecCenter.z);
		glVertex3f(vXAxis.x, vXAxis.y, vXAxis.z);
		glEnd();

		glPopMatrix();
	}
}

void COpenGLView::RecalculateBrokenBonds()
{
	if (!m_viewSettings->BrokenBonds().show) return;
	m_vBrokenBonds.resize(m_pSystemStructure->GetTotalObjectsCount());
	for (size_t i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); i++)
	{
		m_vBrokenBonds[i] = false;
		CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(i);
		if (object == NULL) continue;
		if ((object->GetObjectType() == SOLID_BOND) && (object->GetActivityEnd() >= m_viewSettings->BrokenBonds().startTime) &&
			object->GetActivityEnd() <= m_viewSettings->BrokenBonds().endTime)
			m_vBrokenBonds[i] = true;
	}
}

void COpenGLView::DrawBonds()
{
	if (!m_viewSettings->Visibility().bonds) return;

	// cutting by materials
	const auto visibleBondMaterials = m_viewSettings->VisibleBondMaterials();

	const SPBC pbc = m_pSystemStructure->GetPBC();
	CColor tempColor;
	for (size_t i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); i++)
	{
		CPhysicalObject* object = m_pSystemStructure->GetObjectByIndex(i);
		if (object == NULL) continue;
		bool bShowRealBond = !m_viewSettings->BrokenBonds().show && object->IsActive(m_dCurrentTime);
		bool bShowBrokenBond = m_viewSettings->BrokenBonds().show && m_vBrokenBonds[i];
		if (!bShowRealBond && !bShowBrokenBond) continue;
		if ((object->GetObjectType() != SOLID_BOND) && (object->GetObjectType() != LIQUID_BOND)) continue;

		CBond* pTempBond = (CBond*)m_pSystemStructure->GetObjectByIndex(i);
		CPhysicalObject* objectLeftBonded = m_pSystemStructure->GetObjectByIndex(pTempBond->m_nLeftObjectID);
		CPhysicalObject* objectRightBonded = m_pSystemStructure->GetObjectByIndex(pTempBond->m_nRightObjectID);
		if ((objectLeftBonded == NULL) || (objectRightBonded == NULL)) continue;

		if (!bShowBrokenBond && !SetContains(visibleBondMaterials, object->GetCompoundKey())) continue;

		double angle, dLengthVec;
		CVector3 vecLeft, vecRight, p, t, z;

		vecLeft = objectLeftBonded->GetCoordinates(m_dCurrentTime);
		vecRight = objectRightBonded->GetCoordinates(m_dCurrentTime);

		if (IsPointCuttedByPlanes(vecLeft)) continue;
		if (IsPointCuttedByPlanes(vecRight)) continue;
		z.Init(0, 0, -1);
		p = -1 * GetSolidBond(vecRight, vecLeft, pbc); // Get solid bond
		t = z * p; // Get CROSS product (the axis of rotation)
		dLengthVec = p.Length();
		// check correctness
		if (dLengthVec <= 0) continue;

		// Get angle. LENGTH is magnitude of the vector
		angle = _180_PI * acos(DotProduct(z, p) / dLengthVec);


		// show real bond
		if (bShowRealBond)
		{
			glPushMatrix();
			GetObjectColor(i, &tempColor);
			SetColorGL(tempColor);

			if (object->GetObjectType() == LIQUID_BOND)
			{
				tempColor.a = 0.5; tempColor.r = 0; tempColor.g = 0; tempColor.b = 1;
				SetColorGL(tempColor);
			}
			glTranslatef((GLfloat)vecRight.x, (GLfloat)vecRight.y, (GLfloat)vecRight.z);
			glRotatef((GLfloat)angle, (GLfloat)t.x, (GLfloat)t.y, (GLfloat)t.z);
			gluQuadricOrientation(m_pQuadObj, GLU_OUTSIDE);
			gluCylinder(m_pQuadObj, pTempBond->GetDiameter() / 2, pTempBond->GetDiameter() / 2, dLengthVec, m_cylinderSlices, m_cylinderStacks);

			glPopMatrix();
		}
		if (bShowBrokenBond)
		{
			glPushMatrix();
			qglColor(m_viewSettings->BrokenBonds().color);
			glTranslatef((GLfloat)vecRight.x, (GLfloat)vecRight.y, (GLfloat)vecRight.z);
			glRotatef((GLfloat)angle, (GLfloat)t.x, (GLfloat)t.y, (GLfloat)t.z);
			gluQuadricOrientation(m_pQuadObj, GLU_OUTSIDE);
			gluCylinder(m_pQuadObj, pTempBond->GetDiameter() / 2, pTempBond->GetDiameter() / 2, dLengthVec, m_cylinderSlices, m_cylinderStacks);
			glPopMatrix();
		}
	}
}
void COpenGLView::ZoomView(int _nZoomIn)
{
	//m_pCameraTranslation[2] += (float)_nZoomIn / WHEEL_DELTA * 0.05f * fabs(m_pCameraTranslation[2]);
	m_cameraTranslation[2] += static_cast<float>(_nZoomIn) / 120 * 0.05f * std::fabs(m_cameraTranslation[2]);
	update();
}

void COpenGLView::wheelEvent(QWheelEvent * event)
{
	ZoomView(event->delta());
}

void COpenGLView::SetColorGL(const CColor& _color)
{
	glColor4f(_color.r, _color.g, _color.b, _color.a);
}

void COpenGLView::mouseMoveEvent(QMouseEvent *event)
{
	const float dx = (event->x() - m_lastMousePos.x()) / static_cast<float>(m_windowSize.height());
	const float dy = (event->y() - m_lastMousePos.y()) / static_cast<float>(m_windowSize.width());

	if ((event->buttons() & Qt::LeftButton) && (event->modifiers() & Qt::ShiftModifier))
	{
		m_cameraRotation[2] += dx * 100;
		//	m_Axes.AnglesWereChanged();
		update();
	}
	else if (event->buttons() & Qt::LeftButton)
	{
		m_cameraRotation[0] += dy * 100;
		m_cameraRotation[1] += dx * 100;
		//	m_Axes.AnglesWereChanged();
		update();
	}
	else if (event->buttons() & Qt::RightButton)
	{
		m_cameraTranslation[0] += dx / 5.0f * std::fabs(m_cameraTranslation[2]);
		m_cameraTranslation[1] -= dy / 5.0f * std::fabs(m_cameraTranslation[2]);
		update();
	}
	m_lastMousePos = event->pos();
}

void COpenGLView::mousePressEvent(QMouseEvent *event)
{
	m_lastMousePos = event->pos();
}

void COpenGLView::UpdateView()
{
	RecalculateBrokenBonds();
	update();
}

QSize COpenGLView::minimumSizeHint() const
{
	return QSize(50, 50);
}

QSize COpenGLView::sizeHint() const
{
	return QSize(400, 400);
}

void COpenGLView::Redraw()
{
	update();
}

void COpenGLView::SetCurrentTime(double _dTime)
{
	m_dCurrentTime = _dTime;
	m_bCoordNumberReady = false;
	m_bOverlapsReady = false;
	m_bAgglSizeReady = false;
	RecalculateColoringProperties();
	UpdateView();
}

void COpenGLView::RedrawScene()
{
	//repaint();
	updateGL();
}

SBox COpenGLView::WinCoord2LineOfSight(const QPoint& _pos) const
{
	GLdouble modelViewMatrix[16];
	GLdouble projectionMatrix[16];
	GLint viewPort[4];
	glGetDoublev(GL_MODELVIEW_MATRIX, modelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, projectionMatrix);
	glGetIntegerv(GL_VIEWPORT, viewPort);
	const GLdouble winX = _pos.x();
	const GLdouble winY = static_cast<GLdouble>(m_windowSize.height() - _pos.y() - 1);
	const GLdouble winZMin = -1.0;	// zNear
	const GLdouble winZMax = 1.0;	// zFar
	GLdouble minX, minY, minZ, maxX, maxY, maxZ;
	gluUnProject(winX, winY, winZMin, modelViewMatrix, projectionMatrix, viewPort, &minX, &minY, &minZ);
	gluUnProject(winX, winY, winZMax, modelViewMatrix, projectionMatrix, viewPort, &maxX, &maxY, &maxZ);
	return SBox{ QVector3D{ float(minX), float(minY), float(minZ) }, QVector3D{ float(maxX), float(maxY), float(maxZ) } };
}

// TODO: remove after unification of axes.
void COpenGLView::SetCameraStandardView(const SBox& _box, const QVector3D& _cameraDirection)
{
	CBaseGLView::SetCameraStandardView(_box, _cameraDirection);
	resizeGL(m_windowSize.width(), m_windowSize.height());
}

QImage COpenGLView::Snapshot(uint8_t _scaling)
{
	// adjust scaling factor for rendering
	m_scaling = _scaling;
	// save current sizes
	const QSize oldSize = m_windowSize;
	// resize with scaling and update
	resize(oldSize * m_scaling);
	repaint();
	// get current image
	QImage image = grabFrameBuffer();
	// draw text
	QPainter painter(&image);
	painter.setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);
	DrawAxesText(painter);
	DrawTimeText(painter);
	DrawLegendText(painter);
	painter.end();
	// restore sizes
	m_scaling = 1;
	resize(oldSize);
	update();

	return image;
}

void COpenGLView::RecalculateColoringProperties()
{
	if (m_viewSettings->Coloring().type == EColoring::COORD_NUMBER)
	{
		m_vCoordNumber = m_pSystemStructure->GetCoordinationNumbers(m_dCurrentTime);
		m_bCoordNumberReady = true;
	}
	else if (m_viewSettings->Coloring().type == EColoring::OVERLAP)
	{
		m_vOverlaps = m_pSystemStructure->GetMaxOverlaps(m_dCurrentTime);
		m_bOverlapsReady = true;
	}
	else if (m_viewSettings->Coloring().type == EColoring::AGGL_SIZE)
	{
		CAgglomeratesAnalyzer agglomAnalyzer;
		agglomAnalyzer.SetSystemStructure(m_pSystemStructure);
		agglomAnalyzer.FindAgglomerates(m_dCurrentTime);
		const std::vector<std::vector<size_t>>& vAgglomParts = agglomAnalyzer.GetAgglomeratesParticles();
		const std::vector<std::vector<size_t>>& vAgglomBonds = agglomAnalyzer.GetAgglomeratesBonds();
		m_vAgglSize.resize(m_pSystemStructure->GetTotalObjectsCount());
		std::fill(m_vAgglSize.begin(), m_vAgglSize.end(), 1);
		for (size_t i = 0; i < vAgglomParts.size(); ++i)
			for (size_t j = 0; j < vAgglomParts[i].size(); ++j)
				m_vAgglSize[vAgglomParts[i][j]] = vAgglomParts[i].size();
		for (size_t i = 0; i < vAgglomBonds.size(); ++i)
			for (size_t j = 0; j < vAgglomBonds[i].size(); j++)
				m_vAgglSize[vAgglomBonds[i][j]] = vAgglomParts[i].size();

		m_bAgglSizeReady = true;
	}
}

void COpenGLView::SetAxesVisible(bool _bVisible)
{
	m_bShowAxes = _bVisible;
}

void COpenGLView::SetTimeVisible(bool _bVisible)
{
	m_bShowTime = _bVisible;
}

void COpenGLView::SetLegendVisible(bool _bVisible)
{
	m_bShowLegend = _bVisible;
}

void COpenGLView::SetRenderQuality(uint8_t _quality)
{
	m_sphereSlices = 3 + static_cast<GLint>(_quality / 100.0 * 12);
	m_sphereStacks = 2 + static_cast<GLint>(_quality / 100.0 * 12);
	m_cylinderSlices = 3 + static_cast<GLint>(_quality / 100.0 * 17);
	m_cylinderStacks = 1;
}

void COpenGLView::SetOrientationVisible(bool _bVisible)
{
	m_bShowOrientation = _bVisible;
}

void COpenGLView::DisableOpenGLView()
{
	m_bUpdating = false;
}

void COpenGLView::EnableOpenGLView()
{
	m_bUpdating = true;
}

void COpenGLView::DrawAxes() const
{
	if (!m_bShowAxes) return;

	const double kSize = 20;
	const double dAxisLength = m_dWinSize / kSize;			// GL
	const double dAxisRadius = m_dWinSize / kSize / 20;		// GL
	const double dArrowLength = dAxisLength / 5;			// GL
	const double dArrowRadius = dAxisRadius * 1.5;			// GL
	const double dOriginPos = m_dWinSizePx / kSize * 1.4;	// px
	const CVector3 vPos = WinCoord2GL(dOriginPos, dOriginPos, 0.37 / kSize);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	glTranslated(vPos.x, vPos.y, vPos.z);

	// X
	qglColor(QColor(Qt::red));
	glRotatef(90, 0, 1, 0);
	gluCylinder(m_pQuadObj, dAxisRadius, dAxisRadius, dAxisLength, 20, 1);
	glTranslated(0, 0, 0 + dAxisLength);
	gluCylinder(m_pQuadObj, dArrowRadius, 0, dArrowLength, 20, 1);
	glTranslated(0, 0, -dAxisLength);

	// Y
	qglColor(QColor(Qt::green));
	glRotatef(-90, 1, 0, 0);
	gluCylinder(m_pQuadObj, dAxisRadius, dAxisRadius, dAxisLength, 20, 1);
	glTranslated(0, 0, 0 + dAxisLength);
	gluCylinder(m_pQuadObj, dArrowRadius, 0, dArrowLength, 20, 1);
	glTranslated(0, 0, -dAxisLength);

	// Z
	qglColor(QColor(Qt::blue));
	glRotatef(-90, 0, 1, 0);
	gluCylinder(m_pQuadObj, dAxisRadius, dAxisRadius, dAxisLength, 20, 1);
	glTranslated(0, 0, 0 + dAxisLength);
	gluCylinder(m_pQuadObj, dArrowRadius, 0, dArrowLength, 20, 1);
	glTranslated(0, 0, -dAxisLength);

	glPopMatrix();
}

void COpenGLView::DrawAxesText(QPainter& _painter) const
{
	if (!m_bShowAxes) return;

	const double kSize = 20;
	const double dAxisLength = m_dWinSize / kSize ;				// GL
	const double dArrowLength = dAxisLength / 5;				// GL
	const double dOriginPos = m_dWinSizePx / kSize * 1.4;		// px
	const CVector3 vPos = WinCoord2GL(dOriginPos, dOriginPos, 0.37 / kSize);

	double pModelViewMatrix[16];
	double pProjectionMatrix[16];
	GLint pViewPortParameters[4];
	GLdouble x, y, z;

	// prepare view
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glTranslated(vPos.x, vPos.y, vPos.z);
	glGetDoublev(GL_PROJECTION_MATRIX, pProjectionMatrix);
	glGetIntegerv(GL_VIEWPORT, pViewPortParameters);

	// setup painter
	SetupPainter(&_painter, m_fontAxes);

	// X
	glRotatef(90, 0, 1, 0);
	glGetDoublev(GL_MODELVIEW_MATRIX, pModelViewMatrix);
	gluProject(0, 0, dAxisLength + dArrowLength * 2, pModelViewMatrix, pProjectionMatrix, pViewPortParameters, &x, &y, &z);
	_painter.drawText(x, m_windowSize.height() - static_cast<int>(y), "X");

	// Y
	glRotatef(-90, 1, 0, 0);
	glGetDoublev(GL_MODELVIEW_MATRIX, pModelViewMatrix);
	gluProject(0, 0, dAxisLength + dArrowLength * 2, pModelViewMatrix, pProjectionMatrix, pViewPortParameters, &x, &y, &z);
	_painter.drawText(x, m_windowSize.height() - static_cast<int>(y), "Y");

	// Z
	glRotatef(-90, 0, 1, 0);
	glGetDoublev(GL_MODELVIEW_MATRIX, pModelViewMatrix);
	gluProject(0, 0, dAxisLength + dArrowLength * 2, pModelViewMatrix, pProjectionMatrix, pViewPortParameters, &x, &y, &z);
	_painter.drawText(x, m_windowSize.height() - static_cast<int>(y), "Z");

	glPopMatrix();
}

void COpenGLView::DrawTime() const
{
	if (!m_bShowTime) return;

	//qglColor(QColor(0, 130, 200));
	//glDisable(GL_DEPTH_TEST);
	//renderText(5, 15, "Time: " + QString::number(m_dCurrentTime), m_fontTime);
	//glEnable(GL_DEPTH_TEST);
}

void COpenGLView::DrawTimeText(QPainter& _painter) const
{
	if (!m_bShowTime) return;

	const int nMargin = 5;

	// setup painter
	SetupPainter(&_painter, m_fontTime);

	// draw text
	_painter.drawText(QRect{ {nMargin * m_scaling, nMargin * m_scaling}, m_windowSize }, Qt::AlignLeft | Qt::AlignTop, "Time [s]: " + QString::number(m_dCurrentTime));
}

void COpenGLView::DrawLegend() const
{
	if (!m_bShowLegend)	return;
	if (m_viewSettings->Coloring().type == EColoring::NONE)	return;

	const GLdouble leftMargin = 10 * m_scaling;
	const GLfloat width = 15 * m_scaling;
	const GLfloat height = static_cast<GLfloat>(m_windowSize.height()) / 2;
	const GLfloat middle = static_cast<GLfloat>(m_windowSize.height()) / 2.5;
	const GLfloat top = middle - height / 2;
	const GLfloat bottom = middle + height / 2;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, m_windowSize.width(), m_windowSize.height(), 0, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	glTranslated(0, 0, 1);

	glBegin(GL_QUADS);

	qglColor(m_viewSettings->Coloring().maxColor);
	glVertex2f(leftMargin, top);
	qglColor(m_viewSettings->Coloring().midColor);
	glVertex2f(leftMargin, middle);
	glVertex2f(leftMargin + width, middle);
	qglColor(m_viewSettings->Coloring().maxColor);
	glVertex2f(leftMargin + width, top);

	qglColor(m_viewSettings->Coloring().midColor);
	glVertex2f(leftMargin, middle);
	qglColor(m_viewSettings->Coloring().minColor);
	glVertex2f(leftMargin, bottom);
	glVertex2f(leftMargin + width, bottom);
	qglColor(m_viewSettings->Coloring().midColor);
	glVertex2f(leftMargin + width, middle);

	glEnd();

	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
}

void COpenGLView::DrawLegendText(QPainter& _painter) const
{
	if (!m_bShowLegend) return;
	if (m_viewSettings->Coloring().type == EColoring::NONE)	return;

	const int leftMargin = 10 * m_scaling;
	const int width = 15 * m_scaling;
	const int height = m_windowSize.height() / 2;
	const int middle = static_cast<int>(m_windowSize.height() / 2.5);
	const int top = middle - height / 2;
	const int bottom = middle + height / 2;

	const double topText = m_viewSettings->Coloring().maxDisplayValue;
	const double middleText = (m_viewSettings->Coloring().maxDisplayValue + m_viewSettings->Coloring().minDisplayValue) / 2;
	const double bottomText = m_viewSettings->Coloring().minDisplayValue;
	const double semiTopText = m_viewSettings->Coloring().maxDisplayValue == m_viewSettings->Coloring().minDisplayValue ? m_viewSettings->Coloring().maxDisplayValue
		: (m_viewSettings->Coloring().maxDisplayValue + middleText) / 2;
	const double semiBottomText = m_viewSettings->Coloring().maxDisplayValue == m_viewSettings->Coloring().minDisplayValue ? m_viewSettings->Coloring().maxDisplayValue
		: (middleText + m_viewSettings->Coloring().minDisplayValue) / 2;

	// setup painter
	SetupPainter(&_painter, m_fontLegend);

	// draw text
	_painter.drawText(leftMargin + width + 5, top, QString::number(topText));
	_painter.drawText(leftMargin + width + 5, middle - height / 4, QString::number(semiTopText));
	_painter.drawText(leftMargin + width + 5, middle, QString::number(middleText));
	_painter.drawText(leftMargin + width + 5, middle + height / 4, QString::number(semiBottomText));
	_painter.drawText(leftMargin + width + 5, bottom, QString::number(bottomText));
}

void COpenGLView::DrawQuad(const CVector3& _v1, const CVector3& _v2, const CVector3& _v3, const CVector3& _v4, const CColor& _color)
{
	const CVector3 p1(_v1);
	const CVector3 p2(_v2 - _v1);
	const CVector3 p3(_v4 - _v1);
	const CVector3 p4(_v3 - _v1);

	glColor4f(_color.r, _color.g, _color.b, _color.a);

	glPushMatrix();
	glTranslatef(GLfloat(p1.x), GLfloat(p1.y), GLfloat(p1.z));
	glBegin(GL_TRIANGLE_STRIP);

	glVertex3f(0.0f, -0.0f, -0.0f);
	glVertex3f(GLfloat(p2.x), GLfloat(p2.y), GLfloat(p2.z));
	glVertex3f(GLfloat(p3.x), GLfloat(p3.y), GLfloat(p3.z));
	glVertex3f(GLfloat(p4.x), GLfloat(p4.y), GLfloat(p4.z));

	glEnd();
	glPopMatrix();
}

void COpenGLView::DrawTriangularPlane(const STriangleType& _triangle, const CVector3& _normal, const CColor& _color) const
{
	if (IsPointCuttedByPlanes(_triangle.p1) || IsPointCuttedByPlanes(_triangle.p2) || IsPointCuttedByPlanes(_triangle.p3)) return;

	glColor4f(_color.r, _color.g, _color.b, _color.a);
	glBegin(GL_TRIANGLES);
	glNormal3f(GLfloat(_normal.x), GLfloat(_normal.y), GLfloat(_normal.z));
	glVertex3f(GLfloat(_triangle.p1.x), GLfloat(_triangle.p1.y), GLfloat(_triangle.p1.z));
	glVertex3f(GLfloat(_triangle.p2.x), GLfloat(_triangle.p2.y), GLfloat(_triangle.p2.z));
	glVertex3f(GLfloat(_triangle.p3.x), GLfloat(_triangle.p3.y), GLfloat(_triangle.p3.z));
	glEnd();
}

