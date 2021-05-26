/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ViewManager.h"
#include "AgglomeratesAnalyzer.h"
#include "MeshGenerator.h"
#include "OpenGLViewGlu.h"
#include "OpenGLViewMixed.h"
#include "OpenGLViewShader.h"

CViewManager::CViewManager(QWidget* _showWidget, QLayout* _layout, CViewSettings* _viewSettings, QObject* _parent) :
	QObject(_parent),
	m_widget(_showWidget),
	m_layout(_layout),
	m_eventMonitor(this),
	m_viewSettings{_viewSettings},
	m_time(0)
{
	//Initialize(); // TODO: should be called here

	connect(&m_eventMonitor, &CEventMonitor::ParticleSelected,	this, &CViewManager::SelectObject);
	connect(&m_eventMonitor, &CEventMonitor::GroupSelected,		this, &CViewManager::SelectGroup);
	connect(&m_eventMonitor, &CEventMonitor::CameraChanged,		this, &CViewManager::CameraChanged);
}

void CViewManager::SetPointers(CSystemStructure* _systemStructure, const CSampleAnalyzerTab* _sampleAnalyzer)
{
	m_systemStructure = _systemStructure;
	m_sampleAnalyzer = _sampleAnalyzer;
	Initialize(); // TODO: should not be called here
}

void CViewManager::Initialize()
{
	UpdateRenderType();
}

void CViewManager::UpdateRenderType()
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:		SetRenderGlu();		break;
	case ERenderType::MIXED:	SetRenderMixed();	break;
	case ERenderType::SHADER:	SetRenderShader();	break;
	case ERenderType::NONE:		ClearWidget();		break;
	}
	m_widget->installEventFilter(&m_eventMonitor);
}

void CViewManager::UpdateViewQuality() const
{
	if (m_viewSettings->RenderType() == ERenderType::NONE) return;
	auto w = dynamic_cast<CBaseGLView*>(m_widget);
	w->SetRenderQuality(m_viewSettings->RenderQuality());
	if (m_viewSettings->RenderType() == ERenderType::SHADER)
		DoUpdateBonds();
	w->Redraw();
}

void CViewManager::UpdateParticleTexture() const
{
	if (m_viewSettings->RenderType() == ERenderType::NONE) return;
	auto w = dynamic_cast<CBaseGLView*>(m_widget);
	w->SetParticleTexture(m_viewSettings->ParticleTexture());
	if (m_viewSettings->RenderType() == ERenderType::SHADER)
		DoUpdateParticles();
	w->Redraw();
}

void CViewManager::UpdateParticles() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:
	case ERenderType::MIXED:	dynamic_cast<COpenGLView*>(m_widget)->SetOrientationVisible(m_viewSettings->Visibility().orientations); break;
	case ERenderType::SHADER:	DoUpdateParticles(); break;
	case ERenderType::NONE:		break;
	}
	dynamic_cast<CBaseGLView*>(m_widget)->Redraw();
}

void CViewManager::UpdateBonds() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:
	case ERenderType::MIXED:	dynamic_cast<COpenGLView*>(m_widget)->RecalculateBrokenBonds(); break;
	case ERenderType::SHADER:	DoUpdateBonds(); break;
	case ERenderType::NONE:		break;
	}
	dynamic_cast<CBaseGLView*>(m_widget)->Redraw();
}

void CViewManager::UpdateGeometries() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:
	case ERenderType::MIXED:	break;
	case ERenderType::SHADER:	DoUpdateWalls(); break;
	case ERenderType::NONE:		break;
	}
	dynamic_cast<CBaseGLView*>(m_widget)->Redraw();
}

void CViewManager::UpdateVolumes() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:
	case ERenderType::MIXED:	break;
	case ERenderType::SHADER:	DoUpdateVolumes(); break;
	case ERenderType::NONE:		break;
	}
	dynamic_cast<CBaseGLView*>(m_widget)->Redraw();
}

void CViewManager::UpdateSlices() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:		break;
	case ERenderType::MIXED:	break;
	case ERenderType::SHADER:	DoUpdateSlices(); break;
	case ERenderType::NONE:		break;
	}
	dynamic_cast<CBaseGLView*>(m_widget)->Redraw();
}

void CViewManager::UpdateDomain() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:
	case ERenderType::MIXED:	break;
	case ERenderType::SHADER:	DoUpdateSimulationDomain(); break;
	case ERenderType::NONE:		break;
	}
	dynamic_cast<CBaseGLView*>(m_widget)->Redraw();
}

void CViewManager::UpdatePBC() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:
	case ERenderType::MIXED:	break;
	case ERenderType::SHADER:	DoUpdatePBC(); break;
	case ERenderType::NONE:		break;
	}
	dynamic_cast<CBaseGLView*>(m_widget)->Redraw();
}

void CViewManager::UpdateAxes() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:
	case ERenderType::MIXED:	dynamic_cast<COpenGLView*>(m_widget)->SetAxesVisible(m_viewSettings->Visibility().axes);	break;
	case ERenderType::SHADER:	DoUpdateAxes(); break;
	case ERenderType::NONE:		break;
	}
	dynamic_cast<CBaseGLView*>(m_widget)->Redraw();
}

void CViewManager::UpdateTime() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:
	case ERenderType::MIXED:	dynamic_cast<COpenGLView*>(m_widget)->SetTimeVisible(m_viewSettings->Visibility().time);	break;
	case ERenderType::SHADER:	DoUpdateTime(); break;
	case ERenderType::NONE:		break;
	}
	dynamic_cast<CBaseGLView*>(m_widget)->Redraw();
}

void CViewManager::UpdateLegend() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:
	case ERenderType::MIXED:	dynamic_cast<COpenGLView*>(m_widget)->SetLegendVisible(m_viewSettings->Visibility().legend);	break;
	case ERenderType::SHADER:	DoUpdateLegend(); break;
	case ERenderType::NONE:		break;
	}
	dynamic_cast<CBaseGLView*>(m_widget)->Redraw();
}

void CViewManager::UpdateFontAxes() const
{
	if (m_viewSettings->RenderType() == ERenderType::NONE) return;
	auto w = dynamic_cast<CBaseGLView*>(m_widget);
	w->SetFontAxes(m_viewSettings->FontAxes().font, m_viewSettings->FontAxes().color);
	w->Redraw();
}

void CViewManager::UpdateFontTime() const
{
	if (m_viewSettings->RenderType() == ERenderType::NONE) return;
	auto w = dynamic_cast<CBaseGLView*>(m_widget);
	w->SetFontTime(m_viewSettings->FontTime().font, m_viewSettings->FontTime().color);
	w->Redraw();
}

void CViewManager::UpdateFontLegend() const
{
	if (m_viewSettings->RenderType() == ERenderType::NONE) return;
	auto w = dynamic_cast<CBaseGLView*>(m_widget);
	w->SetFontLegend(m_viewSettings->FontLegend().font, m_viewSettings->FontLegend().color);
	w->Redraw();
}

void CViewManager::UpdateColors() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:		dynamic_cast<COpenGLView*>(m_widget)->UpdateView();  break;
	case ERenderType::MIXED:	dynamic_cast<COpenGLView*>(m_widget)->UpdateView();  break;
	case ERenderType::SHADER:
		DoUpdateParticles();
		DoUpdateBonds();
		DoUpdateSlices();
		DoUpdateLegend();
		dynamic_cast<COpenGLViewShader*>(m_widget)->Redraw();
		break;
	case ERenderType::NONE:     break;
	}
}

void CViewManager::SetTime(double _time)
{
	if (m_time == _time) return;
	m_time = _time;
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:		dynamic_cast<COpenGLView*>(m_widget)->SetCurrentTime(m_time); break;
	case ERenderType::MIXED:	dynamic_cast<COpenGLView*>(m_widget)->SetCurrentTime(m_time); break;
	case ERenderType::SHADER:
		DoUpdateParticles();
		DoUpdateBonds();
		DoUpdateWalls();
		DoUpdateVolumes();
		DoUpdateSlices();
		DoUpdateSimulationDomain();
		DoUpdatePBC();
		DoUpdateTime();
		DoUpdateLegend();
		dynamic_cast<COpenGLViewShader*>(m_widget)->Redraw();
		break;
	case ERenderType::NONE:     break;
	}
}

QImage CViewManager::GetSnapshot(uint8_t _scaling) const
{
	if (m_viewSettings->RenderType() == ERenderType::NONE) return {};
	return dynamic_cast<CBaseGLView*>(m_widget)->Snapshot(_scaling);
}

void CViewManager::SetCameraStandardView(const CVector3& _position) const
{
	CVector3 minCoord{ -0.001 };
	CVector3 maxCoord{ 0.001 };

	if (m_systemStructure->GetTotalObjectsCount() || m_systemStructure->AnalysisVolumesNumber())
	{
		minCoord.Init(std::numeric_limits<double>::max());
		maxCoord.Init(std::numeric_limits<double>::lowest());

		// for all particles
		for (const auto* part : m_systemStructure->GetAllSpheres(m_time, true))
		{
			const double length = part->GetRadius();
			const CVector3 coord = part->GetCoordinates(m_time);
			minCoord = Min(minCoord, coord - length);
			maxCoord = Max(maxCoord, coord + length);
		}
		// for all walls
		for (const auto* wall : m_systemStructure->GetAllWalls(m_time, true))
		{
			const CTriangle t = wall->GetCoords(m_time);
			minCoord = Min(minCoord, Min(t.p1, t.p2, t.p3));
			maxCoord = Max(maxCoord, Max(t.p1, t.p2, t.p3));
		}
		// for all analysis volumes
		for (const auto* volume : m_systemStructure->AllAnalysisVolumes())
			for (const auto& t : volume->Mesh(m_time).Triangles())
			{
				minCoord = Min(minCoord, Min(t.p1, t.p2, t.p3));
				maxCoord = Max(maxCoord, Max(t.p1, t.p2, t.p3));
			}
	}

	dynamic_cast<CBaseGLView*>(m_widget)->SetCameraStandardView(SBox{ C2Q(minCoord), C2Q(maxCoord) }, C2Q(_position));
	dynamic_cast<CBaseGLView*>(m_widget)->Redraw();
}

SCameraSettings CViewManager::GetCameraSettings() const
{
	if (const auto* viewer = dynamic_cast<CBaseGLView*>(m_widget))
		return viewer->GetCameraSettings();
	return {};
}

void CViewManager::SetCameraSettings(const SCameraSettings& _settings) const
{
	if (auto* viewer = dynamic_cast<CBaseGLView*>(m_widget))
		return viewer->SetCameraSettings(_settings);
}

std::vector<double> CViewManager::GetColoringValues(const std::vector<CSphere*>& _parts, const std::vector<CSolidBond*>& _bonds) const
{
	if (m_viewSettings->Coloring().type == EColoring::NONE || m_viewSettings->Coloring().type == EColoring::MATERIAL)
		return {};

	const EColorComponent component = m_viewSettings->Coloring().component; // get selected component for faster access
	// function to get selected component from vector
	const auto Component = [&](const CVector3& _vec)
	{
		switch (component)
		{
		case EColorComponent::TOTAL: return _vec.Length();
		case EColorComponent::X:     return _vec.x;
		case EColorComponent::Y:     return _vec.y;
		case EColorComponent::Z:     return _vec.z;
		default: return 0.0;
		}
	};

	std::vector<double> values;
	values.reserve(_parts.size() + _bonds.size());
	m_systemStructure->PrepareTimePointForRead(m_time);

	switch (m_viewSettings->Coloring().type)
	{
	case EColoring::NONE: break;
	case EColoring::MATERIAL: break;
	case EColoring::AGGL_SIZE:
	{
		CAgglomeratesAnalyzer analyzer;
		analyzer.SetSystemStructure(m_systemStructure);
		analyzer.FindAgglomerates(m_time);
		const auto& aggParts = analyzer.GetAgglomeratesParticles(); // vector of agglomerates, each agglomerate contains a list of particles within
		const auto& aggBonds = analyzer.GetAgglomeratesBonds();		// vector of agglomerates, each agglomerate contains a list of bonds within
		std::vector<size_t> agglSizes(m_systemStructure->GetTotalObjectsCount(), 1); // size of the agglomerate this object is in
		for (size_t i = 0; i < aggParts.size(); ++i)
			for (size_t j = 0; j < aggParts[i].size(); ++j)
				agglSizes[aggParts[i][j]] = aggParts[i].size();
		for (size_t i = 0; i < aggBonds.size(); ++i)
			for (size_t j = 0; j < aggBonds[i].size(); j++)
				agglSizes[aggBonds[i][j]] = aggParts[i].size();
		for (const auto& p : _parts)
			values.push_back(agglSizes[p->m_lObjectID]);
		for (const auto& b : _bonds)
			values.push_back(agglSizes[b->m_lObjectID]);
		break;
	}
	case EColoring::ANGLE_VELOCITY:
		for (const auto& p : _parts)
			values.push_back(Component(p->GetAngleVelocity()));
		break;
	case EColoring::BOND_NORMAL_STRESS:
		for (const auto& b : _bonds)
			values.push_back(-1 * DotProduct(b->GetForce(), m_systemStructure->GetBond(m_time, b->m_lObjectID).Normalized()) / b->m_dCrossCutSurface); // projection of total force on bond
		break;
	case EColoring::BOND_STRAIN:
		for (const auto& b : _bonds)
			values.push_back((m_systemStructure->GetBond(m_time, b->m_lObjectID).Length() - b->GetInitLength()) / b->GetInitLength());
		break;
	case EColoring::BOND_TOTAL_FORCE:
		for (const auto& b : _bonds)
		{
			double val = b->GetForce(m_time).Length();
			if (m_systemStructure->GetBond(m_time, b->m_lObjectID).Length() > b->GetInitLength()) // pulling state
				val *= -1;
			values.push_back(val);
		}
		break;
	case EColoring::CONTACT_DIAMETER:
		for (const auto& p : _parts)
			values.push_back(p->GetContactRadius() * 2);
		break;
	case EColoring::COORDINATE:
		for (const auto& p : _parts)
			values.push_back(Component(p->GetCoordinates()));
		for (const auto& b : _bonds)
			values.push_back(Component(m_systemStructure->GetBondCoordinate(m_time, b->m_lObjectID)));
		break;
	case EColoring::COORD_NUMBER:
	{
		const std::vector<unsigned> coordinations = m_systemStructure->GetCoordinationNumbers(m_time);
		for (const auto& p : _parts)
			values.push_back(coordinations[p->m_lObjectID]);
		break;
	}
	case EColoring::DIAMETER:
		for (const auto& p : _parts)
			values.push_back(p->GetRadius() * 2);
		for (const auto& b : _bonds)
			values.push_back(b->GetDiameter());
		break;
	case EColoring::FORCE:
		for (const auto& p : _parts)
			values.push_back(Component(p->GetForce()));
		for (const auto& b : _bonds)
			values.push_back(Component(b->GetForce()));
		break;
	case EColoring::OVERLAP:
	{
		const std::vector<double> overlaps = m_systemStructure->GetMaxOverlaps(m_time);
		for (const auto& p : _parts)
			values.push_back(overlaps[p->m_lObjectID]);
		break;
	}
	case EColoring::STRESS:
		for (const auto& p : _parts)
			values.push_back(Component(p->GetNormalStress()));
		break;
	case EColoring::PRINCIPAL_STRESS:
		for (const auto& p : _parts)
			values.push_back(Component(p->GetStressTensor().GetPrincipalStresses()));
		break;
	case EColoring::VELOCITY:
		for (const auto& p : _parts)
			values.push_back(Component(p->GetVelocity()));
		for (const auto& b : _bonds)
			values.push_back(Component(m_systemStructure->GetBondVelocity(m_time, b->m_lObjectID)));
		break;
	case EColoring::TEMPERATURE:
		for (const auto& p : _parts)
			 values.push_back(p->GetTemperature());
		break;
	case EColoring::DISPLACEMENT:
		for (const auto& p : _parts)
			values.push_back(Component(p->GetCoordinates()));
		for (const auto& b : _bonds)
			values.push_back(Component(m_systemStructure->GetBondCoordinate(m_time, b->m_lObjectID)));
		m_systemStructure->PrepareTimePointForRead(0.0);
		for (size_t i = 0; i < _parts.size(); ++i)
			values[i] -= Component(_parts[i]->GetCoordinates());
		for (size_t i = 0; i < _bonds.size(); ++i)
			values[i + _parts.size()] -= Component(m_systemStructure->GetBondCoordinate(0.0, _bonds[i]->m_lObjectID));
		break;
	}

	return values;
}

void CViewManager::UpdateAllObjects() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:	 dynamic_cast<COpenGLView*>(m_widget)->UpdateView();  break;
	case ERenderType::MIXED: dynamic_cast<COpenGLView*>(m_widget)->UpdateView();  break;
	case ERenderType::SHADER:
		DoUpdateParticles();
		DoUpdateBonds();
		DoUpdateWalls();
		DoUpdateVolumes();
		DoUpdateSlices();
		DoUpdateSimulationDomain();
		DoUpdatePBC();
		DoUpdateTime();
		DoUpdateLegend();
		DoUpdateAxes();
		dynamic_cast<COpenGLViewShader*>(m_widget)->Redraw();
		break;
	case ERenderType::NONE:     break;
	}
}

void CViewManager::EnableView() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:	  dynamic_cast<COpenGLView*>(m_widget)->EnableOpenGLView();  break;
	case ERenderType::MIXED:  dynamic_cast<COpenGLView*>(m_widget)->EnableOpenGLView();  break;
	case ERenderType::SHADER:
		UpdateAllObjects();
		break;
	case ERenderType::NONE:     break;
	}
}

void CViewManager::DisableView() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:	 dynamic_cast<COpenGLView*>(m_widget)->DisableOpenGLView();  break;
	case ERenderType::MIXED: dynamic_cast<COpenGLView*>(m_widget)->DisableOpenGLView();  break;
	case ERenderType::SHADER:
		// remove all objects from view
		dynamic_cast<COpenGLViewShader*>(m_widget)->SetParticles({});
		dynamic_cast<COpenGLViewShader*>(m_widget)->SetBonds({});
		dynamic_cast<COpenGLViewShader*>(m_widget)->SetWalls({});
		dynamic_cast<COpenGLViewShader*>(m_widget)->SetVolumes({});
		dynamic_cast<COpenGLViewShader*>(m_widget)->SetOrientations({});
		dynamic_cast<COpenGLViewShader*>(m_widget)->SetPeriodic({});
		break;
	case ERenderType::NONE:     break;
	}
}

void CViewManager::UpdateSelectedObjects() const
{
	switch (m_viewSettings->RenderType())
	{
	case ERenderType::GLU:		dynamic_cast<COpenGLView*>(m_widget)->UpdateView();							break;
	case ERenderType::MIXED:	dynamic_cast<COpenGLView*>(m_widget)->UpdateView();							break;
	case ERenderType::SHADER:   DoUpdateParticles(); dynamic_cast<COpenGLViewShader*>(m_widget)->Redraw();	break;
	case ERenderType::NONE:     break;
	}
}

void CViewManager::SelectObject(const QPoint& _pos)
{
	const size_t id = GetPointedObject(_pos);
	if (id != size_t(-1))
		m_viewSettings->SelectedObjects({ id });
	else
		m_viewSettings->SelectedObjects({});
	UpdateParticles();
	emit ObjectsSelected();
}

void CViewManager::SelectGroup(const QPoint& _pos)
{
	const size_t id = GetPointedObject(_pos);
	if (id != size_t(-1))
		m_viewSettings->SelectedObjects({ m_systemStructure->GetAgglomerate(m_time, id) });
	else
		m_viewSettings->SelectedObjects({});
	UpdateParticles();
	emit ObjectsSelected();
}

size_t CViewManager::GetPointedObject(const QPoint& _pos) const
{
	const SBox box = dynamic_cast<CBaseGLView*>(m_widget)->WinCoord2LineOfSight(_pos);
	const SVolumeType coords{ Q2C(box.minCoord), Q2C(box.maxCoord) };

	size_t nearestObjectID = -1;
	double nearestDistance = std::numeric_limits<double>::max();
	const CVector3 lineVector = coords.coordEnd - coords.coordBeg;
	const double squaredLineLength = lineVector.SquaredLength();

	// volume of the lineVector
	const CVector3 min = Min(coords.coordBeg, coords.coordEnd);
	const CVector3 max = Max(coords.coordBeg, coords.coordEnd);

	// go through all visible particles
	for (const auto& part : GetVisibleParticles())
	{
		const CVector3 partCoord = part->GetCoordinates(m_time);
		const double partRadius = part->GetRadius();
		const CVector3 maxPartCoord = partCoord + partRadius;
		const CVector3 minPartCoord = partCoord - partRadius;
		if (maxPartCoord.x < min.x || minPartCoord.x > max.x ||
			maxPartCoord.y < min.y || minPartCoord.y > max.y ||
			maxPartCoord.z < min.z || minPartCoord.z > max.z) continue;

		const CVector3 vecToBegin = coords.coordBeg - partCoord;
		const double distToBegin = vecToBegin.SquaredLength(); // squared distance from start line point to the center of particle
		const double projectionLen = DotProduct(vecToBegin, lineVector);
		const double sqrDistToPartCenter = (distToBegin * squaredLineLength - projectionLen * projectionLen) / squaredLineLength;

		if (sqrDistToPartCenter < partRadius * partRadius) // if particle and line have an overlap point
		{
			const double distance = std::sqrt(distToBegin) - partRadius;
			if (distance > 0)
			{
				const double dLength = distance * distance * (distToBegin - sqrDistToPartCenter) / distToBegin;
				if (dLength < nearestDistance)
				{
					nearestObjectID = part->m_lObjectID;
					nearestDistance = dLength;
				}
			}
		}
	}

	return nearestObjectID;
}

void CViewManager::SetRenderGlu()
{
	switch (WidgetRenderType())
	{
	case ERenderType::GLU:
		return;
	case ERenderType::MIXED:
	{
		auto* oldViewer = dynamic_cast<COpenGLViewMixed*>(m_widget);

		auto* newViewer = new COpenGLViewGlu(*oldViewer, m_viewSettings, m_widget->parentWidget());
		newViewer->SetSystemStructure(m_systemStructure);
		newViewer->m_pSampleAnalyzerTab = const_cast<CSampleAnalyzerTab*>(m_sampleAnalyzer);

		// replace widgets on form
		m_widget->setUpdatesEnabled(false);
		m_layout->replaceWidget(oldViewer, newViewer);
		m_widget = newViewer;
		m_widget->setUpdatesEnabled(true);

		delete oldViewer;

		SetViewSettings();
		newViewer->SetCurrentTime(m_time);

		break;
	}
	case ERenderType::SHADER:
	{
		auto* oldViewer = dynamic_cast<COpenGLViewShader*>(m_widget);

		auto* newViewer = new COpenGLViewGlu(*oldViewer, m_viewSettings, m_widget->parentWidget());
		newViewer->SetSystemStructure(m_systemStructure);
		newViewer->m_pSampleAnalyzerTab = const_cast<CSampleAnalyzerTab*>(m_sampleAnalyzer);

		// replace widgets on form
		m_widget->setUpdatesEnabled(false);
		m_layout->replaceWidget(oldViewer, newViewer);
		m_widget = newViewer;
		m_widget->setUpdatesEnabled(true);

		delete oldViewer;

		SetViewSettings();
		newViewer->SetCurrentTime(m_time);

		break;
	}
	case ERenderType::NONE:
	{
		auto* oldViewer = m_widget;

		auto* newViewer = new COpenGLViewGlu(m_viewSettings, m_widget->parentWidget());
		newViewer->SetSystemStructure(m_systemStructure);
		newViewer->m_pSampleAnalyzerTab = const_cast<CSampleAnalyzerTab*>(m_sampleAnalyzer);

		// replace widgets on form
		m_widget->setUpdatesEnabled(false);
		m_layout->replaceWidget(oldViewer, newViewer);
		m_widget = newViewer;
		m_widget->setUpdatesEnabled(true);

		delete oldViewer;

		SetViewSettings();
		newViewer->SetCurrentTime(m_time);

		break;
	}
	}
}

void CViewManager::SetRenderMixed()
{
	switch (WidgetRenderType())
	{
	case ERenderType::MIXED:
		return;
	case ERenderType::GLU:
	{
		auto* oldViewer = dynamic_cast<COpenGLViewGlu*>(m_widget);

		auto* newViewer = new COpenGLViewMixed(*oldViewer, m_viewSettings, m_widget->parentWidget());
		newViewer->SetSystemStructure(m_systemStructure);
		newViewer->m_pSampleAnalyzerTab = const_cast<CSampleAnalyzerTab*>(m_sampleAnalyzer);
		newViewer->SetParticleTexture(m_viewSettings->ParticleTexture());

		// replace widgets on form
		m_widget->setUpdatesEnabled(false);
		m_layout->replaceWidget(oldViewer, newViewer);
		m_widget = newViewer;
		m_widget->setUpdatesEnabled(true);

		delete oldViewer;

		SetViewSettings();
		newViewer->SetCurrentTime(m_time);

		break;
	}
	case ERenderType::SHADER:
	{
		auto* oldViewer = dynamic_cast<COpenGLViewShader*>(m_widget);

		auto* newViewer = new COpenGLViewMixed(*oldViewer, m_viewSettings, m_widget->parentWidget());
		newViewer->SetSystemStructure(m_systemStructure);
		newViewer->m_pSampleAnalyzerTab = const_cast<CSampleAnalyzerTab*>(m_sampleAnalyzer);
		newViewer->SetParticleTexture(m_viewSettings->ParticleTexture());

		// replace widgets on form
		m_widget->setUpdatesEnabled(false);
		m_layout->replaceWidget(oldViewer, newViewer);
		m_widget = newViewer;
		m_widget->setUpdatesEnabled(true);

		delete oldViewer;

		SetViewSettings();
		newViewer->SetCurrentTime(m_time);

		break;
	}
	case ERenderType::NONE:
	{
		auto* oldViewer = m_widget;

		auto* newViewer = new COpenGLViewMixed(m_viewSettings, m_widget->parentWidget());
		newViewer->SetSystemStructure(m_systemStructure);
		newViewer->m_pSampleAnalyzerTab = const_cast<CSampleAnalyzerTab*>(m_sampleAnalyzer);
		newViewer->SetParticleTexture(m_viewSettings->ParticleTexture());

		// replace widgets on form
		m_widget->setUpdatesEnabled(false);
		m_layout->replaceWidget(oldViewer, newViewer);
		m_widget = newViewer;
		m_widget->setUpdatesEnabled(true);

		delete oldViewer;

		SetViewSettings();
		newViewer->SetCurrentTime(m_time);

		break;
	}
	}
}

void CViewManager::SetRenderShader()
{
	switch (WidgetRenderType())
	{
	case ERenderType::SHADER:
		return;
	case ERenderType::GLU:
	{
		auto* oldViewer = dynamic_cast<COpenGLViewGlu*>(m_widget);

		auto* newViewer = new COpenGLViewShader(*oldViewer, m_widget->parentWidget());

		// replace widgets on form
		m_widget->setUpdatesEnabled(false);
		m_layout->replaceWidget(oldViewer, newViewer);
		m_widget = newViewer;
		m_widget->setUpdatesEnabled(true);

		delete oldViewer;

		newViewer->show(); // to force proper OpenGL initialization prior to setting any data

		break;
	}
	case ERenderType::MIXED:
	{
		auto* oldViewer = dynamic_cast<COpenGLViewMixed*>(m_widget);

		auto* newViewer = new COpenGLViewShader(*oldViewer, m_widget->parentWidget());

		// replace widgets on form
		m_widget->setUpdatesEnabled(false);
		m_layout->replaceWidget(oldViewer, newViewer);
		m_widget = newViewer;
		m_widget->setUpdatesEnabled(true);

		delete oldViewer;

		newViewer->show(); // to force proper OpenGL initialization prior to setting any data

		break;
	}
	case ERenderType::NONE:
	{
		auto* oldViewer = m_widget;

		auto* newViewer = new COpenGLViewShader(m_widget->parentWidget());

		// replace widgets on form
		m_widget->setUpdatesEnabled(false);
		m_layout->replaceWidget(oldViewer, newViewer);
		m_widget = newViewer;
		m_widget->setUpdatesEnabled(true);

		delete oldViewer;

		newViewer->show(); // to force proper OpenGL initialization prior to setting any data

		break;
	}
	}

	SetViewSettings();
	UpdateAllObjects();
}

ERenderType CViewManager::WidgetRenderType() const
{
	if (dynamic_cast<COpenGLViewGlu*>(m_widget))	return ERenderType::GLU;
	if (dynamic_cast<COpenGLViewMixed*>(m_widget))	return ERenderType::MIXED;
	if (dynamic_cast<COpenGLViewShader*>(m_widget))	return ERenderType::SHADER;
	return ERenderType::NONE;
}

void CViewManager::SetViewSettings() const
{
	if (WidgetRenderType() == ERenderType::NONE) return;

	auto* viewer = dynamic_cast<CBaseGLView*>(m_widget);
	viewer->SetRenderQuality(m_viewSettings->RenderQuality());
	viewer->SetParticleTexture(m_viewSettings->ParticleTexture());
	viewer->SetFontAxes(m_viewSettings->FontAxes().font, m_viewSettings->FontAxes().color);
	viewer->SetFontTime(m_viewSettings->FontTime().font, m_viewSettings->FontTime().color);
	viewer->SetFontLegend(m_viewSettings->FontLegend().font, m_viewSettings->FontLegend().color);

	switch (WidgetRenderType()) // TODO: change to m_viewSettings->RenderType()
	{
	case ERenderType::GLU:
	case ERenderType::MIXED:
	{
		auto* viewerOld = dynamic_cast<COpenGLView*>(m_widget);
		viewerOld->SetOrientationVisible(m_viewSettings->Visibility().orientations);
		viewerOld->SetAxesVisible(m_viewSettings->Visibility().axes);
		viewerOld->SetLegendVisible(m_viewSettings->Visibility().legend);
		viewerOld->SetTimeVisible(m_viewSettings->Visibility().time);
		break;
	}
	case ERenderType::SHADER:
	{
		auto* viewerNew = dynamic_cast<COpenGLViewShader*>(m_widget);
		break;
	}
	case ERenderType::NONE:
		break;
	}
}

void CViewManager::ClearWidget()
{
	auto* newViewer = new QWidget();
	m_widget->setUpdatesEnabled(false);
	auto* oldViewer = dynamic_cast<COpenGLViewGlu*>(m_widget);
	m_layout->replaceWidget(oldViewer, newViewer);
	m_widget = newViewer;
	delete oldViewer;
	m_widget->setUpdatesEnabled(true);
}

void CViewManager::DoUpdateParticles() const
{
	// to skip unnecessary rendering attempts
	if (!dynamic_cast<COpenGLViewShader*>(m_widget)->IsValid()) return;

	// get all particles that must be shown
	const auto particles = GetVisibleParticles();
	// prepare particles vector for viewer
	std::vector<COpenGLViewShader::SParticle> data;
	data.reserve(particles.size());
	// prepare particles orientations vector for viewer, if necessary
	std::vector<COpenGLViewShader::SOrientation> data2;
	if (m_viewSettings->Visibility().orientations)
		data2.reserve(particles.size());
	// obtain colors
	const std::vector<QColor> colors = GetObjectsColors(particles, {});

	// for all visible particles
	for (const auto& p : particles)
	{
		const CVector3 coord = p->GetCoordinates(m_time);
		const auto r = static_cast<float>(p->GetRadius());
		const QColor color = colors[&p - &particles[0]];
		data.emplace_back(COpenGLViewShader::SParticle{ C2Q(coord), color, r });
		// add orientations for this particle if necessary
		if (m_viewSettings->Visibility().orientations)
			data2.emplace_back(COpenGLViewShader::SOrientation{ C2Q(coord), C2Q(p->GetOrientation(m_time)), r });
	}

	// add selected particles disregarding any filters and cuttings
	for (const auto& id : m_viewSettings->SelectedObjects())
		if (const CSphere* p = dynamic_cast<CSphere*>(m_systemStructure->GetObjectByIndex(id)))
			data.emplace_back(COpenGLViewShader::SParticle{ C2Q(p->GetCoordinates(m_time)), Qt::yellow, static_cast<float>(p->GetRadius()) * 1.1f });

	dynamic_cast<COpenGLViewShader*>(m_widget)->SetParticles(data);
	dynamic_cast<COpenGLViewShader*>(m_widget)->SetOrientations(data2);
}

void CViewManager::DoUpdateBonds() const
{
	// to skip unnecessary rendering attempts
	if (!dynamic_cast<COpenGLViewShader*>(m_widget)->IsValid()) return;

	// get all bonds that must be shown
	const auto bonds = GetVisibleBonds();
	// prepare bonds vector for viewer
	std::vector<COpenGLViewShader::SBond> data;
	data.reserve(bonds.size());
	// obtain colors
	const std::vector<QColor> colors = GetObjectsColors({}, bonds);

	// periodic boundary conditions
	const SPBC& pbc = m_systemStructure->GetPBC();

	// for all active bonds
	for (const auto& b : bonds)
	{
		// gather coordinates
		const CVector3 p1 = m_systemStructure->GetObjectByIndex(b->m_nLeftObjectID)->GetCoordinates(m_time);
		const CVector3 p2 = m_systemStructure->GetObjectByIndex(b->m_nRightObjectID)->GetCoordinates(m_time);
		const CVector3 coord1 = p1;
		const CVector3 coord2 = !pbc.bEnabled ? p2 : p1 + GetSolidBond(p1, p2, pbc); // to consider PBC, calculate the second coordinate through the bond's length
		// add bond
		const auto r = static_cast<float>(b->GetDiameter() / 2.);
		const QColor color = !m_viewSettings->BrokenBonds().show ? colors[&b - &bonds[0]] : m_viewSettings->BrokenBonds().color;
		data.emplace_back(COpenGLViewShader::SBond{ C2Q(coord1), C2Q(coord2), color, r });
	}

	dynamic_cast<COpenGLViewShader*>(m_widget)->SetBonds(data);
}

void CViewManager::DoUpdateWalls() const
{
	// to skip unnecessary rendering attempts
	if (!dynamic_cast<COpenGLViewShader*>(m_widget)->IsValid()) return;

	if (!m_viewSettings->Visibility().geometries)
	{
		dynamic_cast<COpenGLViewShader*>(m_widget)->SetWalls({});
		return;
	}

	// prepare walls vector for viewer
	std::vector<COpenGLViewShader::STriangle> data;
	data.reserve(m_systemStructure->GetNumberOfSpecificObjects(TRIANGULAR_WALL));

	// in a case of cutting by volumes
	std::vector<bool> visible;
	if (m_viewSettings->Cutting().cutByVolumes)
	{
		visible.resize(m_systemStructure->GetTotalObjectsCount(), false);
		for (const auto& v : m_viewSettings->Cutting().volumes)
			for (size_t i : m_systemStructure->AnalysisVolume(v)->GetWallIndicesInside(m_time))
				visible[i] = true;
	}

	// list of visible geometries
	const auto visibleGeometries = m_viewSettings->VisibleGeometries();

	for (const auto& geometry : m_systemStructure->AllGeometries())
	{
		if (!SetContains(visibleGeometries, geometry->Key())) continue;							// filter by selected geometries
		const QColor color = C2Q(CColor{ geometry->Color(), 1.f - m_viewSettings->GeometriesTransparency() });

		// for all active walls
		for (const auto& w : m_systemStructure->GetAllWallsForGeometry(m_time, geometry->Key()))
		{
			if (m_viewSettings->Cutting().cutByVolumes && !visible[w->m_lObjectID]) continue;	// cut by volume
			const CTriangle t = w->GetCoords(m_time);
			if (m_viewSettings->Cutting().CutByPlanes() && (IsCutByPlanes(t.p1) || IsCutByPlanes(t.p2) || IsCutByPlanes(t.p3))) continue;	// cut by planes
			data.emplace_back(COpenGLViewShader::STriangle{ C2Q(t.p1), C2Q(t.p2), C2Q(t.p3), color });
		}
	}

	dynamic_cast<COpenGLViewShader*>(m_widget)->SetWalls(data);
}

void CViewManager::DoUpdateVolumes() const
{
	// to skip unnecessary rendering attempts
	if (!dynamic_cast<COpenGLViewShader*>(m_widget)->IsValid()) return;

	std::vector<COpenGLViewShader::STriangle> data;

	if (m_viewSettings->Visibility().volumes)
	{
		// list of visible volumes
		const auto visibleVolumes = m_viewSettings->VisibleVolumes();

		// common analysis volumes
		for (const auto& volume : m_systemStructure->AllAnalysisVolumes())
		{
			if (!SetContains(visibleVolumes, volume->Key())) continue;				// filter by selected volumes
			const auto& mesh = volume->Mesh(m_time);
			data.reserve(data.size() + mesh.TrianglesNumber());
			const QColor color = C2Q(volume->Color());
			for (const auto& t : mesh.Triangles())
				data.emplace_back(COpenGLViewShader::STriangle{ C2Q(t.p1), C2Q(t.p2), C2Q(t.p3), color });
		}
	}

	// sample analysis volume
	if (m_sampleAnalyzer && m_sampleAnalyzer->isVisible())
	{
		CTriangularMesh mesh = CMeshGenerator::Sphere(m_sampleAnalyzer->m_dRadius, 3);
		mesh.Shift(m_sampleAnalyzer->m_vCenter);

		data.reserve(data.size() + mesh.TrianglesNumber());
		const QColor color = C2Q(CColor::DefaultSampleAnalyzerColor());
		for (const auto& t : mesh.Triangles())
			data.emplace_back(COpenGLViewShader::STriangle{ C2Q(t.p1), C2Q(t.p2), C2Q(t.p3), color });
	}

	dynamic_cast<COpenGLViewShader*>(m_widget)->SetVolumes(data);
}

void CViewManager::DoUpdateSimulationDomain() const
{
	// to skip unnecessary rendering attempts
	if (!dynamic_cast<COpenGLViewShader*>(m_widget)->IsValid()) return;

	if (!m_viewSettings->Visibility().domain)
	{
		dynamic_cast<COpenGLViewShader*>(m_widget)->SetDomain({});
		return;
	}

	const SVolumeType& domain = m_systemStructure->GetSimulationDomain();
	const COpenGLViewShader::SDomain box{ C2Q(domain.coordBeg), C2Q(domain.coordEnd), C2Q(CColor::DefaultSimulationDomainColor()) };

	dynamic_cast<COpenGLViewShader*>(m_widget)->SetDomain(box);
}

void CViewManager::DoUpdatePBC() const
{
	// to skip unnecessary rendering attempts
	if (!dynamic_cast<COpenGLViewShader*>(m_widget)->IsValid()) return;

	SPBC pbc = m_systemStructure->GetPBC();

	if (!m_viewSettings->Visibility().pbc || !pbc.bEnabled)
	{
		dynamic_cast<COpenGLViewShader*>(m_widget)->SetPeriodic({});
		return;
	}

	pbc.UpdatePBC(m_time);

	const CVector3 posFlags{ pbc.bX ? 1. : 0., pbc.bY ? 1. : 0., pbc.bZ ? 1. : 0. };
	const CVector3 negFlags{ pbc.bX ? 0. : 1., pbc.bY ? 0. : 1., pbc.bZ ? 0. : 1. };
	const SVolumeType domain = m_systemStructure->GetSimulationDomain();
	const CVector3 coordBeg = EntryWiseProduct(domain.coordBeg, negFlags) + EntryWiseProduct(pbc.currentDomain.coordBeg, posFlags);
	const CVector3 coordEnd = EntryWiseProduct(domain.coordEnd,   negFlags) + EntryWiseProduct(pbc.currentDomain.coordEnd,   posFlags);

	dynamic_cast<COpenGLViewShader*>(m_widget)->SetPeriodic(COpenGLViewShader::SPeriodic{ pbc.bX, pbc.bY, pbc.bZ, C2Q(coordBeg), C2Q(coordEnd) });
}

void CViewManager::DoUpdateAxes() const
{
	// to skip unnecessary rendering attempts
	if (!dynamic_cast<COpenGLViewShader*>(m_widget)->IsValid()) return;

	dynamic_cast<COpenGLViewShader*>(m_widget)->SetAxes(m_viewSettings->Visibility().axes);
}

void CViewManager::DoUpdateTime() const
{
	// to skip unnecessary rendering attempts
	if (!dynamic_cast<COpenGLViewShader*>(m_widget)->IsValid()) return;

	dynamic_cast<COpenGLViewShader*>(m_widget)->SetTime(m_time, m_viewSettings->Visibility().time);
}

void CViewManager::DoUpdateLegend() const
{
	// to skip unnecessary rendering attempts
	if (!dynamic_cast<COpenGLViewShader*>(m_widget)->IsValid()) return;

	if (m_viewSettings->Coloring().type == EColoring::NONE)
		dynamic_cast<COpenGLViewShader*>(m_widget)->SetLegend({}, false);
	else
		dynamic_cast<COpenGLViewShader*>(m_widget)->SetLegend(COpenGLViewShader::SLegend{ m_viewSettings->Coloring().minDisplayValue, m_viewSettings->Coloring().maxDisplayValue,
		m_viewSettings->Coloring().minColor, m_viewSettings->Coloring().midColor, m_viewSettings->Coloring().maxColor }, m_viewSettings->Visibility().legend);
}

void CViewManager::DoUpdateSlices() const
{
	// to skip unnecessary rendering attempts
	if (!dynamic_cast<COpenGLViewShader*>(m_widget)->IsValid()) return;

	// prepare particles vector for viewer
	std::vector<COpenGLViewShader::SDiscs> data;

	if (m_viewSettings->Slicing().active && m_viewSettings->Slicing().plane != ESlicePlane::NONE)
	{
		// get all particles that must be shown
		const auto particles = GetVisibleSlices();
		data.reserve(particles.size());

		// obtain colors
		std::vector<QColor> colors = GetObjectsColors(particles, {});
		for (auto& c : colors)
			if (c == C2Q(CColor::DefaultParticleColor()))
				c = Qt::black;

		// for all visible particles
		for (const auto& p : particles)
		{
			const CVector3 partCoord = p->GetCoordinates(m_time);										// real coordinate of the particles
			const double partR = p->GetRadius();														// real radius of the particles
			const size_t i = E2I(m_viewSettings->Slicing().plane) - E2I(ESlicePlane::X);				// index in CVector3 to access coordinate of the selected plane
			CVector3 coord = partCoord;
			coord[i] = m_viewSettings->Slicing().coordinate;											// coordinate of the projected disc
			const double h = partR - std::fabs(partCoord[i] - m_viewSettings->Slicing().coordinate);	// height of the sphere's cap
			const double r = std::sqrt(2 * partR * h - std::pow(h, 2));									// radius of the sphere's cap - the projected radius
			const QColor color = colors[&p - &particles[0]];											// color
			data.emplace_back(COpenGLViewShader::SDiscs{ C2Q(coord), color, static_cast<float>(r), static_cast<Qt::Axis>(i) });
		}
	}

	dynamic_cast<COpenGLViewShader*>(m_widget)->SetDiscs(data);
}

std::vector<CSphere*> CViewManager::GetVisibleParticles() const
{
	if (!m_viewSettings->Visibility().particles) return {};

	std::vector<CSphere*> visibleParticles;
	const std::vector<CSphere*> allParticles = m_systemStructure->GetAllSpheres(m_time);
	visibleParticles.reserve(allParticles.size());

	// in a case of cutting by volumes
	std::vector<bool> show;
	if (m_viewSettings->Cutting().cutByVolumes)
	{
		show.resize(m_systemStructure->GetTotalObjectsCount(), false);
		for (const auto& v : m_viewSettings->Cutting().volumes)
			for (auto i : m_systemStructure->AnalysisVolume(v)->GetParticleIndicesInside(m_time))
				show[i] = true;
	}

	// list of visible materials for particles
	const auto visiblePartMaterials = m_viewSettings->VisiblePartMaterials();

	// for all active particles
	for (const auto& p : allParticles)
	{
		if (!SetContains(visiblePartMaterials, p->GetCompoundKey())) continue;								// filter by selected materials
		if (m_viewSettings->Cutting().cutByVolumes && !show[p->m_lObjectID]) continue;						// cut by volume
		if (m_viewSettings->Cutting().CutByPlanes() && IsCutByPlanes(p->GetCoordinates(m_time))) continue;	// cut by planes
		// add particle
		visibleParticles.push_back(p);
	}

	return visibleParticles;
}

std::vector<CSolidBond*> CViewManager::GetVisibleBonds() const
{
	if (!m_viewSettings->Visibility().bonds) return {};

	std::vector<CSolidBond*> visibleBonds;
	const std::vector<CSolidBond*> allBonds = m_systemStructure->GetAllSolidBonds(m_time);
	visibleBonds.reserve(allBonds.size());

	// in a case of cutting by volumes
	std::vector<bool> show;
	if (m_viewSettings->Cutting().cutByVolumes)
	{
		show.resize(m_systemStructure->GetTotalObjectsCount(), false);
		for (const auto& v : m_viewSettings->Cutting().volumes)
			for (auto i : m_systemStructure->AnalysisVolume(v)->GetBondIndicesInside(m_time))
				show[i] = true;
	}

	// periodic boundary conditions
	const SPBC& pbc = m_systemStructure->GetPBC();

	// list of visible materials for bonds
	const auto visibleBondMaterials = m_viewSettings->VisibleBondMaterials();

	// for all active bonds
	for (const auto& b : allBonds)
	{
		if (!SetContains(visibleBondMaterials, b->GetCompoundKey())) continue;			// filter by selected materials
		if (m_viewSettings->Cutting().cutByVolumes && !show[b->m_lObjectID]) continue;	// cut by volume
		if (m_viewSettings->BrokenBonds().show && !m_viewSettings->BrokenBonds().IsInInterval(b->GetActivityEnd())) continue;	// filter by broken bonds
		if (m_viewSettings->Cutting().CutByPlanes())									// cutting by planes enabled
		{
			const CVector3 p1 = m_systemStructure->GetObjectByIndex(b->m_nLeftObjectID)->GetCoordinates(m_time);
			const CVector3 p2 = m_systemStructure->GetObjectByIndex(b->m_nRightObjectID)->GetCoordinates(m_time);
			if (pbc.bEnabled)
			{
				const CVector3 coord1 = p1;
				const CVector3 coord2 = p1 + GetSolidBond(p1, p2, pbc); // to consider PBC, calculate the second coordinate through the bond's length
				if (IsCutByPlanes(coord1) || IsCutByPlanes(coord2)) continue;			// cut by planes
			}
			else
				if (IsCutByPlanes(p1) || IsCutByPlanes(p2)) continue;					// cut by planes
		}
		// add bond
		visibleBonds.push_back(b);
	}

	return visibleBonds;
}

std::vector<CSphere*> CViewManager::GetVisibleSlices() const
{
	std::vector<CSphere*> visibleParticles;
	const std::vector<CSphere*> allParticles = m_systemStructure->GetAllSpheres(m_time);
	visibleParticles.reserve(allParticles.size());

	// in a case of cutting by volumes
	std::vector<bool> show;
	if (m_viewSettings->Cutting().cutByVolumes)
	{
		show.resize(m_systemStructure->GetTotalObjectsCount(), false);
		for (const auto& v : m_viewSettings->Cutting().volumes)
			for (auto i : m_systemStructure->AnalysisVolume(v)->GetParticleIndicesInside(m_time))
				show[i] = true;
	}

	// list of visible materials for particles
	const auto visiblePartMaterials = m_viewSettings->VisiblePartMaterials();

	// for all active particles
	for (const auto& p : allParticles)
	{
		if (!SetContains(visiblePartMaterials, p->GetCompoundKey())) continue;											// filter by selected materials
		if (m_viewSettings->Cutting().cutByVolumes && !show[p->m_lObjectID]) continue;									// cut by volume
		const CVector3 coord = p->GetCoordinates(m_time);
		if (m_viewSettings->Cutting().CutByPlanes() && IsCutByPlanes(coord)) continue;									// cut by planes
		const size_t i = E2I(m_viewSettings->Slicing().plane) - E2I(ESlicePlane::X);
		if (std::abs(coord[i] - m_viewSettings->Slicing().coordinate) >= static_cast<float>(p->GetRadius())) continue;	// slice
		// add particle
		visibleParticles.push_back(p);
	}

	return visibleParticles;
}

bool CViewManager::IsCutByPlanes(const CVector3& _coord) const
{
	return m_viewSettings->Cutting().cutByX && (_coord.x < m_viewSettings->Cutting().minX || _coord.x > m_viewSettings->Cutting().maxX) ||
		   m_viewSettings->Cutting().cutByY && (_coord.y < m_viewSettings->Cutting().minY || _coord.y > m_viewSettings->Cutting().maxY) ||
		   m_viewSettings->Cutting().cutByZ && (_coord.z < m_viewSettings->Cutting().minZ || _coord.z > m_viewSettings->Cutting().maxZ);
}

std::vector<QColor> CViewManager::GetObjectsColors(const std::vector<CSphere*>& _parts, const std::vector<CSolidBond*>& _bonds) const
{
	static const QColor partDefaultColor = C2Q(CColor::DefaultParticleColor());
	static const QColor bondDefaultColor = C2Q(CColor::DefaultBondColor());

	// builds a vector of default colors
	const auto DefaultColors = [&](size_t _nPart, size_t _nBond)
	{
		std::vector<QColor> colors;
		colors.resize(_nPart, partDefaultColor);
		colors.resize(_nPart + _nBond, bondDefaultColor);
		return colors;
	};

	// special case for EColoring::NONE - all have default color
	if (m_viewSettings->Coloring().type == EColoring::NONE)
		return DefaultColors(_parts.size(), _bonds.size());

	// special case for EColoring::MATERIAL - coloring by materials
	if (m_viewSettings->Coloring().type == EColoring::MATERIAL)
	{
		// prepare vector of resulting colors
		std::vector<QColor> colors;
		colors.reserve(_parts.size() + _bonds.size());
		// gather colors
		for (const auto& p : _parts) colors.push_back(C2Q(p->GetColor()));
		for (const auto& b : _bonds) colors.push_back(C2Q(b->GetColor()));
		return colors;
	}

	// get values, according to which coloring is applied
	const std::vector<double> values = GetColoringValues(_parts, _bonds);

	// if returned nothing - apply default color
	if (values.empty())
		return DefaultColors(_parts.size(), _bonds.size());

	// prepare vector of resulting colors
	std::vector<QColor> colors;
	colors.reserve(values.size());

	// apply interpolation of colors
	for (double v : values)
		colors.push_back(InterpolateColor(v));

	return colors;
}

QColor CViewManager::InterpolateColor(double _value) const
{
	const QColor& minC = m_viewSettings->Coloring().minColor;
	const QColor& midC = m_viewSettings->Coloring().midColor;
	const QColor& maxC = m_viewSettings->Coloring().maxColor;

	// bring value to range [0..1], 0 - minColor, 1 - maxColor
	double value = 0;
	if (m_viewSettings->Coloring().maxValue - m_viewSettings->Coloring().minValue != 0)
		value = (_value - m_viewSettings->Coloring().minValue) / (m_viewSettings->Coloring().maxValue - m_viewSettings->Coloring().minValue);

	// if if falls into boundary - just return it
	const double eps = 0.00001;
	if (value < eps)							return minC;
	if (value > 0.5 - eps && value < 0.5 + eps)	return midC;
	if (value > 1.0 - eps)						return maxC;

	// interpolate
	const double x1 = value;
	const double x2 = std::fabs(0.5 - value);
	const double x3 = 1 - value;
	const double sumX = x1 + x2 + x3;
	const double f1 = sumX / x1;
	const double f2 = sumX / x2;
	const double f3 = sumX / x3;
	const double sumF = f1 + f2 + f3;
	const double k1 = f1 / sumF;
	const double k2 = f2 / sumF;
	const double k3 = f3 / sumF;

	// construct final color
	const double r = ClampFunction(k1 * minC.redF()   + k2 * midC.redF()   + k3 * maxC.redF(),   0.0, 1.0);
	const double g = ClampFunction(k1 * minC.greenF() + k2 * midC.greenF() + k3 * maxC.greenF(), 0.0, 1.0);
	const double b = ClampFunction(k1 * minC.blueF()  + k2 * midC.blueF()  + k3 * maxC.blueF(),  0.0, 1.0);

	QColor color;
	color.setRgbF(r, g, b);
	return color;
}

QQuaternion CViewManager::C2Q(const CQuaternion& _q)
{
	return  QQuaternion{ static_cast<float>(_q.q0), static_cast<float>(_q.q1), static_cast<float>(_q.q2), static_cast<float>(_q.q3) };
}

QVector3D CViewManager::C2Q(const CVector3& _v)
{
	return QVector3D{ static_cast<float>(_v.x), static_cast<float>(_v.y), static_cast<float>(_v.z) };
}

QColor CViewManager::C2Q(const CColor& _c)
{
	QColor res;
	res.setRgbF(_c.r, _c.g, _c.b, _c.a);
	return res;
}

CVector3 CViewManager::Q2C(const QVector3D& _v)
{
	return CVector3{ _v.x(), _v.y(), _v.z() };
}