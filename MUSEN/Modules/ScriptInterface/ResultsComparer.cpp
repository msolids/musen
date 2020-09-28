/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */


#include "ResultsComparer.h"

CResultsComparer::CResultsComparer()
{
	m_dRelTolerance = 1e-3;
}


bool CResultsComparer::CompareTwoValues(std::string _sMessage, double _dValue1, double _dValue2, double _dRelTol)
{
	if (fabs(_dValue1 - _dValue2) > std::min(fabs(_dValue1), fabs(_dValue2))*_dRelTol )
	{
		(*m_pOutStream) << m_pScene1->GetFileName() << " vs " << m_pScene2->GetFileName() << " || Different " << _sMessage << ": " << _dValue1 << " vs " << _dValue2 << std::endl;
		return false;
	}
	return true;
}

bool CResultsComparer::CompareTwoValues(std::string _sMessage, CVector3 _vVec1, CVector3 _vVec2, double _dRelTol)
{
	CVector3 vDiff = _vVec1 - _vVec2;
	if ((fabs(vDiff.x) > std::min(fabs(_vVec1.x), fabs(_vVec1.x))*_dRelTol) ||
		(fabs(vDiff.y) > std::min(fabs(_vVec1.y), fabs(_vVec1.y))*_dRelTol) ||
		(fabs(vDiff.z) > std::min(fabs(_vVec1.z), fabs(_vVec1.z))*_dRelTol))
	{
		(*m_pOutStream) << m_pScene1->GetFileName() << " vs " << m_pScene2->GetFileName() << " || Different " << _sMessage << ": " << _vVec1 << " vs " << _vVec2 << std::endl;
		return false;
	}
	return true;
}

bool CResultsComparer::CompareTwoValues(std::string _sMessage, CVector3 _vVec1, CVector3 _vVec2, double _dRelTol, double _dAbsTol )
{
	CVector3 vDiff = _vVec1 - _vVec2;
	if ((fabs(vDiff.x) > std::min(fabs(_vVec1.x), fabs(_vVec1.x))*_dRelTol + _dAbsTol) ||
		(fabs(vDiff.y) > std::min(fabs(_vVec1.y), fabs(_vVec1.y))*_dRelTol + _dAbsTol) ||
		(fabs(vDiff.z) > std::min(fabs(_vVec1.z), fabs(_vVec1.z))*_dRelTol + _dAbsTol))
	{
		(*m_pOutStream) << m_pScene1->GetFileName() << " vs " << m_pScene2->GetFileName() << " || Different " << _sMessage << ": " << _vVec1 << " vs " << _vVec2 << std::endl;
		return false;
	}
	return true;
}
bool CResultsComparer::CompareTimeIndependentData()
{
	if (!CompareTwoValues("total number of objects", m_pScene1->GetTotalObjectsCount(), m_pScene2->GetTotalObjectsCount())) return false;
	if (!CompareTwoValues("total number of particles", m_pScene1->GetNumberOfSpecificObjects(SPHERE), m_pScene2->GetNumberOfSpecificObjects(SPHERE))) return false;
	if (!CompareTwoValues("total number of solid bonds", m_pScene1->GetNumberOfSpecificObjects(SOLID_BOND), m_pScene2->GetNumberOfSpecificObjects(SOLID_BOND))) return false;
	if (!CompareTwoValues("total number of geometries", m_pScene1->GeometriesNumber(), m_pScene2->GeometriesNumber())) return false;
	if (!CompareTwoValues("total number of triangular walls", m_pScene1->GetNumberOfSpecificObjects(TRIANGULAR_WALL), m_pScene2->GetNumberOfSpecificObjects(TRIANGULAR_WALL))) return false;
	if (!CompareTwoValues("end time points", m_pScene1->GetMaxTime(), m_pScene2->GetMaxTime())) return false;
	return true;
}

bool CResultsComparer::CompareWallForces(double _dTime)
{
	double dMaxTime = m_pScene1->GetMaxTime();
	for (size_t i = 0; i < m_pScene1->GeometriesNumber(); i++)
	{
		CVector3 vForce1(0), vForce2(0);
		for (unsigned int plane : m_pScene1->Geometry(i)->Planes())
			vForce1 += m_pScene1->GetObjectByIndex(plane)->GetForce(dMaxTime);
		for (unsigned int plane : m_pScene2->Geometry(i)->Planes())
			vForce2 += m_pScene1->GetObjectByIndex(plane)->GetForce(dMaxTime);
		if (!CompareTwoValues("forces on walls", vForce1, vForce2, m_dRelTolerance)) return false;
	}
	return true;
}

bool CResultsComparer::CompareKineticEnergies(double _dTime)
{
	double dMaxTime = m_pScene1->GetMaxTime();
	double dKinEnergy1 = 0;
	double dKinEnergy2 = 0;
	std::vector<CSphere*> vAllSpheres1 = m_pScene1->GetAllSpheres(dMaxTime);
	std::vector<CSphere*> vAllSpheres2 = m_pScene2->GetAllSpheres(dMaxTime);
	for (CSphere* pSphere: vAllSpheres1 )
		dKinEnergy1 += pSphere->GetMass()*pSphere->GetVelocity(dMaxTime).SquaredLength();
	for (CSphere* pSphere : vAllSpheres2)
		dKinEnergy2 += pSphere->GetMass()*pSphere->GetVelocity(dMaxTime).SquaredLength();
	if (!CompareTwoValues( "kinetic energies", dKinEnergy1, dKinEnergy2, m_dRelTolerance)) return false;
	return true;
}

bool CResultsComparer::CompareCoordinates(double _dTime)
{
	CVector3 vCOM1 = m_pScene1->GetCenterOfMass(_dTime);
	CVector3 vCOM2 = m_pScene2->GetCenterOfMass(_dTime);
	CVector3 v1 = m_pScene1->GetSimulationDomain().coordEnd - m_pScene1->GetSimulationDomain().coordBeg;
	double dAbsTol = 1e-2*std::max(std::max(v1.x, v1.y), v1.z);
	if (!CompareTwoValues("centers of masses", vCOM1, vCOM2, m_dRelTolerance, dAbsTol)) return false;
	return true;
}

bool CResultsComparer::CompareInterparticleContacts(double _dTime)
{
	// compare coordination numbers
	std::vector<unsigned> vCoordNums1 = m_pScene1->GetCoordinationNumbers(_dTime);
	std::vector<unsigned> vCoordNums2 = m_pScene2->GetCoordinationNumbers(_dTime);
	double dAverCoordNum1 = 0, dAverCoordNum2 = 0;
	if (!vCoordNums1.empty())
		dAverCoordNum1 = std::accumulate(vCoordNums1.begin(), vCoordNums1.end(), 0) / vCoordNums1.size();
	if (!vCoordNums2.empty())
		dAverCoordNum2 = std::accumulate(vCoordNums2.begin(), vCoordNums2.end(), 0) / vCoordNums2.size();
	if (!CompareTwoValues("average coordination number", dAverCoordNum1, dAverCoordNum2, m_dRelTolerance )) return false;

	// compare overlaps
	std::vector<double> vOverlaps1 = m_pScene1->GetMaxOverlaps(_dTime);
	std::vector<double> vOverlaps2 = m_pScene2->GetMaxOverlaps(_dTime);
	double dAverageMaxOverlap1 = 0, dAverageMaxOverlap2 = 0;
	if (!vOverlaps1.empty())
		dAverageMaxOverlap1 = std::accumulate(vOverlaps1.begin(), vOverlaps1.end(), 0) / vOverlaps1.size();
	if (!vOverlaps2.empty())
		dAverageMaxOverlap2 = std::accumulate(vOverlaps2.begin(), vOverlaps2.end(), 0) / vOverlaps2.size();
	if (!CompareTwoValues("average overlap", dAverageMaxOverlap1, dAverageMaxOverlap2, m_dRelTolerance)) return false;
	return true;
}

void CResultsComparer::CompareScenes(std::ofstream& _outStream, CSystemStructure* _pScene1, CSystemStructure* _pScene2)
{
	m_pScene1 = _pScene1;
	m_pScene2 = _pScene2;
	m_pOutStream = &_outStream;
	if (!CompareTimeIndependentData()) return;

	// for init time point
	if (!CompareWallForces(0)) return;
	if (!CompareKineticEnergies(0)) return;
	if (!CompareCoordinates(0)) return;
	if (!CompareInterparticleContacts(0)) return;

	// for last time point
	if (!CompareWallForces(m_pScene1->GetMaxTime())) return;
	if (!CompareKineticEnergies(m_pScene1->GetMaxTime())) return;
	if (!CompareCoordinates(m_pScene1->GetMaxTime())) return;
	if (!CompareInterparticleContacts(m_pScene1->GetMaxTime())) return;
}