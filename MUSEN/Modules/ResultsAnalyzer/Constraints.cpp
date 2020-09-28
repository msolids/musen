/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "Constraints.h"

CConstraints::CConstraints()
{
	m_pSystemStructure = nullptr;
	m_pMaterialsDB = nullptr;
	m_bDiametersActive = false;
	m_bVolumesActive = false;
	m_bMaterialsActive = false;
	m_bGeometriesActive = false;
	ClearVolumes();
	ClearMaterials();
	ClearMaterials2();
	ClearDiameters();
	ClearGeometries();
}

CConstraints::~CConstraints()
{
}

void CConstraints::UpdateSettings()
{
	for(auto g : m_vGeometries)
		if (g >= m_pSystemStructure->GeometriesNumber())
		{
			m_bGeometriesActive = false;
			ClearGeometries();
			break;
		}
	for (auto v : m_vVolumes)
		if (v >= m_pSystemStructure->AnalysisVolumesNumber())
		{
			m_bVolumesActive = false;
			ClearVolumes();
			break;
		}
}

void CConstraints::SetPointers(CSystemStructure* _pSystemStructure, CMaterialsDatabase* _pMaterialsDB)
{
	m_pSystemStructure = _pSystemStructure;
	m_pMaterialsDB = _pMaterialsDB;
}

bool CConstraints::IsAllVolumeSelected() const
{
	return (!m_bVolumesActive || m_vVolumes.empty());
}

bool CConstraints::IsAllMaterialsSelected() const
{
	return (!m_bMaterialsActive || m_MaterialsSet.empty() || m_MaterialsSet.size() == m_pMaterialsDB->CompoundsNumber());
}

bool CConstraints::IsAllMaterials2Selected() const
{
	if (!m_bMaterialsActive)
		return true;
	if (m_MaterialsMap.empty())
		return true;
	if (m_MaterialsMap.size() != m_pMaterialsDB->CompoundsNumber())
		return false;
	for (auto it = m_MaterialsMap.begin(); it != m_MaterialsMap.end(); ++it)
		if (it->second.size() != m_pMaterialsDB->CompoundsNumber())
			return false;
	return true;
}

bool CConstraints::IsAllMaterials2Selected(const std::string& _sKey) const
{
	if (m_MaterialsMap.find(_sKey) == m_MaterialsMap.end())
		return false;
	if (m_MaterialsMap.at(_sKey).size() == m_pMaterialsDB->CompoundsNumber())
		return true;
	return false;
}

bool CConstraints::IsAllGeometriesSelected() const
{
	return (!m_bGeometriesActive || m_vGeometries.empty() || m_vGeometries.size() == m_pSystemStructure->GeometriesNumber());
}

bool CConstraints::IsAllDiametersSelected() const
{
	return (!m_bDiametersActive || m_Diameters.dMax < m_Diameters.dMin || m_Diameters2.dMax < m_Diameters2.dMin );
}

bool CConstraints::IsVolumesActive() const
{
	return m_bVolumesActive;
}

std::vector<unsigned> CConstraints::GetVolumes() const
{
	std::vector<unsigned> vec;
	for (auto it = m_vVolumes.begin(); it != m_vVolumes.end(); ++it)
		vec.push_back(*it);
	return vec;
}

bool CConstraints::IsMaterialsActive() const
{
	return m_bMaterialsActive;
}

std::vector<std::string> CConstraints::GetMaterials() const
{
	std::vector<std::string> vec;
	for (auto it = m_MaterialsSet.begin(); it != m_MaterialsSet.end(); ++it)
		vec.push_back(*it);
	return vec;
}

std::vector<std::string> CConstraints::GetMaterials2(const std::string& _sKey) const
{
	if (m_MaterialsMap.find(_sKey) == m_MaterialsMap.end())
		return std::vector<std::string>();
	std::set<std::string> tmp = m_MaterialsMap.at(_sKey);
	std::vector<std::string> vec;
	for (auto it = tmp.begin(); it != tmp.end(); ++it)
		vec.push_back(*it);
	return vec;
}

std::vector<std::string> CConstraints::GetCommonMaterials() const
{
	if (m_MaterialsMap.size() == 0) return std::vector<std::string>();

	std::set<std::string> commonSet;
	for (unsigned i = 0; i < m_pMaterialsDB->CompoundsNumber(); ++i)
		commonSet.insert(m_pMaterialsDB->GetCompoundKey(i));

	for (auto it = m_MaterialsMap.begin(); it != m_MaterialsMap.end(); ++it)
	{
		std::set<std::string> currSet = it->second;
		size_t nSize = currSet.size();
		std::set<std::string> bufSet;
		std::set_intersection(commonSet.begin(), commonSet.end(), currSet.begin(), currSet.end(), std::inserter(bufSet, bufSet.begin()));
		commonSet = bufSet;
	}

	std::vector<std::string> vec;
	for (auto it = commonSet.begin(); it != commonSet.end(); ++it)
		vec.push_back(*it);
	return vec;
}

bool CConstraints::IsGeometriesActive() const
{
	return m_bGeometriesActive;
}

std::vector<unsigned> CConstraints::GetGeometries() const
{
	std::vector<unsigned> vec;
	for (auto it = m_vGeometries.begin(); it != m_vGeometries.end(); ++it)
		vec.push_back(*it);
	return vec;
}

std::set<size_t> CConstraints::GetGeometriesPlanes() const
{
	std::set<size_t> vPlanes;
	for (auto it = m_vGeometries.begin(); it != m_vGeometries.end(); ++it)
	{
		CRealGeometry* pGeoObj = m_pSystemStructure->Geometry(*it);
		const auto p = pGeoObj->Planes();
		vPlanes.insert(p.begin(), p.end());
	}
	return vPlanes;
}

bool CConstraints::IsDiametersActive() const
{
	return m_bDiametersActive;
}

CConstraints::SInterval CConstraints::GetDiameter() const
{
	return m_Diameters;
}

double CConstraints::GetDiameterMin() const
{
	return m_Diameters.dMin;
}

double CConstraints::GetDiameterMax() const
{
	return m_Diameters.dMax;
}

CConstraints::SInterval CConstraints::GetDiameter2() const
{
	return m_Diameters2;
}

double CConstraints::GetDiameter2Min() const
{
	return m_Diameters2.dMin;
}

double CConstraints::GetDiameter2Max() const
{
	return m_Diameters2.dMax;
}

void CConstraints::SetVolumesActive(bool _bActive)
{
	m_bVolumesActive = _bActive;
}

void CConstraints::AddVolume(unsigned _nIndex)
{
	m_vVolumes.insert(_nIndex);
}

void CConstraints::RemoveVolume(unsigned _nIndex)
{
	m_vVolumes.erase(_nIndex);
}

void CConstraints::SetVolumes(const std::vector<unsigned>& _vIndexes)
{
	m_vVolumes.clear();
	m_vVolumes.insert(_vIndexes.begin(), _vIndexes.end());
}

void CConstraints::ClearVolumes()
{
	m_vVolumes.clear();
}

void CConstraints::SetMateralsActive(bool _bActive)
{
	m_bMaterialsActive = _bActive;
}

void CConstraints::AddMaterial(const std::string& _sKey)
{
	if (_sKey.empty()) return;
	m_MaterialsSet.insert(_sKey);
}

void CConstraints::RemoveMaterial(const std::string& _sKey)
{
	m_MaterialsSet.erase(_sKey);
}

void CConstraints::SetMaterials(const std::vector<std::string>& _vKeys)
{
	m_MaterialsSet.clear();
	m_MaterialsSet.insert(_vKeys.begin(), _vKeys.end());
}

void CConstraints::ClearMaterials()
{
	m_MaterialsSet.clear();
}

void CConstraints::AddMaterial2(const std::string& _sKey)
{
	if (_sKey.empty()) return;

	if ((m_MaterialsMap.find(_sKey) == m_MaterialsMap.end()) || (m_MaterialsMap[_sKey].empty()))
		m_MaterialsMap.insert(std::pair<std::string, std::set<std::string>>(_sKey, std::set<std::string>()));
	for (unsigned i = 0; i < m_pMaterialsDB->CompoundsNumber(); ++i)
		if ((m_MaterialsMap.find(m_pMaterialsDB->GetCompoundKey(i)) == m_MaterialsMap.end()) || (m_MaterialsMap[m_pMaterialsDB->GetCompoundKey(i)].empty()))
			m_MaterialsMap.insert(std::pair<std::string, std::set<std::string>>(m_pMaterialsDB->GetCompoundKey(i), std::set<std::string>()));
	for (unsigned i = 0; i < m_pMaterialsDB->CompoundsNumber(); ++i)
		m_MaterialsMap[_sKey].insert(m_pMaterialsDB->GetCompoundKey(i));
	for (auto it = m_MaterialsMap.begin(); it != m_MaterialsMap.end(); ++it)
		it->second.insert(_sKey);
}

void CConstraints::AddMaterial2(const std::string& _sKey1, const std::string& _sKey2)
{
	if (_sKey1.empty() || _sKey2.empty()) return;

	if ((m_MaterialsMap.find(_sKey1) == m_MaterialsMap.end()) || (m_MaterialsMap[_sKey1].empty()))
		m_MaterialsMap.insert(std::pair<std::string, std::set<std::string>>(_sKey1, std::set<std::string>()));
	if ((m_MaterialsMap.find(_sKey2) == m_MaterialsMap.end()) || (m_MaterialsMap[_sKey2].empty()))
		m_MaterialsMap.insert(std::pair<std::string, std::set<std::string>>(_sKey2, std::set<std::string>()));
	m_MaterialsMap[_sKey1].insert(_sKey2);
	m_MaterialsMap[_sKey2].insert(_sKey1);
}

void CConstraints::RemoveMaterial2(const std::string& _sKey)
{
	for (auto it = m_MaterialsMap.begin(); it != m_MaterialsMap.end(); ++it)
		it->second.erase(_sKey);
	auto it1 = m_MaterialsMap.find(_sKey);
	if (it1 != m_MaterialsMap.end())
		it1->second.clear();
}

void CConstraints::RemoveMaterial2(const std::string& _sKey1, const std::string& _sKey2)
{
	auto it1 = m_MaterialsMap.find(_sKey1);
	if (it1 != m_MaterialsMap.end())
		it1->second.erase(_sKey2);
	auto it2 = m_MaterialsMap.find(_sKey2);
	if (it2 != m_MaterialsMap.end())
		it2->second.erase(_sKey1);
}

void CConstraints::SetMaterials2(const std::string& _sKey1, const std::vector<std::string>& _vKeys2)
{
	if (_sKey1.empty()) return;

	auto it1 = m_MaterialsMap.find(_sKey1);
	if (it1 != m_MaterialsMap.end())
		it1->second.clear();
	if ((m_MaterialsMap.find(_sKey1) == m_MaterialsMap.end()) || (m_MaterialsMap[_sKey1].empty()))
		m_MaterialsMap.insert(std::pair<std::string, std::set<std::string>>(_sKey1, std::set<std::string>()));
	for (unsigned i = 0; i < _vKeys2.size(); ++i)
		if ((m_MaterialsMap.find(_vKeys2[i]) == m_MaterialsMap.end()) || (m_MaterialsMap[_vKeys2[i]].empty()))
			m_MaterialsMap.insert(std::pair<std::string, std::set<std::string>>(_vKeys2[i], std::set<std::string>()));
	for (auto key = _vKeys2.begin(); key != _vKeys2.end(); ++key)
		if (!key->empty())
			m_MaterialsMap[_sKey1].insert(*key);
	for (auto key = _vKeys2.begin(); key != _vKeys2.end(); ++key)
		if (!key->empty())
			m_MaterialsMap[*key].insert(_sKey1);
}

void CConstraints::ClearMaterials2()
{
	m_MaterialsMap.clear();
}

void CConstraints::ClearMaterials2(const std::string& _sKey)
{
	if (m_MaterialsMap.find(_sKey) != m_MaterialsMap.end())
		m_MaterialsMap[_sKey].clear();
}

void CConstraints::SetGeometriesActive(bool _bActive)
{
	m_bGeometriesActive = _bActive;
}

void CConstraints::AddGeometry(unsigned _nIndex)
{
	m_vGeometries.insert(_nIndex);
	m_vGeometriesPlanes = GetGeometriesPlanes();
}

void CConstraints::RemoveGeometry(unsigned _nIndex)
{
	m_vGeometries.erase(_nIndex);
	m_vGeometriesPlanes = GetGeometriesPlanes();
}

void CConstraints::SetGeometries(const std::vector<unsigned>& _vIndexes)
{
	m_vGeometries.clear();
	m_vGeometries.insert(_vIndexes.begin(), _vIndexes.end());
	m_vGeometriesPlanes = GetGeometriesPlanes();
}

void CConstraints::ClearGeometries()
{
	m_vGeometries.clear();
	m_vGeometriesPlanes.clear();
}

void CConstraints::SetDiametersActive(bool _bActive)
{
	m_bDiametersActive = _bActive;
}

void CConstraints::SetDiameter(double _dMin, double _dMax)
{
	m_Diameters.dMin = _dMin;
	m_Diameters.dMax = _dMax;
}

void CConstraints::SetDiameter2(double _dMin, double _dMax)
{
	m_Diameters2.dMin = _dMin;
	m_Diameters2.dMax = _dMax;
}

void CConstraints::ClearDiameters()
{
	m_Diameters.dMin = 0;
	m_Diameters.dMax = 0;
	m_Diameters2.dMin = 0;
	m_Diameters2.dMax = 0;
}

bool CConstraints::CheckVolume(unsigned _nIndex) const
{
	return m_vVolumes.find(_nIndex) != m_vVolumes.end();
}

bool CConstraints::CheckMaterial(const std::string& _sKey) const
{
	return m_MaterialsSet.find(_sKey) != m_MaterialsSet.end();
}

bool CConstraints::CheckMaterial(size_t _nIndex) const
{
	return CheckMaterial(m_pSystemStructure->GetObjectByIndex(_nIndex)->GetCompoundKey());
}

bool CConstraints::CheckMaterial(const std::string& _sKey1, const std::string& _sKey2) const
{
	if (m_MaterialsMap.find(_sKey1) == m_MaterialsMap.end()) return false;
	return m_MaterialsMap.at(_sKey1).find(_sKey2) != m_MaterialsMap.at(_sKey1).end();
}

bool CConstraints::CheckMaterial(size_t _nIndex1, size_t _nIndex2) const
{
	return CheckMaterial(m_pSystemStructure->GetObjectByIndex(_nIndex1)->GetCompoundKey(), m_pSystemStructure->GetObjectByIndex(_nIndex2)->GetCompoundKey());
}

bool CConstraints::CheckGeometry(size_t _nIndex) const
{
	return (m_vGeometriesPlanes.find(_nIndex) != m_vGeometriesPlanes.end());
}

bool CConstraints::CheckDiameter(double _dDiam) const
{
	return ((_dDiam <= m_Diameters.dMax) && (_dDiam >= m_Diameters.dMin));
}

bool CConstraints::CheckDiameter(size_t _nIndex) const
{
	CSphere *pSphere = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(_nIndex));
	return CheckDiameter(pSphere->GetRadius() * 2);
}

bool CConstraints::CheckDiameter(double _dDiam1, double _dDiam2) const
{
	if ((_dDiam1 <= m_Diameters.dMax) && (_dDiam1 >= m_Diameters.dMin) &&
		(_dDiam2 <= m_Diameters2.dMax) && (_dDiam2 >= m_Diameters2.dMin))
		return true;
	else if ((_dDiam1 <= m_Diameters2.dMax) && (_dDiam1 >= m_Diameters2.dMin) &&
		(_dDiam2 <= m_Diameters.dMax) && (_dDiam2 >= m_Diameters.dMin))
		return true;
	return false;
}

bool CConstraints::CheckDiameter(size_t _nIndex1, size_t _nIndex2) const
{
	CSphere *pSphere1 = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(_nIndex1));
	CSphere *pSphere2 = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(_nIndex2));
	return CheckDiameter(pSphere1->GetRadius() * 2, pSphere2->GetRadius() * 2);
}

std::set<size_t> CConstraints::ApplyVolumeFilter(double _dTime, unsigned _nObjType, std::set<size_t>* _vIndexes /*= nullptr*/) const
{
	std::set<size_t> vFiltered;
	for (std::set<unsigned>::iterator it = m_vVolumes.begin(); it != m_vVolumes.end(); ++it)
	{
		std::vector<size_t> vNewIndexes;
		switch (_nObjType)
		{
		case SPHERE:
			vNewIndexes = m_pSystemStructure->AnalysisVolume(*it)->GetParticleIndicesInside(_dTime, false);
			break;
		case SOLID_BOND:
			vNewIndexes = m_pSystemStructure->AnalysisVolume(*it)->GetSolidBondIndicesInside(_dTime);
			break;
		case LIQUID_BOND:
			vNewIndexes = m_pSystemStructure->AnalysisVolume(*it)->GetLiquidBondIndicesInside(_dTime);
			break;
		case TRIANGULAR_WALL:
			vNewIndexes = m_pSystemStructure->AnalysisVolume(*it)->GetWallIndicesInside(_dTime);
			break;
		default: break;
		}
		vFiltered.insert(vNewIndexes.begin(), vNewIndexes.end());
	}
	if (!_vIndexes)
	{
		return vFiltered;
	}
	else
	{
		std::set<size_t> vIntersectFiltered;
		std::set_intersection(vFiltered.begin(), vFiltered.end(), _vIndexes->begin(), _vIndexes->end(), std::inserter(vIntersectFiltered, vIntersectFiltered.begin()));
		return vIntersectFiltered;
	}
}

std::set<size_t> CConstraints::ApplyVolumeFilter(double _dTime, const std::vector<CVector3>& _vCoords) const
{
	std::set<size_t> vFiltered;
	for (std::set<unsigned>::iterator it = m_vVolumes.begin(); it != m_vVolumes.end(); ++it)
	{
		std::vector<size_t> vNewIndexes = m_pSystemStructure->AnalysisVolume(*it)->GetObjectIndicesInside(_dTime, _vCoords);
		vFiltered.insert(vNewIndexes.begin(), vNewIndexes.end());
	}
	return vFiltered;
}

std::set<size_t> CConstraints::ApplyMaterialFilter(double _dTime, unsigned _nObjType, std::set<size_t>* _vIndexes /*= nullptr*/) const
{
	std::set<size_t> vFiltered;
	if (!_vIndexes)
		for (size_t i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); ++i)
		{
			CPhysicalObject* pObj = m_pSystemStructure->GetObjectByIndex(i);
			if (!pObj) continue;
			if (!pObj->IsActive(_dTime)) continue;
			if (pObj->GetObjectType() != _nObjType) continue;
			if (CheckMaterial(pObj->GetCompoundKey()))
				vFiltered.insert(vFiltered.end(), i);
		}
	else
		for (auto it = _vIndexes->begin(); it != _vIndexes->end(); ++it)
		{
			CPhysicalObject* pObj = m_pSystemStructure->GetObjectByIndex(*it);
			if (!pObj) continue;
			if (!pObj->IsActive(_dTime)) continue;
			if (pObj->GetObjectType() != _nObjType) continue;
			if (CheckMaterial(pObj->GetCompoundKey()))
				vFiltered.insert(vFiltered.end(), *it);
		}
	return vFiltered;
}

std::set<size_t> CConstraints::ApplyDiameterFilter(double _dTime, unsigned _nObjType, std::set<size_t>* _vIndexes /*= nullptr*/) const
{
	std::set<size_t> vFiltered;
	if (!_vIndexes)
		for (size_t i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); ++i)
		{
			CPhysicalObject* pObj = m_pSystemStructure->GetObjectByIndex(i);
			if (!pObj) continue;
			if (!pObj->IsActive(_dTime)) continue;
			if (pObj->GetObjectType() != _nObjType) continue;
			switch (_nObjType)
			{
			case SPHERE:
			{
				auto* pSphere = dynamic_cast<CSphere*>(pObj);
				if (!pSphere) continue;
				if (CheckDiameter(pSphere->GetRadius() * 2))
					vFiltered.insert(vFiltered.end(), i);
				break;
			}
			case SOLID_BOND:
			case LIQUID_BOND:
			{
				auto* pBond = dynamic_cast<CBond*>(pObj);
				if (!pBond) continue;
				if (CheckDiameter(pBond->GetDiameter()))
					vFiltered.insert(vFiltered.end(), i);
				break;
			}
			default: ;
				vFiltered.insert(vFiltered.end(), i);
			}
		}
	else
		for (auto it = _vIndexes->begin(); it != _vIndexes->end(); ++it)
		{
			CPhysicalObject* pObj = m_pSystemStructure->GetObjectByIndex(*it);
			if (!pObj) continue;
			if (!pObj->IsActive(_dTime)) continue;
			if (pObj->GetObjectType() != _nObjType) continue;
			switch (_nObjType)
			{
			case SPHERE:
			{
				auto* pSphere = dynamic_cast<CSphere*>(pObj);
				if (!pSphere) continue;
				if (CheckDiameter(pSphere->GetRadius() * 2))
					vFiltered.insert(vFiltered.end(), *it);
			}
			case SOLID_BOND:
			case LIQUID_BOND:
			{
				auto* pBond = dynamic_cast<CBond*>(pObj);
				if (!pBond) continue;
				if (CheckDiameter(pBond->GetDiameter()))
					vFiltered.insert(vFiltered.end(), *it);
				break;
			}
			default:;
				vFiltered.insert(vFiltered.end(), *it);
			}
		}
	return vFiltered;
}

std::vector<size_t> CConstraints::FilteredParticles(double _dTime) const
{
	std::set<size_t> vFilteredParticles;
	std::set<size_t> *pResVector = nullptr;

	if (!IsAllVolumeSelected())
	{
		vFilteredParticles = ApplyVolumeFilter(_dTime, SPHERE);
		pResVector = &vFilteredParticles;
	}

	if(!IsAllMaterialsSelected())
	{
		vFilteredParticles = ApplyMaterialFilter(_dTime, SPHERE, pResVector);
		pResVector = &vFilteredParticles;
	}

	if(!IsAllDiametersSelected())
	{
		vFilteredParticles = ApplyDiameterFilter(_dTime, SPHERE, pResVector);
		pResVector = &vFilteredParticles;
	}

	if (!pResVector)
	{
		for (size_t i = 0; i<m_pSystemStructure->GetTotalObjectsCount(); ++i)
		{
			CPhysicalObject* pObj = m_pSystemStructure->GetObjectByIndex(i);
			if (!pObj) continue;
			if (!pObj->IsActive(_dTime)) continue;
			if (pObj->GetObjectType() != SPHERE) continue;
			vFilteredParticles.insert(vFilteredParticles.end(), i);
		}
	}

	return std::vector<size_t>(vFilteredParticles.begin(), vFilteredParticles.end());
}

std::vector<size_t> CConstraints::FilteredBonds(double _dTime) const
{
	std::set<size_t> vFilteredBonds;
	std::set<size_t> *pResVector = nullptr;

	if (!IsAllVolumeSelected())
	{
		vFilteredBonds = ApplyVolumeFilter( _dTime, SOLID_BOND );
		pResVector = &vFilteredBonds;
	}

	if (!IsAllMaterialsSelected())
	{
		vFilteredBonds = ApplyMaterialFilter( _dTime, SOLID_BOND, pResVector );
		pResVector = &vFilteredBonds;
	}

	if (!IsAllDiametersSelected())
	{
		vFilteredBonds = ApplyDiameterFilter( _dTime, SOLID_BOND, pResVector );
		pResVector = &vFilteredBonds;
	}

	if (!pResVector)
	{
		for (size_t i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); ++i)
		{
			CPhysicalObject* pObj = m_pSystemStructure->GetObjectByIndex(i);
			if ( (!pObj) || (!pObj->IsActive( _dTime )) || (pObj->GetObjectType() != SOLID_BOND) ) continue;
			vFilteredBonds.insert(vFilteredBonds.end(), i);
		}
	}

	return std::vector<size_t>(vFilteredBonds.begin(), vFilteredBonds.end());
}