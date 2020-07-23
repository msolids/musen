/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "SystemStructure.h"

CSystemStructure::CSystemStructure()
{
	Reset();
}

CSystemStructure::~CSystemStructure()
{
	DeleteAllObjects();
}

void CSystemStructure::Reset()
{
	m_storage.reset(new CDemStorage());
	m_sDEMFileName.clear();
	ClearAllData();
}

void CSystemStructure::ClearAllData()
{
	m_SimulationDomain.coordBeg.Init(-5e-3);
	m_SimulationDomain.coordEnd.Init(5e-3);

	m_PBC.SetDefaultValues();

	m_bAnisotropy = false;
	m_bContactRadius = false;

	m_MaterialDatabase.ClearData();
}

void CSystemStructure::DeleteObjects(const std::vector<size_t>& _vIndexes)
{
	std::vector<std::vector<unsigned>> vConnectedBonds;
	vConnectedBonds.resize(objects.size());
	for (unsigned i = 0; i < objects.size(); i++)
		if (objects[i] != NULL)
			if ((objects[i]->GetObjectType() == SOLID_BOND) || (objects[i]->GetObjectType() == LIQUID_BOND))
			{
				vConnectedBonds[((CBond*)objects[i])->m_nLeftObjectID].push_back(i);
				vConnectedBonds[((CBond*)objects[i])->m_nRightObjectID].push_back(i);
			}

	for (size_t i = 0; i < _vIndexes.size(); i++)
	{
		size_t nObjIndex = _vIndexes[i];
		if (objects[nObjIndex] == NULL) continue;

		// remove all connected bonds
		if (objects[_vIndexes[i]]->GetObjectType() == SPHERE) // delete all bonds which connects this particle
			for (size_t j = 0; j < vConnectedBonds[nObjIndex].size(); j++)
				if (objects[vConnectedBonds[nObjIndex][j]] != NULL)
				{
					delete objects[vConnectedBonds[nObjIndex][j]];
					objects[vConnectedBonds[nObjIndex][j]] = NULL;
					m_storage->RemoveObject(vConnectedBonds[nObjIndex][j]);
				}
		// delete object itself
		delete objects[_vIndexes[i]];
		objects[_vIndexes[i]] = NULL;
		m_storage->RemoveObject(_vIndexes[i]);
	}
}

void CSystemStructure::DeleteAllObjects()
{
	DeleteAllGeometries();
	for (unsigned long i = 0; i < objects.size(); i++)
		if (objects[i] != NULL)
		{
			delete objects[i];
			m_storage->RemoveObject(i);
		}
	for (unsigned long i = 0; i < m_AnalysisVolumes.size(); i++)
		delete m_AnalysisVolumes[i];
	m_AnalysisVolumes.clear();
	m_Multispheres.clear();
	objects.clear();
}

void CSystemStructure::DeleteAllBonds()
{
	for (unsigned long i = 0; i < objects.size(); i++)
		if (objects[i] != NULL)
			if ((objects[i]->GetObjectType() == SOLID_BOND) || (objects[i]->GetObjectType() == LIQUID_BOND))
			{
				delete objects[i];
				objects[i] = NULL;
				m_storage->RemoveObject(i);
			}
	Compress();
}

void CSystemStructure::Compress()
{
	// delete all empty objects from the end of array
	while (!objects.empty())
		if (objects.back() == NULL)
			objects.pop_back();
		else
			return;
}

void CSystemStructure::DeleteAllParticles()
{
	for (unsigned long i = 0; i < objects.size(); i++)
		if (objects[i] != NULL)
			if ((objects[i]->GetObjectType() == SPHERE))
			{
				delete objects[i];
				objects[i] = NULL;
				m_storage->RemoveObject(i);
			}
	DeleteAllBonds();
	m_Multispheres.clear();
	Compress();
	ClearAllStatesFrom(0);
}

size_t CSystemStructure::DeleteAllNonConnectedParticles()
{
	std::vector<size_t> vPartToDelete;
	std::vector<size_t> vConnectedParticles;
	for (unsigned long i = 0; i < objects.size(); i++)
	{
		if (objects[i] == NULL) continue;
		if (objects[i]->GetObjectType() == SPHERE)
		{
			vPartToDelete.push_back(i);
		}
		else if ((objects[i]->GetObjectType() == SOLID_BOND) || (objects[i]->GetObjectType() == LIQUID_BOND))
		{
			vConnectedParticles.push_back(((CBond*)objects[i])->m_nLeftObjectID);
			vConnectedParticles.push_back(((CBond*)objects[i])->m_nRightObjectID);
		}
	}
	for (size_t i = 0; i < vConnectedParticles.size(); i++)
	{
		std::vector<size_t>::iterator it = std::find(vPartToDelete.begin(), vPartToDelete.end(), vConnectedParticles[i]);
		if (it != vPartToDelete.end())
		{
			*it = vPartToDelete.back();
			vPartToDelete.pop_back();
		}
	}
	for (unsigned long i = 0; i < vPartToDelete.size(); i++)
	{
		delete objects[vPartToDelete[i]];
		objects[vPartToDelete[i]] = NULL;
		m_storage->RemoveObject(vPartToDelete[i]);
	}

	Compress();
	ClearAllStatesFrom(0);
	return vPartToDelete.size();
}

size_t CSystemStructure::DeleteAllParticlesWithNoContacts()
{
	std::vector<unsigned> vCoordNumbers = GetCoordinationNumbers(0);
	size_t nRemovedParts = 0;
	for (unsigned long i = 0; i < vCoordNumbers.size(); i++)
	{
		if ((objects[i] != NULL) && (objects[i]->GetObjectType()==SPHERE) && (vCoordNumbers[i] == 0) )
		{
			delete objects[i];
			objects[i] = NULL;
			m_storage->RemoveObject(i);
			nRemovedParts++;
		}
	}

	Compress();
	ClearAllStatesFrom(0);
	return nRemovedParts;
}

size_t CSystemStructure::GetTotalObjectsCount() const
{
	return objects.size();
}

size_t CSystemStructure::GetNumberOfSpecificObjects(unsigned _nObjectType)
{
	size_t nObjectsNumber = 0;
	for (auto& obj : objects)
		if (obj && obj->GetObjectType() == _nObjectType)
			nObjectsNumber++;
	return nObjectsNumber;
}

void CSystemStructure::GetAllObjectsOfSpecifiedCompound(double _dTime, std::vector<CPhysicalObject*>* _vecIndexes, unsigned _nObjectType, const std::string& _sCompoundKey/*="" */)
{
	_vecIndexes->clear();
	for (unsigned i = 0; i < objects.size(); i++)
		if (objects[i] != NULL)
			if (objects[i]->GetObjectType() == _nObjectType)
				if (objects[i]->IsActive(_dTime))
					if (_sCompoundKey.empty()) // consider all spheres
						_vecIndexes->push_back(objects[i]);
					else
						if (objects[i]->GetCompoundKey() == _sCompoundKey)
							_vecIndexes->push_back(objects[i]);
}

std::vector<CPhysicalObject*> CSystemStructure::GetAllActiveObjects(double _dTime, unsigned _nObjectType)
{
	std::vector<CPhysicalObject*> vResult;
	for (unsigned i = 0; i < objects.size(); i++)
		if ((objects[i] != NULL) && (objects[i]->IsActive(_dTime)))
			if ((_nObjectType != UNKNOWN_OBJECT) && (objects[i]->GetObjectType() == _nObjectType))
				vResult.push_back(objects[i]);
	return vResult;
}

std::vector<CSphere*> CSystemStructure::GetAllSpheres(double _time, bool _onlyActive/* = true*/)
{
	std::vector<CSphere*> res;
	for (auto& o : objects)
		if (o && dynamic_cast<CSphere*>(o) && (_onlyActive && o->IsActive(_time) || !_onlyActive))
			res.push_back(dynamic_cast<CSphere*>(o));
	return res;
}

std::vector<CSolidBond*> CSystemStructure::GetAllSolidBonds(double _time, bool _onlyActive/* = true*/)
{
	std::vector<CSolidBond*> res;
	for (auto& o : objects)
		if (o && dynamic_cast<CSolidBond*>(o) && (_onlyActive && o->IsActive(_time) || !_onlyActive))
			res.push_back(dynamic_cast<CSolidBond*>(o));
	return res;
}

std::vector<CLiquidBond*> CSystemStructure::GetAllLiquidBonds(double _time, bool _onlyActive/* = true*/)
{
	std::vector<CLiquidBond*> res;
	for (auto& o : objects)
		if (o && dynamic_cast<CLiquidBond*>(o) && (_onlyActive && o->IsActive(_time) || !_onlyActive))
			res.push_back(dynamic_cast<CLiquidBond*>(o));
	return res;
}

std::vector<CBond*> CSystemStructure::GetAllBonds(double _time, bool _onlyActive/* = true*/)
{
	std::vector<CBond*> res;
	for (auto& o : objects)
		if (o && dynamic_cast<CBond*>(o) && (_onlyActive && o->IsActive(_time) || !_onlyActive))
			res.push_back(dynamic_cast<CBond*>(o));
	return res;
}

std::vector<CTriangularWall*> CSystemStructure::GetAllWalls(double _time, bool _onlyActive/* = true*/)
{
	std::vector<CTriangularWall*> res;
	for (auto& o : objects)
		if (o && dynamic_cast<CTriangularWall*>(o) && (_onlyActive && o->IsActive(_time) || !_onlyActive))
			res.push_back(dynamic_cast<CTriangularWall*>(o));
	return res;
}

std::vector<CTriangularWall*> CSystemStructure::GetAllWallsForGeometry(double _time, const std::string& _geometryKey, bool _onlyActive/* = true*/)
{
	const auto geometry = GetGeometry(_geometryKey);
	std::vector<CTriangularWall*> res;
	res.reserve(geometry->vPlanes.size());
	for (size_t iWall : geometry->vPlanes)
	{
		auto* wall = dynamic_cast<CTriangularWall*>(GetObjectByIndex(iWall));
		if (wall && (_onlyActive && wall->IsActive(_time) || !_onlyActive))
			res.push_back(wall);
	}
	return res;
}

bool CSystemStructure::IsParticlesExist() const
{
	for (auto& obj : objects)
		if (obj && obj->GetObjectType() == SPHERE)
			return true;
	return false;
}

bool CSystemStructure::IsSolidBondsExist() const
{
	for (auto& obj : objects)
		if (obj && obj->GetObjectType() == SOLID_BOND)
			return true;
	return false;
}

bool CSystemStructure::IsLiquidBondsExist() const
{
	for (auto& obj : objects)
		if (obj && obj->GetObjectType() == LIQUID_BOND)
			return true;
	return false;
}

bool CSystemStructure::IsBondsExist() const
{
	for (auto& obj : objects)
		if (obj && (obj->GetObjectType() == SOLID_BOND || obj->GetObjectType() == LIQUID_BOND))
			return true;
	return false;
}

bool CSystemStructure::IsWallsExist() const
{
	for (auto& obj : objects)
		if (obj && obj->GetObjectType() == TRIANGULAR_WALL)
			return true;
	return false;
}

CPhysicalObject* CSystemStructure::AddObject(unsigned _objectType)
{
	// if some element is empty
	for (size_t i = 0; i < objects.size(); ++i)
		if (objects[i] == nullptr)
			return AddObject(_objectType, i);
	// add to the end
	return AddObject(_objectType, objects.size());
}

CPhysicalObject* CSystemStructure::AddObject(unsigned _objectType, size_t _nObjectID)
{
	CPhysicalObject* newObject = nullptr;

	// increase the size of the objects
	while (objects.size() <= _nObjectID)
		objects.push_back(nullptr);

	// if the object was already allocated
	if (objects[_nObjectID] != nullptr)
		return objects[_nObjectID];

	// allocate new object
	switch (_objectType)
	{
	case SPHERE:		  newObject = new CSphere(_nObjectID, &*m_storage); break;
	case SOLID_BOND:	  newObject = new CSolidBond(_nObjectID, &*m_storage); break;
	case LIQUID_BOND:	  newObject = new CLiquidBond(_nObjectID, &*m_storage); break;
	case TRIANGULAR_WALL: newObject = new CTriangularWall(_nObjectID, &*m_storage); break;
	default:
		throw std::invalid_argument("SystemStructure::AddObject - Unrecognized object type ");
	}
	objects[_nObjectID] = newObject;
	return newObject;
}

std::vector<CPhysicalObject*> CSystemStructure::AddSeveralObjects(unsigned _objectType, size_t _nObjectsCount)
{
	std::vector<size_t> vFreeIDs = GetFreeIDs(_nObjectsCount);
	std::vector<CPhysicalObject*> vObjects;
	vObjects.reserve(_nObjectsCount);
	for (size_t i = 0; i < _nObjectsCount; ++i)
		vObjects.push_back(AddObject(_objectType, vFreeIDs[i]));
	return vObjects;
}

std::vector<size_t> CSystemStructure::GetFreeIDs(size_t _objectsCount)
{
	std::vector<size_t> vFreeIDs;
	vFreeIDs.reserve(_objectsCount);
	// if some elements are empty, fill them first
	for (size_t i = 0; i < objects.size() && vFreeIDs.size() != _objectsCount; ++i)
		if (!objects[i])
			vFreeIDs.push_back(i);
	// if it is not enough, attach some to the end
	size_t id = objects.size();
	while (vFreeIDs.size() != _objectsCount)
		vFreeIDs.push_back(id++);
	return vFreeIDs;
}

void CSystemStructure::AddMultisphere(const std::vector<size_t>& _vIndexes)
{
	m_Multispheres.push_back(_vIndexes);
}

double CSystemStructure::GetRecommendedTimeStep(double _dTime /*= 0*/) const
{
	// calculate Rayleigh time step
	std::vector<double> steps(objects.size(), 10);
	if (steps.empty()) return {};
	PrepareTimePointForRead(_dTime);
	ParallelFor(objects.size(), [&](size_t i)
	{
		if (!objects[i]) return;
		const CCompound* compound = m_MaterialDatabase.GetCompound(objects[i]->GetCompoundKey());
		if (!compound) return;

		if (auto* part = dynamic_cast<CSphere*>(objects[i]))	// for particles
			steps[i] = PI * part->GetContactRadius() *
				sqrt(2 * compound->GetPropertyValue(PROPERTY_DENSITY) * (1 + compound->GetPropertyValue(PROPERTY_POISSON_RATIO)) /
					compound->GetPropertyValue(PROPERTY_YOUNG_MODULUS)) / (0.1631 * compound->GetPropertyValue(PROPERTY_POISSON_RATIO) + 0.8766);
		else if (auto* bond = dynamic_cast<CBond*>(objects[i]))	// for bonds
		{
			// find the smallest mass of contact partner
			auto* part1 = objects[bond->m_nLeftObjectID];
			auto* part2 = objects[bond->m_nRightObjectID];
			const double dMinMass = std::min(part1->GetMass(), part2->GetMass());
			const double dCrossCut = PI * std::pow(bond->GetDiameter(), 2) / 4.0;
			const double dYoungModulus = compound->GetPropertyValue(PROPERTY_YOUNG_MODULUS);
			const CVector3 bondVec = GetBond(_dTime, i);
			if (!bondVec.IsSignificant()) return;
			const double dLength = bondVec.Length();
			const double dPoisson = compound->GetPropertyValue(PROPERTY_POISSON_RATIO);
			const double KbMax = std::max(dYoungModulus*dCrossCut / dLength, dYoungModulus * dCrossCut / (2 * dLength * (1 + dPoisson)));
			steps[i] = 2 * std::sqrt(dMinMass / KbMax);
		}
	});

	// calculate time step for bonds (see Documentation)
	return *std::min_element(steps.begin(), steps.end()) * 0.1;
}

void CSystemStructure::ImportFromEDEMTextFormat(const std::string& _sFileName)
{
	if (_sFileName.empty()) return;
	ClearAllStatesFrom(0);
	DeleteAllObjects();

	std::ifstream inFile;
	inFile.open(UnicodePath(_sFileName)); //open a file
	std::string line, sTemp;
	int nTemp;

	while (safeGetLine(inFile, line).good() && line != "EXTRACTED DATA");	// skip headers

																			// get all IDs
	std::set<size_t> IDs;
	while (!inFile.eof())
	{
		if (findStringAfter(inFile, ": Particle ID: ", sTemp) == std::string::npos) // cannot find IDs
			continue;
		if (sTemp == "no data") continue;	// data for this TP is not available
											// get all IDs fro string
		std::stringstream tempStream;
		tempStream << sTemp;
		while (tempStream.good())
		{
			tempStream >> nTemp;
			if (nTemp < 0) continue;
			IDs.insert(nTemp);
		}
	}

	// Create all spheres
	for (std::set<size_t>::iterator it = IDs.begin(); it != IDs.end(); ++it)
		AddObject(SPHERE, *it);

	// navigate to the file begin
	inFile.seekg(0, std::ios::beg);
	while (safeGetLine(inFile, line).good() && line != "EXTRACTED DATA");	// skip headers

	std::set<size_t> vNotYetActive = IDs;
	std::set<size_t> vActive;

	double dTimePoint = 0, dPrevTimePoint = 0, dTimeShift = -1;
	std::stringstream tempStream;
	while (!inFile.eof())
	{
		dPrevTimePoint = dTimePoint;
		std::vector<size_t> vID;

		if ((findStringAfter(inFile, "TIME: ", sTemp) == std::string::npos) || (sTemp == "no data")) continue;
		tempStream << sTemp;
		tempStream >> dTimePoint;
		tempStream.clear();

		getVecFromFile(inFile, "Particle ID: ", vID);
		if (vID.empty()) continue; // no particles at this time point

		std::vector<double> vDiam, vCoordX, vCoordY, vCoordZ, vVeloX, vVeloY, vVeloZ, vAngVx, vAngVy, vAngVz;
		getVecFromFile(inFile, "Particle Diameter: ", vDiam);
		getVecFromFile(inFile, "Particle Position X: ", vCoordX);
		getVecFromFile(inFile, "Particle Position Y: ", vCoordY);
		getVecFromFile(inFile, "Particle Position Z: ", vCoordZ);
		getVecFromFile(inFile, "Particle Velocity X: ", vVeloX);
		getVecFromFile(inFile, "Particle Velocity Y: ", vVeloY);
		getVecFromFile(inFile, "Particle Velocity Z: ", vVeloZ);

		getVecFromFile(inFile, "Particle Angular Velocity X: ", vAngVx);
		getVecFromFile(inFile, "Particle Angular Velocity Y: ", vAngVy);
		getVecFromFile(inFile, "Particle Angular Velocity Z: ", vAngVz);

		if (dTimeShift == -1)
			dTimeShift = dTimePoint;

		for (size_t i = 0; i < vID.size(); ++i)
		{
			dynamic_cast<CSphere*>(objects[vID[i]])->SetRadius(vDiam[i] / 2);
			CVector3 vec(vCoordX[i], vCoordY[i], vCoordZ[i]);
			objects[vID[i]]->SetCoordinates(dTimePoint - dTimeShift, vec);
			if (!vVeloX.empty() && !vVeloY.empty() && !vVeloZ.empty())
			{
				vec.Init(vVeloX[i], vVeloY[i], vVeloZ[i]);
				objects[vID[i]]->SetVelocity(dTimePoint - dTimeShift, vec);
				vec.Init(vAngVx[i], vAngVy[i], vAngVz[i]);
				objects[vID[i]]->SetAngleVelocity(dTimePoint - dTimeShift, vec);
			}
		}

		std::set<size_t> vCurrent(vID.begin(), vID.end());
		std::set<size_t> vToActivate = SetIntersection(vNotYetActive, vCurrent);
		for (std::set<size_t>::iterator it = vToActivate.begin(); it != vToActivate.end(); ++it)
			objects[*it]->SetStartActivityTime(dTimePoint - dTimeShift);
		std::set<size_t> vToDeactivate = SetDifference(vActive, vCurrent);
		for (std::set<size_t>::iterator it = vToDeactivate.begin(); it != vToDeactivate.end(); ++it)
			objects[*it]->SetEndActivityTime(dPrevTimePoint - dTimeShift);
		vActive = SetDifference(vActive, vToDeactivate);
		vActive = SetUnion(vActive, vToActivate);
		vNotYetActive = SetDifference(vNotYetActive, vToActivate);
	}

	UpdateAllObjectsCompoundsProperties();
}

void CSystemStructure::ExportGeometriesToSTL(const std::string& _sFileName, double _dTime)
{
	if (m_Geometries.empty()) return;

	CTriangularMesh geometry;
	geometry.sName = m_sDEMFileName + " " + std::to_string(_dTime) + "s";
	for (size_t i = 0; i < m_Geometries.size(); ++i)
	{
		for (size_t j = 0; j < m_Geometries[i]->vPlanes.size(); ++j)
		{
			CTriangularWall* pWall = static_cast<CTriangularWall*>(objects[m_Geometries[i]->vPlanes[j]]);
			if (!pWall || !pWall->IsActive(_dTime)) continue;
			geometry.vTriangles.push_back(pWall->GetCoords(_dTime));
		}
	}
	CSTLFileHandler writer;
	writer.WriteToFile(geometry, _sFileName);
}

void CSystemStructure::CreateFromSystemStructure(CSystemStructure* _pSource, double _dTime)
{
	m_MaterialDatabase = _pSource->m_MaterialDatabase;

	DeleteAllObjects();
	SetSimulationDomain(_pSource->GetSimulationDomain());
	SetPBC(_pSource->GetPBC());
	EnableAnisotropy(_pSource->IsAnisotropyEnabled());
	EnableContactRadius(_pSource->IsContactRadiusEnabled());

	for (unsigned i = 0; i < _pSource->GetGeometriesNumber(); i++)
	{
		SGeometryObject* pNewObject = AddGeometry();
		*pNewObject = *_pSource->GetGeometry(i);
	}

	for (unsigned i = 0; i < _pSource->GetAnalysisVolumesNumber(); ++i)
	{
		CAnalysisVolume* pNewVolume = AddAnalysisVolume();
		*pNewVolume = *_pSource->GetAnalysisVolume(i);
	}

	bool bIsNewFormat = false;
	for (size_t i = 0; i < _pSource->GetTotalObjectsCount(); i++)
	{
		CPhysicalObject* pOldObject = _pSource->GetObjectByIndex(i);
		if (pOldObject == NULL) continue;
		if (!pOldObject->IsActive(_dTime)) continue;

		if (pOldObject->IsQuaternionSet(_dTime)) bIsNewFormat = true; // if _pSource is new format file

		CPhysicalObject* pNewObject = AddObject(pOldObject->GetObjectType(), i);

		pNewObject->SetObjectActivity(0, pOldObject->IsActive(_dTime));
		pNewObject->SetTemperature(0, pOldObject->GetTemperature(_dTime));
		pNewObject->SetCompoundKey(pOldObject->GetCompoundKey());

		switch (pNewObject->GetObjectType())
		{
		case SPHERE:
		{
			pNewObject->SetCoordinates(0, pOldObject->GetCoordinates(_dTime));
			pNewObject->SetVelocity(0, pOldObject->GetVelocity(_dTime));
			pNewObject->SetAngleVelocity(0, pOldObject->GetAngleVelocity(_dTime));
			pNewObject->SetForce(0, pOldObject->GetForce(_dTime));
			if (bIsNewFormat)
				pNewObject->SetOrientation(0, pOldObject->GetOrientation(_dTime));
			else
				pNewObject->SetOrientation(0, CQuaternion(pOldObject->GetAngles(_dTime)));

			pNewObject->SetObjectGeometryBin(pOldObject->GetObjectGeometryBin());
			break;
		}
		case SOLID_BOND:
		{
			CSolidBond* pSBond = static_cast<CSolidBond*>(_pSource->GetObjectByIndex(i));
			CSolidBond* pSBondNew = static_cast<CSolidBond*>(this->GetObjectByIndex(i));

			pSBondNew->SetForce(0, pSBond->GetForce(_dTime));
			pSBondNew->SetTotalTorque(0, pSBond->GetTotalTorque(_dTime));
			if (bIsNewFormat)
				pSBondNew->SetTangentialOverlap(0, pSBond->GetTangentialOverlap(_dTime));
			else
				pSBondNew->SetTangentialOverlap(0, pSBond->GetOldTangentialOverlap(_dTime));

			pNewObject->SetObjectGeometryBin(pOldObject->GetObjectGeometryBin());
			break;
		}
		case LIQUID_BOND:
		{
			CSolidBond* pSBond = static_cast<CSolidBond*>(_pSource->GetObjectByIndex(i));
			CSolidBond* pSBondNew = static_cast<CSolidBond*>(this->GetObjectByIndex(i));

			pSBondNew->SetForce(0, pSBond->GetForce(_dTime));

			pNewObject->SetObjectGeometryBin(pOldObject->GetObjectGeometryBin());
			break;
		}
		case TRIANGULAR_WALL:
		{
			CTriangularWall* pWall = static_cast<CTriangularWall*>(_pSource->GetObjectByIndex(i));
			CTriangularWall* pWallNew = static_cast<CTriangularWall*>(this->GetObjectByIndex(i));

			pWallNew->SetForce(0, pWall->GetForce(_dTime));
			pWallNew->SetVelocity(0, pWall->GetVelocity(_dTime));
			if (bIsNewFormat)
				pWallNew->SetPlaneCoord(0, pWall->GetCoordVertex1(_dTime), pWall->GetCoordVertex2(_dTime), pWall->GetCoordVertex3(_dTime));
			else
				pWallNew->SetPlaneCoord(0, pWall->GetCoordVertex1(_dTime), pWall->GetOldCoordVertex2(_dTime), pWall->GetOldCoordVertex3(_dTime));
			break;
		}
		default:break;
		}
	}

	UpdateAllObjectsCompoundsProperties();
}

void CSystemStructure::UpdateObjectsCompoundProperties(const std::vector<size_t>& _IDs)
{
	ParallelFor(_IDs.size(), [&](size_t i)
	{
		if (!objects[_IDs[i]]) return;
		if (CCompound* compound = m_MaterialDatabase.GetCompound(objects[_IDs[i]]->GetCompoundKey()))
		{
			objects[_IDs[i]]->SetColor(compound->GetColor());
			objects[_IDs[i]]->UpdateCompoundProperties(compound);
		}
	});
}

void CSystemStructure::UpdateAllObjectsCompoundsProperties()
{
	std::vector<size_t> IDs(objects.size());	// vector with all IDs.
	std::iota(IDs.begin(), IDs.end(), 0);		// fill with 0, 1, 2,...
	UpdateObjectsCompoundProperties(IDs);
}

std::set<std::string> CSystemStructure::GetAllParticlesCompounds() const
{
	std::set<std::string> res;
	for (const auto o : objects)
		if (const auto* p = dynamic_cast<const CSphere*>(o))
			res.insert(p->GetCompoundKey());
	return { res.begin(), res.end() };
}

std::set<std::string> CSystemStructure::GetAllBondsCompounds() const
{
	std::set<std::string> res;
	for (const auto o : objects)
		if (const auto* p = dynamic_cast<const CBond*>(o))
			res.insert(p->GetCompoundKey());
	return { res.begin(), res.end() };
}

std::vector<size_t> CSystemStructure::GetParticlesOutsideVolume(double _dTime, const CVector3& vBottomLeft, const CVector3& vTopRight) const
{
	std::vector<size_t> vResultVector;
	for (size_t i = 0; i < objects.size(); ++i)
		if (objects[i] && (objects[i]->GetObjectType() == SPHERE))
		{
			double dRadius = ((CSphere*)objects[i])->GetRadius();
			CVector3 vCoord = objects[i]->GetCoordinates(_dTime);
			if ((vCoord.x - dRadius < vBottomLeft.x) || (vCoord.y - dRadius < vBottomLeft.y) || (vCoord.z - dRadius < vBottomLeft.z))
				vResultVector.push_back(i);
			else if ((vCoord.x + dRadius > vTopRight.x) || (vCoord.y + dRadius > vTopRight.y) || (vCoord.z + dRadius > vTopRight.z))
				vResultVector.push_back(i);
		}
	return vResultVector;
}

void CSystemStructure::ClearAllStatesFrom(double _dTime)
{
	std::vector<size_t> vObjectsToDelete;
	double dStart, dEnd;
	for (size_t i = 0; i < objects.size(); i++)
	{
		if (objects[i] == NULL) continue;
		objects[i]->GetActivityTimeInterval(&dStart, &dEnd);
		if (dStart > _dTime)
			vObjectsToDelete.push_back(i);
	}
	m_storage->RemoveAllAfterTime(_dTime);
	DeleteObjects(vObjectsToDelete);
}

double CSystemStructure::GetMinParticleDiameter() const
{
	double dSmallestRadius = -1;
	for (unsigned i = 0; i < objects.size(); i++)
		if (objects[i] != NULL)
			if (objects[i]->GetObjectType() == SPHERE)
				if (dSmallestRadius == -1)
					dSmallestRadius = ((CSphere*)objects[i])->GetRadius();
				else
					if (dSmallestRadius > ((CSphere*)objects[i])->GetRadius())
						dSmallestRadius = ((CSphere*)objects[i])->GetRadius();
	return dSmallestRadius * 2;
}

double CSystemStructure::GetMaxParticleDiameter() const
{
	double dMaxRadius = -1;
	for (unsigned i = 0; i < objects.size(); i++)
		if (objects[i] != NULL)
			if (objects[i]->GetObjectType() == SPHERE)
				if (dMaxRadius == -1)
					dMaxRadius = ((CSphere*)objects[i])->GetRadius();
				else
					if (dMaxRadius < ((CSphere*)objects[i])->GetRadius())
						dMaxRadius = ((CSphere*)objects[i])->GetRadius();
	return dMaxRadius * 2;
}

std::vector<unsigned> CSystemStructure::GetFragmentsPSD(double _dTime, double _dMin, double _dMax, unsigned _nClasses)
{
	std::vector<unsigned> vecResult(_nClasses, 0);

	std::vector<unsigned> vecConsideredParticles; // 0 - particle was not considered, 1 - particle was considered
	vecConsideredParticles.resize(objects.size(), 0);
	double dClassSize = (_dMax - _dMin) / _nClasses;

	for (unsigned i = 0; i < objects.size(); i++)
	{
		if (objects[i] == NULL) continue;
		if ((objects[i]->GetObjectType() != SPHERE) || (objects[i]->IsActive(_dTime) == false)) continue;

		if (vecConsideredParticles[i] == 0)
		{
			std::vector<size_t> vecParticlesChain;
			GetGroup(_dTime, i, &vecParticlesChain);
			double VParticleTot = 0; // the volume of accorded particle
			for (size_t j = 0; j < vecParticlesChain.size(); j++)
			{
				vecConsideredParticles[vecParticlesChain[j]] = 1;
				VParticleTot += ((CSphere*)objects[vecParticlesChain[j]])->GetVolume();
				for (size_t j1 = 0; j1 < objects.size(); j1++)
					if (objects[j1] != NULL)
						if ((objects[j1]->GetObjectType() == SOLID_BOND) && (objects[j1]->IsActive(_dTime) == true))
							if (((CSolidBond*)objects[j1])->m_nLeftObjectID == vecParticlesChain[j])
								VParticleTot += GetBondVolume(_dTime, j1);
			}
			double dResultDiameter = 2 * pow(3 * VParticleTot / (4 * PI), 1.0 / 3);
			size_t nDestinationClass = (size_t)((dResultDiameter - _dMin) / dClassSize);

			// check on the upper and lower sizes
			if (nDestinationClass >= vecResult.size())
				nDestinationClass = vecResult.size() - 1;
			vecResult[nDestinationClass]++;
		}
	}
	return vecResult;
}

std::vector<unsigned> CSystemStructure::GetPrimaryPSD(double _dTime, double _dMin, double _dMax, unsigned _nClasses)
{
	std::vector<unsigned> vResult;
	vResult.resize(_nClasses, 0);
	double dClassSize = (_dMax - _dMin) / _nClasses;

	for (unsigned i = 0; i < objects.size(); i++)
		if (objects[i] != NULL)
			if ((objects[i]->GetObjectType() == SPHERE) && (objects[i]->IsActive(_dTime) == true))
			{
				unsigned nDestinationClass = (unsigned)(((((CSphere*)objects[i])->GetRadius() * 2) - _dMin) / dClassSize);
				if (nDestinationClass < vResult.size())
					vResult[nDestinationClass]++;
			}
	return vResult;
}

double CSystemStructure::GetBondVolume(double _dTime, size_t _nBondID)
{
	CPhysicalObject* pBond = GetObjectByIndex(_nBondID);
	if (pBond == NULL) return 0;
	if (pBond->GetObjectType() == SOLID_BOND)
		return PI * ((CBond*)pBond)->GetDiameter() / 2 * ((CBond*)pBond)->GetDiameter() / 2 * GetBond(_dTime, _nBondID).Length();

	if (pBond->GetObjectType() != LIQUID_BOND) return 0;

	unsigned nLeftObjectID = ((CBond*)pBond)->m_nLeftObjectID;
	unsigned nRightObjectID = ((CBond*)pBond)->m_nRightObjectID;

	CPhysicalObject* pLeftSphere = GetObjectByIndex(nLeftObjectID);
	CPhysicalObject* pRightSphere = GetObjectByIndex(nRightObjectID);
	if ((pLeftSphere == NULL) || (pRightSphere == NULL)) return 0;
	if ((pLeftSphere->GetObjectType() != SPHERE) || (pRightSphere->GetObjectType() != SPHERE)) return 0;

	double dR1 = ((CSphere*)pLeftSphere)->GetRadius();
	double dR2 = ((CSphere*)pRightSphere)->GetRadius();

	CVector3 vecCenter1 = ((CSphere*)pLeftSphere)->GetCoordinates(0);
	CVector3 vecCenter2 = ((CSphere*)pRightSphere)->GetCoordinates(0);

	double dL = Length(vecCenter1 - vecCenter2) - dR1 - dR2;
	if (dL < 0) dL = 0;

	double dRb = ((CBond*)pBond)->GetDiameter() / 2;

	double h1, h2;
	if (dRb <= dR1)
		h1 = dR1 - sqrt(dR1*dR1 - dRb * dRb);
	else
		h1 = 0;
	if (dRb <= dR2)
		h2 = dR2 - sqrt(dR2*dR2 - dRb * dRb);
	else
		h2 = 0;

	double Vss1 = PI * h1*h1*(3 * dR1 - h1) / 3.0; //objem sharovogo segmenta
	double Vss2 = PI * h2*h2*(3 * dR2 - h2) / 3.0; //objem sharovogo segmenta
	double VCyl = PI * dRb*dRb*(dL + h1 + h2);
	return VCyl - Vss1 - Vss2;
}

CVector3 CSystemStructure::GetBondVelocity(double _dTime, size_t _nBondID)
{
	CVector3 vResult(0, 0, 0);
	if (objects[_nBondID] == NULL) return vResult;
	if ((objects[_nBondID]->GetObjectType() != SOLID_BOND) && (objects[_nBondID]->GetObjectType() != LIQUID_BOND)) return vResult;
	CBond* pBond = (CBond*)objects[_nBondID];
	if ((objects[pBond->m_nLeftObjectID] == NULL) || (objects[pBond->m_nRightObjectID] == NULL)) return vResult;
	return (objects[pBond->m_nLeftObjectID]->GetVelocity(_dTime) + objects[pBond->m_nRightObjectID]->GetVelocity(_dTime)) / 2.0;

}

CVector3 CSystemStructure::GetBondCoordinate(double _dTime, size_t _nBondID)
{
	CVector3 vResult(0, 0, 0);
	if (objects[_nBondID] == NULL) return vResult;
	if ((objects[_nBondID]->GetObjectType() != SOLID_BOND) && (objects[_nBondID]->GetObjectType() != LIQUID_BOND)) return vResult;
	CBond* pBond = (CBond*)objects[_nBondID];
	if ((objects[pBond->m_nLeftObjectID] == NULL) || (objects[pBond->m_nRightObjectID] == NULL)) return vResult;
	return (objects[pBond->m_nLeftObjectID]->GetCoordinates(_dTime) + objects[pBond->m_nRightObjectID]->GetCoordinates(_dTime)) / 2.0;

}

CVector3 CSystemStructure::GetBond(double _dTime, size_t _nBondID) const
{
	CVector3 vResult(0, 0, 0);
	if (!objects[_nBondID]) return vResult;
	if ((objects[_nBondID]->GetObjectType() != SOLID_BOND) && (objects[_nBondID]->GetObjectType() != LIQUID_BOND)) return vResult;
	const CBond* pBond = static_cast<const CBond*>(objects[_nBondID]);
	const CPhysicalObject* pLParticle = objects[pBond->m_nLeftObjectID];
	const CPhysicalObject* pRParticle = objects[pBond->m_nRightObjectID];
	if (!pLParticle->IsActive(_dTime) || !pRParticle->IsActive(_dTime) || !pBond->IsActive(_dTime)) return vResult;
	if (m_PBC.bEnabled)
		m_PBC.UpdatePBC(_dTime);
	return GetSolidBond(pRParticle->GetCoordinates(_dTime), pLParticle->GetCoordinates(_dTime), m_PBC);
}

void CSystemStructure::GetGroup(double _dTime, size_t _nSourceParticleID, std::vector<size_t>* _pVecGroup)
{
	_pVecGroup->clear();
	_pVecGroup->push_back(_nSourceParticleID);
	if (objects[_nSourceParticleID] == NULL) return;
	if (objects[_nSourceParticleID]->GetObjectType() != SPHERE) return;

	size_t nNewFoundParticles = 1;
	while (nNewFoundParticles != 0)
	{
		nNewFoundParticles = 0;
		// go through all bonds
		for (size_t i = 0; i < objects.size(); i++)
			if (objects[i] != NULL)
				if ((objects[i]->GetObjectType() == SOLID_BOND) || (objects[i]->GetObjectType() == LIQUID_BOND))
					if (objects[i]->IsActive(_dTime))
					{
						size_t nLeftObjectID = ((CBond*)objects[i])->m_nLeftObjectID;
						size_t nRightObjectID = ((CBond*)objects[i])->m_nRightObjectID;

						bool bLeftWasFound = false; // indicates that the particle with LeftID accorded to the chain
						bool bRightWasFound = false;

						for (size_t j = 0; j < _pVecGroup->size(); j++)
							if (_pVecGroup->at(j) == nLeftObjectID)
							{
								bLeftWasFound = true;
								break;
							}
						for (size_t j = 0; j < _pVecGroup->size(); j++)
							if (_pVecGroup->at(j) == nRightObjectID)
							{
								bRightWasFound = true;
								break;
							}
						if ((bLeftWasFound == true) && (bRightWasFound == false))
						{
							_pVecGroup->push_back(nRightObjectID);
							nNewFoundParticles++;
						}
						else if ((bLeftWasFound == false) && (bRightWasFound == true))
						{
							_pVecGroup->push_back(nLeftObjectID);
							nNewFoundParticles++;
						}
					}
	}
}

std::vector<size_t> CSystemStructure::GetAgglomerate(double _time, size_t _particleID)
{
	if (!objects[_particleID] || objects[_particleID]->GetObjectType() != SPHERE) return {};

	const auto bonds = GetAllBonds(_time);
	std::vector<size_t> agglomerate{ _particleID };
	size_t foundParticles = 1;
	while (foundParticles != 0)
	{
		foundParticles = 0;
		for (const auto& bond : bonds)
		{
			bool bLFound = false; // indicates that the particle with LeftID was added to the chain
			bool bRFound = false; // indicates that the particle with RightID was added to the chain
			for (auto id : agglomerate)
			{
				if (!bLFound && id == bond->m_nLeftObjectID)	bLFound = true;
				if (!bRFound && id == bond->m_nRightObjectID)	bRFound = true;
			}
			if (bLFound && !bRFound)
			{
				agglomerate.push_back(bond->m_nRightObjectID);
				foundParticles++;
			}
			else if (!bLFound && bRFound)
			{
				agglomerate.push_back(bond->m_nLeftObjectID);
				foundParticles++;
			}
		}
	}

	return agglomerate;
}

void CSystemStructure::UpdateBondedIDs()
{
	// delete all existing bonds
	for (unsigned int i = 0; i < objects.size(); i++)
		if (objects[i]->GetObjectType() != SOLID_BOND)
			objects[i]->DeleteAllBonds();


	for (unsigned int i = 0; i < objects.size(); i++)
		if (objects[i]->GetObjectType() == SOLID_BOND)
		{
			unsigned int nObjectID1 = ((CSolidBond*)objects[i])->m_nLeftObjectID;
			unsigned int nObjectID2 = ((CSolidBond*)objects[i])->m_nRightObjectID;

			// add bond to the first object
			if (nObjectID1 < objects.size())
				if (objects[nObjectID1] != NULL)
					objects[nObjectID1]->AddBond(objects[i]->m_lObjectID);

			// add bond to the first object
			if (nObjectID2 < objects.size())
				if (objects[nObjectID2] != NULL)
					objects[nObjectID2]->AddBond(objects[i]->m_lObjectID);
		}
}

void CSystemStructure::FlushToStorage() // Setting values to the protofile
{
	// save objects geomety
	for (unsigned i = 0; i < objects.size(); i++)
		if (objects[i] != NULL)
			objects[i]->Save();

	ProtoSimulationInfo* si = m_storage->SimulationInfo();
	// save info about geometries
	si->clear_geomobjects();
	for (unsigned i = 0; i < m_Geometries.size(); i++)
	{
		SGeometryObject* pGeom = m_Geometries[i];
		ProtoGeometryInfo* pGeomInfo = si->add_geomobjects();
		pGeomInfo->set_name(pGeom->sName);
		pGeomInfo->set_key(pGeom->sKey);
		pGeomInfo->set_mass(pGeom->dMass);
		pGeomInfo->set_rotate_aroundmasscenter(pGeom->bRotateAroundCenter);
		pGeomInfo->set_forcedependentvel(pGeom->bForceDepVel);
		VectorToProtoVector(pGeomInfo->mutable_vfreemotion(), pGeom->vFreeMotion);
		for (unsigned j = 0; j < pGeom->vPlanes.size(); j++)
			pGeomInfo->add_planes(pGeom->vPlanes[j]);
		for (unsigned j = 0; j < pGeom->vIntervals.size(); j++)
		{
			ProtoTimeDepVel* pTDVel = pGeomInfo->add_tdval();
			pTDVel->set_time(pGeom->vIntervals[j].dCriticalValue);
			VectorToProtoVector(pTDVel->mutable_velocity(), pGeom->vIntervals[j].vVel);
			VectorToProtoVector(pTDVel->mutable_rotcenter(), pGeom->vIntervals[j].vRotCenter);
			VectorToProtoVector(pTDVel->mutable_rotvelocity(), pGeom->vIntervals[j].vRotVel);
		}
		pGeomInfo->set_type((int)pGeom->nVolumeType);
		for (unsigned j = 0; j < pGeom->vProps.size(); j++)
			pGeomInfo->add_props(pGeom->vProps[j]);
		MatrixToProtoMatrix(pGeomInfo->mutable_rotation(), pGeom->mRotation);
		ColorToProtoColor(pGeomInfo->mutable_color(), pGeom->color);
	}

	// save info about analysisVolumes
	si->clear_analysisvolumes();
	for (unsigned i = 0; i < m_AnalysisVolumes.size(); i++)
	{
		CAnalysisVolume* pVol = m_AnalysisVolumes[i];
		ProtoAnalysisVolumeInfo* pVolInfo = si->add_analysisvolumes();
		pVolInfo->set_name(pVol->sName);
		pVolInfo->set_key(pVol->sKey);
		pVolInfo->set_type((int)pVol->nVolumeType);
		for (unsigned j = 0; j < pVol->vProps.size(); j++)
			pVolInfo->add_vprops(pVol->vProps[j]);
		for (unsigned j = 0; j < pVol->vTriangles.size(); j++)
			TriangleToProtoTriangle(pVolInfo->add_triangles(), pVol->vTriangles[j]);
		for (unsigned j = 0; j < pVol->m_vIntervals.size(); j++)
		{
			ProtoTimeDepVel * pTDVel = pVolInfo->add_tdval();
			pTDVel->set_time(pVol->m_vIntervals[j].dTime);
			VectorToProtoVector(pTDVel->mutable_velocity(), pVol->m_vIntervals[j].vVel);
		}
		MatrixToProtoMatrix(pVolInfo->mutable_rotation(), pVol->mRotation);
		ColorToProtoColor(pVolInfo->mutable_color(), pVol->color);
	}

	// save info about multispheres
	si->clear_multispheres();
	for (unsigned i = 0; i < m_Multispheres.size(); i++)
	{
		ProtoMultisphere* protoMultiSphere = si->add_multispheres();
		for (unsigned j = 0; j < m_Multispheres[i].size(); j++)
			protoMultiSphere->add_id(m_Multispheres[i][j]);
	}

	VectorToProtoVector(si->mutable_simulation_volume_min(), m_SimulationDomain.coordBeg);
	VectorToProtoVector(si->mutable_simulation_volume_max(), m_SimulationDomain.coordEnd);

	si->set_periodic_conditions_enabled(m_PBC.bEnabled);
	si->set_periodic_conditions_x(m_PBC.bX);
	si->set_periodic_conditions_y(m_PBC.bY);
	si->set_periodic_conditions_z(m_PBC.bZ);
	VectorToProtoVector(si->mutable_periodic_conditions_min(), m_PBC.initDomain.coordBeg);
	VectorToProtoVector(si->mutable_periodic_conditions_max(), m_PBC.initDomain.coordEnd);
	VectorToProtoVector(si->mutable_periodic_conditions_vel(), m_PBC.vVel);

	si->set_anisotripy(m_bAnisotropy);
	si->set_contact_radius(m_bContactRadius);

	m_storage->FlushToDisk(true);
}

void CSystemStructure::FinalFileTruncate()
{
	m_storage->FinalTruncate();
}

void CSystemStructure::NewFile()
{
	Reset();
	DeleteAllObjects();
}

void CSystemStructure::SaveToFile(const std::string& _sFileName /*= ""*/)
{
	// save materials database
	m_MaterialDatabase.SaveToProtobufFile(*m_storage->ModulesData()->mutable_materials_database());

	if (m_sDEMFileName.empty() && _sFileName.empty()) return;

	if (_sFileName.empty() || FileNamesAreSame(m_sDEMFileName, _sFileName))
	{	// if rewrite current saved file
		FlushToStorage();
	}
	else if (m_sDEMFileName.empty())
	{   // if first saving
		m_storage.reset(new CDemStorage());
		m_sDEMFileName = _sFileName;
		m_storage->CreateNewFile(_sFileName);
	}
	else
	{   // if save as.. to the new file
		auto newStorage = new CDemStorage();
		newStorage->CreateNewFile(_sFileName);
		newStorage->CopyFrom(&*m_storage);
		m_storage.reset(newStorage);
		m_sDEMFileName = _sFileName;
		for (unsigned i = 0; i < objects.size(); i++)
			if (objects[i] != NULL)
				objects[i]->m_storage = &*m_storage;
		FlushToStorage();
	}

	m_storage->SetFileVersion(MDEM_FILE_VERSION);

	FinalFileTruncate();
}

CSystemStructure::ELoadFileResult CSystemStructure::LoadFromFile(const std::string& _sFileName)
{
	DeleteAllObjects();

	m_storage.reset(new CDemStorage());
	// open file
	m_storage->OpenFile(_sFileName);
	const bool binFileLoadResult = m_storage->LoadFromFile(); // load data from file to protobuf

	if (!binFileLoadResult)
		return ELoadFileResult::IsNotDEMFile;

	m_sDEMFileName = _sFileName;

	// load info about simulation
	const ProtoSimulationInfo* si = m_storage->SimulationInfo();
	m_SimulationDomain.coordBeg = ProtoVectorToVector(si->simulation_volume_min());
	m_SimulationDomain.coordEnd = ProtoVectorToVector(si->simulation_volume_max());

	m_PBC.bEnabled = si->periodic_conditions_enabled();
	m_PBC.bX = si->periodic_conditions_x();
	m_PBC.bY = si->periodic_conditions_y();
	m_PBC.bZ = si->periodic_conditions_z();
	m_PBC.SetDomain(ProtoVectorToVector(si->periodic_conditions_min()), ProtoVectorToVector(si->periodic_conditions_max()));
	m_PBC.vVel = ProtoVectorToVector(si->periodic_conditions_vel());

	m_bAnisotropy = si->anisotripy();
	m_bContactRadius = si->contact_radius();

	// load info about geometries
	for (int i = 0; i < si->geomobjects_size(); i++)
	{
		SGeometryObject* pGeometryObject = AddGeometry();
		const ProtoGeometryInfo& geomInfo = si->geomobjects(i);
		pGeometryObject->sName = geomInfo.name();
		pGeometryObject->sKey = geomInfo.key();
		pGeometryObject->dMass = geomInfo.mass();
		pGeometryObject->vFreeMotion = ProtoVectorToVector(geomInfo.vfreemotion());
		pGeometryObject->bRotateAroundCenter = geomInfo.rotate_aroundmasscenter();
		pGeometryObject->bForceDepVel = geomInfo.forcedependentvel();
		pGeometryObject->vPlanes.resize(geomInfo.planes_size());
		for (unsigned j = 0; j < pGeometryObject->vPlanes.size(); j++)
			pGeometryObject->vPlanes[j] = geomInfo.planes(j);
		pGeometryObject->vIntervals.resize(geomInfo.tdval_size());
		for (unsigned j = 0; j < pGeometryObject->vIntervals.size(); j++)
		{
			const ProtoTimeDepVel& tdvProto = geomInfo.tdval(j);
			pGeometryObject->vIntervals[j].dCriticalValue = tdvProto.time();
			pGeometryObject->vIntervals[j].vRotCenter = ProtoVectorToVector(tdvProto.rotcenter());
			pGeometryObject->vIntervals[j].vVel = ProtoVectorToVector(tdvProto.velocity());
			pGeometryObject->vIntervals[j].vRotVel = ProtoVectorToVector(tdvProto.rotvelocity());
		}

		pGeometryObject->nVolumeType = static_cast<EVolumeType>(geomInfo.type());
		pGeometryObject->vProps.resize(geomInfo.props_size());
		for (unsigned j = 0; j < pGeometryObject->vProps.size(); j++)
			pGeometryObject->vProps[j] = geomInfo.props(j);
		pGeometryObject->mRotation = ProtoMatrixToMatrix(geomInfo.rotation());

		if (geomInfo.has_color())
			pGeometryObject->color = ProtoColorToColor(geomInfo.color());
		else // compatibility
			pGeometryObject->color = CColor(0.5, 0.5, 1.0);
	}

	// load info about analysis volumes
	for (int i = 0; i < m_storage->SimulationInfo()->analysisvolumes_size(); i++)
	{
		CAnalysisVolume* pVol = AddAnalysisVolume();
		ProtoAnalysisVolumeInfo volInfo = m_storage->SimulationInfo()->analysisvolumes(i);
		pVol->sName = volInfo.name();
		pVol->sKey = volInfo.key();
		pVol->nVolumeType = static_cast<EVolumeType>(volInfo.type());
		pVol->vProps.resize(volInfo.vprops_size());
		for (unsigned j = 0; j < pVol->vProps.size(); j++)
			pVol->vProps[j] = volInfo.vprops(j);
		pVol->vTriangles.resize(volInfo.triangles_size());
		for (unsigned j = 0; j < pVol->vTriangles.size(); j++)
			pVol->vTriangles[j] = ProtoTriangleToTriangle(volInfo.triangles(j));
		if (volInfo.has_rotation())
			pVol->mRotation = ProtoMatrixToMatrix(volInfo.rotation());
		else if (pVol->nVolumeType != EVolumeType::VOLUME_STL) // compatibility with older versions
			pVol->vTriangles = CTriangularMesh::GenerateMesh(pVol->nVolumeType, pVol->vProps, ProtoVectorToVector(volInfo.vcenter()), CMatrix3::Diagonal()).vTriangles;
		if (volInfo.has_color())
			pVol->color = ProtoColorToColor(volInfo.color());
		else // compatibility
			pVol->color = CColor(0.6f, 1.0f, 0.6f, 1.0f);

		pVol->m_vIntervals.resize(volInfo.tdval_size());

		for (unsigned j = 0; j < pVol->m_vIntervals.size(); j++)
		{
			const ProtoTimeDepVel& tdvProto = volInfo.tdval(j);
			pVol->m_vIntervals[j].dTime = tdvProto.time();
			pVol->m_vIntervals[j].vVel = ProtoVectorToVector(tdvProto.velocity());
		}
	}

	// load info about multispheres
	for (int i = 0; i < m_storage->SimulationInfo()->multispheres_size(); i++)
	{
		std::vector<size_t> multSphere;
		ProtoMultisphere protoMultisphere = m_storage->SimulationInfo()->multispheres(i);
		for (int j = 0; j < protoMultisphere.id_size(); j++)
			multSphere.push_back(protoMultisphere.id(j));
		AddMultisphere(multSphere);
	}

	// load objects and geometry for each object
	objects.resize(m_storage->ObjectsCount(), nullptr);
	ParallelFor(m_storage->ObjectsCount(), [&](size_t i)
	{
		if (m_storage->SimulationInfo()->particles(i).type() == 0) return; // unknown or empty object
		AddObject(m_storage->Object(i)->type(), i);	     // add object to all physical objects in the system
		objects[i]->Load();								 // load object geometry
	});

	// load material data base
	const ProtoModulesData* pProtoMessage = GetProtoModulesData();
	if (pProtoMessage->has_materials_database()) // for correct handling of old versions
		m_MaterialDatabase.LoadFromProtobufFile(pProtoMessage->materials_database());	// load new data
	else
		m_MaterialDatabase.ClearData();

	UpdateAllObjectsCompoundsProperties();

	// check if file was partially saved or not
	if (m_storage->GetRealTimeEnd() < GetMaxTime())
	{
		m_storage->SimulationInfo()->set_end_time(m_storage->GetRealTimeEnd());
		return ELoadFileResult::PartiallyLoaded;
	}

	// check if selective saving mode was used or not
	if (m_storage->SimulationInfo()->selective_saving() && GetAllTimePoints().size() > 1)
		return ELoadFileResult::SelectivelySaved;

	return ELoadFileResult::OK;
}

bool CSystemStructure::IsOldFileVersion(const std::string& _sFileName)
{
	// temp storage
	CDemStorage* pStorage = new CDemStorage();

	// open file
	pStorage->OpenFile(_sFileName);

	// check file version from file header
	if (pStorage->GetFileVersion() < 1)
	{
		// read headers: load time-independent data and first two blocks with time-dependent data from .mdem file
		if (!pStorage->LoadFromFile())
			return false;

		// check file version from time-independent data (for older files)
		if (pStorage->SimulationInfo()->file_version() < 1)
		{
			// check file version by the quaternions existence (for the oldest files)
			for (unsigned i = 0; i < pStorage->ObjectsCount(); i++)
			{
				//if (m_storage->SimulationInfo()->particles(i).has_id() == false) //proto2
				if (pStorage->SimulationInfo()->particles(i).type() == 0)		   //proto3
					continue;

				if (pStorage->Object(i)->type() == SPHERE || pStorage->Object(i)->type() == TRIANGULAR_WALL)
				{
					std::vector<double> vAllTimePoints = pStorage->GetAllTimePoints();
					if (vAllTimePoints.empty())
						vAllTimePoints = pStorage->GetAllTimePointsOldFormat();

					if (pStorage->GetTimePointR(vAllTimePoints.front(), i)->IsQuaternionSet(i))
						break;
					else
					{
						delete pStorage;
						return true;
					}
				}
			}
		}
		// set current file version to file header
		pStorage->SetFileVersion(1);
	}
	delete pStorage;
	return false;
}

uint32_t CSystemStructure::FileVersion(const std::string& _sFileName)
{
	auto* storage = new CDemStorage();
	storage->OpenFile(_sFileName);
	const uint32_t version = storage->GetFileVersion();
	delete storage;
	return version;
}

uint32_t CSystemStructure::FileVersion() const
{
	return m_storage->GetFileVersion();
}

CPhysicalObject* CSystemStructure::GetObjectByIndex(size_t _ObjectID)
{
	if (_ObjectID >= objects.size())
		return NULL;
	return objects[_ObjectID];
}

const CPhysicalObject* CSystemStructure::GetObjectByIndex(size_t _ObjectID) const
{
	if (_ObjectID >= objects.size())
		return NULL;
	return objects[_ObjectID];
}

double CSystemStructure::GetMaxTime() const
{
	return m_storage->SimulationInfo()->end_time();
}

std::vector<double> CSystemStructure::GetAllTimePoints() const
{
	return m_storage->GetAllTimePoints();
}

std::vector<double> CSystemStructure::GetAllTimePointsOldFormat()
{
	return m_storage->GetAllTimePointsOldFormat();
}

CVector3 CSystemStructure::GetMaxCoordinate(const double _dTime)
{
	CVector3 vCurrentCoord;
	CVector3 vMaxCoord(0, 0, 0);
	bool bInitialized = false;
	for (unsigned i = 0; i < objects.size(); i++)
	{
		if (objects[i] == NULL) continue;
		if (!objects[i]->IsActive(_dTime)) continue;
		if ((objects[i]->GetObjectType() != SPHERE) && (objects[i]->GetObjectType() != TRIANGULAR_WALL)) continue;

		if (!bInitialized)
		{
			vMaxCoord = objects[i]->GetCoordinates(_dTime);
			bInitialized = true;
		}

		vCurrentCoord = objects[i]->GetCoordinates(_dTime);
		if (objects[i]->GetObjectType() == SPHERE)
			vMaxCoord = Max(vMaxCoord, vCurrentCoord + ((CSphere*)objects[i])->GetRadius());
		else if (objects[i]->GetObjectType() == TRIANGULAR_WALL)
		{
			vMaxCoord = Max(vMaxCoord, ((CTriangularWall*)objects[i])->GetCoordVertex1(_dTime), ((CTriangularWall*)objects[i])->GetCoordVertex2(_dTime));
			vMaxCoord = Max(vMaxCoord, ((CTriangularWall*)objects[i])->GetCoordVertex3(_dTime));
		}
	}
	return vMaxCoord;
}

CVector3 CSystemStructure::GetMinCoordinate(const double _dTime)
{
	CVector3 vCurrentCoord;
	CVector3 vMinCoord(0, 0, 0);
	bool bInitialized = false;

	for (unsigned i = 0; i < objects.size(); i++)
	{
		if (objects[i] == NULL) continue;
		if (!objects[i]->IsActive(_dTime)) continue;
		if ((objects[i]->GetObjectType() != SPHERE) && (objects[i]->GetObjectType() != TRIANGULAR_WALL)) continue;

		if (!bInitialized)
		{
			vMinCoord = objects[i]->GetCoordinates(_dTime);
			bInitialized = true;
		}

		vCurrentCoord = objects[i]->GetCoordinates(_dTime);
		if (objects[i]->GetObjectType() == SPHERE)
			vMinCoord = Min(vMinCoord, vCurrentCoord - ((CSphere*)objects[i])->GetRadius());
		else if (objects[i]->GetObjectType() == TRIANGULAR_WALL)
		{
			vMinCoord = Min(vMinCoord, ((CTriangularWall*)objects[i])->GetCoordVertex1(_dTime), ((CTriangularWall*)objects[i])->GetCoordVertex2(_dTime));
			vMinCoord = Min(vMinCoord, ((CTriangularWall*)objects[i])->GetCoordVertex3(_dTime));
		}
	}
	return vMinCoord;
}

double CSystemStructure::GetMinTime() const
{
	return m_storage->SimulationInfo()->begin_time();
}

CVector3 CSystemStructure::GetCenterOfMass(double _dTime, size_t _nFirstObjectID /*= 0*/, size_t _nLastObjectID /*= 0*/)
{
	if ((_nFirstObjectID == _nLastObjectID) && (_nLastObjectID == 0))
		_nLastObjectID = GetTotalObjectsCount();

	CVector3 vecResult(0);
	double dTotalMass = 0; // total mass of a system
	for (size_t i = _nFirstObjectID; i < _nLastObjectID; ++i)
	{
		CPhysicalObject* pTempPhysicalObject = GetObjectByIndex(i);
		if (pTempPhysicalObject != NULL)
			if (pTempPhysicalObject->GetObjectType() == SPHERE)
			{
				double dMass1 = ((CSphere*)pTempPhysicalObject)->GetMass();
				CVector3 vecPosition = pTempPhysicalObject->GetCoordinates(_dTime);
				vecResult = (vecResult*dTotalMass + vecPosition * dMass1) / (dMass1 + dTotalMass);
				dTotalMass += dMass1;
			}
	}
	return vecResult;
}

void CSystemStructure::RotateSystem(double _dTime, const CVector3& _RotCenter, const CVector3& _RotAngleRad, size_t _nFirstObjectID /*= 0*/, size_t _nLastObjectID /*= 0 */)
{
	if ((_nFirstObjectID == _nLastObjectID) && (_nLastObjectID == 0))
		_nLastObjectID = GetTotalObjectsCount();

	CMatrix3 RotMatrix = CQuaternion(_RotAngleRad).ToRotmat();

	for (size_t i = _nFirstObjectID; i < _nLastObjectID; i++)
	{
		CPhysicalObject* pTempPhysicalObject = GetObjectByIndex(i);
		if (pTempPhysicalObject != NULL)
			if (pTempPhysicalObject->GetObjectType() == SPHERE)
			{
				CVector3 vecNewCoord = RotMatrix * (pTempPhysicalObject->GetCoordinates(_dTime) - _RotCenter);
				pTempPhysicalObject->SetCoordinates(_dTime, vecNewCoord + _RotCenter);
			}
			else if (pTempPhysicalObject->GetObjectType() == TRIANGULAR_WALL)
			{
				CTriangularWall* pWall = (CTriangularWall*)pTempPhysicalObject;
				CVector3 vertex1 = RotMatrix * (pWall->GetCoordVertex1(_dTime) - _RotCenter) + _RotCenter;
				CVector3 vertex2 = RotMatrix * (pWall->GetCoordVertex2(_dTime) - _RotCenter) + _RotCenter;
				CVector3 vertex3 = RotMatrix * (pWall->GetCoordVertex3(_dTime) - _RotCenter) + _RotCenter;
				pWall->SetPlaneCoord(_dTime, vertex1, vertex2, vertex3);
			}
	}
}

void CSystemStructure::MoveSystem(double _dTime, const CVector3& _vOffset)
{
	for (unsigned i = 0; i < objects.size(); i++)
		if (objects[i] != NULL)
			objects[i]->SetCoordinates(_dTime, objects[i]->GetCoordinates(_dTime) + _vOffset);
}

void CSystemStructure::SetSystemVelocity(double _dTime, const CVector3& _vNewVelocity)
{
	for (unsigned i = 0; i < objects.size(); i++)
		if (objects[i] != NULL)
			if (objects[i]->GetObjectType() == SPHERE)
				objects[i]->SetVelocity(_dTime, _vNewVelocity);
}

void CSystemStructure::GetOverlaps(const double& _dTime, std::vector<unsigned>& _vID1, std::vector<unsigned>& _vID2, std::vector<double>& _vOverlap)
{
	CContactCalculator calculator;
	for (unsigned i = 0; i < objects.size(); ++i)
		if ((objects[i]) && (objects[i]->IsActive(_dTime)) && (objects[i]->GetObjectType() == SPHERE))
			calculator.AddParticle(objects[i]->m_lObjectID, objects[i]->GetCoordinates(_dTime), dynamic_cast<CSphere*>(objects[i])->GetContactRadius());
	calculator.GetAllOverlaps(_vID1, _vID2, _vOverlap, m_PBC);
}

std::vector<double> CSystemStructure::GetMaxOverlaps(const double& _dTime)
{
	std::vector<unsigned> vID1, vID2;
	std::vector<double> vOverlap;
	std::vector<double> vRes(objects.size(), 0);
	GetOverlaps(_dTime, vID1, vID2, vOverlap);
	for (size_t i = 0; i < vOverlap.size(); ++i)
	{
		vRes[vID1[i]] = std::max(vOverlap[i], vRes[vID1[i]]);
		vRes[vID2[i]] = std::max(vOverlap[i], vRes[vID2[i]]);
	}
	return vRes;
}

std::vector<double> CSystemStructure::GetMaxOverlaps(const double& _dTime, const std::vector<size_t>& _vIDs)
{
	std::vector<unsigned> vID1, vID2;
	std::vector<double> vOverlap;
	std::vector<double> vRes(objects.size(), 0);

	CContactCalculator calculator;
	for (size_t i = 0; i < _vIDs.size(); ++i)
		if ((_vIDs[i] < objects.size()) && objects[_vIDs[i]] && objects[_vIDs[i]]->IsActive(_dTime) && (objects[_vIDs[i]]->GetObjectType() == SPHERE))
			calculator.AddParticle(objects[_vIDs[i]]->m_lObjectID, objects[_vIDs[i]]->GetCoordinates(_dTime), dynamic_cast<CSphere*>(objects[_vIDs[i]])->GetContactRadius());
	calculator.GetAllOverlaps(vID1, vID2, vOverlap, m_PBC);

	for (size_t i = 0; i < vOverlap.size(); ++i)
	{
		vRes[vID1[i]] = std::max(vOverlap[i], vRes[vID1[i]]);
		vRes[vID2[i]] = std::max(vOverlap[i], vRes[vID2[i]]);
	}
	return vRes;
}

void CSystemStructure::GetOverlaps(double& _dMaxOverlap, unsigned& _nMaxOverlapParticleID1, unsigned& _nMaxOverlapParticleID2, double& _dTotalOverlap, const double& _dTime)
{
	_dMaxOverlap = 0;
	_dTotalOverlap = 0;
	_nMaxOverlapParticleID1 = 0;
	_nMaxOverlapParticleID2 = 0;

	std::vector<unsigned> vID1, vID2;
	std::vector<double> vOverlap;
	GetOverlaps(_dTime, vID1, vID2, vOverlap);

	for (unsigned i = 0; i < vOverlap.size(); ++i)
	{
		_dTotalOverlap += vOverlap[i];
		if (vOverlap[i] > _dMaxOverlap)
		{
			_dMaxOverlap = vOverlap[i];
			_nMaxOverlapParticleID1 = vID1[i];
			_nMaxOverlapParticleID2 = vID2[i];
		}
	}
}

double CSystemStructure::GetMaxOverlap(const double& _dTime)
{
	double dMaxOverlap, dTotalOverlap;
	unsigned nID1, nID2;
	GetOverlaps(dMaxOverlap, nID1, nID2, dTotalOverlap, _dTime);
	return dMaxOverlap;
}

std::vector<unsigned> CSystemStructure::GetCoordinationNumbers(double _dTime)
{
	std::vector<unsigned> vResult(GetTotalObjectsCount(), 0);
	std::vector<CPhysicalObject*> vSpheres = GetAllActiveObjects(_dTime, SPHERE);
	CContactCalculator calculator;
	for (unsigned i = 0; i < vSpheres.size(); ++i)
		calculator.AddParticle(vSpheres[i]->m_lObjectID, vSpheres[i]->GetCoordinates(_dTime), dynamic_cast<CSphere*>(vSpheres[i])->GetContactRadius());

	std::vector<unsigned> vID1, vID2;
	std::vector<double> vOverlaps;
	calculator.GetAllOverlaps(vID1, vID2, vOverlaps, m_PBC);
	for (size_t i = 0; i < vOverlaps.size(); i++)
	{
		vResult[vID1[i]] ++;
		vResult[vID2[i]] ++;
	}

	for (unsigned i = 0; i < objects.size(); i++)
	{
		if (objects[i] == NULL) continue;
		if (!objects[i]->IsActive(_dTime)) continue;
		if (objects[i]->GetObjectType() == SOLID_BOND) // add bonds
		{
			vResult[((CSolidBond*)objects[i])->m_nLeftObjectID]++;
			vResult[((CSolidBond*)objects[i])->m_nRightObjectID]++;
		}
	}
	return vResult;
}

std::vector<CSolidBond*> CSystemStructure::GetParticleBonds(unsigned _nID)
{
	std::vector<CSolidBond*> vRes;
	for (unsigned i = 0; i < GetTotalObjectsCount(); ++i)
	{
		CSolidBond *pBond = dynamic_cast<CSolidBond*>(GetObjectByIndex(i));
		if (!pBond) continue;
		if ((pBond->m_nLeftObjectID == _nID) || (pBond->m_nRightObjectID == _nID))
			vRes.push_back(pBond);
	}
	return vRes;
}

std::string CSystemStructure::IsAllCompoundsDefined() const
{
	// check materials for particles and bonds
	for (size_t i = 0; i < objects.size(); i++)
		if ((objects[i]) && (m_MaterialDatabase.GetCompoundIndex(objects[i]->GetCompoundKey()) < 0))
		{
			if (objects[i]->GetObjectType() == SPHERE)
				return "Materials not for all particles are defined in database";
			else if ((objects[i]->GetObjectType() == SOLID_BOND) || (objects[i]->GetObjectType() == LIQUID_BOND))
				return "Materials not for all bonds are defined in database";
		}
	// check materials for geometries
	for (size_t i = 0; i < m_Geometries.size(); i++)
		for (size_t j = 0; j < m_Geometries[i]->vPlanes.size(); j++)
			if ((objects[m_Geometries[i]->vPlanes[j]]) && (m_MaterialDatabase.GetCompoundIndex(objects[m_Geometries[i]->vPlanes[j]]->GetCompoundKey()) < 0))
				return "Undefined material for geometry: " + m_Geometries[i]->sName;
	return "";
}

ProtoModulesData* CSystemStructure::GetProtoModulesData() const
{
	return m_storage->ModulesData();
}

ProtoSimulationInfo* CSystemStructure::GetSimulationInfo() const
{
	return m_storage->SimulationInfo();
}

void CSystemStructure::PrepareTimePointForRead(double _time) const
{
	if(m_storage->IsValid())
		m_storage->PrepareTimePointForRead(_time, objects.size());
}

void CSystemStructure::PrepareTimePointForWrite(double _time) const
{
	if (m_storage->IsValid())
		m_storage->PrepareTimePointForWrite(_time, objects.size());
}

std::vector<size_t> CSystemStructure::GetMultisphere(unsigned _nIndex)
{
	if (m_Multispheres.size() > _nIndex)
		return m_Multispheres[_nIndex];
	return {};
}

long long int CSystemStructure::GetMultisphereIndex(unsigned _nParticleIndex)
{
	for (size_t i = 0; i < m_Multispheres.size(); i++)
		for (unsigned j = 0; j < m_Multispheres[i].size(); j++)
			if (m_Multispheres[i][j] == _nParticleIndex)
				return i;
	return -1; // particle does not corresponds to any multisphere
}

SVolumeType CSystemStructure::GetSimulationDomain() const
{
	return m_SimulationDomain;
}

void CSystemStructure::SetSimulationDomain(const SVolumeType& _domain)
{
	m_SimulationDomain = _domain;
}

SPBC CSystemStructure::GetPBC() const
{
	return m_PBC;
}

void CSystemStructure::SetPBC(const SPBC& _PBC)
{
	m_PBC = _PBC;
}

bool CSystemStructure::IsAnisotropyEnabled() const
{
	return m_bAnisotropy;
}

void CSystemStructure::EnableAnisotropy(bool _bEnable)
{
	m_bAnisotropy = _bEnable;
}

bool CSystemStructure::IsContactRadiusEnabled() const
{
	return m_bContactRadius;
}

void CSystemStructure::EnableContactRadius(bool _bEnable)
{
	m_bContactRadius = _bEnable;
}

size_t CSystemStructure::GetGeometriesNumber() const
{
	return m_Geometries.size();
}

std::vector<SGeometryObject*> CSystemStructure::GetAllGeometries() const
{
	return m_Geometries;
}

const SGeometryObject* CSystemStructure::GetGeometry(size_t _iGeometry) const
{
	if (_iGeometry < m_Geometries.size())
		return m_Geometries[_iGeometry];
	else
		return nullptr;
}

SGeometryObject* CSystemStructure::GetGeometry(size_t _iGeometry)
{
	return const_cast<SGeometryObject*>(static_cast<const CSystemStructure&>(*this).GetGeometry(_iGeometry));
}

const SGeometryObject* CSystemStructure::GetGeometry(const std::string& _sKey) const
{
	for (auto g : m_Geometries)
		if (g->sKey == _sKey)
			return g;
	return nullptr;
}

SGeometryObject* CSystemStructure::GetGeometry(const std::string& _sKey)
{
	return const_cast<SGeometryObject*>(static_cast<const CSystemStructure&>(*this).GetGeometry(_sKey));
}

const SGeometryObject* CSystemStructure::GetGeometryByName(const std::string& _name) const
{
	for (auto g : m_Geometries)
		if (g->sName == _name)
			return g;
	return nullptr;
}

SGeometryObject* CSystemStructure::GetGeometryByName(const std::string& _name)
{
	return const_cast<SGeometryObject*>(static_cast<const CSystemStructure&>(*this).GetGeometryByName(_name));
}

size_t CSystemStructure::GetGeometryIndex(const std::string& _sKey) const
{
	for (size_t i = 0; i < m_Geometries.size(); ++i)
		if (m_Geometries[i]->sKey == _sKey)
			return i;
	return (std::numeric_limits<size_t>::max)();
}

std::vector<CTriangularWall*> CSystemStructure::GetGeometryWalls(size_t _iGeometry)
{
	std::vector<CTriangularWall*> vRes;

	if (_iGeometry >= m_Geometries.size()) return vRes;

	vRes.reserve(m_Geometries[_iGeometry]->vPlanes.size());
	for (size_t i = 0; i < m_Geometries[_iGeometry]->vPlanes.size(); ++i)
	{
		CTriangularWall* pWall = static_cast<CTriangularWall*>(GetObjectByIndex(m_Geometries[_iGeometry]->vPlanes[i]));
		if (pWall) vRes.push_back(pWall);
	}
	return vRes;
}

std::vector<const CTriangularWall*> CSystemStructure::GetGeometryWalls(size_t _iGeometry) const
{
	std::vector<const CTriangularWall*> vRes;

	if (_iGeometry >= m_Geometries.size()) return vRes;

	vRes.reserve(m_Geometries[_iGeometry]->vPlanes.size());
	for (size_t i = 0; i < m_Geometries[_iGeometry]->vPlanes.size(); ++i)
	{
		const CTriangularWall* pWall = static_cast<const CTriangularWall*>(GetObjectByIndex(m_Geometries[_iGeometry]->vPlanes[i]));
		if (pWall)	vRes.push_back(pWall);
	}
	return vRes;
}

SGeometryObject* CSystemStructure::AddGeometry()
{
	SGeometryObject* pGeometry = new SGeometryObject;
	pGeometry->sKey = GenerateNewKey();
	pGeometry->sName = "Unspecified";
	pGeometry->dMass = 0;
	pGeometry->vFreeMotion.Init(0, 0, 0);
	pGeometry->nVolumeType = EVolumeType::VOLUME_STL;
	pGeometry->mRotation = CMatrix3::Diagonal();
	pGeometry->color = CColor(0.5f, 0.5f, 1.0f);
	pGeometry->bRotateAroundCenter = false;
	pGeometry->bForceDepVel = false;
	pGeometry->nCurrentInterval = 0;
	m_Geometries.push_back(pGeometry);
	return pGeometry;
}



SGeometryObject* CSystemStructure::AddGeometry(const CTriangularMesh& _mesh)
{
	SGeometryObject* pGeometry = AddGeometry();
	pGeometry->sName = _mesh.sName;
	pGeometry->nVolumeType = EVolumeType::VOLUME_STL;

	for (auto tri : _mesh.vTriangles)
	{
		CTriangularWall* pWall = static_cast<CTriangularWall*>(AddObject(TRIANGULAR_WALL));
		pWall->SetPlaneCoord(0, tri);
		pGeometry->vPlanes.push_back(pWall->m_lObjectID);
	}

	return pGeometry;
}

SGeometryObject* CSystemStructure::AddGeometry(const EVolumeType& _type, const std::vector<double>& _params, const CVector3& _center, const CMatrix3& _rotation, size_t _accuracy /*= 0*/)
{
	const CTriangularMesh mesh = CTriangularMesh::GenerateMesh(_type, _params, _center, _rotation, _accuracy);

	SGeometryObject* pGeometry = AddGeometry(mesh);
	pGeometry->nVolumeType = _type;
	pGeometry->vProps = _params;
	pGeometry->mRotation = _rotation * pGeometry->mRotation;

	return pGeometry;
}

void CSystemStructure::DeleteGeometry(size_t _iGeometry)
{
	if (_iGeometry >= m_Geometries.size()) return;
	DeleteObjects(m_Geometries[_iGeometry]->vPlanes);
	delete m_Geometries[_iGeometry];
	m_Geometries.erase(m_Geometries.begin() + _iGeometry);
}

void CSystemStructure::DeleteAllGeometries()
{
	for (auto& g : m_Geometries)
	{
		for (auto& p : g->vPlanes)
			if (p < objects.size())
			{
				delete objects[p];
				objects[p] = nullptr;
			}
		delete g;
	}
	m_Geometries.clear();
	Compress();
}

void CSystemStructure::UpGeometry(size_t _iGeometry)
{
	if (_iGeometry < m_Geometries.size() && _iGeometry != 0)
		std::iter_swap(m_Geometries.begin() + _iGeometry, m_Geometries.begin() + _iGeometry - 1);
}

void CSystemStructure::DownGeometry(size_t _iGeometry)
{
	if ((_iGeometry < m_Geometries.size()) && (_iGeometry != (m_Geometries.size() - 1)))
		std::iter_swap(m_Geometries.begin() + _iGeometry, m_Geometries.begin() + _iGeometry + 1);
}

CVector3 CSystemStructure::GetGeometryCenter(double _dTime, size_t _iGeometry) const
{
	CVector3 vCenter(0, 0, 0);
	for (auto wall : GetGeometryWalls(_iGeometry))
		vCenter += (wall->GetCoordVertex1(_dTime) + wall->GetCoordVertex2(_dTime) + wall->GetCoordVertex3(_dTime)) / (3.0*m_Geometries[_iGeometry]->vPlanes.size());
	return vCenter;
}

void CSystemStructure::SetGeometryCenter(double _dTime, size_t _iGeometry, const CVector3& _vCoord)
{
	if (m_Geometries.size() <= _iGeometry) return;
	CVector3 vOldCenter = GetGeometryCenter(_dTime, _iGeometry);
	for (auto& wall : GetGeometryWalls(_iGeometry))
	{
		CVector3 vVert1 = wall->GetCoordVertex1(_dTime) - vOldCenter + _vCoord;
		CVector3 vVert2 = wall->GetCoordVertex2(_dTime) - vOldCenter + _vCoord;
		CVector3 vVert3 = wall->GetCoordVertex3(_dTime) - vOldCenter + _vCoord;
		wall->SetPlaneCoord(_dTime, vVert1, vVert2, vVert3);
	}
}

void CSystemStructure::SetGeometryMaterial(size_t _iGeometry, const CCompound* _pCompound)
{
	for (auto& wall : GetGeometryWalls(_iGeometry))
		wall->SetCompound(_pCompound);
}

void CSystemStructure::SetGeometryMaterial(const std::string& _sKey, const CCompound* _pCompound)
{
	SetGeometryMaterial(GetGeometryIndex(_sKey), _pCompound);
}

void CSystemStructure::ScaleGeometry(double _dTime, size_t _iGeometry, double _factor)
{
	if (!GetGeometry(_iGeometry)) return;

	CVector3 vCenter = GetGeometryCenter(_dTime, _iGeometry);
	for (auto& wall : GetGeometryWalls(_iGeometry))
		wall->SetPlaneCoord(0,
			vCenter + (wall->GetCoordVertex1(_dTime) - vCenter)*_factor,
			vCenter + (wall->GetCoordVertex2(_dTime) - vCenter)*_factor,
			vCenter + (wall->GetCoordVertex3(_dTime) - vCenter)*_factor);
	for (auto& p : GetGeometry(_iGeometry)->vProps)
		p *= _factor;
}

void CSystemStructure::RotateGeometry(double _dTime, size_t _iGeometry, const CMatrix3& _rotation)
{
	if (!GetGeometry(_iGeometry)) return;

	CVector3 vCenter = GetGeometryCenter(_dTime, _iGeometry);
	for (auto& wall : GetGeometryWalls(_iGeometry))
	{
		wall->SetPlaneCoord(_dTime,
			vCenter + _rotation * (wall->GetCoordVertex1(_dTime) - vCenter),
			vCenter + _rotation * (wall->GetCoordVertex2(_dTime) - vCenter),
			vCenter + _rotation * (wall->GetCoordVertex3(_dTime) - vCenter));
	}
	GetGeometry(_iGeometry)->mRotation = _rotation * GetGeometry(_iGeometry)->mRotation;
}

size_t CSystemStructure::GetAnalysisVolumesNumber() const
{
	return m_AnalysisVolumes.size();
}

std::vector<CAnalysisVolume*> CSystemStructure::GetAllAnalysisVolumes() const
{
	return m_AnalysisVolumes;
}

const CAnalysisVolume* CSystemStructure::GetAnalysisVolume(size_t _iVolume) const
{
	if (_iVolume < m_AnalysisVolumes.size())
		return m_AnalysisVolumes[_iVolume];
	return nullptr;
}

CAnalysisVolume* CSystemStructure::GetAnalysisVolume(size_t _iVolume)
{
	return const_cast<CAnalysisVolume*>(static_cast<const CSystemStructure&>(*this).GetAnalysisVolume(_iVolume));
}

const CAnalysisVolume* CSystemStructure::GetAnalysisVolume(const std::string& _volumeKey) const
{
	for (auto g : m_AnalysisVolumes)
		if (g->sKey == _volumeKey)
			return g;
	return nullptr;
}

CAnalysisVolume* CSystemStructure::GetAnalysisVolume(const std::string& _volumeKey)
{
	return const_cast<CAnalysisVolume*>(static_cast<const CSystemStructure&>(*this).GetAnalysisVolume(_volumeKey));
}

const CAnalysisVolume* CSystemStructure::GetAnalysisVolumeByName(const std::string& _volumeName) const
{
	for (auto g : m_AnalysisVolumes)
		if (g->sName == _volumeName)
			return g;
	return nullptr;
}

CAnalysisVolume* CSystemStructure::GetAnalysisVolumeByName(const std::string& _volumeName)
{
	return const_cast<CAnalysisVolume*>(static_cast<const CSystemStructure&>(*this).GetAnalysisVolumeByName(_volumeName));
}

size_t CSystemStructure::GetAnalysisVolumeIndex(const std::string& _volumeKey) const
{
	for (size_t i = 0; i < m_AnalysisVolumes.size(); ++i)
		if (m_AnalysisVolumes[i]->sKey == _volumeKey)
			return i;
	return std::numeric_limits<size_t>::max();
}

std::vector<size_t> CSystemStructure::GetParticleIndicesInVolume(double _time, size_t _iVolume, bool _bTotallyInVolume /*= true*/)
{
	const CAnalysisVolume* volume = GetAnalysisVolume(_iVolume);
	if (!volume) return {};

	const auto particles = GetAllSpheres(_time);
	std::vector<size_t> IDs;		IDs.reserve(particles.size());
	std::vector<double> radiuses;	radiuses.reserve(particles.size());
	std::vector<CVector3> coords;	coords.reserve(particles.size());
	for (const auto& p : particles)
	{
		IDs.push_back(p->m_lObjectID);
		if (_bTotallyInVolume)
			radiuses.push_back(p->GetRadius());
		coords.push_back(p->GetCoordinates(_time));
	}

	return CInsideVolumeChecker{ volume, _time }.GetSpheresTotallyInside(coords, radiuses, IDs);
}

std::vector<size_t> CSystemStructure::GetParticleIndicesInVolume(double _time, const std::string& _volumeKey, bool _bTotallyInVolume /*= true*/)
{
	return GetParticleIndicesInVolume(_time, GetAnalysisVolumeIndex(_volumeKey), _bTotallyInVolume);
}

std::vector<CSphere*> CSystemStructure::GetParticlesInVolume(double _time, const std::string& _volumeKey, bool _bTotallyInVolume)
{
	const std::vector<size_t> indices = GetParticleIndicesInVolume(_time, _volumeKey, _bTotallyInVolume);
	std::vector<CSphere*> res;
	res.reserve(indices.size());
	for (size_t i : indices)
		res.push_back(dynamic_cast<CSphere*>(objects[i]));
	return res;
}

std::vector<size_t> CSystemStructure::GetSolidBondIndicesInVolume(double _time, size_t _iVolume)
{
	const CAnalysisVolume* volume = GetAnalysisVolume(_iVolume);
	if (!volume) return {};

	const auto bonds = GetAllSolidBonds(_time);
	std::vector<size_t> IDs;		IDs.reserve(bonds.size());
	std::vector<CVector3> coords;	coords.reserve(bonds.size());
	for (const auto& b : bonds)
	{
		IDs.push_back(b->m_lObjectID);
		coords.push_back(GetBondCoordinate(_time, b->m_lObjectID));
	}

	return CInsideVolumeChecker{ volume, _time }.GetObjectsInside(coords, IDs);
}

std::vector<size_t> CSystemStructure::GetSolidBondIndicesInVolume(double _time, const std::string& _volumeKey)
{
	return GetSolidBondIndicesInVolume(_time, GetAnalysisVolumeIndex(_volumeKey));
}

std::vector<CSolidBond*> CSystemStructure::GetSolidBondsInVolume(double _time, const std::string& _volumeKey)
{
	const std::vector<size_t> indices = GetSolidBondIndicesInVolume(_time, _volumeKey);
	std::vector<CSolidBond*> res;
	res.reserve(indices.size());
	for (size_t i : indices)
		res.push_back(dynamic_cast<CSolidBond*>(objects[i]));
	return res;
}

std::vector<size_t> CSystemStructure::GetLiquidBondIndicesInVolume(double _time, size_t _iVolume)
{
	const CAnalysisVolume* volume = GetAnalysisVolume(_iVolume);
	if (!volume) return {};

	const auto bonds = GetAllLiquidBonds(_time);
	std::vector<size_t> IDs;		IDs.reserve(bonds.size());
	std::vector<CVector3> coords;	coords.reserve(bonds.size());
	for (const auto& b : bonds)
	{
		IDs.push_back(b->m_lObjectID);
		coords.push_back(GetBondCoordinate(_time, b->m_lObjectID));
	}

	return CInsideVolumeChecker{ volume, _time }.GetObjectsInside(coords, IDs);
}

std::vector<size_t> CSystemStructure::GetLiquidBondIndicesInVolume(double _time, const std::string& _volumeKey)
{
	return GetLiquidBondIndicesInVolume(_time, GetAnalysisVolumeIndex(_volumeKey));
}

std::vector<CLiquidBond*> CSystemStructure::GetLiquidBondsInVolume(double _time, const std::string& _volumeKey)
{
	const std::vector<size_t> indices = GetLiquidBondIndicesInVolume(_time, _volumeKey);
	std::vector<CLiquidBond*> res;
	res.reserve(indices.size());
	for (size_t i : indices)
		res.push_back(dynamic_cast<CLiquidBond*>(objects[i]));
	return res;
}

std::vector<size_t> CSystemStructure::GetBondIndicesInVolume(double _time, size_t _iVolume)
{
	const CAnalysisVolume* volume = GetAnalysisVolume(_iVolume);
	if (!volume) return {};

	const auto bonds = GetAllBonds(_time);
	std::vector<size_t> IDs;		IDs.reserve(bonds.size());
	std::vector<CVector3> coords;	coords.reserve(bonds.size());
	for (const auto& b : bonds)
	{
		IDs.push_back(b->m_lObjectID);
		coords.push_back(GetBondCoordinate(_time, b->m_lObjectID));
	}

	return CInsideVolumeChecker{ volume, _time }.GetObjectsInside(coords, IDs);
}

std::vector<size_t> CSystemStructure::GetBondIndicesInVolume(double _time, const std::string& _volumeKey)
{
	return GetBondIndicesInVolume(_time, GetAnalysisVolumeIndex(_volumeKey));
}

std::vector<CBond*> CSystemStructure::GetBondsInVolume(double _time, const std::string& _volumeKey)
{
	const std::vector<size_t> indices = GetBondIndicesInVolume(_time, _volumeKey);
	std::vector<CBond*> res;
	res.reserve(indices.size());
	for (size_t i : indices)
		res.push_back(dynamic_cast<CBond*>(objects[i]));
	return res;
}

std::vector<size_t> CSystemStructure::GetWallIndicesInVolume(double _time, size_t _iVolume)
{
	const CAnalysisVolume* volume = GetAnalysisVolume(_iVolume);
	if (!volume) return {};

	const auto walls = GetAllWalls(_time);
	std::vector<size_t> vWallsID;		vWallsID.reserve(walls.size());
	std::vector<CVector3> vWallsCoordX;	vWallsCoordX.reserve(walls.size());
	std::vector<CVector3> vWallsCoordY;	vWallsCoordY.reserve(walls.size());
	std::vector<CVector3> vWallsCoordZ;	vWallsCoordZ.reserve(walls.size());
	for (const auto& w : walls)
	{
		vWallsID.push_back(w->m_lObjectID);
		const STriangleType t = w->GetCoords(_time);
		vWallsCoordX.push_back(t.p1);
		vWallsCoordY.push_back(t.p2);
		vWallsCoordZ.push_back(t.p3);
	}
	const CInsideVolumeChecker insideVolumeChecher(volume, _time);
	const std::vector<size_t> resultsIDx = insideVolumeChecher.GetObjectsInside(vWallsCoordX, vWallsID);
	const std::vector<size_t> resultsIDy = insideVolumeChecher.GetObjectsInside(vWallsCoordY, vWallsID);
	const std::vector<size_t> resultsIDz = insideVolumeChecher.GetObjectsInside(vWallsCoordZ, vWallsID);
	return VectorIntersection({ resultsIDx, resultsIDy, resultsIDz });
}

std::vector<size_t> CSystemStructure::GetWallIndicesInVolume(double _time, const std::string& _volumeKey)
{
	return GetWallIndicesInVolume(_time, GetAnalysisVolumeIndex(_volumeKey));
}

std::vector<CTriangularWall*> CSystemStructure::GetWallsInVolume(double _time, const std::string& _volumeKey)
{
	const std::vector<size_t> indices = GetWallIndicesInVolume(_time, _volumeKey);
	std::vector<CTriangularWall*> res;
	res.reserve(indices.size());
	for (size_t i : indices)
		res.push_back(dynamic_cast<CTriangularWall*>(objects[i]));
	return res;
}

std::vector<size_t> CSystemStructure::GetObjectIndicesInVolume(double _time, const std::vector<CVector3>& _coords, size_t _iVolume)
{
	const CAnalysisVolume* volume = GetAnalysisVolume(_iVolume);
	if (!volume) return {};

	return CInsideVolumeChecker{ GetAnalysisVolume(_iVolume), _time }.GetObjectsInside(_coords);
}

CAnalysisVolume* CSystemStructure::AddAnalysisVolume()
{
	CAnalysisVolume* pNewVolume = new CAnalysisVolume();
	pNewVolume->sKey = GenerateNewKey();
	pNewVolume->sName = "Unspecified";
	pNewVolume->nVolumeType = EVolumeType::VOLUME_STL;
	pNewVolume->mRotation = CMatrix3::Diagonal();
	pNewVolume->color = CColor(0.6f, 1.0f, 0.6f, 1.0f);
	m_AnalysisVolumes.push_back(pNewVolume);
	return pNewVolume;
}

CAnalysisVolume* CSystemStructure::AddAnalysisVolume(const CTriangularMesh& _mesh)
{
	CAnalysisVolume* pVolume = AddAnalysisVolume();
	pVolume->sName = _mesh.sName;
	pVolume->nVolumeType = EVolumeType::VOLUME_STL;
	pVolume->vTriangles = _mesh.vTriangles;
	return pVolume;
}

CAnalysisVolume* CSystemStructure::AddAnalysisVolume(const EVolumeType& _type, const std::vector<double>& _params, const CVector3& _center, const CMatrix3& _rotation, size_t _accuracy /*= 0*/)
{
	const CTriangularMesh mesh = CTriangularMesh::GenerateMesh(_type, _params, _center, _rotation, _accuracy);

	CAnalysisVolume* pVolume = AddAnalysisVolume(mesh);
	pVolume->nVolumeType = _type;
	pVolume->vProps = _params;
	pVolume->mRotation = _rotation * pVolume->mRotation;

	return pVolume;
}

void CSystemStructure::DeleteAnalysisVolume(size_t _iVolume)
{
	if (_iVolume >= m_AnalysisVolumes.size()) return;
	delete m_AnalysisVolumes[_iVolume];
	m_AnalysisVolumes.erase(m_AnalysisVolumes.begin() + _iVolume);
}

void CSystemStructure::UpAnalysisVolume(size_t _iVolume)
{
	if (_iVolume < m_AnalysisVolumes.size() && _iVolume != 0)
		std::iter_swap(m_AnalysisVolumes.begin() + _iVolume, m_AnalysisVolumes.begin() + _iVolume - 1);
}

void CSystemStructure::DownAnalysisVolume(size_t _iVolume)
{
	if ((_iVolume < m_AnalysisVolumes.size()) && (_iVolume != (m_AnalysisVolumes.size() - 1)))
		std::iter_swap(m_AnalysisVolumes.begin() + _iVolume, m_AnalysisVolumes.begin() + _iVolume + 1);
}

CVector3 CSystemStructure::GetAnalysisVolumeCenter(size_t _iVolume, double _dTime) const
{
	const CAnalysisVolume* pVolume = GetAnalysisVolume(_iVolume);
	if (!pVolume) return CVector3(0);
	return pVolume->GetCenter(_dTime);
}

void CSystemStructure::SetAnalysisVolumeCenter(size_t _iVolume, const CVector3& _vCoord)
{
	CAnalysisVolume* pVolume = GetAnalysisVolume(_iVolume);
	if (!pVolume) return;
	pVolume->SetCenter(_vCoord);
}

void CSystemStructure::ScaleAnalysisVolume(size_t _iVolume, double _factor)
{
	CAnalysisVolume* pVolume = GetAnalysisVolume(_iVolume);
	if (!pVolume) return;
	pVolume->Scale(_factor);
}

void CSystemStructure::RotateAnalysisVolume(size_t _iVolume, const CMatrix3& _rotation)
{
	CAnalysisVolume* pVolume = GetAnalysisVolume(_iVolume);
	if (!pVolume) return;
	pVolume->Rotate(_rotation);
}

void CSystemStructure::ClearAllTDData()
{
	size_t nNumberOfObjects = this->GetTotalObjectsCount();		   // total number of objects in the old file

	for (auto j = 0; j < nNumberOfObjects; j++)
	{
		CPhysicalObject* pObject = this->GetObjectByIndex(j);

		if (pObject)
		{
			// clear all protobuf fields for 0 time point for current object
			pObject->ClearAllTDData(0);
		}
	}
}