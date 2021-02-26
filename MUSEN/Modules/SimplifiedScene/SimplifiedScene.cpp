/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "SimplifiedScene.h"
#include "GeometricFunctions.h"

CSimplifiedScene::CSimplifiedScene()
{
	m_Objects.vParticles = std::make_shared<SParticleStruct>();
	m_Objects.vSolidBonds = std::make_shared<SSolidBondStruct>();
	m_Objects.vLiquidBonds = std::make_shared<SLiquidBondStruct>();
	m_Objects.vWalls = std::make_shared<SWallStruct>();
	m_Objects.vMultiSpheres = std::make_shared<SMultiSphere>();

	m_vInteractProps = std::make_shared<std::vector<SInteractProps>>();
	m_vParticlesToSolidBonds = std::make_shared<std::vector<std::vector<unsigned>>>();
	m_pSystemStructure = nullptr;
}

CSimplifiedScene::~CSimplifiedScene()
{
	ClearAllData();
}

void CSimplifiedScene::SetSystemStructure(CSystemStructure* _pSystemStructure)
{
	ClearAllData();
	m_pSystemStructure = _pSystemStructure;
	m_PBC = m_pSystemStructure->GetPBC();
	if (m_PBC.bEnabled)
		m_vPBCVirtShift.clear();
}

void CSimplifiedScene::InitializeScene(double _dStartTime, const SOptionalVariables& _activeOptionalVariables)
{
	ClearAllData();

	m_ActiveVariables = _activeOptionalVariables;

	// obtain indexes of simulated objects
	std::vector<size_t> viParticles;
	std::vector<size_t> viSolidBonds;
	std::vector<size_t> viLiquidBonds;
	std::vector<size_t> viWalls;
	m_vNewIndexes.clear();
	m_Objects.nVirtualParticles = 0;

	for (size_t i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); ++i)
	{
		CPhysicalObject* pObject = m_pSystemStructure->GetObjectByIndex(i);
		if (pObject && pObject->IsActive(_dStartTime))
		{
			switch (pObject->GetObjectType())
			{
			case SPHERE:		viParticles.push_back(i);	break;
			case SOLID_BOND:	viSolidBonds.push_back(i);	break;
			case LIQUID_BOND:	viLiquidBonds.push_back(i); break;
			}
		}
	}

	std::multimap<uint64_t,size_t> indices;
	const double maxD = m_pSystemStructure->GetMaxParticleDiameter();
	for (size_t iPart : viParticles)
	{
		const CVector3 pos = m_pSystemStructure->GetObjectByIndex(iPart)->GetCoordinates(_dStartTime) / maxD;
		indices.insert({ MortonEncode((size_t)std::round(pos.x), (size_t)std::round(pos.y), (size_t)std::round(pos.z)), iPart});
	}
	viParticles.clear();
	for (auto const& pair : indices)
		viParticles.push_back(pair.second);

	// add geometrical objects
	for (size_t i = 0; i < m_pSystemStructure->GeometriesNumber(); ++i)
	{
		CRealGeometry* pGeom = m_pSystemStructure->Geometry(i);
		for (const auto& plane : pGeom->Planes())
			if (m_pSystemStructure->GetObjectByIndex(plane))
				viWalls.push_back(plane);
		pGeom->Motion()->ResetMotionInfo();
	}


	// fill in memory for particles
	for (size_t i = 0; i < viParticles.size(); ++i)
		AddParticle(viParticles[i], _dStartTime);

	// fill in memory for bonds
	for (size_t i = 0; i < viSolidBonds.size(); ++i)
		AddSolidBond(viSolidBonds[i], _dStartTime);

	// fill in memory for liquid bonds
	for (size_t i = 0; i < viLiquidBonds.size(); ++i)
		AddLiquidBond(viLiquidBonds[i], _dStartTime);

	// fill in memory for walls bonds
	for (size_t i = 0; i < viWalls.size(); ++i)
		AddWall(viWalls[i], _dStartTime);

	// fill in memory for multispheres
	for (unsigned i = 0; i < m_pSystemStructure->GetMultispheresNumber(); ++i)
		AddMultisphere(m_pSystemStructure->GetMultisphere(i));

	InitializeLiquidBondsCharacteristics(_dStartTime);
	InitializeGeometricalObjects(_dStartTime);
	InitializeMaterials();
	UpdateParticlesToBonds();
	FindAdjacentWalls();
}

void CSimplifiedScene::UpdateParticlesToBonds()
{
	m_vParticlesToSolidBonds->resize(m_Objects.vParticles->Size());
	for (size_t i = 0; i < m_vParticlesToSolidBonds->size(); i++)
		(*m_vParticlesToSolidBonds)[i].clear();
	for (unsigned i = 0; i < static_cast<unsigned>(m_Objects.vSolidBonds->Size()); i++)
	{
		if (m_Objects.vSolidBonds->Active(i))
		{
			if ((m_Objects.vParticles->Active(m_Objects.vSolidBonds->LeftID(i))) && (m_Objects.vParticles->Active(m_Objects.vSolidBonds->RightID(i))))
			{
				(*m_vParticlesToSolidBonds)[m_Objects.vSolidBonds->LeftID(i)].push_back(i);
				(*m_vParticlesToSolidBonds)[m_Objects.vSolidBonds->RightID(i)].push_back(i);
			}
		}
	}
}

void  CSimplifiedScene::AddParticle(size_t _index, double _dTime)
{
	CSphere* pSphere = dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(_index));
	if (!pSphere) return;

	m_Objects.vParticles->AddParticle(
		pSphere->IsActive(_dTime),
		pSphere->GetCoordinates(_dTime),
		pSphere->GetRadius(),
		static_cast<unsigned>(_index),
		pSphere->GetMass(),
		pSphere->GetInertiaMoment(),
		pSphere->GetVelocity(_dTime),
		pSphere->GetAngleVelocity(_dTime)
	);
	m_Objects.vParticles->CoordVerlet(m_Objects.vParticles->Size() - 1) = pSphere->GetCoordinates(_dTime); // needed for proper work of dynamic generator
	m_Objects.vParticles->AddQuaternion(pSphere->GetOrientation(_dTime));
	m_Objects.vParticles->AddContactRadius(pSphere->GetContactRadius());

	while (m_vNewIndexes.size() <= _index)
		m_vNewIndexes.emplace_back(0);
	m_vNewIndexes[_index] = m_Objects.vParticles->Size() - 1;
}

void CSimplifiedScene::AddSolidBond(size_t _index, double _dTime)
{
	CSolidBond* pBond = dynamic_cast<CSolidBond*>(m_pSystemStructure->GetObjectByIndex(_index));
	if (!pBond) return;


	m_Objects.vSolidBonds->AddSolidBond(
		pBond->IsActive(_dTime),
		static_cast<unsigned>(_index),
		m_vNewIndexes[pBond->m_nLeftObjectID],
		m_vNewIndexes[pBond->m_nRightObjectID],
		pBond->GetDiameter(),
		pBond->m_dCrossCutSurface,
		pBond->GetInitLength(),
		pBond->GetTangentialOverlap(),
		pBond->m_dAxialMoment,
		pBond->GetYoungModulus(),
		pBond->GetShearModulus(),
		pBond->GetNormalStrength(),
		pBond->GetTangStrnegth()
	);

	m_Objects.vSolidBonds->AddYieldStrength(pBond->GetYieldStrength());
	m_Objects.vSolidBonds->AddViscosity(pBond->GetViscosity());
	m_Objects.vSolidBonds->AddTimeThermExpCoeff(pBond->GetTimeThermExpCoeff());

	CPhysicalObject* pLSphere = m_pSystemStructure->GetObjectByIndex(pBond->m_nLeftObjectID);
	CPhysicalObject* pRSphere = m_pSystemStructure->GetObjectByIndex(pBond->m_nRightObjectID);
	if (!pLSphere || !pRSphere) // one of the contact partner is absent
		pBond->SetObjectActivity(_dTime, false);
	else
	{
		size_t i = m_Objects.vSolidBonds->Size();
		m_Objects.vSolidBonds->PrevBond(i - 1) = GetSolidBond(pLSphere->GetCoordinates(_dTime), pRSphere->GetCoordinates(_dTime), m_PBC);
		m_Objects.vSolidBonds->AddNormalPlasticStrain(0);
		m_Objects.vSolidBonds->AddTangentialPlasticStrain(CVector3(0));
	}


	while (m_vNewIndexes.size() <= _index)
		m_vNewIndexes.push_back(0);
	m_vNewIndexes[_index] = m_Objects.vSolidBonds->Size() - 1;
}

void CSimplifiedScene::AddLiquidBond(size_t _index, double _dTime)
{
	CLiquidBond* pBond = dynamic_cast<CLiquidBond*>(m_pSystemStructure->GetObjectByIndex(_index));
	if (!pBond) return;

	m_Objects.vLiquidBonds->AddLiquidBond(
		pBond->IsActive(_dTime),
		static_cast<unsigned>(_index),
		static_cast<unsigned>(m_vNewIndexes[pBond->m_nLeftObjectID]),
		static_cast<unsigned>(m_vNewIndexes[pBond->m_nRightObjectID]),
		m_pSystemStructure->GetBondVolume(_dTime, _index),
		pBond->GetViscosity(),
		pBond->GetSurfaceTension()
	);

	while (m_vNewIndexes.size() <= _index)
		m_vNewIndexes.push_back(0);
	m_vNewIndexes[_index] = m_Objects.vLiquidBonds->Size() - 1;
}

void CSimplifiedScene::AddWall(size_t _index, double _dTime)
{
	CTriangularWall* pWall = dynamic_cast<CTriangularWall*>(m_pSystemStructure->GetObjectByIndex(_index));
	if (!pWall) return;

	m_Objects.vWalls->AddWall(
		true,
		static_cast<unsigned>(_index),
		pWall->GetCoordinates(_dTime),
		pWall->GetCoordVertex2(_dTime),
		pWall->GetCoordVertex3(_dTime),
		pWall->GetNormalVector(_dTime),
		pWall->GetVelocity(_dTime),
		pWall->GetAngleVelocity(_dTime),
		CVector3(0)
	);

	while (m_vNewIndexes.size() <= _index)
		m_vNewIndexes.push_back(0);
	m_vNewIndexes[_index] = m_Objects.vWalls->Size() - 1;
}

void CSimplifiedScene::AddMultisphere(const std::vector<size_t>& _vIndexes)
{
/*	std::map<size_t, size_t> vOldNewIndexes;
	for (size_t i = 0; i < m_Objects.vParticles->Size(); ++i)
		vOldNewIndexes[m_Objects.vParticles->InitIndex(i)] = i;


	std::vector<size_t> vIndexes;
	CMatrix3 mInertTensor(0);
	CMatrix3 mLMatrix(0);
	CMatrix3 mInvInertTensor(0);
	CMatrix3 mInvLMatrix;
	double dMass = 0;
	CVector3 vCenter(0);
	CVector3 vVelocity(0);
	CVector3 vRotVelocity(0);
	for (size_t j = 0; j < _vIndexes.size(); ++j)
	{
		const size_t nIndex = vOldNewIndexes[_vIndexes[j]];
		const CVector3 vCoord = m_Objects.vParticles->Coord(nIndex);
		double dMass = m_Objects.vParticles->Mass(nIndex);
		vIndexes.push_back(static_cast<unsigned>(nIndex));
		dMass += dMass;
		vCenter += vCoord * dMass;
		vVelocity += m_Objects.vParticles->Vel(nIndex) * dMass;
		m_Objects.vParticles->MultiSphIndex(vOldNewIndexes[_vIndexes[j]]) = static_cast<unsigned>(m_Objects.vMultiSpheres->Size()) - 1; // index of current multisphere

		// calculation of inertial tensor
		mInertTensor.values[0][0] += m_Objects.vParticles->InertiaMoment(nIndex) + dMass * (vCoord.y*vCoord.y + vCoord.z*vCoord.z);
		mInertTensor.values[0][1] += -dMass * vCoord.x*vCoord.y;
		mInertTensor.values[0][2] += -dMass * vCoord.x*vCoord.z;
		mInertTensor.values[1][0] += -dMass * vCoord.x*vCoord.y;
		mInertTensor.values[1][1] += m_Objects.vParticles->InertiaMoment(nIndex) + dMass * (vCoord.x*vCoord.x + vCoord.z*vCoord.z);
		mInertTensor.values[1][2] += -dMass * vCoord.y*vCoord.z;
		mInertTensor.values[2][0] += -dMass * vCoord.x*vCoord.z;
		mInertTensor.values[2][1] += -dMass * vCoord.y*vCoord.z;
		mInertTensor.values[2][2] += m_Objects.vParticles->InertiaMoment(nIndex) + dMass * (vCoord.y*vCoord.y + vCoord.x*vCoord.x);
	}
	vCenter = vCenter / (dMass * _vIndexes.size());
	vVelocity = vVelocity / (dMass * _vIndexes.size());

	CVector3 vEigenValues;
	mInertTensor.EigenDecomposition(mLMatrix, vEigenValues);
	// create diagonal matrix
	mInertTensor.values[0][0] = vEigenValues.x; mInertTensor.values[0][1] = 0; 	mInertTensor.values[0][2] = 0;
	mInertTensor.values[1][1] = vEigenValues.y; mInertTensor.values[1][0] = 0;  mInertTensor.values[1][2] = 0;
	mInertTensor.values[2][2] = vEigenValues.z;	mInertTensor.values[2][0] = 0;	mInertTensor.values[2][1] = 0;
	mInvInertTensor = mInertTensor.Inverse();

	// normalize eigenvectors
	for (size_t i = 0; i < 3; ++i)
	{
		const double dLength = sqrt(pow(mLMatrix.values[0][i], 2) + pow(mLMatrix.values[1][i], 2) + pow(mLMatrix.values[2][i], 2));
		for (size_t j = 0; j < 3; ++j)
			mLMatrix.values[j][i] = mLMatrix.values[j][i] / dLength;
	}
	mInvLMatrix = mLMatrix.Inverse();
	m_Objects.vMultiSpheres->AddMultisphere(std::move(vIndexes),
											std::move(mLMatrix),
											std::move(mInertTensor),
											std::move(mInvLMatrix),
											std::move(mInvInertTensor),
											std::move(vCenter),
											std::move(vVelocity),
											std::move(vRotVelocity),
											std::move(dMass)
	);*/
}

void CSimplifiedScene::InitializeMaterials()
{
	m_pSystemStructure->UpdateAllObjectsCompoundsProperties();
	std::vector<std::vector<SInteractProps>> vTempInteractProps;
	std::vector<std::string> vCompoundKeys; // keys of all current available compounds

	for (size_t i = 0; i < m_Objects.vParticles->Size() + m_Objects.vWalls->Size(); ++i)
	{
		// define compound properties
		size_t iInitID;
		if (i < m_Objects.vParticles->Size())
			iInitID = m_Objects.vParticles->InitIndex(i);
		else
			iInitID = m_Objects.vWalls->InitIndex(i - m_Objects.vParticles->Size());
		const std::string sCompoundKey = m_pSystemStructure->GetObjectByIndex(iInitID)->GetCompoundKey();

		const auto iter = std::find(vCompoundKeys.begin(), vCompoundKeys.end(), sCompoundKey);
		size_t nIndex = iter - vCompoundKeys.begin();
		if (iter == vCompoundKeys.end()) // such compound has not been defined yet
		{
			nIndex = vCompoundKeys.size();
			vCompoundKeys.push_back(sCompoundKey);
			// add  column to the interaction properties
			vTempInteractProps.push_back(std::vector<SInteractProps>());
			// add interaction property to all existing - last entry into each column
			for (size_t k = 0; k < vTempInteractProps.size() - 1; ++k)
			{
				const SInteractProps InterProp = CalculateInteractionProperty(sCompoundKey, vCompoundKeys[k]);
				vTempInteractProps.back().push_back(InterProp);
				vTempInteractProps[k].push_back(InterProp);
			}
			// add last element
			vTempInteractProps.back().push_back(CalculateInteractionProperty(sCompoundKey, sCompoundKey));
		}

		if (i < m_Objects.vParticles->Size())
			m_Objects.vParticles->CompoundIndex(i) = static_cast<unsigned>(nIndex);
		else
			m_Objects.vWalls->CompoundIndex(i - m_Objects.vParticles->Size()) = static_cast<unsigned>(nIndex);
	}
	// transfer from 2D into 1D array
	m_vInteractProps->clear();
	for (size_t i = 0; i < vTempInteractProps.size(); ++i)
		for (size_t j = 0; j < vTempInteractProps[i].size(); ++j)
			m_vInteractProps->push_back(std::move(vTempInteractProps[i][j]));
	m_vCompoundsNumber = vTempInteractProps.size();
}

void CSimplifiedScene::ClearAllForcesAndMoments()
{
	ParallelFor(m_Objects.vParticles->Size(), [&](size_t i)
	{
		m_Objects.vParticles->Force(i).Init(0);
		m_Objects.vParticles->Moment(i).Init(0);
	});

	ParallelFor(m_Objects.vWalls->Size(), [&](size_t i)
	{
		m_Objects.vWalls->Force(i).Init(0);
	});
}

void CSimplifiedScene::AddVirtualParticles(double _dVerletDistance)
{
	if (!m_PBC.bEnabled) return;

	const double dMaxContactRadius = GetMaxParticleContactRadius();
	m_vPBCVirtShift.clear(); // remove all shifts

	// calculate virtual domain
	const SVolumeType virtDomain{
		m_PBC.currentDomain.coordBeg + _dVerletDistance + dMaxContactRadius,
		m_PBC.currentDomain.coordEnd - _dVerletDistance - dMaxContactRadius
	};
	const CVector3& t = m_PBC.boundaryShift;
	const size_t nRealPartCount = m_Objects.vParticles->Size();
	for (size_t i = 0; i < nRealPartCount; ++i)
	{
		const CVector3& coord = m_Objects.vParticles->Coord(i);
		const bool xL = m_PBC.bX && (coord.x - m_Objects.vParticles->ContactRadius(i) <= virtDomain.coordBeg.x);
		const bool yL = m_PBC.bY && (coord.y - m_Objects.vParticles->ContactRadius(i) <= virtDomain.coordBeg.y);
		const bool zL = m_PBC.bZ && (coord.z - m_Objects.vParticles->ContactRadius(i) <= virtDomain.coordBeg.z);

		const bool xG = m_PBC.bX && (coord.x + m_Objects.vParticles->ContactRadius(i) >= virtDomain.coordEnd.x);
		const bool yG = m_PBC.bY && (coord.y + m_Objects.vParticles->ContactRadius(i) >= virtDomain.coordEnd.y);

		if (xL)				AddVirtualParticleBox(i, CVector3(t.x, 0, 0));
		if (yL)				AddVirtualParticleBox(i, CVector3(0, t.y, 0));
		if (zL)				AddVirtualParticleBox(i, CVector3(0, 0, t.z));
		if (xL && yL)		AddVirtualParticleBox(i, CVector3(t.x, t.y, 0));
		if (xL && zL)		AddVirtualParticleBox(i, CVector3(t.x, 0, t.z));
		if (yL && zL)		AddVirtualParticleBox(i, CVector3(0, t.y, t.z));
		if (xL && yL && zL)	AddVirtualParticleBox(i, CVector3(t.x, t.y, t.z));

		if (xG && yL)		AddVirtualParticleBox(i, CVector3(-t.x, t.y, 0));
		if (xG && zL)		AddVirtualParticleBox(i, CVector3(-t.x, 0, t.z));
		if (yG && zL)		AddVirtualParticleBox(i, CVector3(0, -t.y, t.z));

		if (xG && yL && zL)	AddVirtualParticleBox(i, CVector3(-t.x, t.y, t.z));
		if (xL && yG && zL)	AddVirtualParticleBox(i, CVector3(t.x, -t.y, t.z));
		if (xG && yG && zL)	AddVirtualParticleBox(i, CVector3(-t.x, -t.y, t.z));
	}
}



void CSimplifiedScene::GetAllParticlesInVolume(const SVolumeType& _volume, std::vector<unsigned>* _pvIndexes) const
{
	_pvIndexes->clear();
	for (unsigned i = 0; i < static_cast<unsigned>(m_Objects.vParticles->Size()); ++i)
	{
		CVector3 vCoord = m_Objects.vParticles->Coord(i);
		double dR = m_Objects.vParticles->ContactRadius(i);
		if (vCoord.x - dR >= _volume.coordEnd.x) continue;
		if (vCoord.y - dR >= _volume.coordEnd.y) continue;
		if (vCoord.z - dR >= _volume.coordEnd.z) continue;
		if (vCoord.x + dR <= _volume.coordBeg.x) continue;
		if (vCoord.y + dR <= _volume.coordBeg.y) continue;
		if (vCoord.z + dR <= _volume.coordBeg.z) continue;
		_pvIndexes->push_back(i);
	}
}

void CSimplifiedScene::GetAllWallsInVolume(const SVolumeType& _volume, std::vector<unsigned>* _pvIndexes) const
{
	_pvIndexes->clear();
	for (unsigned i = 0; i < static_cast<unsigned>(m_Objects.vWalls->Size()); ++i)
	{
		SVolumeType tempVolume;
		tempVolume.coordBeg = m_Objects.vWalls->MinCoord(i);
		tempVolume.coordEnd = m_Objects.vWalls->MaxCoord(i);
		if (CheckVolumesIntersection(_volume, tempVolume))
			_pvIndexes->push_back(i);
	}
}

void CSimplifiedScene::SaveVerletCoords()
{
	// for particles
	ParallelFor(m_Objects.vParticles->Size(), [&](size_t i)
	{
		m_Objects.vParticles->CoordVerlet(i) = m_Objects.vParticles->Coord(i);
	});
}

double CSimplifiedScene::GetMaxPartVerletDistance()
{
	static std::vector<double> vTempBuffer;
	vTempBuffer.resize(m_Objects.vParticles->Size());
	ParallelFor(m_Objects.vParticles->Size(), [&](size_t i)
	{
		if (m_Objects.vParticles->Active(i))
			vTempBuffer[i] = SquaredLength(m_Objects.vParticles->Coord(i) - m_Objects.vParticles->CoordVerlet(i));
		else
			vTempBuffer[i] = 0;
	});
	return sqrt(VectorMax(vTempBuffer));
}


double CSimplifiedScene::GetMaxParticleVelocity() const
{
	size_t nThreadsNumber = GetThreadsNumber();
	std::vector<double> vMaxVel(nThreadsNumber, 0);
	ParallelFor(m_Objects.vParticles->Size(), [&](size_t i)
	{
		const size_t nIndex = i % nThreadsNumber;
		if (m_Objects.vParticles->Active(i))
		{
			const double dTemp = m_Objects.vParticles->Vel(i).SquaredLength();
			if (vMaxVel[nIndex] < dTemp)
				vMaxVel[nIndex] = dTemp;
		}
	});

	return sqrt(VectorMax(vMaxVel));
}

double CSimplifiedScene::GetMaxParticleRadius() const
{
	if (m_Objects.vParticles->Empty())
		return 0.;
	double dMaxRadius = m_Objects.vParticles->Radius(0);
	for (size_t i = 1; i < m_Objects.vParticles->Size(); ++i)
		dMaxRadius = std::max(dMaxRadius, m_Objects.vParticles->Radius(i));
	return dMaxRadius;
}

double CSimplifiedScene::GetMinParticleRadius() const
{
	if (m_Objects.vParticles->Empty())
		return 0;
	double dMinRadius = m_Objects.vParticles->Radius(0);
	for (size_t i = 1; i < m_Objects.vParticles->Size(); ++i)
		dMinRadius = std::min(m_Objects.vParticles->Radius(i), dMinRadius);
	return dMinRadius;
}

double CSimplifiedScene::GetMaxParticleContactRadius() const
{
	if (m_Objects.vParticles->Empty())
		return 0.;
	double dMaxContactRadius = m_Objects.vParticles->ContactRadius(0);
	for (size_t i = 1; i < m_Objects.vParticles->Size(); ++i)
		dMaxContactRadius = std::max(dMaxContactRadius, m_Objects.vParticles->ContactRadius(i));
	return dMaxContactRadius;
}

double CSimplifiedScene::GetMinParticleContactRadius() const
{
	if (m_Objects.vParticles->Empty())
		return 0;
	double dMinContactRadius = m_Objects.vParticles->ContactRadius(0);
	for (size_t i = 1; i < m_Objects.vParticles->Size(); ++i)
		dMinContactRadius = std::min(m_Objects.vParticles->ContactRadius(i), dMinContactRadius);
	return dMinContactRadius;
}

double CSimplifiedScene::GetMaxWallVelocity() const
{
	double* pTemp = new  double[m_Objects.vWalls->Size()];
	ParallelFor(m_Objects.vWalls->Size(), [&](size_t i)
	{
		const double dLinVel = m_Objects.vWalls->Vel(i).SquaredLength();
		const double dSquaredRotVel = m_Objects.vWalls->RotVel(i).SquaredLength();
		double dMaxSquaredRotVel = 0;
		if (dSquaredRotVel > 0)
			dMaxSquaredRotVel = dSquaredRotVel * std::max({ SquaredLength(m_Objects.vWalls->Vert1(i) - m_Objects.vWalls->RotCenter(i)), SquaredLength(m_Objects.vWalls->Vert2(i) - m_Objects.vWalls->RotCenter(i)), SquaredLength(m_Objects.vWalls->Vert3(i) - m_Objects.vWalls->RotCenter(i)) });
		pTemp[i] = sqrt(dLinVel) + sqrt(dMaxSquaredRotVel);
	});

	double dMaxVelocity = 0;
	for (size_t i = 0; i < m_Objects.vWalls->Size(); ++i)
		if (pTemp[i] > dMaxVelocity)
			dMaxVelocity = pTemp[i];
	delete[] pTemp;
	return dMaxVelocity;
}

double   CSimplifiedScene::GetParticleTemperature(size_t _index) const
{
	if (m_ActiveVariables.bThermals)
		return m_Objects.vParticles->Temperature(_index);
	else
		return 0.;
}


void CSimplifiedScene::ClearAllData()
{
	m_Objects.vParticles->Resize(0);
	m_Objects.vSolidBonds->Resize(0);
	m_Objects.vLiquidBonds->Resize(0);
	m_Objects.vMultiSpheres->Resize(0);
	m_Objects.vWalls->Resize(0);
}

void CSimplifiedScene::InitializeLiquidBondsCharacteristics(double _dTime)
{
	for (size_t i = 0; i < m_Objects.vLiquidBonds->Size(); ++i)
	{
		CLiquidBond* pBond = dynamic_cast<CLiquidBond*>(m_pSystemStructure->GetObjectByIndex(m_Objects.vLiquidBonds->InitIndex(i)));
		if (!pBond) continue;
		CPhysicalObject* pLSphere = m_pSystemStructure->GetObjectByIndex(pBond->m_nLeftObjectID);
		CPhysicalObject* pRSphere = m_pSystemStructure->GetObjectByIndex(pBond->m_nRightObjectID);
		if (!pLSphere || !pRSphere) // one of the contact partner is absent
			pBond->SetObjectActivity(_dTime, false);
	}
}

void CSimplifiedScene::InitializeGeometricalObjects(double _dTime)
{
	for (size_t iGeom = 0; iGeom < m_pSystemStructure->GeometriesNumber(); ++iGeom)
	{
		CRealGeometry* pGeom = m_pSystemStructure->Geometry(iGeom);
		if (pGeom->Planes().empty()) continue;
		CVector3 vVel = pGeom->GetCurrentVelocity();
		CVector3 vRotVel = pGeom->GetCurrentRotVelocity();
		CVector3 vRotCenter = pGeom->GetCurrentRotCenter();
		for (const auto& plane : pGeom->Planes())
		{
			size_t index = m_vNewIndexes[plane];
			m_Objects.vWalls->Vel(index) = vVel;
			m_Objects.vWalls->RotVel(index) = vRotVel;
			m_Objects.vWalls->RotCenter(index) = vRotCenter;
		}
	}
}

SInteractProps CSimplifiedScene::CalculateInteractionProperty(const std::string& _sCompound1, const std::string& _sCompound2) const
{
	const CInteraction* pInteraction = m_pSystemStructure->m_MaterialDatabase.GetInteraction(_sCompound1, _sCompound2);
	SInteractProps InterProp;


	InterProp.dRollingFriction = pInteraction->GetPropertyValue(PROPERTY_ROLLING_FRICTION);
	InterProp.dRestCoeff = pInteraction->GetPropertyValue(PROPERTY_RESTITUTION_COEFFICIENT);
	InterProp.dAlpha = log(InterProp.dRestCoeff) / sqrt(PI*PI + pow(log(InterProp.dRestCoeff), 2));
	InterProp.dSlidingFriction = pInteraction->GetPropertyValue(PROPERTY_STATIC_FRICTION);

	const double dPoisson1 = m_pSystemStructure->m_MaterialDatabase.GetPropertyValue(_sCompound1, PROPERTY_POISSON_RATIO);
	const double dPoisson2 = m_pSystemStructure->m_MaterialDatabase.GetPropertyValue(_sCompound2, PROPERTY_POISSON_RATIO);
	const double dYoungModulus1 = m_pSystemStructure->m_MaterialDatabase.GetPropertyValue(_sCompound1, PROPERTY_YOUNG_MODULUS);
	const double dYoungModulus2 = m_pSystemStructure->m_MaterialDatabase.GetPropertyValue(_sCompound2, PROPERTY_YOUNG_MODULUS);
	const double dSurfaceTension1 = m_pSystemStructure->m_MaterialDatabase.GetPropertyValue(_sCompound1, PROPERTY_SURFACE_TENSION);
	const double dSurfaceTension2 = m_pSystemStructure->m_MaterialDatabase.GetPropertyValue(_sCompound2, PROPERTY_SURFACE_TENSION);
	const double dSurfaceEnergy1 = m_pSystemStructure->m_MaterialDatabase.GetPropertyValue(_sCompound1, PROPERTY_SURFACE_ENERGY);
	const double dSurfaceEnergy2 = m_pSystemStructure->m_MaterialDatabase.GetPropertyValue(_sCompound2, PROPERTY_SURFACE_ENERGY);
	const double dShearModulusPart1 = dYoungModulus1 / (2 * (1 + dPoisson1));
	const double dShearModulusPart2 = dYoungModulus2 / (2 * (1 + dPoisson2));
	InterProp.dEquivYoungModulus = 1.0 / ((1 - pow(dPoisson1, 2)) / dYoungModulus1 + (1 - pow(dPoisson2, 2)) / dYoungModulus2);
	InterProp.dEquivShearModulus = 1.0 / ((2 - dPoisson1) / dShearModulusPart1 + (2 - dPoisson2) / dShearModulusPart2);
	InterProp.dEquivSurfaceTension = sqrt(dSurfaceTension1 * dSurfaceTension2);
	InterProp.dEquivSurfaceEnergy = sqrt(dSurfaceEnergy1 * dSurfaceEnergy2);
	const double dK1 = m_pSystemStructure->m_MaterialDatabase.GetPropertyValue(_sCompound1, PROPERTY_THERMAL_CONDUCTIVITY);
	const double dK2 = m_pSystemStructure->m_MaterialDatabase.GetPropertyValue(_sCompound2, PROPERTY_THERMAL_CONDUCTIVITY);
	InterProp.dEquivThermalConductivity = 1 / (0.5 * 1 / dK1 + 0.5*dK2);

	return std::move(InterProp);
}

void CSimplifiedScene::AddVirtualParticleBox(size_t _nSourceID, const CVector3& _vShift)
{
	if (!m_Objects.vParticles->Active(_nSourceID)) return;

	m_Objects.vParticles->AddBasicParticle(
		m_Objects.vParticles->Active(_nSourceID),
		m_Objects.vParticles->Coord(_nSourceID) + _vShift,
		m_Objects.vParticles->ContactRadius(_nSourceID),
		static_cast<unsigned>(_nSourceID)
	);


	m_Objects.nVirtualParticles++;
	m_vPBCVirtShift.push_back(GetVirtShiftFromVector(_vShift));
}

void CSimplifiedScene::FindAdjacentWalls()
{
	const auto HaveSharedVertex = [](const CTriangle& _t1, const CTriangle& _t2)
	{
		constexpr double tol = 1e-18;
		return SquaredLength(_t1.p1 - _t2.p1) < tol
			|| SquaredLength(_t1.p1 - _t2.p2) < tol
			|| SquaredLength(_t1.p1 - _t2.p3) < tol
			|| SquaredLength(_t1.p2 - _t2.p1) < tol
			|| SquaredLength(_t1.p2 - _t2.p2) < tol
			|| SquaredLength(_t1.p2 - _t2.p3) < tol
			|| SquaredLength(_t1.p3 - _t2.p1) < tol
			|| SquaredLength(_t1.p3 - _t2.p2) < tol
			|| SquaredLength(_t1.p3 - _t2.p3) < tol;
	};

	m_adjacentWalls.clear();								// remove old
	m_adjacentWalls.resize(m_Objects.vWalls->Size(), {});	// resize
	for (const auto& geometry : m_pSystemStructure->AllGeometries())
	{
		const auto& planes = geometry->Planes();
		for (size_t iPlane1 = 0; iPlane1 < planes.size() - 1; ++iPlane1)
		{
			const size_t iWall1 = m_vNewIndexes[planes[iPlane1]];
			const CTriangle triangle1{ m_Objects.vWalls->Vert1(iWall1), m_Objects.vWalls->Vert2(iWall1), m_Objects.vWalls->Vert3(iWall1) };
			for (size_t iPlane2 = iPlane1 + 1; iPlane2 < planes.size(); ++iPlane2)
			{
				const size_t iWall2 = m_vNewIndexes[planes[iPlane2]];
				const CTriangle triangle2{ m_Objects.vWalls->Vert1(iWall2), m_Objects.vWalls->Vert2(iWall2), m_Objects.vWalls->Vert3(iWall2) };
				if (HaveSharedVertex(triangle1, triangle2))
				{
					m_adjacentWalls[iWall1].push_back((unsigned)iWall2);
					m_adjacentWalls[iWall2].push_back((unsigned)iWall1);
				}
			}
		}
	}
}

void CSimplifiedScene::RemoveVirtualParticles()
{
	m_Objects.vParticles->Resize(m_Objects.vParticles->Size() - m_Objects.nVirtualParticles);	// remove old virtual particles
	m_Objects.nVirtualParticles = 0;
}
