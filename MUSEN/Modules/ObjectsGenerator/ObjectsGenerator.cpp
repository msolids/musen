/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ObjectsGenerator.h"
#include "Quaternion.h"

CObjectsGenerator::CObjectsGenerator(CAgglomeratesDatabase* _pAgglomD, CMaterialsDatabase* _pMaterialsDB) :
	m_pAgglomDB( _pAgglomD ), m_pMaterialsDB( _pMaterialsDB )
{
	m_sName = "ObjectsGenerator";
	m_sVolumeKey = "";
	m_bGenerateMixture = true;
	m_sAgglomerateKey = "";
	m_dAgglomerateScaleFactor = 1.0;
	m_vObjInitVel.Init(0);
	m_sMixtureKey = "";
	m_dStartGenerationTime = 0;
	m_dEndGenerationTime = 1.0;
	m_dUpdateStep = 1e-3;
	m_dVelMagnitude = 0;
	m_bVisible = false;
	m_bActive = true;
	m_bRandomVelocity = false;
	m_bInsideGeometries = true;
	m_dLastGenerationTime = 0;
	m_PreLoadedAgglomerate = SAgglomerate{};
}

void CObjectsGenerator::Initialize()
{
	m_dLastGenerationTime = 0;
	unsigned seed = (unsigned)time( 0 );
	switch ( seed % 5 )
	{
	case 4: seed *= 2; break;
	case 3: seed *= seed; break;
	case 2: seed += seed*seed; break;
	case 1: seed = seed / 2; break;
	case 0: break;
	}
	srand( seed );

	if (!m_bGenerateMixture)
	{
		// check that the selected agglomerate exists
		if (!m_pAgglomDB->GetAgglomerate(m_sAgglomerateKey))
			return;

		// load agglomerate
		m_PreLoadedAgglomerate = *m_pAgglomDB->GetAgglomerate(m_sAgglomerateKey);

		// shift agglomerate center of mass into point with coord 0 and automatically scale it
		CVector3 vCenterOfMass(0);
		double dTotalVolume = 0; // total mass of a system
		for ( unsigned i = 0; i < m_PreLoadedAgglomerate.vParticles.size(); i++ )
		{
			double dVolume1 = PI*pow(2 * m_PreLoadedAgglomerate.vParticles[i].dRadius, 3) / 6.0;
			vCenterOfMass = (vCenterOfMass*dTotalVolume + m_PreLoadedAgglomerate.vParticles[i].vecCoord*dVolume1) / (dVolume1 + dTotalVolume);
			dTotalVolume += dVolume1;
		}
		for ( unsigned i = 0; i < m_PreLoadedAgglomerate.vParticles.size(); i++ )
		{
			m_PreLoadedAgglomerate.vParticles[i].vecCoord = (m_PreLoadedAgglomerate.vParticles[i].vecCoord - vCenterOfMass)*m_dAgglomerateScaleFactor;				// NOTE: this changes agglomerate DB in RAM -> will affect all calls to it
			m_PreLoadedAgglomerate.vParticles[i].dRadius *= m_dAgglomerateScaleFactor;
			m_PreLoadedAgglomerate.vParticles[i].dContactRadius *= m_dAgglomerateScaleFactor;
		}
		for (unsigned i = 0; i < m_PreLoadedAgglomerate.vBonds.size(); i++)
			m_PreLoadedAgglomerate.vBonds[i].dRadius *= m_dAgglomerateScaleFactor;
	}

	switch (m_rateType)
	{
	case ERateType::GENERATION_RATE:	m_dGenerationRate = m_rateValue;													break;
	case ERateType::OBJECTS_PER_STEP:	m_dGenerationRate = m_rateValue / m_dUpdateStep;									break;
	case ERateType::OBJECTS_TOTAL:		m_dGenerationRate = m_rateValue / (m_dEndGenerationTime - m_dStartGenerationTime);	break;
	}
}

size_t CObjectsGenerator::Generate(double _dCurrentTime, CSystemStructure* _pSystemStructure, CSimplifiedScene& _Scene)
{
	if (!m_bActive) return 0;
	if ( (_dCurrentTime < m_dStartGenerationTime) || (_dCurrentTime > m_dEndGenerationTime )  ) return 0;
	if (_dCurrentTime < m_dLastGenerationTime + m_dUpdateStep) return 0;
	unsigned nNewObjects = (unsigned)floor((_dCurrentTime - m_dLastGenerationTime)*m_dGenerationRate);
	if ( nNewObjects < 1 ) return 0;

	SPBC m_PBC = _Scene.GetPBC();

	CAnalysisVolume* pGenVolume = _pSystemStructure->AnalysisVolume( m_sVolumeKey );
	if ( pGenVolume == nullptr ) return 0;

	const SVolumeType boundingBox = pGenVolume->BoundingBox(_dCurrentTime);
	const double maxRadius = _Scene.GetMaxParticleContactRadius();
	// enlarge bounding box by maximum contact radius to account for particles just out of box possibly colliding with inserted ones
	const SVolumeType boundingBoxContactSearch{ boundingBox.coordBeg - maxRadius, boundingBox.coordEnd + maxRadius };
	std::vector<unsigned> vPartInVolume; // vector of indexes (in simplified scene) of all particles situated in the specified volume
	std::vector<unsigned> vWallsInVolume; // vector of indexes (in simplified scene) of all walls situated in the specified volume
	_Scene.GetAllParticlesInVolume(boundingBoxContactSearch, &vPartInVolume );
	_Scene.GetAllWallsInVolume(boundingBoxContactSearch, &vWallsInVolume );

	SWallStruct& pWalls = _Scene.GetRefToWalls();
	std::vector<CVector3> vCoordNewPart;
	std::vector<CQuaternion> vQuatNewPart;
	std::vector<double> vRadiiNewPart;
	std::vector<double> vContRadiiNewPart;
	std::vector<std::string> vMaterialsKey;
	size_t nCreatedParticles = 0;
	CInsideVolumeChecker insideChecker( pGenVolume, _dCurrentTime );

	// for checking that new object are not situated in any real volume
	std::vector<std::unique_ptr<CInsideVolumeChecker>> vInRealVolumeCheckers;
	if ( !m_bInsideGeometries )
		for (unsigned i = 0; i < _pSystemStructure->GeometriesNumber(); i++)
		{
			CRealGeometry* pGeom = _pSystemStructure->Geometry(i);
			std::vector<CTriangle> vTriangles;
			for (const auto& plane : pGeom->Planes())
			{
				size_t nIndex = _Scene.m_vNewIndexes[plane];
				vTriangles.emplace_back(pWalls.Vert1(nIndex),  pWalls.Vert2(nIndex),  pWalls.Vert3(nIndex));
			}
			vInRealVolumeCheckers.emplace_back(new CInsideVolumeChecker{ vTriangles });
		}

	for ( unsigned iObj = 0; iObj <nNewObjects; iObj++ )
	{
		int nMaxAttempt = static_cast<int>(m_maxIterations);
		bool bSuccess = false;
		while ( (nMaxAttempt > 0) && (!bSuccess) )
		{
			// generate single object
			GenerateNewObject(&vCoordNewPart, &vQuatNewPart, &vRadiiNewPart, &vContRadiiNewPart, &vMaterialsKey, boundingBox, m_PBC, _dCurrentTime);
			bSuccess = true;

			std::vector<size_t> vID = insideChecker.GetSpheresTotallyInside(vCoordNewPart, vRadiiNewPart);
			if ( vID.size() != vCoordNewPart.size() ) // not all particles in volume
				bSuccess = false;
			if ( bSuccess )
				bSuccess = !IsOverlapped( vCoordNewPart, vContRadiiNewPart, vPartInVolume, vWallsInVolume, _Scene);

			if (( bSuccess ) && (!m_bInsideGeometries))
				for (const auto& checker : vInRealVolumeCheckers)
				{
					vID = checker->GetSpheresTotallyInside(vCoordNewPart, vRadiiNewPart);
					if (!vID.empty()) // some objects in the volume
					{
						bSuccess = false;
						break;
					}
				}
			nMaxAttempt--;
		}
		if ( bSuccess ) // add new objects (due to successful addition of particles )
		{
			nCreatedParticles += vCoordNewPart.size();
			std::vector<size_t> vTempNewIndexes;
			CVector3 initVelocity;
			if (m_bRandomVelocity)
			{
				double dLen;
				do
				{
					initVelocity.x = 2 * (double)rand() / (double)RAND_MAX - 1;
					initVelocity.y = 2 * (double)rand() / (double)RAND_MAX - 1;
					initVelocity.z = 2 * (double)rand() / (double)RAND_MAX - 1;
					dLen = initVelocity.Length();
				} while (dLen == 0 || dLen > 1);
				initVelocity = initVelocity / dLen * m_dVelMagnitude;
			}

			const std::vector<size_t> vFreeIDs = _pSystemStructure->GetFreeIDs(vCoordNewPart.size());
			for ( size_t i = 0; i < vCoordNewPart.size(); i++ )  // add particles
			{
				auto* pNewSphere = dynamic_cast<CSphere*>(_pSystemStructure->AddObject(SPHERE, vFreeIDs[i]));
				vTempNewIndexes.push_back( pNewSphere->m_lObjectID );
				pNewSphere->SetRadius(vRadiiNewPart[i]);
				pNewSphere->SetContactRadius(vContRadiiNewPart[i]);
				if (m_bGenerateMixture) // this is sphere
					pNewSphere->SetCompound(m_pMaterialsDB->GetCompound(vMaterialsKey[i]));
				else // this is agglomerate
					pNewSphere->SetCompound(m_pMaterialsDB->GetCompound(m_partMaterials[m_PreLoadedAgglomerate.vParticles[i].sCompoundAlias]));
				pNewSphere->SetCoordinates( _dCurrentTime, vCoordNewPart[ i ] );
				pNewSphere->SetOrientation(_dCurrentTime, vQuatNewPart[i]);
				if (m_bRandomVelocity)
					pNewSphere->SetVelocity(_dCurrentTime, initVelocity);
				else
					pNewSphere->SetVelocity( _dCurrentTime, m_vObjInitVel );
				pNewSphere->SetStartActivityTime( _dCurrentTime );
				pNewSphere->SetEndActivityTime( 1e+300 );
				_Scene.AddParticle( pNewSphere->m_lObjectID, _dCurrentTime );
				vPartInVolume.push_back( (unsigned)_Scene.GetTotalParticlesNumber()-1 );
			}
			if (!m_bGenerateMixture) // this is agglomerate or multisphere
			{
				if ( m_PreLoadedAgglomerate.nType == MULTISPHERE )
				{
					_pSystemStructure->AddMultisphere( vTempNewIndexes );
					_Scene.AddMultisphere( vTempNewIndexes );
				}
				else
				{
					nCreatedParticles += m_PreLoadedAgglomerate.vBonds.size();
					const std::vector<size_t> vFreeIDsBonds = _pSystemStructure->GetFreeIDs(m_PreLoadedAgglomerate.vBonds.size());
					for (size_t j = 0; j < m_PreLoadedAgglomerate.vBonds.size(); ++j)
					{
						auto* pNewBond = dynamic_cast<CSolidBond*>(_pSystemStructure->AddObject(SOLID_BOND, vFreeIDsBonds[j]));
						const SAggloBond& bond = m_PreLoadedAgglomerate.vBonds[j];
						pNewBond->SetDiameter(2 * bond.dRadius);
						pNewBond->SetCompound(m_pMaterialsDB->GetCompound(m_bondMaterials[bond.sCompoundAlias]));
						pNewBond->m_nLeftObjectID = static_cast<unsigned>(vTempNewIndexes[bond.nLeftID]);
						pNewBond->m_nRightObjectID = static_cast<unsigned>(vTempNewIndexes[bond.nRightID]);
						pNewBond->SetStartActivityTime(_dCurrentTime);
						pNewBond->SetEndActivityTime(1e+300);
						pNewBond->SetInitialLength(_pSystemStructure->GetBond(_dCurrentTime, pNewBond->m_lObjectID).Length());
						pNewBond->SetTangentialOverlap(_dCurrentTime, CVector3{ 0.0 });
						_Scene.AddSolidBond(pNewBond->m_lObjectID, _dCurrentTime);
					}
				}
			}
		}
	}
	m_dLastGenerationTime = _dCurrentTime;
	return nCreatedParticles;
}

bool CObjectsGenerator::IsNeedToBeGenerated(double _dCurrentTime) const
{
	if (!m_bActive) return false;
	if ((_dCurrentTime < m_dStartGenerationTime) || (_dCurrentTime > m_dEndGenerationTime)) return false;
	if (_dCurrentTime < m_dLastGenerationTime + m_dUpdateStep) return false;
	const unsigned nNewObjects = (unsigned)floor((_dCurrentTime - m_dLastGenerationTime)*m_dGenerationRate);
	if (nNewObjects < 1) return false;
	return true;
}

void CObjectsGenerator::GenerateNewObject(std::vector<CVector3>* _pCoordPart, std::vector<CQuaternion>* _pQuatPart, std::vector<double>* _pPartRad, std::vector<double>* _pPartContRad,
	std::vector<std::string>* _sMaterialsKey, const SVolumeType& _boundBox, SPBC& _PBC, const double _dCurrentTime)
{
	CVector3 newCoord;
	CQuaternion rotQuat;
	_pCoordPart->clear();
	_pQuatPart->clear();
	_pPartRad->clear();
	_pPartContRad->clear();
	_sMaterialsKey->clear();

	// account for case when pbc is smaller than bounding box
	SVolumeType boundBoxInclPBC = _boundBox;
	CVector3 pbcSides{ (double)_PBC.bX, (double)_PBC.bY, (double)_PBC.bZ };
	if (_PBC.bEnabled)
		for (size_t i = 0; i < pbcSides.Size(); ++i)
			if (pbcSides[i] != 0.0)
			{
				if (boundBoxInclPBC.coordBeg[i] < _PBC.currentDomain.coordBeg[i])
					boundBoxInclPBC.coordBeg[i] = _PBC.currentDomain.coordBeg[i];
				if (boundBoxInclPBC.coordEnd[i] > _PBC.currentDomain.coordEnd[i])
					boundBoxInclPBC.coordEnd[i] = _PBC.currentDomain.coordEnd[i];
			}
	CreateRandomPoint(&newCoord, boundBoxInclPBC);
	if (m_bGenerateMixture)
	{
		const CMixture* pMixture = m_pMaterialsDB->GetMixture(m_sMixtureKey);
		if (!pMixture) return;
		const double dRand = ((double)rand()) / RAND_MAX; // generate random value
		double dTempSum = 0;
		size_t nIndex = 0;
		while (nIndex < pMixture->FractionsNumber())
		{
			dTempSum += pMixture->GetFractionValue(nIndex);
			if (dTempSum >= dRand) break;
			nIndex++;
		}

		_pCoordPart->push_back(newCoord);
		rotQuat.RandomGenerator();
		_pQuatPart->push_back(rotQuat);
		_pPartRad->push_back(pMixture->GetFractionDiameter(nIndex) / 2.0);
		_pPartContRad->push_back(pMixture->GetFractionContactDiameter(nIndex) / 2.0);
		_sMaterialsKey->push_back(pMixture->GetFractionCompound(nIndex));
	}
	else
	{
		rotQuat.RandomGenerator();
		for (const auto& particle : m_PreLoadedAgglomerate.vParticles)
		{
			CVector3 partNewCoord = newCoord + QuatRotateVector(rotQuat, particle.vecCoord);
			if (_PBC.bEnabled && !_PBC.IsCoordInPBC(partNewCoord, _dCurrentTime)) // if PBC is enabled in certain direction, move back into PBC domain
				for (size_t i = 0; i < pbcSides.Size(); ++i)
					if (pbcSides[i] != 0.0)
					{
						while (partNewCoord[i] < _PBC.currentDomain.coordBeg[i])
							partNewCoord[i] += _PBC.boundaryShift[i];
						while (partNewCoord[i] > _PBC.currentDomain.coordEnd[i])
							partNewCoord[i] -= _PBC.boundaryShift[i];
					}
			_pCoordPart->push_back(partNewCoord);
			CQuaternion partNewQuat = rotQuat * particle.qQuaternion;
			partNewQuat.Normalize();
			_pQuatPart->push_back(partNewQuat);

			_pPartRad->push_back(particle.dRadius);
			_pPartContRad->push_back(particle.dContactRadius);
			_sMaterialsKey->push_back(""); // for agglomerates specified separately
		}
	}
}

void CObjectsGenerator::CreateRandomPoint(CVector3* _pResult, const SVolumeType& _boundBox)
{
	const CVector3 vSize = _boundBox.coordEnd-_boundBox.coordBeg;
	_pResult->x = _boundBox.coordBeg.x + (double)(rand() + 1)*vSize.x/RAND_MAX;
	_pResult->y = _boundBox.coordBeg.y + (double)(rand() + 1)*vSize.y/RAND_MAX;
	_pResult->z = _boundBox.coordBeg.z + (double)(rand() + 1)*vSize.z/RAND_MAX;
}

bool CObjectsGenerator::IsOverlapped(const std::vector<CVector3>& _partCoords, const std::vector<double>& _partRadii,
	const std::vector<unsigned>& _existingPartID, const std::vector<unsigned>& _existingWallID, const CSimplifiedScene& _scene)
{
	const SPBC pbc = _scene.GetPBC();
	const SParticleStruct& parts = _scene.GetRefToParticles();
	const SWallStruct& walls = _scene.GetRefToWalls();

	// add new particles to calculator
	CContactCalculator calculator;
	for (size_t i = 0; i < _partCoords.size(); ++i)
		calculator.AddParticle(static_cast<unsigned>(i), _partCoords[i], _partRadii[i]);

	// check self-overlapping over PBC boundaries
	if (pbc.bEnabled)
	{
		std::vector<double> overlapsTarget = calculator.GetOverlaps(SPBC{});	// overlaps without PBC
		std::vector<double> overlapsActual = calculator.GetOverlaps(pbc);		// overlaps with PBC
		if (overlapsActual.size() > overlapsTarget.size())						// more overlaps than without PBC
			return true;
	}

	// to distinguish between old and new particles
	const unsigned limit = (unsigned)_partCoords.size() + 1;
	// add old particles to calculator
	for (auto id : _existingPartID)
		calculator.AddParticle(id + limit, parts.Coord(id), parts.ContactRadius(id));
	// calculate all overlaps
	const auto [ID1, ID2] = calculator.GetOverlappingIDs(pbc);

	// check if there are overlaps between old and new particles
	for (size_t i = 0; i < ID1.size(); ++i)
		if (ID1[i] < limit && ID2[i] >= limit || ID1[i] >= limit && ID2[i] < limit)
			return true;

	// check PW overlaps
	for (auto id : _existingWallID)
		for (size_t j = 0; j < _partCoords.size(); ++j)
			if (IsSphereIntersectTriangle(walls.Coordinates(id), walls.NormalVector(id), _partCoords[j], _partRadii[j]).first != EIntersectionType::NO_CONTACT)
				return true;

	return false;
}
