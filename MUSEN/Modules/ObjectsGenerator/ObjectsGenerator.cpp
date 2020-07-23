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
	m_dGenerationRate = 100;
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
	case 1: seed = seed/2;
	case 0: break;
	}
	srand( seed );
	if ( m_pAgglomDB->GetAgglomerate( m_sAgglomerateKey )==NULL ) return;
	m_PreLoadedAgglomerate = *m_pAgglomDB->GetAgglomerate( m_sAgglomerateKey );

	if ( !m_bGenerateMixture) // shift agglomerate center of mass into point with coord 0 and automatically scale it
	{
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
		}
		for (unsigned i = 0; i < m_PreLoadedAgglomerate.vBonds.size(); i++)
			m_PreLoadedAgglomerate.vBonds[i].dRadius *= m_dAgglomerateScaleFactor;
	}
}

size_t CObjectsGenerator::Generate(double _dCurrentTime, CSystemStructure* _pSystemStructure, CSimplifiedScene& _Scene)
{
	if (!m_bActive) return 0;
	if ( (_dCurrentTime < m_dStartGenerationTime) || (_dCurrentTime > m_dEndGenerationTime )  ) return 0;
	if (_dCurrentTime < m_dLastGenerationTime + m_dUpdateStep) return 0;
	unsigned nNewObjects = (unsigned)floor((_dCurrentTime - m_dLastGenerationTime)*m_dGenerationRate);
	if ( nNewObjects < 1 ) return 0;

	CAnalysisVolume* pGenVolume = _pSystemStructure->GetAnalysisVolume( m_sVolumeKey );
	if ( pGenVolume == NULL ) return 0;

	SVolumeType boundingBox = pGenVolume->BoundingBox(_dCurrentTime);
	std::vector<unsigned> vPartInVolume; // vector of indexes (in simplified scene) of all particles situated in the specified volume
	std::vector<unsigned> vWallsInVolume; // vector of indexes (in simplified scene) of all walls situated in the specified volume
	_Scene.GetAllParticlesInVolume( boundingBox, &vPartInVolume );
	_Scene.GetAllWallsInVolume( boundingBox, &vWallsInVolume );

	SWallStruct& pWalls = _Scene.GetRefToWalls();
	std::vector<CVector3> vCoordNewPart;
	std::vector<CQuaternion> vQuatNewPart;
	std::vector<double> vRadiiNewPart;
	std::vector<double> vContRadiiNewPart;
	std::vector<std::string> vMaterialsKey;
	size_t nCreatedParticles = 0;
	CInsideVolumeChecker insideChecker( pGenVolume, _dCurrentTime );

	// for checking that new object are not situtated in any real volume
	std::vector<CInsideVolumeChecker> vInRealVolumeCheckers;
	if ( !m_bInsideGeometries )
		for (unsigned i = 0; i < _pSystemStructure->GetGeometriesNumber(); i++)
		{
			SGeometryObject* pGeom = _pSystemStructure->GetGeometry(i);
			std::vector<STriangleType> vTriangles;
			for (size_t i = 0; i < pGeom->vPlanes.size(); ++i)
			{
				size_t nIndex = _Scene.m_vNewIndexes[pGeom->vPlanes[i]];
				vTriangles.push_back(STriangleType{ pWalls.Vert1(nIndex),  pWalls.Vert2(nIndex),  pWalls.Vert3(nIndex) });
			}
			CInsideVolumeChecker newChecker;
			newChecker.SetTriangles(vTriangles, CVector3(0));
			vInRealVolumeCheckers.push_back(newChecker);
		}

	for ( unsigned i = 0; i <nNewObjects; i++ )
	{
		unsigned nMaxAttempt = 30;
		bool bSuccess = false;
		while ( (nMaxAttempt > 0) && (!bSuccess) )
		{
			// generate single object
			GenerateNewObject(&vCoordNewPart, &vQuatNewPart, &vRadiiNewPart, &vContRadiiNewPart, &vMaterialsKey, boundingBox);
			bSuccess = true;

			std::vector<size_t> vID = insideChecker.GetSpheresTotallyInside(vCoordNewPart, vRadiiNewPart);
			if ( vID.size() != vCoordNewPart.size() ) // not all particles in volume
				bSuccess = false;
			if ( bSuccess )
				bSuccess = !IsOverlapped( vCoordNewPart, vContRadiiNewPart, vPartInVolume, vWallsInVolume, _Scene );

			if (( bSuccess ) && (!m_bInsideGeometries))
				for (size_t j = 0; j < vInRealVolumeCheckers.size(); j++)
				{
					vID = vInRealVolumeCheckers[j].GetSpheresTotallyInside(vCoordNewPart, vRadiiNewPart);
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
			nCreatedParticles += (unsigned)vCoordNewPart.size();
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
			for ( size_t j = 0; j < vCoordNewPart.size(); j++ )  // add particles
			{
				CSphere* pNewSphere = (CSphere*)_pSystemStructure->AddObject(SPHERE, vFreeIDs[j]);
				vTempNewIndexes.push_back( pNewSphere->m_lObjectID );
				pNewSphere->SetRadius(vRadiiNewPart[j]);
				pNewSphere->SetContactRadius(vContRadiiNewPart[j]);
				if (m_bGenerateMixture) // this is sphere
					pNewSphere->SetCompound(m_pMaterialsDB->GetCompound(vMaterialsKey[j]));
				else // this is agglomerate
					pNewSphere->SetCompound(m_pMaterialsDB->GetCompound(m_partMaterials[m_PreLoadedAgglomerate.vParticles[j].sCompoundAlias]));
				pNewSphere->SetCoordinates( _dCurrentTime, vCoordNewPart[ j ] );
				pNewSphere->SetOrientation(_dCurrentTime, vQuatNewPart[j]);
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
						CSolidBond* pNewBond = static_cast<CSolidBond*>(_pSystemStructure->AddObject(SOLID_BOND, vFreeIDsBonds[j]));
						const SAggloBond& bond = m_PreLoadedAgglomerate.vBonds[j];
						pNewBond->SetDiameter(2 * bond.dRadius);
						pNewBond->SetCompound(m_pMaterialsDB->GetCompound(m_bondMaterials[bond.sCompoundAlias]));
						pNewBond->m_nLeftObjectID = vTempNewIndexes[bond.nLeftID];
						pNewBond->m_nRightObjectID = vTempNewIndexes[bond.nRightID];
						pNewBond->SetStartActivityTime(_dCurrentTime);
						pNewBond->SetEndActivityTime(1e+300);
						pNewBond->SetInitialLength(_pSystemStructure->GetBond(_dCurrentTime, pNewBond->m_lObjectID).Length());
						_Scene.AddSolidBond(pNewBond->m_lObjectID, _dCurrentTime);
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
	unsigned nNewObjects = (unsigned)floor((_dCurrentTime - m_dLastGenerationTime)*m_dGenerationRate);
	if (nNewObjects < 1) return false;
	return true;
}

void CObjectsGenerator::GenerateNewObject(std::vector<CVector3>* _pCoordPart, std::vector<CQuaternion>* _pQuatPart, std::vector<double>* _pPartRad, std::vector<double>* _pPartContRad,
	std::vector<std::string>* _sMaterialsKey, const SVolumeType& _boundBox)
{
	CVector3 newCoord;
	CQuaternion rotQuat;
	_pCoordPart->clear();
	_pQuatPart->clear();
	_pPartRad->clear();
	_pPartContRad->clear();
	_sMaterialsKey->clear();
	CreateRandomPoint( &newCoord, _boundBox );
	if ( m_bGenerateMixture )
	{
		const CMixture* pMixture = m_pMaterialsDB->GetMixture(m_sMixtureKey);
		if (!pMixture) return;
		double dRand = ((double)rand()) / RAND_MAX; // generate random value
		double dTempSum = 0;
		size_t nIndex = 0;
		while (nIndex < pMixture->FractionsNumber())
		{
			dTempSum += pMixture->GetFractionValue(nIndex);
			if (dTempSum >= dRand) break;
			nIndex++;
		}

		_pCoordPart->push_back( newCoord );
		rotQuat.RandomGenerator();
		_pQuatPart->push_back(rotQuat);
		_pPartRad->push_back(pMixture->GetFractionDiameter(nIndex) / 2.0);
		_pPartContRad->push_back(pMixture->GetFractionContactDiameter(nIndex) / 2.0);
		_sMaterialsKey->push_back(pMixture->GetFractionCompound(nIndex));
	}
	else
	{
		rotQuat.RandomGenerator();
		for (size_t i = 0; i<m_PreLoadedAgglomerate.vParticles.size(); i++)
		{
			// NO ROTATION
			/*_pCoordPart->push_back(newCoord + m_PreLoadedAgglomerate.vParticles[i].vecCoord);
			_pQuatPart->push_back(m_PreLoadedAgglomerate.vParticles[i].qQuaternion);*/

			// ROTATION WITH QUATERNION
			_pCoordPart->push_back(newCoord + QuatRotateVector(rotQuat, m_PreLoadedAgglomerate.vParticles[i].vecCoord));
			CQuaternion partNewQuat = rotQuat * m_PreLoadedAgglomerate.vParticles[i].qQuaternion;
			partNewQuat.Normalize();
			_pQuatPart->push_back(partNewQuat);

			_pPartRad->push_back(m_PreLoadedAgglomerate.vParticles[i].dRadius);
			_pPartContRad->push_back(m_PreLoadedAgglomerate.vParticles[i].dContactRadius);
			_sMaterialsKey->push_back(""); // for agglomerates specified separately
		}
	}
}

void inline CObjectsGenerator::CreateRandomPoint(CVector3* _pResult, const SVolumeType& _boundBox)
{
	CVector3 vSize = _boundBox.coordEnd-_boundBox.coordBeg;
	_pResult->x = _boundBox.coordBeg.x + (double)(rand() + 1)*vSize.x/RAND_MAX;
	_pResult->y = _boundBox.coordBeg.y + (double)(rand() + 1)*vSize.y/RAND_MAX;
	_pResult->z = _boundBox.coordBeg.z + (double)(rand() + 1)*vSize.z/RAND_MAX;
}

void CObjectsGenerator::CreateRandomAngle( CVector3* _pResult )				// probably does not create uniform distribution over all possible rotations -> use CreateRandomQuaternion
{
	_pResult->x = ((double)(rand() + 1)*360/RAND_MAX)*PI/180.0;
	_pResult->y = ((double)(rand() + 1)*360/RAND_MAX)*PI/180.0;
	_pResult->z = ((double)(rand() + 1)*360/RAND_MAX)*PI/180.0;
}

bool CObjectsGenerator::IsOverlapped(const std::vector<CVector3>& _vCoordPart, const std::vector<double>& _vPartContRad, const std::vector<unsigned>& _vExistedPartID,
	const std::vector<unsigned>& _nExistedWallsID, const CSimplifiedScene& _Scene)
{
	const SParticleStruct& parts = _Scene.GetRefToParticles();
	const SWallStruct& walls = _Scene.GetRefToWalls();

	bool bOverlap = false;
	// firstly check PP overlap
	for ( unsigned i = 0; i < _vExistedPartID.size(); i++ )
		for ( unsigned j = 0; j< _vCoordPart.size(); j++ )
			if ( !bOverlap )
				if ( SquaredLength( _vCoordPart[ j ], parts.Coord(_vExistedPartID[i])) < pow( parts.ContactRadius(_vExistedPartID[i]) + _vPartContRad[ j ], 2 ) )
					bOverlap = true;

	if ( bOverlap ) return true;
	// check PW overlap
	CVector3 vContactPoint;
	for (unsigned i = 0; i < _nExistedWallsID.size(); i++)
		for (unsigned j = 0; j < _vCoordPart.size(); j++)
			if (!bOverlap)
				if (IsSphereIntersectTriangle(walls.Coordinates(_nExistedWallsID[i]), walls.NormalVector(_nExistedWallsID[i]), _vCoordPart[j], _vPartContRad[j]).first != EIntersectionType::NO_CONTACT)
					bOverlap = true;
	return bOverlap;
}
