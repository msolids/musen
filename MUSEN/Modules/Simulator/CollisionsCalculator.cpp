/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "CollisionsCalculator.h"

CCollisionsCalculator::CCollisionsCalculator(CSimplifiedScene& _scene, CVerletList& _list, CCollisionsAnalyzer& _analyzer) :
	m_Scene{ _scene },
	m_verletList{ _list },
	m_collisionsAnalyzer{ _analyzer }

{
}

CCollisionsCalculator::~CCollisionsCalculator()
{
	ClearCollMatrixes();
	ClearFinishedCollisionMatrixes();
}

void CCollisionsCalculator::ClearCollMatrixes()
{
	ClearCollisionMatrix( m_vCollMatrixPP );
	ClearCollisionMatrix( m_vCollMatrixPW );
}

void CCollisionsCalculator::ResizeCollMatrixes()
{
	ResizeCollisionMatrix( m_vCollMatrixPP );
	ResizeCollisionMatrix( m_vCollMatrixPW );
}

void CCollisionsCalculator::ClearCollisionMatrix( std::vector<std::vector<SCollision*>>& _matrix )
{
	ParallelFor(_matrix.size(), [&](size_t i)
	{
		for (size_t j = 0; j < _matrix[i].size(); j++)
			if (_matrix[i][j] != nullptr)
				delete _matrix[i][j]; 			// deallocate memory and remove first pointer
		_matrix[i].clear();
	});
	_matrix.clear();
}

void CCollisionsCalculator::ResizeCollisionMatrix( std::vector<std::vector<SCollision*>>& _pMatrix )
{
	if (_pMatrix.size() != m_Scene.GetTotalParticlesNumber())  // if the matrix was not initialized
	{
		ClearCollisionMatrix(_pMatrix);
		_pMatrix.resize(m_Scene.GetTotalParticlesNumber());
		for (size_t i = 0; i < _pMatrix.size(); ++i)
			_pMatrix[i].clear();
	}
}

void CCollisionsCalculator::RemoveOldCollisions( std::vector<std::vector<SCollision*>>& _pMatrix )
{
	ParallelFor(_pMatrix.size(), [&](size_t i )
	{
		size_t j = 0;
		while ( j < _pMatrix[ i ].size() )
		{
			if ( _pMatrix[ i ][ j ] == nullptr )
				_pMatrix[ i ].erase( _pMatrix[ i ].begin() + j );
			else
			{
				if ( _pMatrix[ i ][ j ]->bContactStillExist == false )
				{
					if ( !m_bAnalyzeCollisions )	// otherwise will be really removed in ClearFinishedCollisionMatrix()
						delete _pMatrix[ i ][ j ];
					_pMatrix[ i ].erase( _pMatrix[ i ].begin() + j );
				}
				else
					j = j + 1;
			}
		}
	});
}

void CCollisionsCalculator::EnableCollisionsAnalysis( bool _bEnable )
{
	m_bAnalyzeCollisions = _bEnable;
}

void CCollisionsCalculator::UpdateCollisionMatrixes( double _dTimeStep, double _dCurrentTime )
{
	ResizeCollMatrixes();

	if ( m_bAnalyzeCollisions )
		ParallelFor(m_Scene.GetTotalParticlesNumber(), [&](size_t i)
		{
			for ( size_t j = 0; j < m_vCollMatrixPP[ i ].size(); j++ )
			{
				m_vCollMatrixPP[ i ][ j ]->pSave->dTimeEnd = _dCurrentTime;
				m_vCollMatrixPP[ i ][ j ]->bContactStillExist = false;
			}
			for ( size_t j = 0; j < m_vCollMatrixPW[ i ].size(); j++ )
			{
				m_vCollMatrixPW[ i ][ j ]->pSave->dTimeEnd = _dCurrentTime;
				m_vCollMatrixPW[ i ][ j ]->bContactStillExist = false;
			}
		});
	else
		ParallelFor(m_Scene.GetTotalParticlesNumber(), [&](size_t i)
		{
			for ( size_t j = 0; j < m_vCollMatrixPP[ i ].size(); j++ )
				m_vCollMatrixPP[ i ][ j ]->bContactStillExist = false;
			for (size_t j = 0; j < m_vCollMatrixPW[i].size(); j++)
				m_vCollMatrixPW[i][j]->bContactStillExist = false;
		});

	ParallelFor(m_verletList.m_PPList.size(), [&](size_t i)
	{
		for (size_t j = 0; j < m_verletList.m_PPList[i].size(); ++j)
			CheckPPCollision(i, j, _dCurrentTime);
		CheckPWCollisions( i, _dCurrentTime );
	});


	// remove unnecessary contacts
	if ( m_bAnalyzeCollisions )
	{
		CopyFinishedPPCollisions();
		CopyFinishedPWCollisions();
	}
	RemoveOldCollisions( m_vCollMatrixPP );
	RemoveOldCollisions( m_vCollMatrixPW );
}

void CCollisionsCalculator::CheckPWCollisions( size_t _nParticle, double _dCurrentTime )
{
	if ( m_verletList.m_PWList[ _nParticle ].empty() ) return;
	const SParticleStruct& pParticles = m_Scene.GetRefToParticles();
	const SWallStruct& pWalls = m_Scene.GetRefToWalls();

	std::vector<EIntersectionType> vIntersectionType;
	std::vector<CVector3> vContactPoint;
	m_verletList.GetPWContacts( _nParticle, vIntersectionType, vContactPoint );

	auto& partColls = m_vCollMatrixPW[_nParticle]; // vector of collisions for particle _nParticle

	std::vector<size_t> newColls; // list of collisions' ID that have been activated at this time step
	for (size_t iWall = 0; iWall < vIntersectionType.size(); ++iWall)
	{
		if (vIntersectionType[iWall] == EIntersectionType::NO_CONTACT) continue;
		const unsigned nWall = m_verletList.m_PWList[_nParticle][iWall];
		SCollision* pCollision = nullptr;
		// check if this collision have been exists in the previous contact
		for (auto pOldColl : partColls)
		{
			// collision between these particles already exists, it stays virtual, or stays real, or changes its virtuality
			if (pOldColl->nSrcID == nWall && (!m_Scene.m_PBC.bEnabled || pOldColl->nVirtShift == m_verletList.m_PWVirtShift[_nParticle][iWall]))
			{
				pCollision = pOldColl;
				if (m_Scene.m_PBC.bEnabled)
					pCollision->nVirtShift = m_verletList.m_PWVirtShift[_nParticle][iWall]; // update shift info in case if real collision became virtual or vice versa
				break; // such contact was in previous step
			}
		}

		if (pCollision == nullptr) // create new contact
		{
			pCollision = new SCollision();
			pCollision->nSrcID = nWall;
			pCollision->nDstID = static_cast<unsigned>(_nParticle);
			pCollision->pSave = nullptr;
			pCollision->nInteractProp = static_cast<uint16_t>(pWalls.CompoundIndex(nWall) * m_Scene.GetCompoundsNumber() + pParticles.CompoundIndex(_nParticle));
			if (m_Scene.m_PBC.bEnabled)
				pCollision->nVirtShift = m_verletList.m_PWVirtShift[_nParticle][iWall];

			// TODO: to conform with new PBC
			if ( m_bAnalyzeCollisions )
			{
				const int nGeomIndex = GetGeometryIndex( nWall );
				for ( size_t i = 0; i < partColls.size(); ++i )
					if (partColls[ i ]->pSave->nGeomID == nGeomIndex )
					{
						// collision between these particle and geometry already exists
						pCollision->pSave = partColls[ i ]->pSave;
						pCollision->pSave->nCnt++;
						pCollision->pSave->vPtr.push_back( pCollision );
						if ( pCollision->pSave->dTimeStart == _dCurrentTime ) // it is another contact in first time point of collision
						{
							CVector3 vecNormV, vecTangV;
							CalculatePWContactVelocity( nWall, _nParticle, vContactPoint[ i ], vecNormV, vecTangV );
							pCollision->pSave->vNormVelocity = MaxLength( vecNormV, pCollision->pSave->vNormVelocity );
							pCollision->pSave->vTangVelocity = MaxLength(vecTangV, pCollision->pSave->vTangVelocity);
						}
						break;
					}

				// create completely new collision
				if ( !pCollision->pSave )
				{
					pCollision->pSave = new SSavedCollision();
					pCollision->pSave->nCnt = 1;
					pCollision->pSave->vPtr.push_back( pCollision );
					pCollision->pSave->nGeomID = nGeomIndex;
					pCollision->pSave->dTimeStart = _dCurrentTime;
					CalculatePWContactVelocity( nWall, _nParticle, vContactPoint[ iWall ], pCollision->pSave->vNormVelocity, pCollision->pSave->vTangVelocity );
					pCollision->pSave->vContactPoint = vContactPoint[ iWall ];
				}
			}

			newColls.push_back(partColls.size());
			partColls.push_back(pCollision);
		}
		pCollision->vContactVector = vContactPoint[ iWall ];
		pCollision->bContactStillExist = true;
	}

	if (!newColls.empty())
	{
		// gather deactivated collisions
		std::vector<size_t> oldColls;
		for (size_t i = 0; i < partColls.size(); ++i)
			if (!partColls[i]->bContactStillExist)
				oldColls.push_back(i);

		for (size_t iCollNew : newColls)								// iterate all new activated collisions
		{
			const auto wallIDNew = partColls[iCollNew]->nSrcID;			// wall ID in new activated collision
			bool found = false;
			for (size_t iCollOld : oldColls)							// iterate all old deactivated collisions
			{
				if (found) break;
				const auto wallIDOld = partColls[iCollOld]->nSrcID;		// wall ID in old deactivated collision
				for (auto iWall : m_Scene.m_adjacentWalls[wallIDNew])	// iterate all walls adjacent to new wall
					if (iWall == wallIDOld)								// ID of the wall in this deactivated collision belongs to the list of adjacent walls
					{
						partColls[iCollNew]->vTangOverlap = partColls[iCollOld]->vTangOverlap;	// copy data
						found = true;
						break;
					}
			}
		}
	}
}

void CCollisionsCalculator::CheckPPCollision(size_t _iPart1, size_t _iPart2, double _dCurrentTime)
{
	const size_t nPart1 = _iPart1;
	const size_t nPart2 = m_verletList.m_PPList[_iPart1][_iPart2];
	const bool bVirtContact = m_Scene.m_PBC.bEnabled && (m_verletList.m_PPVirtShift[_iPart1][_iPart2] != 0);
	const SParticleStruct& pParticles = m_Scene.GetRefToParticles();

	const CVector3 vContactVector = !bVirtContact ?
		pParticles.Coord(nPart2) - pParticles.Coord(nPart1) :
		GetVirtualProperty(pParticles.Coord(nPart2), m_verletList.m_PPVirtShift[_iPart1][_iPart2], m_Scene.m_PBC) - pParticles.Coord(nPart1);
	const double dSquaredDistance = SquaredLength(vContactVector);
	if ((pParticles.ContactRadius(nPart1) + pParticles.ContactRadius(nPart2))*(pParticles.ContactRadius(nPart1) + pParticles.ContactRadius(nPart2)) > dSquaredDistance)
	{
		SCollision* pCollision = nullptr;
		// check if this collision have been existed in the previous contact
		for (auto pOldColl : m_vCollMatrixPP[nPart1])
		{
			if (pOldColl->nDstID == nPart2 && (!m_Scene.m_PBC.bEnabled || pOldColl->nVirtShift == m_verletList.m_PPVirtShift[_iPart1][_iPart2]))
			{
				pCollision = pOldColl;
				if (m_Scene.m_PBC.bEnabled)
					pCollision->nVirtShift = m_verletList.m_PPVirtShift[_iPart1][_iPart2]; // update shift info in case if real collision became virtual or vice versa
				break; // such contact was in previous step
			}
		}

		if (pCollision == nullptr)  // allocate memory for a new collision
		{
			pCollision = new SCollision();
			pCollision->nSrcID = static_cast<unsigned>(nPart1);
			pCollision->nDstID = static_cast<unsigned>(nPart2);
			pCollision->pSave = nullptr;
			pCollision->nInteractProp = static_cast<uint16_t>(pParticles.CompoundIndex(nPart1) * m_Scene.GetCompoundsNumber() + pParticles.CompoundIndex(nPart2));
			pCollision->dEquivRadius = pParticles.ContactRadius(nPart1)*pParticles.ContactRadius(nPart2) / (pParticles.ContactRadius(nPart1) + pParticles.ContactRadius(nPart2));
			pCollision->dEquivMass = pParticles.Mass(nPart1)*pParticles.Mass(nPart2) / (pParticles.Mass(nPart1) + pParticles.Mass(nPart2));
			if (m_Scene.m_PBC.bEnabled)
				pCollision->nVirtShift = bVirtContact ? m_verletList.m_PPVirtShift[_iPart1][_iPart2] : 0;

			// TODO: to conform with new PBC
			if (m_bAnalyzeCollisions)
			{
				pCollision->pSave = new SSavedCollision();
				pCollision->pSave->nCnt = 1;
				pCollision->pSave->dTimeStart = _dCurrentTime;

				//obtain contact point
				const CVector3 vecContactPoint = pParticles.Coord(nPart1) + (pParticles.Coord(nPart2) - pParticles.Coord(nPart1))*pParticles.ContactRadius(nPart1) / (pParticles.ContactRadius(nPart1) + pParticles.ContactRadius(nPart2));
				const CVector3 vecRc = vecContactPoint - pParticles.Coord(nPart1);
				const CVector3 vecRc2 = vecContactPoint - pParticles.Coord(nPart2);
				const CVector3 vecNormalVector = vecRc / vecRc.Length();
				// relative velocity (normal and tangential)
				const CVector3 vecRelVelocity = pParticles.Vel(nPart2) - pParticles.Vel(nPart1) + vecRc * pParticles.AnglVel(nPart1) - vecRc2 * pParticles.AnglVel(nPart2);
				const CVector3 vecRelVelNormal = vecNormalVector * (DotProduct(vecNormalVector, vecRelVelocity));
				const CVector3 vecRelVelTang = vecRelVelocity - vecRelVelNormal;

				pCollision->pSave->vNormVelocity = vecRelVelNormal;
				pCollision->pSave->vTangVelocity = vecRelVelTang;
				pCollision->pSave->vContactPoint = vecContactPoint;
			}

			m_vCollMatrixPP[nPart1].push_back(pCollision);
		}
		pCollision->dNormalOverlap = pParticles.ContactRadius(nPart1) + pParticles.ContactRadius(nPart2) - sqrt(dSquaredDistance);
		pCollision->vContactVector = vContactVector;

		// indicates that contact is still exists and should not be removed
		pCollision->bContactStillExist = true;
	}
}

int CCollisionsCalculator::GetGeometryIndex( unsigned _nWallIndex ) const
{
	const unsigned ind = m_Scene.GetRefToWalls().InitIndex(_nWallIndex);
	for ( size_t i = 0; i < m_pSystemStructure->GeometriesNumber(); ++i )
	{
		CRealGeometry *pGeo = m_pSystemStructure->Geometry(i);
		const auto& planes = pGeo->Planes();
		if ( std::find(planes.begin(), planes.end(), ind ) != planes.end() )
			return static_cast<int>(i);
	}
	return -1;
}

void CCollisionsCalculator::CalculatePWContactVelocity( size_t _nWall, size_t _nPart, const CVector3& _vecContactPoint, CVector3& _vecNormV, CVector3& _vecTangV ) const
{
	SWallStruct& pWall = m_Scene.GetRefToWalls();
	// obtain additional parameters
	SParticleStruct& pParticles = m_Scene.GetRefToParticles();
	const CVector3 vecRc = pParticles.Coord(_nPart) - _vecContactPoint;
	const CVector3 vecNormalVector = vecRc / vecRc.Length();
	// relative velocity (normal and tangential)
	CVector3 vecRelVelocity = pParticles.Vel(_nPart) - pWall.Vel(_nWall) + vecNormalVector*pParticles.AnglVel(_nPart)*pParticles.Radius(_nPart);
	if (!pWall.RotVel(_nWall).IsZero()) // velocity due to rotation
		vecRelVelocity += (_vecContactPoint - pWall.RotCenter(_nWall))*pWall.RotVel(_nWall);
	_vecNormV = vecNormalVector*(DotProduct(vecNormalVector, vecRelVelocity));
	_vecTangV = vecRelVelocity - _vecNormV;
}

void CCollisionsCalculator::ClearFinishedCollisionMatrixes()
{
	ClearFinishedCollisionMatrix( m_vFinishedCollisionsPP );
	ClearFinishedCollisionMatrix( m_vFinishedCollisionsPW );
}

void CCollisionsCalculator::ClearFinishedCollisionMatrix( std::vector<SCollision*>& _matrix )
{
	for (size_t i = 0; i < _matrix.size(); ++i)
		if (_matrix[i])
		{
			if (_matrix[i]->pSave)
				delete _matrix[i]->pSave;
			delete _matrix[i];
		}
	_matrix.clear();
}

void CCollisionsCalculator::SaveCollisions()
{
	if ( m_bAnalyzeCollisions && (m_vFinishedCollisionsPP.size() + m_vFinishedCollisionsPW.size() > COLLISIONS_NUMBER_TO_SAVE))
	{
		RecalculateSavedIDs();
		m_collisionsAnalyzer.AddCollisions( m_vFinishedCollisionsPP, m_vFinishedCollisionsPW );
		ClearFinishedCollisionMatrix( m_vFinishedCollisionsPP );
		ClearFinishedCollisionMatrix( m_vFinishedCollisionsPW );
	}
}

void CCollisionsCalculator::SaveRestCollisions()
{
	if ( !m_bAnalyzeCollisions ) return;
	ParallelFor(m_Scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		for ( size_t j = 0; j < m_vCollMatrixPP[ i ].size(); j++ )
		{
			m_vCollMatrixPP[ i ][ j ]->pSave->dTimeEnd = -1;	// not finished collision
			m_vCollMatrixPP[ i ][ j ]->bContactStillExist = false;
		}
		for ( size_t j = 0; j < m_vCollMatrixPW[ i ].size(); j++ )
		{
			m_vCollMatrixPW[ i ][ j ]->pSave->dTimeEnd = -1; 	// not finished collision
			m_vCollMatrixPW[ i ][ j ]->bContactStillExist = false;
		}
	});
	CopyFinishedPPCollisions();
	CopyFinishedPWCollisions();
	RemoveOldCollisions( m_vCollMatrixPP );
	RemoveOldCollisions( m_vCollMatrixPW );

	RecalculateSavedIDs();
	m_collisionsAnalyzer.AddCollisions( m_vFinishedCollisionsPP, m_vFinishedCollisionsPW );
	m_collisionsAnalyzer.Finalize();

	ClearFinishedCollisionMatrix( m_vFinishedCollisionsPP );
	ClearFinishedCollisionMatrix( m_vFinishedCollisionsPW );
}

void CCollisionsCalculator::CopyFinishedPPCollisions()
{
	if (!m_pSystemStructure->GetPBC().bEnabled)	// no PBC
	{
		for (size_t i = 0; i < m_vCollMatrixPP.size(); ++i)
			for (size_t j = 0; j < m_vCollMatrixPP[i].size(); ++j)
				if ((!m_vCollMatrixPP[i][j]->bContactStillExist) && (m_vCollMatrixPP[i][j]->pSave->dTimeStart != m_vCollMatrixPP[i][j]->pSave->dTimeEnd))
					m_vFinishedCollisionsPP.push_back(m_vCollMatrixPP[i][j]);
	}
	// TODO: collisions saving with new PBC.
	else // if PBC exists
	{
	//	for (size_t i = 0; i < m_vCollMatrixPP.size(); ++i)
	//		for (size_t j = 0; j < m_vCollMatrixPP[i].size(); ++j)
	//			if ((!m_vCollMatrixPP[i][j]->bContactStillExist) && (m_vCollMatrixPP[i][j]->pSave->dTimeStart != m_vCollMatrixPP[i][j]->pSave->dTimeEnd))
	//			{
	//				unsigned iSrc = m_vCollMatrixPP[i][j]->nSrcID;
	//				unsigned iDst = m_vCollMatrixPP[i][j]->nDstID;
	//				const bool srcIsReal = iSrc <= m_scene.GetRealParticlesNumber();
	//				const bool dstIsReal = iDst <= m_scene.GetRealParticlesNumber();
	//				if (srcIsReal && dstIsReal)	// real with real
	//					m_vFinishedCollisionsPP.push_back(m_vCollMatrixPP[i][j]);
	//				else // virtual with real (virtual with virtual is not possible)
	//				{
	//					std::vector<SParticleStruct>& parts = m_scene.GetRefToParticles();
	//					unsigned iReal = srcIsReal ? iSrc : iDst;	// index of a real particle
	//					unsigned iRealOfVirtual = srcIsReal ? parts[iDst].nInitIndex : parts[iSrc].nInitIndex; // real index of virtual particle
	//					if (iReal <= iRealOfVirtual)	// to omit duplication of collision
	//					{
	//						m_vFinishedCollisionsPP.push_back(m_vCollMatrixPP[i][j]);
	//						(srcIsReal ? m_vFinishedCollisionsPP.back()->nDstID : m_vFinishedCollisionsPP.back()->nSrcID) = iRealOfVirtual;	// replace index of virtual particle with the real one
	//					}
	//				}
	//			}
	}
}

void CCollisionsCalculator::CopyFinishedPWCollisions()
{
	for ( size_t i = 0; i < m_vCollMatrixPW.size(); ++i )
		for (size_t j = 0; j < m_vCollMatrixPW[i].size(); ++j)
		{
			SCollision* coll = m_vCollMatrixPW[i][j];
			if ((!coll->bContactStillExist) && (coll->pSave->dTimeStart != coll->pSave->dTimeEnd))
			{
				if (coll->pSave->nCnt == 1)	// collision is ended
					m_vFinishedCollisionsPW.push_back(coll);
				else	// not fully finished collision
				{
					coll->pSave->nCnt--;
					// remove pointer to itself
					std::vector<SCollision*>::iterator it = std::find(coll->pSave->vPtr.begin(), coll->pSave->vPtr.end(), coll);
					if (it != coll->pSave->vPtr.end())
						coll->pSave->vPtr.erase(it);
					// delete collision
					delete m_vCollMatrixPW[i][j];
					m_vCollMatrixPW[i][j] = nullptr;
				}
			}
		}
}

void CCollisionsCalculator::RecalculateSavedIDs()
{
	for (unsigned i = 0; i < m_vFinishedCollisionsPP.size(); ++i)
	{
		m_vFinishedCollisionsPP[i]->nSrcID = m_Scene.GetRefToParticles().InitIndex(m_vFinishedCollisionsPP[i]->nSrcID);
		m_vFinishedCollisionsPP[i]->nDstID = m_Scene.GetRefToParticles().InitIndex(m_vFinishedCollisionsPP[i]->nDstID);
	}
	for ( unsigned i = 0; i < m_vFinishedCollisionsPW.size(); ++i )
	{
		m_vFinishedCollisionsPW[ i ]->nSrcID = m_Scene.GetRefToWalls().InitIndex(m_vFinishedCollisionsPW[i]->nSrcID);
		m_vFinishedCollisionsPW[ i ]->nDstID = m_Scene.GetRefToParticles().InitIndex(m_vFinishedCollisionsPW[i]->nDstID);
	}
}

void CCollisionsCalculator::CalculateStatisticInfo( std::vector<std::vector<SCollision*>>& _matrix )
{
	ParallelFor(_matrix.size(), [&](size_t i)
	{
		for ( size_t j = 0; j < _matrix[ i ].size(); j++ )
		{
			SCollision* pColl = _matrix[i][j];
			if (pColl->pSave->nCnt < 2 )
			{
				pColl->pSave->vMaxTotalForce = MaxLength(pColl->vTotalForce, pColl->pSave->vMaxTotalForce);
				pColl->pSave->vMaxTangForce = MaxLength(pColl->vTangForce, pColl->pSave->vMaxTangForce);
				CVector3 vNormF = pColl->vTotalForce - pColl->vTangForce;
				pColl->pSave->vMaxNormForce = MaxLength(vNormF, pColl->pSave->vMaxNormForce);
			}
			else
			{
				CVector3 vSumTotF(0), vSumTanF(0);
				for ( size_t k = 0; k < pColl->pSave->vPtr.size(); ++k )
				{
					vSumTotF += pColl->pSave->vPtr[ k ]->vTotalForce;
					vSumTanF += pColl->pSave->vPtr[ k ]->vTangForce;
				}
				pColl->pSave->vMaxTotalForce = MaxLength(vSumTotF, pColl->pSave->vMaxTotalForce);
				pColl->pSave->vMaxTangForce = MaxLength(vSumTanF, pColl->pSave->vMaxTangForce);
				CVector3 vSumNormF = vSumTotF - vSumTanF;
				pColl->pSave->vMaxNormForce = MaxLength(vSumNormF, pColl->pSave->vMaxNormForce);
			}
		}
	});
}

void CCollisionsCalculator::CalculateTotalStatisticsInfo()
{
	if ( m_bAnalyzeCollisions )
	{
		CalculateStatisticInfo( m_vCollMatrixPP );
		CalculateStatisticInfo( m_vCollMatrixPW );
	}
}