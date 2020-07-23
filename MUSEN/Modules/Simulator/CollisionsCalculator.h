/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "SystemStructure.h"
#include "AbstractDEMModel.h"
#include "MUSENComponent.h"
#include "ThreadPool.h"
#include "VerletList.h"
#include "CollisionsAnalyzer.h"

class CCollisionsCalculator : public CMusenComponent
{
protected:
	std::vector<SCollision*> m_vFinishedCollisionsPP;	// list of finished particle-particle
	std::vector<SCollision*> m_vFinishedCollisionsPW;	// list of finished particle-wall collisions

	CSimplifiedScene& m_Scene;
	CVerletList* m_pVerletList;
	CCollisionsAnalyzer* m_pCollisionsAnalyzer;
	bool m_bAnalyzeCollisions;

public:
	std::vector< std::vector<SCollision*>> m_vCollMatrixPP;
	std::vector< std::vector<SCollision*>> m_vCollMatrixPW;

private:
	void ClearCollisionMatrix( std::vector<std::vector<SCollision*>>& _pMatrix ); // removes all entries from collision matrix
	void ResizeCollisionMatrix( std::vector<std::vector<SCollision*>>& _pMatrix );

	// remove all contacts which are not active in the current time step
	void RemoveOldCollisions( std::vector<std::vector<SCollision*>>& _pMatrix );

	void CheckPPCollision(size_t _iPart1, size_t _iPart2, double _dCurrentTime); // check the collision between two particles
	void CheckPWCollisions( size_t _nParticle, double _dCurrentTime ); // check the between particle and wall

	/// Return index of a geometry, which contains triangular wall with index _nWallIndex.
	int GetGeometryIndex( unsigned _nWallIndex ) const;
	void CalculatePWContactVelocity(size_t _nWall, size_t _nPart, const CVector3& _vecContactPoint, CVector3& _vecNormV, CVector3& _vecTangV) const;
	// remove all finished contacts from temporary matrix
	void ClearFinishedCollisionMatrix( std::vector<SCollision*>& _matrix );
	void CalculateStatisticInfo( std::vector<std::vector<SCollision*>>& _matrix );

	// saves pointers to finished collisions
	void CopyFinishedPPCollisions();
	void CopyFinishedPWCollisions();

public:
	CCollisionsCalculator(CSimplifiedScene& _Scene);
	~CCollisionsCalculator();

	void SetPointers(CVerletList* _pList, CCollisionsAnalyzer* _pCollAnalyzer );

	void ClearCollMatrixes();
	void ResizeCollMatrixes();

	// update the matrix of collision between particles
	void UpdateCollisionMatrixes( double _dTimeStep, double _dCurrentTime );

	void EnableCollisionsAnalysis( bool _bEnable );
	void ClearFinishedCollisionMatrixes();

	void CalculateTotalStatisticsInfo();
	void SaveRestCollisions();
	void SaveCollisions();
	void RecalculateSavedIDs();
};