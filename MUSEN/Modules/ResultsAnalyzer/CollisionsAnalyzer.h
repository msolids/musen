/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ResultsAnalyzer.h"
#include "CollisionsStorage.h"
#include "ProtoFunctions.h"
#include "SceneTypes.h"

class CCollisionsAnalyzer : public CResultsAnalyzer
{
private:
	CCollisionsStorage m_Storage;
	bool m_bAllVolumes;
	bool m_bAllMaterials;
	bool m_bAllGeometries;
	bool m_bAllDiameters;
	std::string m_sMsgTime;

public:
	CCollisionsAnalyzer();
	~CCollisionsAnalyzer();

	// Set pointer to a system structure. File name where the system structure is stored will be used to construct the name of file with collisions.
	void SetSystemStructure(CSystemStructure* _pSystemStructure) override;

	// Initializes analyzer by removing all old data, prepares for writing new data.
	void ResetAndClear();
	// Flushes all data onto disc and closes file.
	void Finalize();

	// Add new set of collisions to analyzer.
	void AddCollisions(const std::vector<SCollision*>& _vCollisionsPP, const std::vector<SCollision*>& _vCollisionsPW);

	// Save some data into text file
	bool Export() override;

private:
	bool IsParticleInVolume(const std::set<size_t>& _vParticles, size_t _nIndex1, size_t _nIndex2) const;
	// Filters collisions by chosen volumes.
	std::vector<ProtoCollision*> FilterCollisionsByVolume(const std::vector<ProtoCollision*>& _vCollisions) const;
	// Returns filtered vector of collisions, which exist or appear on specified time point or time interval.
	void GetCollisionsForTime(std::vector<ProtoCollision*>& _vCollisions, double _dT1, double _dT2 = -1);
	// Returns position of collision.
	CVector3 CalculateCollisionCoordinate(ProtoCollision& _collision, double _dTime);
	// Returns energy of collision.
	double CalculateCollisionEnergy(ProtoCollision& _collision);
};

