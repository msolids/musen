/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "BasicTypes.h"

class CContactCalculator
{
public:
	CContactCalculator();

	// Add new particle to contact calculator.
	void AddParticle(unsigned _nID, const CVector3& _vCoord, double _dRadius);
	// Returns list of all overlaps in the system: {ID1, ID2, overlap}.
	void GetAllOverlaps(std::vector<unsigned>& _vID1, std::vector<unsigned>& _vID2, std::vector<double>& _vOverlap, const SPBC& _PBC);
	// Sets new maximum allowed number of cells in each direction.
	void SetMaxCellsNumber(unsigned _nMaxCells);

private:
	struct SParticle
	{
		CVector3 vCoord;
		double dRadius;
		unsigned nID;
		SParticle(unsigned _nID, const CVector3& _vCoord, double _dRadius) : vCoord(_vCoord), dRadius(_dRadius), nID(_nID) { }
	};
	struct SOverlap
	{
		unsigned nID1;
		unsigned nID2;
		double dOverlap;
		SOverlap(unsigned _nID1, unsigned _nID2, double _dOverlap) : nID1(_nID1), nID2(_nID2), dOverlap(_dOverlap) { }
	};

	std::vector<SParticle> m_vParticles;	// Temporary list of particles to store info until RecalculatePositions() will be called.
	std::vector<std::vector<std::vector<std::vector<size_t>>>> m_Grid;
	double m_dCellSize;		// Current size of the grid's cells.
	unsigned m_nCellsX;		// Number of cells in direction X.
	unsigned m_nCellsY;		// Number of cells in direction Y.
	unsigned m_nCellsZ;		// Number of cells in direction Z.
	unsigned m_nCellsMax;	// Maximum allowed number of cells in each direction.
	unsigned m_nMaxID;		// Maximum ID of the particles.
	SVolumeType m_vSimDomain;


	// Calculates grid sizes, assign particles to cells, calculates all internal parameters. Should be called after all particles are added and before Get*() functions.
	void RecalculatePositions();

	// Calculates all overlaps of particles in specified cells.
	void CalculateOverlaps_p(std::vector<SOverlap*>& _vOverlapsList, unsigned _nX1, unsigned _nY1, unsigned _nZ1, unsigned _nX2, unsigned _nY2, unsigned _nZ2, bool _bSameCell = false);

	double GetMaxParticleRadius() const;
	void AddVirtualParticles(const SPBC& _PBC, unsigned _nMaxPartID);
};

