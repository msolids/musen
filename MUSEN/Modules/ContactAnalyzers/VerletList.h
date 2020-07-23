/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ThreadPool.h"
#include "SimplifiedScene.h"

struct SCalcPerfmMetric
{
	double dAnalysisTime; // last time when analysis has been done
	double dVerletDistance;
	double dCalcTimeCoeff;
};

#define DEFAULT_TEOR_DISTANCE			1e+12
#define DEFAULT_VERLET_DISTANCE_COEFF	2

class CVerletList
{
public:
	// Contains indexes of connecting objects.
	std::vector<std::vector<unsigned>> m_PPList;
	std::vector<std::vector<unsigned>> m_PWList;

	/* Information to calculate between real-virtual particles for PP-contacts.
	 * For BOX: shifts {x, y, z}; for CYLINDER: angle of rotation {cos(a), sin(a), 0}; for not virtual contact: {0, 0, 0}.
	 * The length is equal to [partNum][collNumber].*/
	std::vector<std::vector<uint8_t>> m_PPVirtShift;
	/* Information to calculate between real-virtual particles for PW-contacts.
	 * For BOX: shifts {x, y, z}; for CYLINDER: angle of rotation {cos(a), sin(a), 0}; for not virtual contact: {0, 0, 0}.
	 * The length is equal to [partNum][collNumber].*/
	std::vector<std::vector<uint8_t>> m_PWVirtShift;

	size_t m_nThreadsNumber;	/// Number of available parallel threads.

private:
	struct SGridCell
	{
		std::vector<unsigned> vMainPartIDs; // particles which are large and p-p collisions directly calculated on this level
		std::vector<unsigned> vSecondaryPartIDs; // smaller particles which interactions are calculated on lower levels
		std::vector<unsigned> vWallIDs; // memory which has been allocated for walls
	};

	struct SGridLevel
	{
		std::vector<std::vector<std::vector<SGridCell>>> grid;
		double dCellSize;		// cell size in current grid
		double dMaxPartRadius;	// max radius of particles are being placed into this grid
		double dMinPartRadius;	// min radius of particles are being considered in contacts in this grid
		unsigned nCellsX;		// number of cells in direction X
		unsigned nCellsY;		// number of cells in direction Y
		unsigned nCellsZ;		// number of cells in direction Z
	};

	struct SEntry
	{
		unsigned ID;
		double dVal;
		friend bool operator<(const SEntry& _Inp1, const SEntry& _Inp2)
		{
			return _Inp1.dVal < _Inp2.dVal;
		}
	};
	enum class ESortCoord : unsigned { X , Y , Z, XY, YZ, XZ };
	enum class ESortDir : unsigned { Left, Right };

	SParticleStruct& m_vParticles;
	const SWallStruct& m_vWalls;
	SVolumeType m_SimDomain;
	double m_dMaxParticleRadius;
	double m_dMinParticleRadius;
	double m_dVerletDistance;
	double m_dMaxTheorWallDistance; // the maximal theoretical distance which has been overcome by particles
	bool m_bConnectedPPContact; // consider contact between already connected particles
	std::vector<SGridLevel> m_vGrid;
	uint32_t m_nCellsMax;					/// Maximum allowed number of cells in each direction.
	double m_dVerletDistanceCoeff;		/// A coefficient to calculate verlet distance.
	bool m_bAutoAdjustVerletDistance;	/// If set to true - the verlet distance will be automatically adjusted during the simulation.

	CSimplifiedScene& m_Scene;

	clock_t m_LastCPUTime;
	clock_t m_DisregardingTimeInterval;	// This time interval will not be taken into account during adjustment of verlet distance.
	double m_dLastRealTime; // last time of real process
	unsigned m_nAutoVerletDistNumerator; // AutoUpdate verlet distance called after each 10 recalculation steps
	std::vector<SCalcPerfmMetric> m_PerformHistory; // performance history

public:
	CVerletList(CSimplifiedScene& _Scene);
	void InitializeList();
	void SetPointers(const std::vector<SWallStruct>& _vWalls );
	void SetSceneInfo(const SVolumeType& _simDomain, double _dMinPartRadius, double _dMaxPartRadius, uint32_t _dMaxCellsNumber, double _dVerletCoeff, bool _bAutoAdjust);
	void SetConnectedPPContact(bool _bPPContact) { m_bConnectedPPContact = _bPPContact;  }

	void ResetCurrentData(); // set current data as not actual
	bool IsNeedToBeUpdated(double _dTimeStep, double _dMaxPartDist, double _dMaxWallVel); // Returns true if verlet list needs to be updated at the current step.
	void UpdateList(double _dCurrTime);
	void GetPWContacts(size_t _iP, std::vector<EIntersectionType>& _vIntersectionType, std::vector<CVector3>& _vContactPoint) const;
	void ReassignVirtualContacts();
	void AddDisregardingTimeInterval(const clock_t& _interval);

private:
	void AutoAdjustVerletDistance( double _dCurrentTime );
	void RecalculateGrid();	// recalculates whole grids
	void EmptyGrid();
	void SortList();		// Sorts current PP verlet list so that the src is always smaller as the dst.

	void RecalcPositions();
	void RecalcParticlesPositions();
	void RecalcWallsPositions();
	void ClearOldPositions();

	void CheckCollisionPP( const SGridLevel& _gridLevel, unsigned _nX1, unsigned _nY1, unsigned _nZ1, unsigned _nX2, unsigned _nY2, unsigned _nZ2, bool _bSameCell = false );
	void CheckCollisionPPSorted(const SGridLevel& _gridLevel, unsigned _nX1, unsigned _nY1, unsigned _nZ1, unsigned _nX2, unsigned _nY2, unsigned _nZ2, ESortCoord _dim);
	void CheckCollisionPW(const SGridLevel& _gridLevel, const SGridCell& _gridCell);

	void AddPossibleContactPP(unsigned _iPart1, unsigned _iPart2);	// Add possible contacts into the list
	void AddPossibleContactPW(unsigned _iPart, unsigned _iWall);	// Add possible contacts into the list

	// remove contacts between particles "directly" connected with bonds
	void RemoveSBContacts();

	// for improved contact detection
	void InsertParticlesToMultiSet(std::multiset<SEntry>& _set, const std::vector<unsigned>& _partIDs, ESortCoord _dim, ESortDir _dir) const;
};