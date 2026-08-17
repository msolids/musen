/* Copyright (c) 2013-2020, MUSEN Development Team.
   Copyright (c) 2026, DyssolTEC GmbH.
   All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "GeometricFunctions.h"
#include "SimplifiedScene.h"
#include "ThreadPool.h"


struct SCalcPerfmMetric
{
	double dAnalysisTime; // last time when analysis has been done
	double dVerletDistance;
	double dCalcTimeCoeff;
};

#define DEFAULT_TEOR_DISTANCE			1e+12
#define DEFAULT_VERLET_DISTANCE_COEFF	2

constexpr uint32_t c_defaultVerletMaxCells = 50;	///< Default cube root of the maximum total number of grid cells for Verlet lists calculation.

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

private:
	/**
	 * @brief Object indices stored for a single grid cell. */
	struct SCellSpan
	{
		const uint32_t* ids;	///< First index of the cell.
		size_t number;			///< Number of indices in the cell.
		[[nodiscard]] const uint32_t* begin() const { return ids; }
		[[nodiscard]] const uint32_t* end() const { return ids + number; }
		[[nodiscard]] size_t size() const { return number; }
		[[nodiscard]] bool empty() const { return number == 0; }
		uint32_t operator[](size_t _i) const { return ids[_i]; }
	};

	/**
	 * @brief Object indices of all cells of one grid level, grouped by cell in a compressed sparse row form. */
	class CCellLists
	{
		std::vector<uint32_t> m_offsets;	///< Position of each cell in m_ids; the size is the number of cells plus one.
		std::vector<uint32_t> m_ids;		///< Object indices, grouped by cell.

	public:
		/**
		 * @brief Removes all objects, leaving the lists valid and empty.
		 * @param _cellsNumber Number of cells in the grid level. */
		void Reset(size_t _cellsNumber);
		/**
		 * @brief Groups object indices by grid cell.
		 * @tparam T Callable void(size_t, uint32_t).
		 * @param _cellsNumber Number of cells in the grid level.
		 * @param _forEachEntry Visitor which calls the function it receives with a cell index and an object index, once per entry. */
		template<typename T>
		void Build(size_t _cellsNumber, const T& _forEachEntry);
		/**
		 * @brief Groups object indices by grid cell, taking the cell of each object from a list.
		 * @param _cellsNumber Number of cells in the grid level.
		 * @param _cellIndex Cell of each object; the objects outside the grid are skipped. */
		void Build(size_t _cellsNumber, const std::vector<uint32_t>& _cellIndex);
		/**
		 * @brief Returns the objects stored for a cell.
		 * @param _iCell Linear index of the cell.
		 * @return Indices of the objects placed into the cell. */
		[[nodiscard]] SCellSpan Cell(size_t _iCell) const;
	};

	/**
	 * @brief Inclusive range of cells covered by an object.
	 * @details A default-constructed range is empty, since its first cell lies behind its last one. */
	struct SCellRange
	{
		uint32_t minX{ 1 }, minY{ 1 }, minZ{ 1 };	///< First covered cell in each direction.
		uint32_t maxX{ 0 }, maxY{ 0 }, maxZ{ 0 };	///< Last covered cell in each direction.
	};

	/**
	 * @brief One level of the grid: a uniform division of the grid domain into cubic cells. */
	struct SGridLevel
	{
		CCellLists mainParts;	///< Particles which are placed into this level and whose contacts are searched here.
		CCellLists secondParts;	///< Smaller particles, kept here only to be paired with the main ones; their mutual contacts are searched on a finer level.
		CCellLists walls;		///< Walls whose bounding box overlaps the cell.
		double cellSize;		///< Edge length of a cell.
		double maxPartRadius;	///< Largest contact radius of a particle placed into this level.
		double minPartRadius;	///< Smallest contact radius of a particle placed into this level; zero on the finest one.
		uint32_t cellsX;		///< Number of cells in direction X.
		uint32_t cellsY;		///< Number of cells in direction Y.
		uint32_t cellsZ;		///< Number of cells in direction Z.

		/**
		 * @brief Returns the total number of cells of the level. */
		[[nodiscard]] size_t CellsNumber() const { return static_cast<size_t>(cellsX) * cellsY * cellsZ; }
		/**
		 * @brief Converts the coordinates of a cell into its linear index.
		 * @param _x Cell coordinate in direction X.
		 * @param _y Cell coordinate in direction Y.
		 * @param _z Cell coordinate in direction Z.
		 * @return Linear index of the cell. */
		[[nodiscard]] size_t CellIndex(uint32_t _x, uint32_t _y, uint32_t _z) const { return (static_cast<size_t>(_x) * cellsY + _y) * cellsZ + _z; }
	};

	/**
	 * @brief Auxiliary struct for flipping wrongly sorted PP list pairs. */
	struct SReversedPair
	{
		uint32_t dst;	///< Index the contact moves to.
		uint32_t src;	///< Index the contact was emitted in.
		uint8_t shift;	///< Periodic shift, inverted for the new direction (for PBC).
	};

	struct SEntry
	{
		unsigned id;
		double val;
		SEntry(unsigned _id, double _val) : id{ _id }, val{ _val }{}
		friend bool operator<(const SEntry& _e1, const SEntry& _e2)
		{
			return _e1.val < _e2.val;
		}
	};
	enum class ESortCoord : unsigned { X , Y , Z, XY, YZ, XZ };
	enum class ESortDir : unsigned { Left, Right };

	SParticleStruct& m_vParticles;
	const SWallStruct& m_vWalls;
	SVolumeType m_SimDomain;
	SVolumeType m_gridDomain{};      ///< Region covered by the grid: a padded bounding box of all particles.
	SVolumeType m_partBoundingBox{}; ///< Bounding box of the centres of all active particles, virtual ones included.
	double m_dMaxParticleRadius;
	double m_dMinParticleRadius;
	double m_dVerletDistance;
	double m_dMaxTheorWallDistance; // the maximal theoretical distance which has been overcome by particles
	bool m_bConnectedPPContact; // consider contact between already connected particles
	std::vector<SGridLevel> m_grid;		///< Grid levels, from the coarsest to the finest.
	uint32_t m_nCellsMax;				///< Cube root of the maximum allowed total number of grid cells.
	double m_dVerletDistanceCoeff;		/// A coefficient to calculate verlet distance.
	bool m_bAutoAdjustVerletDistance;	/// If set to true - the verlet distance will be automatically adjusted during the simulation.

	CSimplifiedScene& m_Scene;

	clock_t m_LastCPUTime;
	clock_t m_DisregardingTimeInterval;	// This time interval will not be taken into account during adjustment of verlet distance.
	double m_dLastRealTime; // last time of real process
	unsigned m_nAutoVerletDistNumerator; // AutoUpdate verlet distance called after each 10 recalculation steps
	std::vector<SCalcPerfmMetric> m_PerformHistory; // performance history

	std::vector<std::vector<SReversedPair>> m_reversedPairs;	///< Contacts which SortList moves between indices, bucketed as [writing thread][receiving thread].

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
	/**
	 * @brief Recalculates the bounding box of all active particles. */
	void UpdateParticlesBoundingBox();
	/**
	 * @brief Checks whether the grid has to be rebuilt to fit the current particles.
	 * @return True if the particles do not fit the grid anymore, or the grid is much larger than needed. */
	[[nodiscard]] bool IsGridRefitNeeded() const;
	void AutoAdjustVerletDistance( double _dCurrentTime );
	/** @brief Invalidates the grid and marks it for a rebuild during the next update. */
	void InvalidateGrid();
	/**
	 * @brief Rebuilds all grid levels around the current particle bounding box. */
	void RecalculateGrid();
	/**
	 * @brief Sorts current PP verlet list so that the src is always smaller as the dst. */
	void SortList();

	void RecalcPositions();
	/**
	 * @brief Places all active particles into the cells of every grid level.
	 * @details A particle is a main one on the level which matches its contact radius, and a secondary one on all coarser levels. */
	void RecalcParticlesPositions();
	/**
	 * @brief Places all walls into those cells of every grid level which their bounding box overlaps. */
	void RecalcWallsPositions();

	void CheckCollisionPP( const SGridLevel& _gridLevel, unsigned _nX1, unsigned _nY1, unsigned _nZ1, unsigned _nX2, unsigned _nY2, unsigned _nZ2, bool _bSameCell = false );
	void CheckCollisionPPSorted(const SGridLevel& _gridLevel, unsigned _nX1, unsigned _nY1, unsigned _nZ1, unsigned _nX2, unsigned _nY2, unsigned _nZ2, ESortCoord _dim);
	void CheckCollisionPW(const SGridLevel& _gridLevel, size_t _iCell);

	void AddPossibleContactPP(unsigned _iPart1, unsigned _iPart2);	// Add possible contacts into the list
	void AddPossibleContactPW(unsigned _iPart, unsigned _iWall);	// Add possible contacts into the list

	// remove contacts between particles "directly" connected with bonds
	void RemoveSBContacts();

	// for improved contact detection
	void InsertParticlesToVector(std::vector<SEntry>& _vec, SCellSpan _partIDs, ESortCoord _dim, ESortDir _dir) const;

	/**
	 * @brief Determines the cells of a grid level which the bounding box of a wall overlaps.
	 * @param _gridLevel Grid level.
	 * @param _iWall Index of the wall.
	 * @return Covered cell range; empty if the wall lies outside the grid. */
	[[nodiscard]] SCellRange WallCellRange(const SGridLevel& _gridLevel, unsigned _iWall) const;
};