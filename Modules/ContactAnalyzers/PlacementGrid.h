/* Copyright (c) 2026, DyssolTEC GmbH.
   All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"

/**
 * @brief Uniform-grid index answering whether a target sphere overlaps any of the indexed spheres.
 * @details Built once over a set of spheres and extended in place as further spheres are added.
 * The cell size is at least twice the largest radius passed to Initialize(). Limitations:
 * - No sphere larger than that radius may be inserted or queried, or overlaps are missed.
 * - Coordinates outside the indexed box are clamped into the edge cells.
 * - No periodic boundaries.
 * - Not thread-safe. */
class CPlacementGrid
{
public:
	/**
	 * @brief Prepares an empty index over the given region.
	 * @details Reuses the existing buckets whenever the cell count does not change.
	 * The number of cells is bounded by the number of spheres expected.
	 * @param _box Region to index.
	 * @param _maxRadius Largest radius that will be inserted or queried; defines the cell size.
	 * @param _expectedNumber Number of spheres the index is expected to hold. */
	void Initialize(const SVolumeType& _box, double _maxRadius, size_t _expectedNumber);
	/**
	 * @brief Adds one sphere to the index.
	 * @param _coord Center of the sphere.
	 * @param _radius Radius of the sphere. */
	void Insert(const CVector3& _coord, double _radius);
	/**
	 * @brief Checks the given sphere against all indexed spheres.
	 * @param _coord Center of the sphere.
	 * @param _radius Radius of the sphere.
	 * @return True if the sphere overlaps any indexed sphere. */
	[[nodiscard]] bool Overlaps(const CVector3& _coord, double _radius) const;
	/**
	 * @brief Returns the largest radius the cells are sized for.
	 * @return Largest radius that may be inserted or queried, as passed to Initialize(). */
	[[nodiscard]] double MaxRadius() const;

private:
	/**
	 * @brief One indexed sphere. */
	struct SEntry
	{
		CVector3 coord;  ///< Center of the sphere.
		double   radius; ///< Radius of the sphere.
	};

	/**
	 * @brief Zero-based coordinates of a cell within the grid. */
	struct SCellCoords
	{
		size_t ix; ///< Cell coordinate along X.
		size_t iy; ///< Cell coordinate along Y.
		size_t iz; ///< Cell coordinate along Z.
	};

	CVector3    m_origin{};                      ///< Lower corner of the indexed region.
	double      m_maxRadius{ 0.0 };              ///< Largest radius the cells are sized for.
	CVector3    m_invCellSize{};                 ///< Cells per unit length along each axis (1/cell size).
	size_t      m_nX{ 1 }, m_nY{ 1 }, m_nZ{ 1 }; ///< Number of cells per axis.
	std::vector<std::vector<SEntry>> m_cells;    ///< Spheres in each cell.

	/**
	 * @brief Returns the cell that the given point falls into.
	 * @details Points outside the indexed region are clamped to the edge cells.
	 * @param _coord Target point.
	 * @return Coordinates of the cell. */
	[[nodiscard]] SCellCoords CellCoords(const CVector3& _coord) const;
	/**
	 * @brief Returns the bucket that holds the given cell.
	 * @param _cell Coordinates of the cell.
	 * @return Index in m_cells. */
	[[nodiscard]] size_t CellIndex(const SCellCoords& _cell) const;
};
