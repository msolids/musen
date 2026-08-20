/* Copyright (c) 2026, DyssolTEC GmbH.
   All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
   See LICENSE file for license and warranty information. */

#include "PlacementGrid.h"

#include <algorithm>
#include <utility>

void CPlacementGrid::Initialize(const SVolumeType& _box, double _maxRadius, size_t _expectedNumber)
{
	m_origin = _box.coordBeg;
	m_maxRadius = _maxRadius;

	const size_t maxCells = std::max(size_t{ 1024 }, 4 * _expectedNumber);

	const double cell = 2 * _maxRadius;
	const CVector3 extent = _box.coordEnd - _box.coordBeg;
	// Number of whole cells of the chosen size that fit along an axis of the given length.
	const auto CellsNumber = [&](double _length)
	{
		if (!(cell > 0.0) || !(_length > 0.0)) return size_t{ 1 };	// also catches NaN
		const double n = _length / cell;
		if (n >= static_cast<double>(maxCells)) return maxCells;	// keep the conversion defined
		return n >= 1.0 ? static_cast<size_t>(n) : size_t{ 1 };
	};
	m_nX = CellsNumber(extent.x);
	m_nY = CellsNumber(extent.y);
	m_nZ = CellsNumber(extent.z);

	// keep the index bounded: coarsen uniformly if needed
	while (static_cast<double>(m_nX) * static_cast<double>(m_nY) * static_cast<double>(m_nZ) > static_cast<double>(maxCells))
	{
		m_nX = (m_nX + 1) / 2;
		m_nY = (m_nY + 1) / 2;
		m_nZ = (m_nZ + 1) / 2;
	}

	// the cells cover the whole extent, so their size follows from their number
	m_invCellSize = CVector3{
		extent.x > 0 ? static_cast<double>(m_nX) / extent.x : 0.0,
		extent.y > 0 ? static_cast<double>(m_nY) / extent.y : 0.0,
		extent.z > 0 ? static_cast<double>(m_nZ) / extent.z : 0.0 };

	// empty the cells keeping the reserved memory
	m_cells.resize(m_nX * m_nY * m_nZ);
	for (auto& c : m_cells)
		c.clear();
}

void CPlacementGrid::Insert(const CVector3& _coord, double _radius)
{
	m_cells[CellIndex(CellCoords(_coord))].push_back(SEntry{ _coord, _radius });
}

bool CPlacementGrid::Overlaps(const CVector3& _coord, double _radius) const
{
	// with a cell size of 2*maxRadius, a sphere can only reach others in its own and in the 26 adjacent cells
	const auto Neighbors = [](size_t _i, size_t _cellsNumber) // inclusive range [first, last]
	{
		const size_t first = _i > 0 ? _i - 1 : _i;
		const size_t last  = _i + 1 < _cellsNumber ? _i + 1 : _i;
		return std::pair<size_t, size_t>{ first, last };
	};

	const SCellCoords cell = CellCoords(_coord);
	const auto [x1, x2] = Neighbors(cell.ix, m_nX);
	const auto [y1, y2] = Neighbors(cell.iy, m_nY);
	const auto [z1, z2] = Neighbors(cell.iz, m_nZ);

	for (size_t ix = x1; ix <= x2; ++ix)
		for (size_t iy = y1; iy <= y2; ++iy)
			for (size_t iz = z1; iz <= z2; ++iz)
				for (const auto& entry : m_cells[CellIndex({ ix, iy, iz })])
				{
					const double distance = entry.radius + _radius;
					if (SquaredLength(entry.coord, _coord) < distance * distance)
						return true;
				}
	return false;
}

double CPlacementGrid::MaxRadius() const
{
	return m_maxRadius;
}

CPlacementGrid::SCellCoords CPlacementGrid::CellCoords(const CVector3& _coord) const
{
	const auto Coord = [](double _distance, double _invCellSize, size_t _cellsNumber)
	{
		const double i = _distance * _invCellSize;
		if (!(i > 0.0)) return size_t{ 0 };
		if (i >= static_cast<double>(_cellsNumber)) return _cellsNumber - 1;
		return static_cast<size_t>(i);
	};
	return SCellCoords{
		Coord(_coord.x - m_origin.x, m_invCellSize.x, m_nX),
		Coord(_coord.y - m_origin.y, m_invCellSize.y, m_nY),
		Coord(_coord.z - m_origin.z, m_invCellSize.z, m_nZ) };
}

size_t CPlacementGrid::CellIndex(const SCellCoords& _cell) const
{
	return (_cell.ix * m_nY + _cell.iy) * m_nZ + _cell.iz;
}
