/* Copyright (c) 2013-2020, MUSEN Development Team.
   Copyright (c) 2026, DyssolTEC GmbH.
   All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
   See LICENSE file for license and warranty information. */

#include "VerletList.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <limits>
#include <numeric>

namespace
{
	constexpr uint32_t c_padCells = 2;									///< Number of empty grid cells kept between the particles and the grid boundary.
	constexpr uint32_t c_noCell = std::numeric_limits<uint32_t>::max();	///< Marks an object which is placed into no cell.
	constexpr uint32_t c_cellsMaxLimit = 1625;							///< Upper limit for the cell number: its cube must stay within uint32_t.
	constexpr size_t c_sortedMinPairs = 256;							///< Particle pairs of two cells above which the sorted neighbour search pays off.
	constexpr double c_padTotal = 2 * c_padCells + 1;					///< Cells each direction holds on top of those covering the particles.
	constexpr double c_diagonalScale = 0.70710678118654752;				///< Scales a diagonal sort key to a unit direction.

	/**
	 * @brief Particle placed on the direction along which a cell is sorted. */
	struct SEntry
	{
		uint32_t id;	///< Index of the particle.
		double val;		///< Position of the surface point of the particle on the sorting direction.
		friend bool operator<(const SEntry& _e1, const SEntry& _e2) { return _e1.val < _e2.val; }
	};
}

void CVerletList::CCellLists::Reset(size_t _cellsNumber)
{
	m_offsets.assign(_cellsNumber + 1, 0);
	m_ids.clear();
}

template<typename T>
void CVerletList::CCellLists::Build(size_t _cellsNumber, const T& _forEachEntry)
{
	Reset(_cellsNumber);
	// counts are gathered shifted by one cell, to turn into offsets by summing them up
	_forEachEntry([&](size_t _iCell, uint32_t) { assert(_iCell < _cellsNumber); ++m_offsets[_iCell + 1]; });
	std::partial_sum(m_offsets.begin(), m_offsets.end(), m_offsets.begin());
	m_ids.resize(m_offsets.back());
	auto nextFree = m_offsets;
	_forEachEntry([&](size_t _iCell, uint32_t _id) { m_ids[nextFree[_iCell]++] = _id; });
}

void CVerletList::CCellLists::Build(size_t _cellsNumber, const std::vector<uint32_t>& _cellIndex)
{
	Build(_cellsNumber, [&](const auto& _addEntry)
		{
			for (size_t i = 0; i < _cellIndex.size(); ++i)
				if (_cellIndex[i] < _cellsNumber)
					_addEntry(_cellIndex[i], static_cast<uint32_t>(i));
		});
}

CVerletList::SCellSpan CVerletList::CCellLists::Cell(size_t _iCell) const
{
	return { m_ids.data() + m_offsets[_iCell], static_cast<size_t>(m_offsets[_iCell + 1] - m_offsets[_iCell]) };
}

CVerletList::CVerletList(CSimplifiedScene& _Scene):
	m_vParticles(_Scene.GetRefToParticles()),
	m_vWalls(_Scene.GetRefToWalls()),
	m_Scene(_Scene)
{
	m_SimDomain.coordBeg.Init(0);
	m_SimDomain.coordEnd.Init(0.5);
	m_gridDomain = m_SimDomain;
	m_partBoundingBox = m_SimDomain;
	m_dMaxParticleRadius = 0;
	m_dMinParticleRadius = 0;
	m_dVerletDistance = 0;

	m_dMaxTheorWallDistance = DEFAULT_TEOR_DISTANCE;
	m_LastCPUTime = 0;
	m_DisregardingTimeInterval = 0;
	m_dLastRealTime = 0;
	m_nAutoVerletDistNumerator = 0;
	m_bConnectedPPContact = false;
	m_nCellsMax = c_defaultVerletMaxCells;
	m_dVerletDistanceCoeff = DEFAULT_VERLET_DISTANCE_COEFF;
	m_bAutoAdjustVerletDistance = true;
}

void CVerletList::InitializeList()
{
	m_PerformHistory.clear();
	m_LastCPUTime = 0;
	m_DisregardingTimeInterval = 0;
	m_nAutoVerletDistNumerator = 0;
	m_dVerletDistance = 0;
	m_partBoundingBox = m_SimDomain;
	InvalidateGrid();
}

void CVerletList::SetSceneInfo(const SVolumeType& _simDomain, double _dMinPartRadius, double _dMaxPartRadius, uint32_t _dMaxCellsNumber, double _dVerletCoeff, bool _bAutoAdjust)
{
	bool bRecalculate = false;
	if (m_SimDomain.coordBeg != _simDomain.coordBeg || m_SimDomain.coordEnd != _simDomain.coordEnd)
	{
		m_SimDomain = _simDomain;
		m_gridDomain = m_SimDomain;
		m_partBoundingBox = m_SimDomain;
		bRecalculate = true;
	}
	if (_dMinPartRadius != m_dMinParticleRadius && _dMinPartRadius > 0)
	{
		m_dMinParticleRadius = _dMinPartRadius;
		bRecalculate = true;
	}
	if (_dMaxPartRadius != m_dMaxParticleRadius && _dMaxPartRadius > 0)
	{
		m_dMaxParticleRadius = _dMaxPartRadius;
		bRecalculate = true;
	}
	if (m_nCellsMax != _dMaxCellsNumber)
	{
		m_nCellsMax = _dMaxCellsNumber;
		bRecalculate = true;
	}
	if (m_dVerletDistance == 0 || m_dVerletDistanceCoeff != _dVerletCoeff)
	{
		m_dVerletDistanceCoeff = _dVerletCoeff;
		m_dVerletDistance = m_dVerletDistanceCoeff * m_dMinParticleRadius;
		bRecalculate = true;
	}
	if (m_bAutoAdjustVerletDistance != _bAutoAdjust)
	{
		m_bAutoAdjustVerletDistance = _bAutoAdjust;
		bRecalculate = true;
	}
	if (bRecalculate)
		InvalidateGrid();
}

void CVerletList::SortList()
{
	const size_t rowsNumber = m_PPList.size();
	const size_t threadsNumber = std::max<size_t>(GetThreadsNumber(), 1);
	const bool pbcEnabled = m_Scene.m_PBC.bEnabled;

	m_reversedPairs.resize(threadsNumber * threadsNumber);
	for (auto& bucket : m_reversedPairs)
		bucket.clear();

	// take the wrongly directed contacts out of their rows
	ParallelFor(threadsNumber, [&](size_t iThread)
	{
		for (size_t iSrc = iThread; iSrc < rowsNumber; iSrc += threadsNumber)
		{
			size_t j = 0;
			while (j < m_PPList[iSrc].size())
			{
				const uint32_t iDst = m_PPList[iSrc][j];
				if (iDst < iSrc)
				{
					const size_t iBucket = iThread * threadsNumber + iDst % threadsNumber;
					const uint8_t shift = pbcEnabled ? InverseVirtShift(m_PPVirtShift[iSrc][j]) : uint8_t{ 0 };
					m_reversedPairs[iBucket].push_back({ iDst, static_cast<uint32_t>(iSrc), shift });
					m_PPList[iSrc][j] = m_PPList[iSrc].back();
					m_PPList[iSrc].pop_back();
					if (pbcEnabled)
					{
						m_PPVirtShift[iSrc][j] = m_PPVirtShift[iSrc].back();
						m_PPVirtShift[iSrc].pop_back();
					}
				}
				else
					++j;
			}
		}
	});

	// put them back into the rows of their destinations
	ParallelFor(threadsNumber, [&](size_t iThread)
	{
		for (size_t iWriter = 0; iWriter < threadsNumber; ++iWriter)
		{
			for (const SReversedPair& pair : m_reversedPairs[iWriter * threadsNumber + iThread])
			{
				m_PPList[pair.dst].push_back(pair.src);
				if (pbcEnabled)
					m_PPVirtShift[pair.dst].push_back(pair.shift);
			}
		}
	});
}

void CVerletList::InvalidateGrid()
{
	m_grid.clear(); // an empty grid is rebuilt during the next update
	ResetCurrentData();
}

void CVerletList::RecalculateGrid()
{
	m_grid.clear();

	double currCellSize = 2 * m_dMaxParticleRadius + m_dVerletDistance;
	if (currCellSize == 0.0)
		return;

	// cover only the occupied region, padded so that no particle falls into an outermost cell
	const CVector3 extent = m_partBoundingBox.coordEnd - m_partBoundingBox.coordBeg;
	// the number of cells is limited in total; the limit must leave room for the padding
	const double cellsMax = std::clamp(static_cast<double>(m_nCellsMax), c_padTotal + 1, static_cast<double>(c_cellsMaxLimit));
	const double cellsBudget = cellsMax * cellsMax * cellsMax;
	// the smallest cell size which keeps the padded box within the budget
	currCellSize = std::max(currCellSize, (extent.x + extent.y + extent.z) / (3 * (cellsMax - c_padTotal)));

	const CVector3 pad{ c_padCells * currCellSize };
	m_gridDomain.coordBeg = m_partBoundingBox.coordBeg - pad;
	m_gridDomain.coordEnd = m_partBoundingBox.coordEnd + pad;

	const CVector3 gridExtent = m_gridDomain.coordEnd - m_gridDomain.coordBeg;
	// upper bound of the number of cells the padded box needs at a given cell size
	const auto CellsNeeded = [&gridExtent](double _cellSize)
		{
			return (gridExtent.x / _cellSize + 1) * (gridExtent.y / _cellSize + 1) * (gridExtent.z / _cellSize + 1);
		};

	do
	{
		m_grid.emplace_back();
		SGridLevel& gl = m_grid.back();
		gl.cellSize = currCellSize;
		gl.maxPartRadius = (gl.cellSize - m_dVerletDistance) / 2;
		currCellSize /= 2; // proceed to the next grid
		gl.minPartRadius = (currCellSize - m_dVerletDistance) / 2;

		gl.cellsX = static_cast<uint32_t>(floor((m_gridDomain.coordEnd.x - m_gridDomain.coordBeg.x) / gl.cellSize)) + 1;
		gl.cellsY = static_cast<uint32_t>(floor((m_gridDomain.coordEnd.y - m_gridDomain.coordBeg.y) / gl.cellSize)) + 1;
		gl.cellsZ = static_cast<uint32_t>(floor((m_gridDomain.coordEnd.z - m_gridDomain.coordBeg.z) / gl.cellSize)) + 1;

		gl.cellsX = std::max(gl.cellsX, 1u);
		gl.cellsY = std::max(gl.cellsY, 1u);
		gl.cellsZ = std::max(gl.cellsZ, 1u);

		// an empty level, until the objects are placed into it
		gl.mainParts.Reset(gl.CellsNumber());
		gl.secondParts.Reset(gl.CellsNumber());
		gl.walls.Reset(gl.CellsNumber());
		// a further level is added only while it still holds smaller particles and fits the budget
	} while (currCellSize > 2 * m_dMinParticleRadius + m_dVerletDistance && CellsNeeded(currCellSize) <= cellsBudget);

	m_grid.back().minPartRadius = 0;
}

bool CVerletList::IsNeedToBeUpdated(double _dTimeStep, double _dMaxPartDist, double _dMaxWallVel)
{
	m_dMaxTheorWallDistance += _dMaxWallVel * _dTimeStep;
	if (m_Scene.m_PBC.bEnabled)
		m_dMaxTheorWallDistance += 2 * std::max({ fabs(m_Scene.m_PBC.vVel.x), fabs(m_Scene.m_PBC.vVel.y), fabs(m_Scene.m_PBC.vVel.z) });
	return _dMaxPartDist+ std::max(m_dMaxTheorWallDistance, _dMaxPartDist) >= m_dVerletDistance;
}

void CVerletList::UpdateList(double _dCurrTime)
{
	if(m_bAutoAdjustVerletDistance)
		AutoAdjustVerletDistance(_dCurrTime);
	m_Scene.AddVirtualParticles(m_dVerletDistance);
	UpdateParticlesBoundingBox();
	if (IsGridRefitNeeded())
		RecalculateGrid();
	RecalcPositions(); // fills the grid anew
	m_PPList.resize(m_vParticles.Size());
	m_PWList.resize(m_vParticles.Size());
	for (size_t i = 0; i < m_vParticles.Size(); ++i)
	{
		m_PPList[i].clear();
		m_PWList[i].clear();
	};

	// resize vectors with shifts in case of PBC
	if (m_Scene.m_PBC.bEnabled)
	{
		m_PPVirtShift.resize(m_vParticles.Size());
		m_PWVirtShift.resize(m_vParticles.Size());
		for (size_t i = 0; i < m_vParticles.Size(); ++i)
		{
			m_PPVirtShift[i].clear();
			m_PWVirtShift[i].clear();
		}
	}
	else
	{
		m_PPVirtShift.clear();
		m_PWVirtShift.clear();
	}

	for (auto& gridLevel : m_grid)
	{
		ParallelFor(gridLevel.CellsNumber(), [&](size_t iCell)
		{
			if (gridLevel.mainParts.Cell(iCell).empty() && gridLevel.secondParts.Cell(iCell).empty())
				return; // an empty cell can hold no contacts
			const uint32_t x = static_cast<uint32_t>(floor(double(iCell) / gridLevel.cellsZ / gridLevel.cellsY));
			const uint32_t y = static_cast<uint32_t>(floor(double(iCell - x * gridLevel.cellsZ * gridLevel.cellsY) / gridLevel.cellsZ));
			const uint32_t z = static_cast<uint32_t>(iCell) - x* gridLevel.cellsZ* gridLevel.cellsY - y* gridLevel.cellsZ;
			CheckCollisionPPInCell(gridLevel, iCell);
			CheckCollisionPW(gridLevel, iCell);
			CheckCollisionPPBetweenCells(gridLevel, iCell, x    , y    , z + 1, ESortCoord::Z);
			CheckCollisionPPBetweenCells(gridLevel, iCell, x    , y + 1, z    , ESortCoord::Y);
			CheckCollisionPPBetweenCells(gridLevel, iCell, x + 1, y    , z    , ESortCoord::X);
			CheckCollisionPPBetweenCells(gridLevel, iCell, x + 1, y + 1, z    , ESortCoord::XY);
			CheckCollisionPPBetweenCells(gridLevel, iCell, x + 1, y + 1, z + 1, ESortCoord::XY);
			CheckCollisionPPBetweenCells(gridLevel, iCell, x    , y + 1, z + 1, ESortCoord::YZ);
			CheckCollisionPPBetweenCells(gridLevel, iCell, x + 1, y    , z + 1, ESortCoord::XZ);
			CheckCollisionPPBetweenCells(gridLevel, iCell, x - 1, y    , z + 1, ESortCoord::Z);
			CheckCollisionPPBetweenCells(gridLevel, iCell, x - 1, y - 1, z + 1, ESortCoord::Z);
			CheckCollisionPPBetweenCells(gridLevel, iCell, x    , y - 1, z + 1, ESortCoord::Z);
			CheckCollisionPPBetweenCells(gridLevel, iCell, x + 1, y - 1, z + 1, ESortCoord::XZ);
			CheckCollisionPPBetweenCells(gridLevel, iCell, x + 1, y - 1, z    , ESortCoord::X);
			CheckCollisionPPBetweenCells(gridLevel, iCell, x + 1, y - 1, z - 1, ESortCoord::X);
		});
	}
	ReassignVirtualContacts(); // shift virtual-real contacts as real-real
	RemoveSBContacts();
	m_Scene.RemoveVirtualParticles();
	SortList();
	m_dMaxTheorWallDistance = 0;
}

void CVerletList::RemoveSBContacts()
{
	if (m_bConnectedPPContact) return; // if it is necessary to consider PP contacts
	auto& vSolidBonds = m_Scene.GetRefToSolidBonds();
	auto& vPartToSolidBonds = *m_Scene.GetPointerToPartToSolidBonds();
	ParallelFor(vPartToSolidBonds.size(), [&](size_t i)
	{
		for (size_t j = 0; j < vPartToSolidBonds[i].size(); j++)
		{
			const unsigned nBondIndex = vPartToSolidBonds[i][j];
			if (vSolidBonds.Active(nBondIndex)) // if bond is active
			{
				// find index of second particle
				size_t nSecondPart = vSolidBonds.LeftID(nBondIndex);
				if (nSecondPart == i)
					nSecondPart = vSolidBonds.RightID(nBondIndex);

				for (size_t i_2 = 0; i_2 < m_PPList[i].size(); i_2++)
					if (m_PPList[i][i_2] == nSecondPart)
					{
						m_PPList[i].erase(m_PPList[i].begin() + i_2);
						if (m_Scene.m_PBC.bEnabled)
							m_PPVirtShift[i].erase(m_PPVirtShift[i].begin() + i_2);
						break;
					}
			}
		}
	});

}

void CVerletList::AddDisregardingTimeInterval(const clock_t& _interval)
{
	m_DisregardingTimeInterval += _interval;
}

void CVerletList::UpdateParticlesBoundingBox()
{
	const size_t particlesNumber = m_vParticles.Size();
	const size_t threadsNumber = std::max<size_t>(GetThreadsNumber(), 1);
	const SVolumeType empty{ CVector3{ DBL_MAX }, CVector3{ -DBL_MAX } };
	std::vector partialBox(threadsNumber, empty);
	ParallelFor(threadsNumber, [&](size_t iThread)
		{
			SVolumeType box = empty;
			for (size_t i = particlesNumber * iThread / threadsNumber; i < particlesNumber * (iThread + 1) / threadsNumber; ++i)
				if (m_vParticles.Active(i) && m_vParticles.Coord(i).IsFinite())
				{
					const CVector3& coord = m_vParticles.Coord(i);
					box.coordBeg = Min(box.coordBeg, coord);
					box.coordEnd = Max(box.coordEnd, coord);
				}
			partialBox[iThread] = box;
		});

	SVolumeType box = empty;
	for (const auto& part : partialBox)
	{
		box.coordBeg = Min(box.coordBeg, part.coordBeg);
		box.coordEnd = Max(box.coordEnd, part.coordEnd);
	}
	// keep the previous box if no active particle with a finite position was found
	if (box.coordBeg.x <= box.coordEnd.x)
		m_partBoundingBox = box;
}

bool CVerletList::IsGridRefitNeeded() const
{
	if (m_grid.empty()) return true;
	const double cellSize = m_grid.front().cellSize;
	const double margin = (c_padCells - 1.0) * cellSize;
	for (size_t d = 0; d < 3; ++d)
	{
		// the outermost grid cells must stay free of particles
		if (m_partBoundingBox.coordBeg[d] < m_gridDomain.coordBeg[d] + margin) return true;
		if (m_partBoundingBox.coordEnd[d] > m_gridDomain.coordEnd[d] - margin) return true;
		// the grid covers much more than the particles occupy
		if (m_gridDomain.coordEnd[d] - m_gridDomain.coordBeg[d] > 2 * (m_partBoundingBox.coordEnd[d] - m_partBoundingBox.coordBeg[d] + 2 * c_padCells * cellSize)) return true;
	}
	return false;
}

void CVerletList::AutoAdjustVerletDistance(double _dCurrentTime)
{
	if (m_vParticles.Empty()) return; // no recalculation if there is no particles
	if (m_LastCPUTime == 0) // first step
	{
		m_LastCPUTime = clock();
		m_dLastRealTime = _dCurrentTime;
	}
	else
	{
		SCalcPerfmMetric newMetric;
		newMetric.dCalcTimeCoeff = (_dCurrentTime - m_dLastRealTime) / (clock() - m_LastCPUTime - m_DisregardingTimeInterval); // do not take into account time for saving
		newMetric.dVerletDistance = m_dVerletDistance;
		newMetric.dAnalysisTime = _dCurrentTime;
		m_DisregardingTimeInterval = 0;
		unsigned nTemp = 0;
		while (nTemp < m_PerformHistory.size()) // overwrite existing value
			if (m_PerformHistory[nTemp].dVerletDistance == m_dVerletDistance)
			{
				m_PerformHistory[nTemp] = newMetric;
				break;
			}
			else
				nTemp++;
		if (nTemp >= m_PerformHistory.size()) // add only this value was not in array
			m_PerformHistory.push_back(newMetric);

		m_LastCPUTime = clock();
		m_dLastRealTime = _dCurrentTime;

		if (m_PerformHistory.size() < 3) // minimal amount of values which should be in the array
		{
			m_dVerletDistance = m_dVerletDistance * 1.1;
			InvalidateGrid();
		}
		else
		{
			if (m_nAutoVerletDistNumerator != 0)
				m_nAutoVerletDistNumerator--;
			else
			{
				m_nAutoVerletDistNumerator = 10; // parameter 10 seems to bee meaningful
				std::sort(m_PerformHistory.begin(), m_PerformHistory.end(), [](const SCalcPerfmMetric& a, const SCalcPerfmMetric& b) { return a.dVerletDistance < b.dVerletDistance; });
				auto iter = std::max_element(m_PerformHistory.begin(), m_PerformHistory.end(), [](const SCalcPerfmMetric& a, const SCalcPerfmMetric& b) { return a.dCalcTimeCoeff < b.dCalcTimeCoeff; });
				size_t nIndex = iter - m_PerformHistory.begin();
				if (nIndex == 0)
				{
					m_dVerletDistance = m_PerformHistory[0].dVerletDistance*0.8;
					InvalidateGrid();
				}
				else if (nIndex == m_PerformHistory.size() - 1)
				{
					m_dVerletDistance = m_PerformHistory[nIndex].dVerletDistance*1.2;
					InvalidateGrid();
				}
				else
				{
					if ((m_PerformHistory[nIndex + 1].dVerletDistance - m_PerformHistory[nIndex - 1].dVerletDistance) / m_PerformHistory[nIndex - 1].dVerletDistance > 5e-2)
					{
						m_dVerletDistance = (m_PerformHistory[nIndex + 1].dVerletDistance + m_PerformHistory[nIndex].dVerletDistance) / 2;
						InvalidateGrid();
					}
				}
			}
			// clear analysis history by removing too old entry
			if (m_PerformHistory.size() > 10)
			{
				std::sort(m_PerformHistory.begin(), m_PerformHistory.end(), [](const SCalcPerfmMetric& a, const SCalcPerfmMetric& b) { return a.dAnalysisTime > b.dAnalysisTime; });
				m_PerformHistory.pop_back();
			}
		}
	}
}

void CVerletList::ReassignVirtualContacts()
{
	if (m_Scene.GetVirtualParticlesNumber() == 0) return;
	const size_t realPartNum = m_Scene.GetRealParticlesNumber();
	for (size_t i = 0; i < m_Scene.GetVirtualParticlesNumber(); i++)
	{
		const size_t iVirt = i + realPartNum;
		// for PP contacts
		for (size_t j = 0; j < m_PPList[iVirt].size(); j++)
		{
			const unsigned nRealID1 = m_PPList[iVirt][j];
			const unsigned nRealID2 = m_vParticles.InitIndex(iVirt);
			uint8_t nVirtShiftPart2 = m_Scene.m_vPBCVirtShift[i];
			if (nRealID1 < nRealID2)
			{
				m_PPList[nRealID1].push_back(nRealID2);
				m_PPVirtShift[nRealID1].push_back(nVirtShiftPart2);
			}
			else
			{
				m_PPList[nRealID2].push_back(nRealID1);
				m_PPVirtShift[nRealID2].push_back(InverseVirtShift(nVirtShiftPart2));
			}
		}
		// for PW contacts
		const size_t iReal = m_vParticles.InitIndex(iVirt);
		for (size_t j = 0; j < m_PWList[iVirt].size(); j++)
		{
			m_PWList[iReal].push_back(m_PWList[iVirt][j]);
			m_PWVirtShift[iReal].push_back(m_PWVirtShift[iVirt][j]);
		}
	}
	m_PPList.resize(realPartNum);
	m_PWList.resize(realPartNum);
	m_PPVirtShift.resize(realPartNum);
	m_PWVirtShift.resize(realPartNum);
}

void CVerletList::CheckCollisionPPInCell(const SGridLevel& _gridLevel, size_t _iCell)
{
	const SCellSpan main = _gridLevel.mainParts.Cell(_iCell);
	const SCellSpan second = _gridLevel.secondParts.Cell(_iCell);
	for (size_t i = 0; i < main.size(); ++i)
	{
		const uint32_t p1 = main[i];
		const double reach = m_dVerletDistance + m_vParticles.ContactRadius(p1);
		const CVector3 pos1 = m_vParticles.Coord(p1);
		for (size_t j = i + 1; j < main.size(); ++j) // each pair of the cell once
			if (IsCloseEnough(pos1, reach, main[j]))
				AddPossibleContactPP(std::min(p1, main[j]), std::max(p1, main[j]));
		for (const uint32_t p2 : second)
			if (IsCloseEnough(pos1, reach, p2))
				AddPossibleContactPP(std::min(p1, p2), std::max(p1, p2));
	}
}

void CVerletList::CheckCollisionPPBetweenCells(const SGridLevel& _gridLevel, size_t _iCell1, uint32_t _x2, uint32_t _y2, uint32_t _z2, ESortCoord _dim)
{
	if (_x2 >= _gridLevel.cellsX || _y2 >= _gridLevel.cellsY || _z2 >= _gridLevel.cellsZ)
		return;
	const size_t iCell2 = _gridLevel.CellIndex(_x2, _y2, _z2);
	const SCellSpan main2 = _gridLevel.mainParts.Cell(iCell2);
	const SCellSpan second2 = _gridLevel.secondParts.Cell(iCell2);
	if (main2.empty() && second2.empty())
		return;	// nothing to pair the first cell with
	const SCellSpan main1 = _gridLevel.mainParts.Cell(_iCell1);
	const SCellSpan second1 = _gridLevel.secondParts.Cell(_iCell1);

	if (second1.empty() && second2.empty() && main1.size() * main2.size() > c_sortedMinPairs)
		PairCellsSorted(main1, main2, _dim);
	else
		PairCellsPlain(main1, second1, main2, second2);
}

void CVerletList::PairCellsSorted(SCellSpan _main1, SCellSpan _main2, ESortCoord _dim)
{
	thread_local std::vector<SEntry> sortedMain2; // kept per thread, so that refills stay off the heap
	sortedMain2.clear();
	sortedMain2.reserve(_main2.size());
	for (const uint32_t p2 : _main2) // the near surface point of each particle along the direction
		sortedMain2.push_back({ p2, SortKey(p2, _dim) - m_vParticles.ContactRadius(p2) });
	std::sort(sortedMain2.begin(), sortedMain2.end());
	for (const uint32_t p1 : _main1)
	{
		const double reach = m_dVerletDistance + m_vParticles.ContactRadius(p1);
		const CVector3 pos1 = m_vParticles.Coord(p1);
		const double sortPos = SortKey(p1, _dim) + m_vParticles.ContactRadius(p1); // its far surface point
		for (const SEntry& neighbour : sortedMain2)
		{
			if (neighbour.val - sortPos > m_dVerletDistance)
				break; // sorted, so no further particle can be close enough either
			if (IsCloseEnough(pos1, reach, neighbour.id))
				AddPossibleContactPP(p1, neighbour.id);
		}
	}
}

void CVerletList::PairCellsPlain(SCellSpan _main1, SCellSpan _second1, SCellSpan _main2, SCellSpan _second2)
{
	for (const uint32_t p1 : _main1)
	{
		const double reach = m_dVerletDistance + m_vParticles.ContactRadius(p1);
		const CVector3 pos1 = m_vParticles.Coord(p1);
		for (const uint32_t p2 : _main2)
			if (IsCloseEnough(pos1, reach, p2))
				AddPossibleContactPP(p1, p2);
		for (const uint32_t p2 : _second2)
			if (IsCloseEnough(pos1, reach, p2))
				AddPossibleContactPP(p1, p2);
	}

	if (_second1.empty())
		return; // nothing left to pair the second cell with
	for (const uint32_t p1 : _main2)
	{
		const double reach = m_dVerletDistance + m_vParticles.ContactRadius(p1);
		const CVector3 pos1 = m_vParticles.Coord(p1);
		for (const uint32_t p2 : _second1)
			if (IsCloseEnough(pos1, reach, p2))
				AddPossibleContactPP(p2, p1); // the row of p2, which belongs to the first cell
	}
}

void CVerletList::CheckCollisionPW(const SGridLevel& _gridLevel, size_t _iCell)
{
	const SCellSpan parts = _gridLevel.mainParts.Cell(_iCell);
	const SCellSpan walls = _gridLevel.walls.Cell(_iCell);
	for (const uint32_t p : parts)
	{
		if (m_vParticles.ContactRadius(p) <= _gridLevel.minPartRadius) continue; // will be considered on another grid level
		for (const uint32_t w : walls)
		{
			if (IsSphereIntersectTriangle(m_vWalls.Coordinates(w), m_vWalls.NormalVector(w), m_vParticles.Coord(p), m_vParticles.ContactRadius(p) + m_dVerletDistance).first != EIntersectionType::NO_CONTACT)
				AddPossibleContactPW(p, w);
		}
	}
}

void CVerletList::AddPossibleContactPP(unsigned _iPart1, unsigned _iPart2)
{
	m_PPList[_iPart1].push_back(_iPart2);
	if (!m_Scene.m_PBC.bEnabled) return;

	// if PBC are enabled
	const size_t nRealPart = m_Scene.GetRealParticlesNumber();
	if (_iPart1 >= nRealPart && _iPart2 >= nRealPart) // virtual-virtual contact
		m_PPList[_iPart1].pop_back();
	else if (( _iPart1 < nRealPart) && (_iPart2 < nRealPart)) // real-real contact
		m_PPVirtShift[_iPart1].push_back(0); // put empty shift to fulfill requirements of same size vectors
	else if ((_iPart1 < nRealPart) && (_iPart2 >= nRealPart)) // real-virtual contact
	{
		m_PPList[_iPart1].back() = m_vParticles.InitIndex(_iPart2);
		m_PPVirtShift[_iPart1].push_back(m_Scene.m_vPBCVirtShift[_iPart2 - nRealPart]);
	}
	// else virtual-real contact - is correctly approximated with first line. afterwards reassign virtual contact will be used
}

void CVerletList::AddPossibleContactPW(unsigned _iPart, unsigned _iWall)
{
	m_PWList[_iPart].push_back(_iWall);
	if (!m_Scene.m_PBC.bEnabled) return;

	// if PBC are enabled and particle which was added is virtual
	if (_iPart >= m_Scene.GetRealParticlesNumber())
		m_PWVirtShift[_iPart].push_back(m_Scene.m_vPBCVirtShift[_iPart - m_Scene.GetRealParticlesNumber()]);
	else
		m_PWVirtShift[_iPart].push_back(0); // put empty shift to fulfill requirements of same size vectors
}

void CVerletList::RecalcPositions()
{
	RecalcParticlesPositions();
	RecalcWallsPositions();
}

void CVerletList::RecalcParticlesPositions()
{
	const size_t particlesNumber = m_vParticles.Size();
	std::vector<uint32_t> partLevel(particlesNumber, 0);
	if (m_grid.size() > 1)
	{
		ParallelFor(particlesNumber, [&](size_t i)
			{
				if (m_vParticles.Active(i))
					for (uint32_t iGrid = 0; iGrid < m_grid.size(); iGrid++)
						if (m_grid[iGrid].maxPartRadius + DBL_EPSILON >= m_vParticles.ContactRadius(i) && m_grid[iGrid].minPartRadius - DBL_EPSILON < m_vParticles.ContactRadius(i))
						{
							partLevel[i] = iGrid;
							break;
						}
			});
	}

	std::vector<uint32_t> mainCell(particlesNumber), secondCell;
	if (m_grid.size() > 1)
		secondCell.resize(particlesNumber);
	for (size_t iGrid = 0; iGrid < m_grid.size(); ++iGrid)
	{
		SGridLevel& gridLevel = m_grid[iGrid];
		const bool finerLevelExists = iGrid + 1 < m_grid.size(); // the finest level holds no secondary particles
		ParallelFor(particlesNumber, [&](size_t i)
		{
			mainCell[i] = c_noCell;
			if (finerLevelExists)
				secondCell[i] = c_noCell;
			if (!m_vParticles.Active(i)) return;
			const CVector3 relCoord = (m_vParticles.Coord(i) - m_gridDomain.coordBeg) / gridLevel.cellSize;
			if (!relCoord.IsFinite()) return; // a particle without a finite position cannot be placed into the grid
			// clamped if the particle lays outside the domain (like newly generated)
			const auto x = static_cast<uint32_t>(std::clamp(std::floor(relCoord.x), 0.0, gridLevel.cellsX - 1.0));
			const auto y = static_cast<uint32_t>(std::clamp(std::floor(relCoord.y), 0.0, gridLevel.cellsY - 1.0));
			const auto z = static_cast<uint32_t>(std::clamp(std::floor(relCoord.z), 0.0, gridLevel.cellsZ - 1.0));
			if (partLevel[i] == iGrid)
				mainCell[i] = static_cast<uint32_t>(gridLevel.CellIndex(x, y, z));
			else if (partLevel[i] > iGrid)
				secondCell[i] = static_cast<uint32_t>(gridLevel.CellIndex(x, y, z));
		});

		gridLevel.mainParts.Build(gridLevel.CellsNumber(), mainCell);
		if (finerLevelExists)
			gridLevel.secondParts.Build(gridLevel.CellsNumber(), secondCell);
	}
}

void CVerletList::RecalcWallsPositions()
{
	std::vector<SCellRange> ranges(m_vWalls.Size());
	for (SGridLevel& gridLevel : m_grid)
	{
		ParallelFor(m_vWalls.Size(), [&](size_t iWall)
			{
				ranges[iWall] = WallCellRange(gridLevel, static_cast<uint32_t>(iWall));
			});
		gridLevel.walls.Build(gridLevel.CellsNumber(), [&](const auto& _addEntry)
			{
				for (uint32_t iWall = 0; iWall < m_vWalls.Size(); ++iWall)
				{
					const SCellRange& range = ranges[iWall];
					for (uint32_t x = range.minX; x <= range.maxX; ++x)
						for (uint32_t y = range.minY; y <= range.maxY; ++y)
							for (uint32_t z = range.minZ; z <= range.maxZ; ++z)
								_addEntry(gridLevel.CellIndex(x, y, z), iWall);
				}
			});
	}
}

void CVerletList::ResetCurrentData()
{
	m_dMaxTheorWallDistance = DEFAULT_TEOR_DISTANCE;
}

void CVerletList::GetPWContacts(size_t _iP, std::vector<EIntersectionType>& _vIntersectionType, std::vector<CVector3>& _vContactPoint) const
{
	if (m_PWList[_iP].empty()) return;

	_vIntersectionType.resize(m_PWList[_iP].size());
	_vContactPoint.resize(m_PWList[_iP].size());

	CVector3 vPartCoord;
	std::vector<bool> bVirtualContacts(m_PWList[_iP].size(), false); // true for all virtual contacts
	for (size_t i = 0; i < m_PWList[_iP].size(); ++i)
	{
		if(!m_PWVirtShift.empty() && (m_PWVirtShift[_iP][i] != 0)) // virtual contact
		{
			vPartCoord = GetVirtualProperty(m_vParticles.Coord(_iP), m_PWVirtShift[_iP][i], m_Scene.m_PBC );
			bVirtualContacts[i] = true;
		}
		else
			vPartCoord = m_vParticles.Coord(_iP);

		const size_t w = m_PWList[_iP][i];
		std::tie(_vIntersectionType[i], _vContactPoint[i]) = IsSphereIntersectTriangle(m_vWalls.Coordinates(w), m_vWalls.NormalVector(w), vPartCoord, m_vParticles.ContactRadius(_iP));
	}

	for (size_t i = 0; i < _vContactPoint.size() - 1; ++i)
		if (_vIntersectionType[i] != EIntersectionType::NO_CONTACT)
			for (size_t j = i + 1; j < _vContactPoint.size(); ++j)
				if (_vIntersectionType[j] != EIntersectionType::NO_CONTACT && SquaredLength(m_vWalls.NormalVector(m_PWList[_iP][i]) - m_vWalls.NormalVector(m_PWList[_iP][j])) < 1e-6) // simplified unique calculation check
					switch (_vIntersectionType[i])
					{
					case EIntersectionType::FACE_CONTACT:
						if (_vIntersectionType[j] == EIntersectionType::EDGE_CONTACT || _vIntersectionType[j] == EIntersectionType::VERTEX_CONTACT)
							_vIntersectionType[j] = EIntersectionType::NO_CONTACT;
						break;
					case EIntersectionType::EDGE_CONTACT:
						if (_vIntersectionType[j] == EIntersectionType::EDGE_CONTACT)
							_vIntersectionType[j] = EIntersectionType::NO_CONTACT;
						else if (_vIntersectionType[j] == EIntersectionType::FACE_CONTACT)
							_vIntersectionType[i] = EIntersectionType::NO_CONTACT;
						else
							_vIntersectionType[j] = EIntersectionType::NO_CONTACT;
						break;
					case EIntersectionType::VERTEX_CONTACT:
						_vIntersectionType[i] = EIntersectionType::NO_CONTACT;
						break;
					default: ;
					}

	for (size_t i = 0; i < m_PWList[_iP].size(); ++i)
		if (bVirtualContacts[i])
			for (size_t j = 0; j < m_PWList[_iP].size(); ++j) // additional check that there is no contact between one wall and real and virtual particles
				if (j != i && _vIntersectionType[j] != EIntersectionType::NO_CONTACT && bVirtualContacts[j] == 0 && m_PWList[_iP][j] == m_PWList[_iP][i])
					_vIntersectionType[i] = EIntersectionType::NO_CONTACT;
}

bool CVerletList::IsCloseEnough(const CVector3& _pos1, double _reach, uint32_t _p2) const
{
	const double contactDist = _reach + m_vParticles.ContactRadius(_p2);
	return SquaredLength(_pos1 - m_vParticles.Coord(_p2)) <= contactDist * contactDist;
}

double CVerletList::SortKey(uint32_t _id, ESortCoord _dim) const
{
	const CVector3 coord = m_vParticles.Coord(_id);
	switch (_dim)
	{
	case ESortCoord::X: return coord.x;
	case ESortCoord::Y: return coord.y;
	case ESortCoord::Z: return coord.z;
	case ESortCoord::XY: return (coord.x + coord.y) * c_diagonalScale;
	case ESortCoord::XZ: return (coord.x + coord.z) * c_diagonalScale;
	case ESortCoord::YZ: return (coord.y + coord.z) * c_diagonalScale;
	}
	return 0.0;	// unreachable
}

CVerletList::SCellRange CVerletList::WallCellRange(const SGridLevel& _gridLevel, unsigned _iWall) const
{
	if (!m_vWalls.MinCoord(_iWall).IsFinite() || !m_vWalls.MaxCoord(_iWall).IsFinite())
		return {};	// a wall without a finite position cannot be placed into the grid

	// Converts a cell coordinate into an index.
	const auto CellBound = [](double _cellCoord, unsigned _cellsCount)
		{
			return static_cast<int>(std::clamp(_cellCoord, -2.0, static_cast<double>(_cellsCount)));
		};

	const CVector3 minCoord = (m_vWalls.MinCoord(_iWall) - m_gridDomain.coordBeg) / _gridLevel.cellSize;
	int nMinX = CellBound(floor(minCoord.x), _gridLevel.cellsX);
	int nMinY = CellBound(floor(minCoord.y), _gridLevel.cellsY);
	int nMinZ = CellBound(floor(minCoord.z), _gridLevel.cellsZ);

	if (nMinX >= static_cast<int>(_gridLevel.cellsX) || nMinY >= static_cast<int>(_gridLevel.cellsY) || nMinZ >= static_cast<int>(_gridLevel.cellsZ)) return {};

	const CVector3 maxCoord = (m_vWalls.MaxCoord(_iWall) - m_gridDomain.coordBeg) / _gridLevel.cellSize;
	int nMaxX = CellBound(ceil(maxCoord.x), _gridLevel.cellsX) + 1;
	int nMaxY = CellBound(ceil(maxCoord.y), _gridLevel.cellsY) + 1;
	int nMaxZ = CellBound(ceil(maxCoord.z), _gridLevel.cellsZ) + 1;

	if (nMaxX < 0 || nMaxY < 0 || nMaxZ < 0) return {};

	nMaxX = std::min(nMaxX, static_cast<int>(_gridLevel.cellsX) - 1);
	nMaxY = std::min(nMaxY, static_cast<int>(_gridLevel.cellsY) - 1);
	nMaxZ = std::min(nMaxZ, static_cast<int>(_gridLevel.cellsZ) - 1);

	nMinX = std::max(nMinX, 0);
	nMinY = std::max(nMinY, 0);
	nMinZ = std::max(nMinZ, 0);

	if (nMinX > 0) nMinX--;
	if (nMinY > 0) nMinY--;
	if (nMinZ > 0) nMinZ--;

	return { static_cast<uint32_t>(nMinX), static_cast<uint32_t>(nMinY), static_cast<uint32_t>(nMinZ),
			 static_cast<uint32_t>(nMaxX), static_cast<uint32_t>(nMaxY), static_cast<uint32_t>(nMaxZ) };
}
