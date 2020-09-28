/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "VerletList.h"

CVerletList::CVerletList(CSimplifiedScene& _Scene): m_Scene(_Scene),
	m_nThreadsNumber(GetThreadsNumber()),
	m_vParticles(_Scene.GetRefToParticles()),
	m_vWalls(_Scene.GetRefToWalls())
{
	m_SimDomain.coordBeg.Init(0);
	m_SimDomain.coordEnd.Init(0.5);
	m_workDomain = m_SimDomain;
	m_dMaxParticleRadius = 0;
	m_dMinParticleRadius = 0;
	m_dVerletDistance = 0;

	m_dMaxTheorWallDistance = DEFAULT_TEOR_DISTANCE;
	m_LastCPUTime = 0;
	m_DisregardingTimeInterval = 0;
	m_dLastRealTime = 0;
	m_nAutoVerletDistNumerator = 0;
	m_bConnectedPPContact = false;
	m_nCellsMax = DEFAULT_MAX_CELLS;
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
	m_nThreadsNumber = GetThreadsNumber();
}

void CVerletList::SetSceneInfo(const SVolumeType& _simDomain, double _dMinPartRadius, double _dMaxPartRadius, uint32_t _dMaxCellsNumber, double _dVerletCoeff, bool _bAutoAdjust)
{
	bool bRecalculate = false;
	if (m_SimDomain.coordBeg != _simDomain.coordBeg || m_SimDomain.coordEnd != _simDomain.coordEnd)
	{
		m_SimDomain = _simDomain;
		m_workDomain = m_SimDomain;
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
		RecalculateGrid();
}


void CVerletList::EmptyGrid()
{
	for (size_t i = 0; i < m_vGrid.size(); ++i)
	{
		for (size_t x = 0; x < m_vGrid[i].grid.size(); ++x)
			for (size_t y = 0; y < m_vGrid[i].grid[x].size(); ++y)
				for (size_t z = 0; z < m_vGrid[i].grid[x][y].size(); ++z)
				{
					m_vGrid[i].grid[x][y][z].vMainPartIDs.clear();
					m_vGrid[i].grid[x][y][z].vSecondaryPartIDs.clear();
					m_vGrid[i].grid[x][y][z].vWallIDs.clear();
				}
		m_vGrid[i].grid.clear();
	}
	m_vGrid.clear();
}

void CVerletList::SortList()
{
	std::vector<std::vector<unsigned>> vTempDst(m_PPList.size());  // iDst
	std::vector<std::vector<uint8_t>> vTempVirt(m_PPList.size()); // virtData
	// remove old contacts and save new ones
	ParallelFor(m_PPList.size(), [&](size_t iSrc)
	{
		int j = 0;
		while (j < m_PPList[iSrc].size())
		{
			const unsigned iDst = m_PPList[iSrc][j];
			if (iDst < iSrc)
			{
				vTempDst[iSrc].push_back(iDst);
				m_PPList[iSrc][j] = m_PPList[iSrc].back();
				m_PPList[iSrc].pop_back();
				if (m_Scene.m_PBC.bEnabled)
				{
					vTempVirt[iSrc].emplace_back(InverseVirtShift(m_PPVirtShift[iSrc][j]));
					m_PPVirtShift[iSrc][j] = m_PPVirtShift[iSrc].back();
					m_PPVirtShift[iSrc].pop_back();
				}
			}
			else
				j++;
		}
	});

	// add new contacts
	ParallelFor([&](size_t iThread)
	{
		for (size_t iSrc = 0; iSrc < vTempDst.size(); ++iSrc)
			for (size_t i = 0; i < vTempDst[iSrc].size(); ++i)
			{
				const unsigned iDst = vTempDst[iSrc][i];
				if (iDst % m_nThreadsNumber == iThread)
				{
					m_PPList[iDst].push_back(static_cast<unsigned>(iSrc));
					if (m_Scene.m_PBC.bEnabled)
						m_PPVirtShift[iDst].push_back(vTempVirt[iSrc][i]);
				}
			}
	});
}

void CVerletList::RecalculateGrid()
{
	EmptyGrid();
	ResetCurrentData();

	double dCurrCellSize = 2 * m_dMaxParticleRadius +  m_dVerletDistance;
	if (dCurrCellSize == 0)	return;

	// if PBC is enabled, the simulation domain must be large enough to allow generation of virtual particles outside PBC, but still inside the domain
	if (m_Scene.m_PBC.bEnabled)
	{
		m_workDomain = m_SimDomain;
		const double delta = m_dVerletDistance + 2 * m_dMaxParticleRadius;
		const CVector3 pbcSides{ (double)m_Scene.m_PBC.bX, (double)m_Scene.m_PBC.bY, (double)m_Scene.m_PBC.bZ };
		for (size_t i = 0; i < 3; ++i)
			if (pbcSides[i] != 0.0)
			{
				m_workDomain.coordBeg[i] = std::min(m_workDomain.coordBeg[i], m_Scene.m_PBC.currentDomain.coordBeg[i] - delta);
				m_workDomain.coordEnd[i] = std::max(m_workDomain.coordEnd[i], m_Scene.m_PBC.currentDomain.coordEnd[i] + delta);
			}
	}

	const double dAverLength = (m_workDomain.coordEnd.x - m_workDomain.coordBeg.x + m_workDomain.coordEnd.y - m_workDomain.coordBeg.y + m_workDomain.coordEnd.z - m_workDomain.coordBeg.z) / 3;
	do
	{
		m_vGrid.emplace_back();
		SGridLevel& gl = m_vGrid.back();
		gl.dCellSize = dCurrCellSize;
		gl.dMaxPartRadius = (gl.dCellSize -  m_dVerletDistance) / 2;
		dCurrCellSize /= 2; // proceed to the next grid
		gl.dMinPartRadius = (dCurrCellSize -  m_dVerletDistance) / 2;
		if (dAverLength / gl.dCellSize > m_nCellsMax)
		{
			gl.dCellSize = dAverLength / m_nCellsMax;
			dCurrCellSize = 0; // to stop loop afterwards
		}

		gl.nCellsX = static_cast<unsigned>(floor((m_workDomain.coordEnd.x - m_workDomain.coordBeg.x) / gl.dCellSize)) + 1;
		gl.nCellsY = static_cast<unsigned>(floor((m_workDomain.coordEnd.y - m_workDomain.coordBeg.y) / gl.dCellSize)) + 1;
		gl.nCellsZ = static_cast<unsigned>(floor((m_workDomain.coordEnd.z - m_workDomain.coordBeg.z) / gl.dCellSize)) + 1;

		if (gl.nCellsX < 1) gl.nCellsX = 1;
		if (gl.nCellsY < 1) gl.nCellsY = 1;
		if (gl.nCellsZ < 1) gl.nCellsZ = 1;

		gl.grid.resize(gl.nCellsX);
		for (unsigned x = 0; x < gl.nCellsX; ++x)
		{
			gl.grid[x].resize(gl.nCellsY);
			for (unsigned y = 0; y < gl.nCellsY; ++y)
				gl.grid[x][y].resize(gl.nCellsZ);
		}
	} while (dCurrCellSize > 2*m_dMinParticleRadius +  m_dVerletDistance);

	m_vGrid.back().dMinPartRadius = 0;
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
	ClearOldPositions();
	RecalcPositions();
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

	for (auto& gridLevel : m_vGrid)
	{
		ParallelFor(gridLevel.nCellsX * gridLevel.nCellsY * gridLevel.nCellsZ, [&](size_t i)
		{
			const unsigned x = static_cast<unsigned>(floor(double(i) / gridLevel.nCellsZ / gridLevel.nCellsY));
			const unsigned y = static_cast<unsigned>(floor(double(i - x * gridLevel.nCellsZ * gridLevel.nCellsY) / gridLevel.nCellsZ));
			const unsigned z = static_cast<unsigned>(i) - x* gridLevel.nCellsZ* gridLevel.nCellsY - y* gridLevel.nCellsZ;
			CheckCollisionPP(gridLevel, x, y, z, x, y, z, true);
			CheckCollisionPW(gridLevel, gridLevel.grid[x][y][z]);
			if (gridLevel.grid[x][y][z].vMainPartIDs.size() > 10 &&  gridLevel.grid[x][y][z].vSecondaryPartIDs.empty())
			{
				CheckCollisionPPSorted(gridLevel, x, y, z, x, y, z + 1, ESortCoord::Z);
				CheckCollisionPPSorted(gridLevel, x, y, z, x, y + 1, z, ESortCoord::Y);
				CheckCollisionPPSorted(gridLevel, x, y, z, x + 1, y, z, ESortCoord::X);
				CheckCollisionPPSorted(gridLevel, x, y, z, x + 1, y + 1, z, ESortCoord::XY);
				CheckCollisionPPSorted(gridLevel, x, y, z, x + 1, y + 1, z + 1, ESortCoord::XY);
				CheckCollisionPPSorted(gridLevel, x, y, z, x, y + 1, z + 1, ESortCoord::YZ);
				CheckCollisionPPSorted(gridLevel, x, y, z, x + 1, y, z + 1, ESortCoord::XZ);

				if (x > 0)
					CheckCollisionPPSorted(gridLevel, x, y, z, x - 1, y, z + 1, ESortCoord::Z);
				if (y > 0)
				{
					if (x > 0)
						CheckCollisionPPSorted(gridLevel, x, y, z, x - 1, y - 1, z + 1, ESortCoord::Z);
					CheckCollisionPPSorted(gridLevel, x, y, z, x, y - 1, z + 1, ESortCoord::Z);
					CheckCollisionPPSorted(gridLevel, x, y, z, x + 1, y - 1, z + 1, ESortCoord::XZ);
					CheckCollisionPPSorted(gridLevel, x, y, z, x + 1, y - 1, z, ESortCoord::X);
					if (z > 0)
						CheckCollisionPPSorted(gridLevel, x, y, z, x + 1, y - 1, z - 1, ESortCoord::X);
				}
			}
			else
			{
				CheckCollisionPP(gridLevel, x, y, z, x, y, z + 1);
				CheckCollisionPP(gridLevel, x, y, z, x, y + 1, z);
				CheckCollisionPP(gridLevel, x, y, z, x + 1, y, z);
				CheckCollisionPP(gridLevel, x, y, z, x, y + 1, z + 1);
				CheckCollisionPP(gridLevel, x, y, z, x + 1, y + 1, z);
				CheckCollisionPP(gridLevel, x, y, z, x + 1, y, z + 1);
				CheckCollisionPP(gridLevel, x, y, z, x + 1, y + 1, z + 1);

				if (x > 0)
					CheckCollisionPP(gridLevel, x, y, z, x - 1, y, z + 1);
				if (y > 0)
				{
					if (x > 0)
						CheckCollisionPP(gridLevel, x, y, z, x - 1, y - 1, z + 1);
					CheckCollisionPP(gridLevel, x, y, z, x, y - 1, z + 1);
					CheckCollisionPP(gridLevel, x, y, z, x + 1, y - 1, z + 1);
					CheckCollisionPP(gridLevel, x, y, z, x + 1, y - 1, z);
					if (z > 0)
						CheckCollisionPP(gridLevel, x, y, z, x + 1, y - 1, z - 1);
				}
			}

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
			RecalculateGrid();
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
					RecalculateGrid();
				}
				else if (nIndex == m_PerformHistory.size() - 1)
				{
					m_dVerletDistance = m_PerformHistory[nIndex].dVerletDistance*1.2;
					RecalculateGrid();
				}
				else
				{
					if ((m_PerformHistory[nIndex + 1].dVerletDistance - m_PerformHistory[nIndex - 1].dVerletDistance) / m_PerformHistory[nIndex - 1].dVerletDistance > 5e-2)
					{
						m_dVerletDistance = (m_PerformHistory[nIndex + 1].dVerletDistance + m_PerformHistory[nIndex].dVerletDistance) / 2;
						RecalculateGrid();
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

void CVerletList::InsertParticlesToMultiSet(std::multiset<SEntry>& _set, const std::vector<unsigned>& _partIDs, ESortCoord _dim, ESortDir _dir) const
{
	for (unsigned int id : _partIDs)
	{
		const double radius = _dir == ESortDir::Right ? m_vParticles.ContactRadius(id) : -m_vParticles.ContactRadius(id);
		switch (_dim)
		{
		case ESortCoord::X:	_set.insert({ id, m_vParticles.Coord(id).x + radius }); break;
		case ESortCoord::Y:	_set.insert({ id, m_vParticles.Coord(id).y + radius }); break;
		case ESortCoord::Z:	_set.insert({ id, m_vParticles.Coord(id).z + radius }); break;
		case ESortCoord::XY: _set.insert({ id, (m_vParticles.Coord(id).x + m_vParticles.Coord(id).y)*0.5 + radius }); break;
		case ESortCoord::XZ: _set.insert({ id, (m_vParticles.Coord(id).x + m_vParticles.Coord(id).z)*0.5 + radius }); break;
		case ESortCoord::YZ: _set.insert({ id, (m_vParticles.Coord(id).y + m_vParticles.Coord(id).z)*0.5 + radius }); break;
		}
	}
}

void CVerletList::CheckCollisionPPSorted(const SGridLevel& _gridLevel, unsigned _nX1, unsigned _nY1, unsigned _nZ1, unsigned _nX2, unsigned _nY2, unsigned _nZ2, ESortCoord _dim)
{
	if (_nX2 >= _gridLevel.nCellsX || _nY2 >= _gridLevel.nCellsY || _nZ2 >= _gridLevel.nCellsZ) return;

	std::multiset<SEntry> setMainRSorted, setMainLSorted;
	InsertParticlesToMultiSet(setMainRSorted, _gridLevel.grid[_nX1][_nY1][_nZ1].vMainPartIDs, _dim, ESortDir::Right);
	InsertParticlesToMultiSet(setMainLSorted, _gridLevel.grid[_nX2][_nY2][_nZ2].vMainPartIDs, _dim, ESortDir::Left);
	for (auto iter1 = setMainRSorted.rbegin(); iter1 != setMainRSorted.rend(); ++iter1) //main-main
	{
		const double dTemp1 = m_dVerletDistance + m_vParticles.ContactRadius(iter1->ID);
		auto iter2 = setMainLSorted.begin();
		for (; iter2 != setMainLSorted.end(); ++iter2)
			if (iter2->dVal - iter1->dVal <= m_dVerletDistance)
			{
				if (SquaredLength(m_vParticles.Coord(iter1->ID) - m_vParticles.Coord(iter2->ID)) <= std::pow(dTemp1 + m_vParticles.ContactRadius(iter2->ID), 2))
					AddPossibleContactPP(iter1->ID, iter2->ID);
			}
			else
				break;
		if (iter2 == setMainLSorted.begin()) break;
	}

	// THIS ALGORITHM IS NOT WELL TESTED FOR MULTIGRID APPROACH
	/*for (auto iter1 = sSetMain1.rbegin(); iter1 != sSetMain1.rend(); iter1++) //main-secondary
	{
		SParticleStruct& part1 = (*m_vParticles.(iter1->ID);
		const double dTemp1 = m_dVerletDistance + part1.dContactRadius;
		auto iter2 = sSetSecond2.begin();

		for (; iter2 != sSetSecond2.end(); iter2++)
		{
			SParticleStruct& part2 = (*m_vParticles.(iter2->ID);
			if (iter2->dVal - iter1->dVal <= m_dVerletDistance)
			{
				if (SquaredLength(part1.vCoord - part2.vCoord) <= std::pow(dTemp1 + part2.dContactRadius, 2))
					AddPossibleContactPP(iter1->ID, iter2->ID);
			}
			else
				break;
		}
		if (iter2 == sSetSecond2.begin()) break;
	}

	for (auto iter1 = sSetMain2.begin(); iter1 != sSetMain2.end(); iter1++) // secondary-main
	{
		SParticleStruct& part1 = (*m_vParticles.(iter1->ID);
		const double dTemp1 = m_dVerletDistance + part1.dContactRadius;
		auto iter2 = sSetSecond2.begin();

		for (; iter2 != sSetSecond2.end(); iter2++)
		{
			SParticleStruct& part2 = (*m_vParticles.(iter2->ID);
			if (iter2->dVal - iter1->dVal <= m_dVerletDistance)
			{
				if (SquaredLength(part1.vCoord - part2.vCoord) <= std::pow(dTemp1 + part2.dContactRadius, 2))
					AddPossibleContactPP(iter1->ID, iter2->ID);
			}
			else
				break;
		}
		if (iter2 == sSetSecond2.begin()) break;
	}

	for (unsigned i = 0; i < _gridLevel.grid[_nX2][_nY2][_nZ2].vMainPartIDs.size(); ++i)
	{
		unsigned p1 = _gridLevel.grid[_nX2][_nY2][_nZ2].vMainPartIDs[i];
		SParticleStruct& part1 = m_vParticles.(p1)
		const double dTemp1 = m_dVerletDistance + part1.dContactRadius;
		for (unsigned j = 0; j < _gridLevel.grid[_nX1][_nY1][_nZ1].vSecondaryPartIDs.size(); ++j)
		{
			unsigned p2 = _gridLevel.grid[_nX1][_nY1][_nZ1].vSecondaryPartIDs[j];
			if (SquaredLength(part1.vCoord - m_vParticles.vCoord(p2)) <= pow(dTemp1 + m_vParticles.dContactRadius(p2), 2))
				AddPossibleContactPP(p2, p1);
		}
	}
	return;*/
}

void CVerletList::CheckCollisionPP(const SGridLevel& _gridLevel, unsigned _nX1, unsigned _nY1, unsigned _nZ1, unsigned _nX2, unsigned _nY2, unsigned _nZ2, bool _bSameCell /*= false*/)
{
	if (_nX2 >= _gridLevel.nCellsX || _nY2 >= _gridLevel.nCellsY || _nZ2 >= _gridLevel.nCellsZ) return;
	const SGridCell& cell1 = _gridLevel.grid[_nX1][_nY1][_nZ1];
	const SGridCell& cell2 = _gridLevel.grid[_nX2][_nY2][_nZ2];
	for (unsigned i = 0; i < cell1.vMainPartIDs.size(); ++i)
	{
		const unsigned p1 = cell1.vMainPartIDs[i];
		const double dTemp1 = m_dVerletDistance + m_vParticles.ContactRadius(p1);
		const CVector3 vPos1 = m_vParticles.Coord(p1);
		unsigned nStartIndex = 0;
		if (_bSameCell)
			nStartIndex = i + 1;
		for (unsigned j = nStartIndex; j < cell2.vMainPartIDs.size(); ++j) //main-main
		{
			unsigned p2 = cell2.vMainPartIDs[j];
			if (SquaredLength(vPos1 - m_vParticles.Coord(p2)) <= std::pow(dTemp1 + m_vParticles.ContactRadius(p2), 2))
				if (( _bSameCell ) && ( p2 < p1))
					AddPossibleContactPP(p2, p1);
				else
					AddPossibleContactPP(p1, p2);
		}
		for (unsigned j = 0; j < cell2.vSecondaryPartIDs.size(); ++j) // main-secondary
		{
			unsigned p2 = cell2.vSecondaryPartIDs[j];
			if (SquaredLength(vPos1 - m_vParticles.Coord(p2)) <= pow(dTemp1 + m_vParticles.ContactRadius(p2), 2))
				if ((_bSameCell) && (p2 < p1))
					AddPossibleContactPP(p2, p1);
				else
					AddPossibleContactPP(p1, p2);
		}
	}

	if (!_bSameCell) // secondary-main
		for (unsigned i = 0; i < cell2.vMainPartIDs.size(); ++i)
		{
			unsigned p1 = cell2.vMainPartIDs[i];
			const CVector3 vPos1 = m_vParticles.Coord(p1);
			const double dTemp1 = m_dVerletDistance + m_vParticles.ContactRadius(p1);
			for (unsigned j = 0; j < cell1.vSecondaryPartIDs.size(); ++j)
			{
				unsigned p2 = cell1.vSecondaryPartIDs[j];
				if (SquaredLength(vPos1 - m_vParticles.Coord(p2)) <= pow(dTemp1 + m_vParticles.ContactRadius(p2), 2))
					AddPossibleContactPP(p2, p1);
			}
		}
}

void CVerletList::CheckCollisionPW(const SGridLevel& _gridLevel, const SGridCell& _gridCell)
{
	for (unsigned iPart = 0; iPart < _gridCell.vMainPartIDs.size(); ++iPart)
	{
		const unsigned p = _gridCell.vMainPartIDs[iPart];
		if (m_vParticles.ContactRadius(p) <= _gridLevel.dMinPartRadius) continue; // will be considered on another grid level
		for (unsigned iWall = 0; iWall < _gridCell.vWallIDs.size(); ++iWall)
		{
			const unsigned w = _gridCell.vWallIDs[iWall];
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
	size_t nParticles= m_vParticles.Size();
	// Here index of grid layers for each specific particle is calculated. Otherwise one particle can be considered twice
	// if it is directly comes to the boundary of grid size
	std::vector<unsigned> vGridLevel(nParticles);
	ParallelFor(m_vParticles.Size(), [&](size_t i)
	{
		if (m_vParticles.Active(i))
			for (unsigned iGrid = 0; iGrid < m_vGrid.size(); iGrid++)
				if (m_vGrid[iGrid].dMaxPartRadius + DBL_EPSILON >= m_vParticles.ContactRadius(i) && m_vGrid[iGrid].dMinPartRadius - DBL_EPSILON < m_vParticles.ContactRadius(i))
				{
					vGridLevel[i] = iGrid;
					break;
				}
	});


	for (size_t iGrid = 0; iGrid < m_vGrid.size(); ++iGrid)
	{
		SGridLevel& gridLevel = m_vGrid[iGrid];
		size_t nMaxIndex = gridLevel.nCellsX*gridLevel.nCellsY* gridLevel.nCellsZ;
		static std::vector<size_t> vTotalIndex, vIDx, vIDy, vIDz;
		vTotalIndex.resize(nParticles); vIDx.resize(nParticles); vIDy.resize(nParticles); vIDz.resize(nParticles);
		ParallelFor(nParticles, [&](size_t i)
		{
			if (m_vParticles.Active(i))
			{
				const CVector3 relCoord = (m_vParticles.Coord(i) - m_workDomain.coordBeg) / gridLevel.dCellSize;
				vIDx[i] = static_cast<unsigned>(floor(relCoord.x));
				vIDy[i] = static_cast<unsigned>(floor(relCoord.y));
				vIDz[i] = static_cast<unsigned>(floor(relCoord.z));
				vTotalIndex[i] = vIDx[i] * gridLevel.nCellsY*gridLevel.nCellsZ + vIDy[i] * gridLevel.nCellsZ + vIDz[i];
			}
			else
				vTotalIndex[i] = nMaxIndex + 1;
		});

		ParallelFor([&](size_t iThread)
		{
			for (unsigned i = 0; i < nParticles; ++i)
				if ((vTotalIndex[i] < nMaxIndex) && (vTotalIndex[i] % m_nThreadsNumber == iThread))
				{
					if (vGridLevel[i] == iGrid)
						gridLevel.grid[vIDx[i]][vIDy[i]][vIDz[i]].vMainPartIDs.push_back(i);
					else if (vGridLevel[i] > iGrid)
						gridLevel.grid[vIDx[i]][vIDy[i]][vIDz[i]].vSecondaryPartIDs.push_back(i);
				}
		});
	}
}


void CVerletList::RecalcWallsPositions()
{
	ParallelFor(m_vGrid.size(), [&](size_t iGrid)
	{
		for (unsigned iWall = 0; iWall < m_vWalls.Size(); ++iWall)
		{
			SGridLevel& gridLevel = m_vGrid[iGrid];

			const CVector3 minCoord = (m_vWalls.MinCoord(iWall) - m_workDomain.coordBeg) / gridLevel.dCellSize;
			int nMinX = static_cast<int>(floor(minCoord.x));
			int nMinY = static_cast<int>(floor(minCoord.y));
			int nMinZ = static_cast<int>(floor(minCoord.z));

			if (nMinX >= static_cast<int>(gridLevel.nCellsX) || nMinY >= static_cast<int>(gridLevel.nCellsY) || nMinZ >= static_cast<int>(gridLevel.nCellsZ)) continue;

			const CVector3 maxCoord = (m_vWalls.MaxCoord(iWall) - m_workDomain.coordBeg) / gridLevel.dCellSize;
			int nMaxX = static_cast<int>(ceil(maxCoord.x)) + 1;
			int nMaxY = static_cast<int>(ceil(maxCoord.y)) + 1;
			int nMaxZ = static_cast<int>(ceil(maxCoord.z)) + 1;

			if (nMaxX < 0 || nMaxY < 0 || nMaxZ < 0) continue;

			nMaxX = std::min(nMaxX, static_cast<int>(gridLevel.nCellsX) - 1);
			nMaxY = std::min(nMaxY, static_cast<int>(gridLevel.nCellsY) - 1);
			nMaxZ = std::min(nMaxZ, static_cast<int>(gridLevel.nCellsZ) - 1);

			nMinX = std::max(nMinX, 0);
			nMinY = std::max(nMinY, 0);
			nMinZ = std::max(nMinZ, 0);

			if (nMinX > 0) nMinX--;
			if (nMinY > 0) nMinY--;
			if (nMinZ > 0) nMinZ--;

			for (int x = nMinX; x <= nMaxX; ++x)
				for (int y = nMinY; y <= nMaxY; ++y)
					for (int z = nMinZ; z <= nMaxZ; ++z)
						gridLevel.grid[x][y][z].vWallIDs.push_back(iWall);
		}
	});
}

void CVerletList::ResetCurrentData()
{
	m_dMaxTheorWallDistance = DEFAULT_TEOR_DISTANCE;
}

void CVerletList::ClearOldPositions()
{
	for (size_t i = 0; i < m_vGrid.size(); ++i)
		ParallelFor(m_vGrid[i].nCellsX, [&](size_t x)
		{
			for (unsigned y = 0; y < m_vGrid[i].nCellsY; ++y)
				for (unsigned z = 0; z < m_vGrid[i].nCellsZ; ++z)
				{
					m_vGrid[i].grid[x][y][z].vMainPartIDs.clear();
					m_vGrid[i].grid[x][y][z].vSecondaryPartIDs.clear();
					m_vGrid[i].grid[x][y][z].vWallIDs.clear();
				}
		});
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

