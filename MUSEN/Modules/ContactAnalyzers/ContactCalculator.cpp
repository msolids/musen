/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ContactCalculator.h"
#include "MixedFunctions.h"
#include "ThreadPool.h"
#include <algorithm>

CContactCalculator::CContactCalculator()
{
	m_nMaxID = 0;
	m_dCellSize = 0;
	m_nCellsX = 0;
	m_nCellsY = 0;
	m_nCellsZ = 0;
	m_nCellsMax = DEFAULT_MAX_CELLS;
}

void CContactCalculator::AddParticle(unsigned _nID, const CVector3& _vCoord, double _dRadius)
{
	m_vParticles.push_back(SParticle{ _nID, _vCoord, _dRadius });
}

void CContactCalculator::GetAllOverlaps(std::vector<unsigned>& _vID1, std::vector<unsigned>& _vID2, std::vector<double>& _vOverlap, const SPBC& _PBC)
{
	unsigned nMaxParticleID=0;
	const size_t nRealPartCount = m_vParticles.size();
	for (size_t i = 0; i < m_vParticles.size(); i++)
		nMaxParticleID = std::max(nMaxParticleID, m_vParticles[i].nID);

	AddVirtualParticles(_PBC, nMaxParticleID);
	RecalculatePositions();

	_vID1.clear();
	_vID2.clear();
	_vOverlap.clear();
	for (unsigned nOffsetX = 0; nOffsetX < 3; ++nOffsetX)
	{
		const unsigned nProcCount = static_cast<unsigned>(ceil(static_cast<double>(m_nCellsX) / 3.0));
		std::vector<std::vector<SOverlap*>> vTempOverlaps(nProcCount);
		ParallelFor(nProcCount, [&](size_t i)
		{
			unsigned nX = static_cast<unsigned>(i) * 3 + nOffsetX;
			if (nX < m_nCellsX)
			{
				for (unsigned nY = 0; nY < m_nCellsY; ++nY)
					for (unsigned nZ = 0; nZ < m_nCellsZ; ++nZ)
					{
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX, nY, nZ, true);
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX, nY, nZ + 1);
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX, nY + 1, nZ);
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX, nY + 1, nZ + 1);
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX + 1, nY, nZ);
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX + 1, nY, nZ + 1);
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX + 1, nY + 1, nZ);
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX + 1, nY + 1, nZ + 1);
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX - 1, nY, nZ + 1);
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX - 1, nY - 1, nZ + 1);
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX, nY - 1, nZ + 1);
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX + 1, nY - 1, nZ + 1);
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX + 1, nY - 1, nZ);
						CalculateOverlaps_p(vTempOverlaps[i], nX, nY, nZ, nX + 1, nY - 1, nZ - 1);
					}
			}
		});
		for (unsigned i = 0; i < vTempOverlaps.size(); ++i)
		{
			for (unsigned j = 0; j < vTempOverlaps[i].size(); ++j)
			{
				if (_PBC.bEnabled )
				{
					//avoid case of contact between two virtual particles
					if (vTempOverlaps[i][j]->nID1 > nMaxParticleID && vTempOverlaps[i][j]->nID2 > nMaxParticleID) continue;

					// if particles were virtual
					if (vTempOverlaps[i][j]->nID1 > nMaxParticleID)
						vTempOverlaps[i][j]->nID1 -= nMaxParticleID;
					if (vTempOverlaps[i][j]->nID2 > nMaxParticleID)
						vTempOverlaps[i][j]->nID2 -= nMaxParticleID;
					if (vTempOverlaps[i][j]->nID2 == vTempOverlaps[i][j]->nID1) continue;

				}
				_vID1.push_back(vTempOverlaps[i][j]->nID1);
				_vID2.push_back(vTempOverlaps[i][j]->nID2);
				_vOverlap.push_back(vTempOverlaps[i][j]->dOverlap);
				delete vTempOverlaps[i][j];
			}
		}
	}

	// reassign virtual contacts
	if (_PBC.bEnabled)
		m_vParticles.erase(m_vParticles.begin() + nRealPartCount, m_vParticles.end());
}

std::tuple<std::vector<unsigned>, std::vector<unsigned>, std::vector<double>> CContactCalculator::GetAllOverlaps(const SPBC& _PBC)
{
	std::vector<unsigned> IDs1;
	std::vector<unsigned> IDs2;
	std::vector<double> overlaps;
	GetAllOverlaps(IDs1, IDs2, overlaps, _PBC);
	return { IDs1 , IDs2, overlaps };
}

std::vector<double> CContactCalculator::GetOverlaps(const SPBC& _pbc)
{
	std::vector<unsigned> IDs1;
	std::vector<unsigned> IDs2;
	std::vector<double> overlaps;
	GetAllOverlaps(IDs1, IDs2, overlaps, _pbc);
	return overlaps;
}

std::pair<std::vector<unsigned>, std::vector<unsigned>> CContactCalculator::GetOverlappingIDs(const SPBC& _pbc)
{
	std::vector<unsigned> IDs1;
	std::vector<unsigned> IDs2;
	std::vector<double> overlaps;
	GetAllOverlaps(IDs1, IDs2, overlaps, _pbc);
	return { IDs1 , IDs2 };
}

double CContactCalculator::GetMaxParticleRadius() const
{
	if (m_vParticles.empty())
		return 0.;
	double dMaxRadius = m_vParticles.front().dRadius;
	for (size_t i = 1; i < m_vParticles.size(); ++i)
		dMaxRadius = std::max(dMaxRadius, m_vParticles[i].dRadius);
	return dMaxRadius;
}

void CContactCalculator::AddVirtualParticles(const SPBC& _PBC, unsigned _nMaxPartID )
{
	if (!_PBC.bEnabled) return;

	const size_t nRealPartCount = m_vParticles.size();
	const double dMaxRadius = GetMaxParticleRadius();
	// calculate virtual domain
	const SVolumeType virtDomain{ _PBC.currentDomain.coordBeg + dMaxRadius, _PBC.currentDomain.coordEnd - dMaxRadius };
	const CVector3& t = _PBC.boundaryShift;
	for (unsigned i = 0; i < nRealPartCount; ++i)
	{
		const CVector3& coord = m_vParticles[i].vCoord;
		const double dRadius = m_vParticles[i].dRadius;
		const bool xL = _PBC.bX && (coord.x - dRadius <= virtDomain.coordBeg.x);
		const bool yL = _PBC.bY && (coord.y - dRadius <= virtDomain.coordBeg.y);
		const bool zL = _PBC.bZ && (coord.z - dRadius <= virtDomain.coordBeg.z);

		const bool xG = _PBC.bX && (coord.x + dRadius >= virtDomain.coordEnd.x);
		const bool yG = _PBC.bY && (coord.y + dRadius >= virtDomain.coordEnd.y);

		if (xL)				m_vParticles.emplace_back(m_vParticles[i].nID + _nMaxPartID, coord + CVector3(t.x, 0, 0), dRadius);
		if (yL)				m_vParticles.emplace_back(m_vParticles[i].nID + _nMaxPartID, coord + CVector3(0, t.y, 0), dRadius);
		if (zL)				m_vParticles.emplace_back(m_vParticles[i].nID + _nMaxPartID, coord + CVector3(0, 0, t.z), dRadius);
		if (xL && yL)		m_vParticles.emplace_back(m_vParticles[i].nID + _nMaxPartID, coord + CVector3(t.x, t.y, 0), dRadius);
		if (xL && zL)		m_vParticles.emplace_back(m_vParticles[i].nID + _nMaxPartID, coord + CVector3(t.x, 0, t.z), dRadius);
		if (yL && zL)		m_vParticles.emplace_back(m_vParticles[i].nID + _nMaxPartID, coord + CVector3(0, t.y, t.z), dRadius);
		if (xL && yL && zL)	m_vParticles.emplace_back(m_vParticles[i].nID + _nMaxPartID, coord + CVector3(t.x, t.y, t.z), dRadius);

		if (xG && yL)		m_vParticles.emplace_back(m_vParticles[i].nID + _nMaxPartID, coord + CVector3(-t.x, t.y, 0), dRadius);
		if (xG && zL)		m_vParticles.emplace_back(m_vParticles[i].nID + _nMaxPartID, coord + CVector3(-t.x, 0, t.z), dRadius);
		if (yG && zL)		m_vParticles.emplace_back(m_vParticles[i].nID + _nMaxPartID, coord + CVector3(0, -t.y, t.z), dRadius);

		if (xG && yL && zL)	m_vParticles.emplace_back(m_vParticles[i].nID + _nMaxPartID, coord + CVector3(-t.x, t.y, t.z), dRadius);
		if (xL && yG && zL)	m_vParticles.emplace_back(m_vParticles[i].nID + _nMaxPartID, coord + CVector3(t.x, -t.y, t.z), dRadius);
		if (xG && yG && zL)	m_vParticles.emplace_back(m_vParticles[i].nID + _nMaxPartID, coord + CVector3(-t.x, -t.y, t.z), dRadius);
	}
}

void CContactCalculator::SetMaxCellsNumber(unsigned _nMaxCells)
{
	m_nCellsMax = _nMaxCells;
}

void CContactCalculator::RecalculatePositions()
{
	if (m_vParticles.empty()) return;

	// remove old data
	m_Grid.clear();

	// calculate simulation domain and max radius of particles
	m_vSimDomain.coordBeg = m_vParticles.front().vCoord;
	m_vSimDomain.coordEnd = m_vParticles.front().vCoord;
	double dMaxPartRadius = 0;
	m_nMaxID = 0;
	for (unsigned i = 0; i < m_vParticles.size(); ++i)
	{
		if (m_vParticles[i].dRadius > dMaxPartRadius)
			dMaxPartRadius = m_vParticles[i].dRadius;
		if (m_vParticles[i].nID > m_nMaxID)
			m_nMaxID = m_vParticles[i].nID;
		m_vSimDomain.coordBeg = Min(m_vSimDomain.coordBeg, m_vParticles[i].vCoord);
		m_vSimDomain.coordEnd = Max(m_vSimDomain.coordEnd, m_vParticles[i].vCoord);
	}

	// calculate grid cells
	m_dCellSize = 2.1 * dMaxPartRadius;
	double dMaxLength = std::max({ m_vSimDomain.coordEnd.x - m_vSimDomain.coordBeg.x, m_vSimDomain.coordEnd.y - m_vSimDomain.coordBeg.y, m_vSimDomain.coordEnd.z - m_vSimDomain.coordBeg.z });
	if (dMaxLength / m_dCellSize > m_nCellsMax) m_dCellSize = dMaxLength / m_nCellsMax;
	m_nCellsX = static_cast<unsigned>(floor((m_vSimDomain.coordEnd.x - m_vSimDomain.coordBeg.x) / m_dCellSize) + 1);
	m_nCellsY = static_cast<unsigned>(floor((m_vSimDomain.coordEnd.y - m_vSimDomain.coordBeg.y) / m_dCellSize) + 1);
	m_nCellsZ = static_cast<unsigned>(floor((m_vSimDomain.coordEnd.z - m_vSimDomain.coordBeg.z) / m_dCellSize) + 1);
	if (m_nCellsX < 1) m_nCellsX = 1;
	if (m_nCellsY < 1) m_nCellsY = 1;
	if (m_nCellsZ < 1) m_nCellsZ = 1;

	//resize grid
	m_Grid.resize(m_nCellsX);
	for (unsigned iX = 0; iX < m_nCellsX; ++iX)
	{
		m_Grid[iX].resize(m_nCellsY);
		for (unsigned iY = 0; iY < m_nCellsY; ++iY)
			m_Grid[iX][iY].resize(m_nCellsZ);
	}

	// set particles to grid
	for (unsigned i = 0; i < m_vParticles.size(); ++i)
	{
		CVector3 tempCoord = m_vParticles[i].vCoord - m_vSimDomain.coordBeg;
		unsigned nIDx = tempCoord.x < 0 ? 0 : static_cast<unsigned>(floor(tempCoord.x / m_dCellSize));
		unsigned nIDy = tempCoord.y < 0 ? 0 : static_cast<unsigned>(floor(tempCoord.y / m_dCellSize));
		unsigned nIDz = tempCoord.z < 0 ? 0 : static_cast<unsigned>(floor(tempCoord.z / m_dCellSize));
		if (nIDx >= m_nCellsX) nIDx = m_nCellsX - 1;
		if (nIDy >= m_nCellsY) nIDy = m_nCellsY - 1;
		if (nIDz >= m_nCellsZ) nIDz = m_nCellsZ - 1;

		m_Grid[nIDx][nIDy][nIDz].push_back(i);
	}
}


void CContactCalculator::CalculateOverlaps_p(std::vector<SOverlap*>& _vOverlapsList, unsigned _nX1, unsigned _nY1, unsigned _nZ1, unsigned _nX2, unsigned _nY2, unsigned _nZ2, bool _bSameCell /*= false*/)
{
	if ((_nX1 >= m_nCellsX) || (_nY1 >= m_nCellsY) || (_nZ1 >= m_nCellsZ)) return;
	if ((_nX2 >= m_nCellsX) || (_nY2 >= m_nCellsY) || (_nZ2 >= m_nCellsZ)) return;
	for (unsigned i = 0; i < m_Grid[_nX1][_nY1][_nZ1].size(); ++i)
	{
		for (unsigned j = 0; j < m_Grid[_nX2][_nY2][_nZ2].size(); ++j)
		{
			if ((_bSameCell) && (j <= i)) continue;
			const SParticle& Part1 = m_vParticles[m_Grid[_nX1][_nY1][_nZ1][i]];
			const SParticle& Part2 = m_vParticles[m_Grid[_nX2][_nY2][_nZ2][j]];

			double dLength = Length(Part1.vCoord, Part2.vCoord);
			double dCurrOverlap = Part1.dRadius + Part2.dRadius - dLength;
			if (dCurrOverlap > 0)
				_vOverlapsList.push_back(new SOverlap(Part1.nID, Part2.nID, dCurrOverlap));
		}
	}
}