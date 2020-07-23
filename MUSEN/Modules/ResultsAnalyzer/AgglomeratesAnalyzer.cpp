/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "AgglomeratesAnalyzer.h"

CAgglomeratesAnalyzer::CAgglomeratesAnalyzer()
{
	SetPropertyType(EPropertyType::Coordinate);
}

size_t CAgglomeratesAnalyzer::GetAgglomeratesNumber() const
{
	return m_vAgglomParticles.size();
}

bool CAgglomeratesAnalyzer::Export()
{
	// request proper material database if needed
	if (!m_Constraints.IsAllVolumeSelected() || (GetProperty() == EPropertyType::Coordinate) || (GetProperty() == EPropertyType::VelocityTotal))
		if (!CheckMaterialsDatabase())
			return false;

	if (m_bConcParam || m_nResultsType != EResultType::Distribution)
		m_nPropSteps = 1;

	if (GetProperty() == EPropertyType::ExportToFile)
	{
		m_bCustomFileWriter = true; // to omit calling of WriteResultsToFile() from CResultsAnalyzer::StartExport
		FindAgglomerates(m_vTimePoints.front());
		std::vector<size_t> vAgglomerates = FilterAgglomerates(m_vTimePoints.front());

		for (auto iAgglom : vAgglomerates)
			ExportCoordinatesAgglomerates(iAgglom, m_vTimePoints.front());
		return true;
	}

	for (size_t iTime = 0; iTime < m_vDistrResults.size(); ++iTime)
	{
		if (CheckTerminationFlag()) return false;

		const double dTime = m_vTimePoints[iTime];

		// status description
		m_sStatusDescr = "Time = " + std::to_string(dTime) + " [s]. Applying constraints. ";

		FindAgglomerates(dTime);
		if (!m_Constraints.IsAllVolumeSelected() || GetProperty() == EPropertyType::Coordinate)
			CalculateAgglomsCenterOfMass(dTime);

		const std::vector<size_t> vAgglomerates = FilterAgglomerates(dTime); // filter agglomerates by constraints

		// status description
		m_sStatusDescr = "Time = " + std::to_string(dTime) + " [s]. Processing agglomerates (" + std::to_string(vAgglomerates.size()) + ")";

		for (auto iAgglom : vAgglomerates)
		{
			if (CheckTerminationFlag()) return false;
			switch (GetProperty())
			{
			case EPropertyType::Number:
				m_vConcResults[iTime]++;
				break;
			case EPropertyType::Coordinate:
				WriteComponentToResults(m_vAgglomeratesMassCenters[iAgglom], iTime);
				break;
			case EPropertyType::VelocityTotal:
				WriteComponentToResults(CalculateAgglomVelocity(iAgglom, dTime), iTime);
				break;
			case EPropertyType::Orientation:
				WriteComponentToResults(CalculateAgglomOrientation(iAgglom, dTime), iTime);
				break;
			case EPropertyType::Diameter:
				WriteValueToResults(CalculateAgglomDiameter(iAgglom, dTime), iTime);
				break;
			case EPropertyType::PartNumber:
				WriteValueToResults(static_cast<double>(m_vAgglomParticles[iAgglom].size()), iTime);
				break;
			case EPropertyType::BondNumber:
				WriteValueToResults(static_cast<double>(m_vAgglomBonds[iAgglom].size()), iTime);
				break;
			default:
				break;
			}
		}
		m_nProgress = static_cast<unsigned>((iTime + 1.) / static_cast<double>(m_vDistrResults.size()) * 100);
	}
	return true;
}

void CAgglomeratesAnalyzer::FindAgglomerates(double _dTime)
{
	std::vector<size_t> vBondsIndexes, vPartIndexes;
	vPartIndexes.reserve(m_pSystemStructure->GetNumberOfSpecificObjects(SPHERE));
	vBondsIndexes.reserve(m_pSystemStructure->GetNumberOfSpecificObjects(SOLID_BOND) + m_pSystemStructure->GetNumberOfSpecificObjects(LIQUID_BOND));
	std::vector<size_t> vPartAssignment(m_pSystemStructure->GetTotalObjectsCount(), -1); // index of particle chain where this particle is considered
	for (size_t i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); ++i)
	{
		CPhysicalObject* pObj = m_pSystemStructure->GetObjectByIndex(i);
		if (!pObj || !pObj->IsActive(_dTime)) continue;
		if (pObj->GetObjectType() == SOLID_BOND || pObj->GetObjectType() == LIQUID_BOND)
			vBondsIndexes.push_back(i);
		else if (pObj->GetObjectType() == SPHERE)
			vPartIndexes.push_back(i);
	}

	std::vector<std::vector<size_t>> vBondChains, vPartChains;  // particle and bonds chains
	vBondChains.resize(vBondsIndexes.size()); // maximal number of chains is vBondsIndexes.size()
	vPartChains.resize(vBondsIndexes.size());

	for (size_t i = 0; i < vBondsIndexes.size(); ++i)
	{
		const auto* pBond = dynamic_cast<CBond*>(m_pSystemStructure->GetObjectByIndex(vBondsIndexes[i]));
		const size_t nPart1ID = pBond->m_nLeftObjectID;
		const size_t nPart2ID = pBond->m_nRightObjectID;
		if (vPartAssignment[nPart1ID] == static_cast<size_t>(-1) && vPartAssignment[nPart2ID] == static_cast<size_t>(-1)) // both particles were not assigned to any bond chain
		{
			vPartChains[i].push_back(nPart1ID);
			vPartChains[i].push_back(nPart2ID);
			vBondChains[i].push_back(vBondsIndexes[i]);
			vPartAssignment[nPart1ID] = i;
			vPartAssignment[nPart2ID] = i;
		}
		else if (vPartAssignment[nPart1ID] != static_cast<size_t>(-1) && vPartAssignment[nPart2ID] != static_cast<size_t>(-1)) // both particles were already assigned to one of chains
		{
			const int64_t nChainDstID = vPartAssignment[nPart1ID];
			const int64_t nChainSrcID = vPartAssignment[nPart2ID];
			// concatenate vectors
			if (nChainDstID != nChainSrcID)
			{
				for (size_t j = 0; j < vPartChains[nChainSrcID].size(); j++) // reassign new index to all secondary particles
					vPartAssignment[vPartChains[nChainSrcID][j]] = nChainDstID;
				vPartChains[nChainDstID].insert(vPartChains[nChainDstID].end(), vPartChains[nChainSrcID].begin(), vPartChains[nChainSrcID].end());
				vPartChains[nChainSrcID].clear();
				vBondChains[nChainDstID].insert(vBondChains[nChainDstID].end(), vBondChains[nChainSrcID].begin(), vBondChains[nChainSrcID].end());
				vBondChains[nChainSrcID].clear();
			}
			vBondChains[nChainDstID].push_back(vBondsIndexes[i]);
		}
		else
		{
			if (vPartAssignment[nPart1ID] == static_cast<size_t>(-1))
			{
				vPartChains[vPartAssignment[nPart2ID]].push_back(nPart1ID);
				vBondChains[vPartAssignment[nPart2ID]].push_back(vBondsIndexes[i]);
				vPartAssignment[nPart1ID] = vPartAssignment[nPart2ID];
			}
			else
			{
				vPartChains[vPartAssignment[nPart1ID]].push_back(nPart2ID);
				vBondChains[vPartAssignment[nPart1ID]].push_back(vBondsIndexes[i]);
				vPartAssignment[nPart2ID] = vPartAssignment[nPart1ID];
			}
		}
	}
	m_vAgglomParticles.clear();
	m_vAgglomBonds.clear();
	for (size_t i = 0; i < vPartChains.size(); i++) // add agglomerates consisting of more than one particle
		if (!vPartChains[i].empty())
		{
			m_vAgglomParticles.push_back(vPartChains[i]);
			m_vAgglomBonds.push_back(vBondChains[i]);
		}
	for (size_t i = 0; i < vPartIndexes.size(); i++) // add separate particles
		if (vPartAssignment[vPartIndexes[i]] == static_cast<size_t>(-1))
		{
			m_vAgglomParticles.push_back(std::vector<size_t>{vPartIndexes[i]});
			m_vAgglomBonds.emplace_back(); //empty vector
		}
}

void CAgglomeratesAnalyzer::FindSuperAgglomerates(double _dTime)
{
	// Constraints
	/*const double openingDP = 0.17453;
	const double pos_upper = 9.647648640742474e-08;
	const double neg_upper = 1.524297034366652e-08;*/
	const double openingDP = 0.1;
	const double pos_upper = 4e-08;
	const double neg_upper = 0;
	const double pos_upper_squared = pos_upper * pos_upper;

	// Calculate COM of all agglomerates (were found using FindAgglomerates)
	CalculateAgglomsCenterOfMass(_dTime);

	// Find super agglomerates by distance of COM
	std::vector<unsigned> vSuperBondsIndexes, vSuperAgglomerateIndexes;
	std::vector<unsigned> vSuperBondsLeftIDs, vSuperBondsRightIDs;				// Saves ID's of left and right agglomerates recognized as super-agglomerate
	std::vector<double> vSuperBondsLengths;
	std::vector<int64_t> vPartAssignment; // index of particle chain where this particle is considered
	vPartAssignment.resize(m_vAgglomParticles.size(), -1);
	unsigned superBondCount = 0;
	for (unsigned i = 0; i < m_vAgglomParticles.size(); ++i)
	{
		vSuperAgglomerateIndexes.push_back(i);
		for (unsigned j = i+1; j < m_vAgglomParticles.size(); j++)
		{
			// Rule out contact of far-away
			if (GetSolidBond(m_vAgglomeratesMassCenters[j], m_vAgglomeratesMassCenters[i], m_pSystemStructure->GetPBC()).SquaredLength() > pos_upper_squared)
				continue;

			// Check for contact inside agglomerates using orientation
			for (size_t p_i = 0; p_i < m_vAgglomParticles[i].size(); p_i++)
			{
				for (size_t p_j = 0; p_j < m_vAgglomParticles[j].size(); p_j++)
				{
					CPhysicalObject* pObj_1 = m_pSystemStructure->GetObjectByIndex(m_vAgglomParticles[i][p_i]);
					CPhysicalObject* pObj_2 = m_pSystemStructure->GetObjectByIndex(m_vAgglomParticles[j][p_j]);

					CVector3 v_xDirPar1 = QuatRotateVector(pObj_1->GetOrientation(_dTime), CVector3(1, 0, 0));								// Vector of x direction of particle 1 (left)
					CVector3 v_xDirPar2 = QuatRotateVector(pObj_2->GetOrientation(_dTime), CVector3(1, 0, 0));								// Vector of x direction of particle 2 (right)

					CVector3 v_BondVec1 = GetSolidBond(pObj_1->GetCoordinates(_dTime), pObj_2->GetCoordinates(_dTime), m_pSystemStructure->GetPBC());		// vector: left -> right
					double d_PartDistance = v_BondVec1.Length();												// particle distance
					v_BondVec1 = v_BondVec1.Normalized();														// normalize
					CVector3 v_BondVec2 = -1 * v_BondVec1;														// vector: right -> left

					double d_DP1 = DotProduct(v_xDirPar1, v_BondVec1);											// dot product left-axis
					double d_DP2 = DotProduct(v_xDirPar2, v_BondVec2);											// dot product right-axis

					if (((d_DP1 > openingDP) && (d_DP2 > openingDP) && (d_PartDistance <= pos_upper)) || (d_PartDistance <= neg_upper))
					{
						vSuperBondsIndexes.push_back(superBondCount);
						superBondCount++;
						vSuperBondsLeftIDs.push_back(i);
						vSuperBondsRightIDs.push_back(j);
						vSuperBondsLengths.push_back(GetSolidBond(m_vAgglomeratesMassCenters[j], m_vAgglomeratesMassCenters[i], m_pSystemStructure->GetPBC()).Length());

						// break out of loops
						p_i = m_vAgglomParticles[i].size();
						break;
					}
				}
			}

			// // Check for contact using only distance
			//double squaredDistAgglom = GetSolidBond(m_vAgglomeratesMassCenters[j], m_vAgglomeratesMassCenters[i], m_pSystemStructure->GetPBC()).SquaredLength();
			//if (squaredDistAgglom <= 4.7E-15)				// consider super agglomerate if distance between COM is less than 68.5 nm
			//{
			//	vSuperBondsIndexes.push_back(superBondCount);
			//	superBondCount++;
			//	vSuperBondsLeftIDs.push_back(i);
			//	vSuperBondsRightIDs.push_back(j);
			//	vSuperBondsLengths.push_back(sqrt(squaredDistAgglom));
			//}
		}
	}

	std::vector<std::vector<size_t>> vBondChains, vPartChains;  // particle and bonds chains
	vBondChains.resize(vSuperBondsIndexes.size()); // maximal number of chains is vBondsIndexes.size()
	vPartChains.resize(vSuperBondsIndexes.size());

	for (int64_t i = 0; i < static_cast<int64_t>(vSuperBondsIndexes.size()); i++)
	{
		const unsigned nPart1ID = vSuperBondsLeftIDs[vSuperBondsIndexes[i]];
		const unsigned nPart2ID = vSuperBondsRightIDs[vSuperBondsIndexes[i]];
		if (vPartAssignment[nPart1ID] == -1 && vPartAssignment[nPart2ID] == -1) // both particles were not assigned to any bond chain
		{
			vPartChains[i].push_back(nPart1ID);
			vPartChains[i].push_back(nPart2ID);
			vBondChains[i].push_back(vSuperBondsIndexes[i]);
			vPartAssignment[nPart1ID] = i;
			vPartAssignment[nPart2ID] = i;
		}
		else if (vPartAssignment[nPart1ID] != -1 && vPartAssignment[nPart2ID] != -1) // both particles were already assigned to one of chains
		{
			const int64_t nChainDstID = vPartAssignment[nPart1ID];
			const int64_t nChainSrcID = vPartAssignment[nPart2ID];
			// concatenate vectors
			if (nChainDstID != nChainSrcID)
			{
				for (size_t j = 0; j < vPartChains[nChainSrcID].size(); j++) // reassign new index to all secondary particles
					vPartAssignment[vPartChains[nChainSrcID][j]] = nChainDstID;
				vPartChains[nChainDstID].insert(vPartChains[nChainDstID].end(), vPartChains[nChainSrcID].begin(), vPartChains[nChainSrcID].end());
				vPartChains[nChainSrcID].clear();
				vBondChains[nChainDstID].insert(vBondChains[nChainDstID].end(), vBondChains[nChainSrcID].begin(), vBondChains[nChainSrcID].end());
				vBondChains[nChainSrcID].clear();
			}
			vBondChains[nChainDstID].push_back(vSuperBondsIndexes[i]);
		}
		else
		{
			if (vPartAssignment[nPart1ID] == -1)
			{
				vPartChains[vPartAssignment[nPart2ID]].push_back(nPart1ID);
				vBondChains[vPartAssignment[nPart2ID]].push_back(vSuperBondsIndexes[i]);
				vPartAssignment[nPart1ID] = vPartAssignment[nPart2ID];
			}
			else
			{
				vPartChains[vPartAssignment[nPart1ID]].push_back(nPart2ID);
				vBondChains[vPartAssignment[nPart1ID]].push_back(vSuperBondsIndexes[i]);
				vPartAssignment[nPart2ID] = vPartAssignment[nPart1ID];
			}
		}
	}
	for (size_t i = 0; i < vPartChains.size(); i++) // add agglomerates consisting of more than one particle
		if (!vPartChains[i].empty())
		{
			m_vSuperAgglomerates.push_back(vPartChains[i]);
			//m_vAgglomBonds.push_back(vBondChains[i]);
		}
	for (size_t i = 0; i < vSuperAgglomerateIndexes.size(); i++) // add separate particles
		if (vPartAssignment[vSuperAgglomerateIndexes[i]] == -1)
		{
			m_vSuperAgglomerates.push_back(std::vector<size_t>{vSuperAgglomerateIndexes[i]});
			//m_vAgglomBonds.emplace_back(); //empty vector
		}

	// Determine number of particles per superAgglomerate
	m_vSuperAgglomeratesNumParticles.resize(m_vSuperAgglomerates.size(),0);
	for (size_t i = 0; i < m_vSuperAgglomerates.size(); i++)
	{
		for (size_t j = 0; j < m_vSuperAgglomerates[i].size(); j++)
		{
			m_vSuperAgglomeratesNumParticles[i] += m_vAgglomParticles[m_vSuperAgglomerates[i][j]].size();
		}
	}

	// Determine COM of super-agglomerates
	CalculateSuperAgglomsCOM(_dTime);

	// Calculate radius of super-agglomerates
	CalculateSuperAgglomsRadiusGyration(_dTime);

	///////////////////////////////////////////////////////
	////////////// PRELIMINARY OUTPUT /////////////////////
	///////////////////////////////////////////////////////
	if (false)
	{
		// Determine SuperAgglomerate size (number of individual particles)
		size_t minSizeSuperAgglo = (std::numeric_limits<size_t>::max)();
		size_t maxSizeSuperAgglo = 0;
		for (auto tempSize : m_vSuperAgglomeratesNumParticles)
		{
			if (tempSize > maxSizeSuperAgglo)
				maxSizeSuperAgglo = tempSize;
			if (tempSize < minSizeSuperAgglo)
				minSizeSuperAgglo = tempSize;
		}
		std::vector<size_t> superAggloHist;
		superAggloHist.resize(maxSizeSuperAgglo, 0);
		// fill histogram
		for (size_t i = 0; i < m_vSuperAgglomerates.size(); i++)
		{
			superAggloHist[m_vSuperAgglomeratesNumParticles[i] - 1]++;
		}

		// output to file - preliminary
		if (_dTime < 1e-15)
		{
			std::remove("Super_Agglomerate_Histogram.txt");
		}
		std::ofstream pFile;
		pFile.open("Super_Agglomerate_Histogram.txt", std::ofstream::out | std::ofstream::app);
		pFile << _dTime << "\t";
		for (size_t i = 0; i < maxSizeSuperAgglo; i++)
		{
			pFile << superAggloHist[i] << "\t";
		}
		pFile << "\n";
		pFile.close();
	}
}

void CAgglomeratesAnalyzer::CalculateMSD(double _dTime)
{
	if (_dTime < 1e-15)
	{
		std::remove("Agglomerate_MSD.txt");
		m_vAgglomeratesMassCenters_Initial = m_vAgglomeratesMassCenters;
	}

	double MSD_current = 0;
	const size_t numAgglom = m_vAgglomeratesMassCenters.size();
	for (size_t i = 0; i < numAgglom; i++)
	{
		MSD_current += (m_vAgglomeratesMassCenters[i] - m_vAgglomeratesMassCenters_Initial[i]).SquaredLength();
	}
	MSD_current /= numAgglom;

	// output to file - preliminary
	std::ofstream pFile;
	pFile.open("Agglomerate_MSD.txt", std::ofstream::out | std::ofstream::app);
	pFile << _dTime << "\t" << MSD_current << "\n";
	pFile.close();
}

std::vector<size_t> CAgglomeratesAnalyzer::FilterAgglomerates(double _dTime) const
{
	std::set<size_t> vFilteredAggloms;

	// filter by volume
	if (!m_Constraints.IsAllVolumeSelected())
		vFilteredAggloms = m_Constraints.ApplyVolumeFilter(_dTime, m_vAgglomeratesMassCenters);
	else
		for (size_t i = 0; i < m_vAgglomParticles.size(); ++i)
			vFilteredAggloms.insert(i);

	// filter by material
	if (!m_Constraints.IsAllMaterialsSelected())
		vFilteredAggloms = ApplyMaterialFilter(_dTime, vFilteredAggloms);

	return std::vector<size_t>(vFilteredAggloms.begin(), vFilteredAggloms.end());
}

std::set<size_t> CAgglomeratesAnalyzer::ApplyMaterialFilter(double _dTime, const std::set<size_t>& _vIndexes) const
{
	std::set<size_t> vFiltered;
	for (const auto iAgg : _vIndexes)
	{
		bool materialIsTrue = true;
		for (auto iObj : VectorUnion(m_vAgglomParticles[iAgg], m_vAgglomBonds[iAgg]))
		{
			const CPhysicalObject* pObj = m_pSystemStructure->GetObjectByIndex(iObj);
			if (!pObj || !pObj->IsActive(_dTime)) continue;
			if (!m_Constraints.CheckMaterial(pObj->GetCompoundKey()))
			{
				materialIsTrue = false;
				break;
			}
		}

		if (materialIsTrue)
			vFiltered.insert(vFiltered.end(), iAgg);
	}
	return vFiltered;
}

void CAgglomeratesAnalyzer::CalculateAgglomsCenterOfMass( double _dTime)
{
	m_vAgglomeratesMassCenters.resize(m_vAgglomParticles.size(), CVector3(0, 0, 0));

	for (size_t i = 0; i < m_vAgglomParticles.size(); ++i)
	{
		CVector3 vCoordRef = m_pSystemStructure->GetObjectByIndex(m_vAgglomParticles[i][0])->GetCoordinates(_dTime); // use first particle as reference point for PBC
		CVector3 vCenter(0, 0, 0);
		double dTotalMass = 0.0;
		for (size_t j = 0; j < m_vAgglomParticles[i].size(); ++j)
		{
			double dMass = m_pSystemStructure->GetObjectByIndex(m_vAgglomParticles[i][j])->GetMass();
			vCenter += (vCoordRef + GetSolidBond(vCoordRef, m_pSystemStructure->GetObjectByIndex(m_vAgglomParticles[i][j])->GetCoordinates(_dTime), m_pSystemStructure->GetPBC())) * dMass;
			dTotalMass += dMass;
		}
		vCenter = vCenter / dTotalMass;
		m_vAgglomeratesMassCenters[i] = vCenter;
	}
}

void CAgglomeratesAnalyzer::CalculateSuperAgglomsCOM(double _dTime)
{
	m_vSuperAgglomCOM.resize(m_vSuperAgglomerates.size(), CVector3(0,0,0));

	for (size_t super_i = 0; super_i < m_vSuperAgglomerates.size(); ++super_i)
	{
		CVector3 vCoordRef = m_pSystemStructure->GetObjectByIndex(m_vAgglomParticles[m_vSuperAgglomerates[super_i][0]][0])->GetCoordinates(_dTime);				// use first particle as reference point for PBC
		CVector3 vCenter(0, 0, 0);
		double dTotalMass = 0.0;
		for (size_t agglom_i = 0; agglom_i < m_vSuperAgglomerates[super_i].size(); ++agglom_i)
		{
			size_t agglom_ID = m_vSuperAgglomerates[super_i][agglom_i];
			for (size_t part_i = 0; part_i < m_vAgglomParticles[agglom_ID].size(); part_i++)
			{
				size_t part_ID = m_vAgglomParticles[agglom_ID][part_i];
				double dMass = m_pSystemStructure->GetObjectByIndex(part_ID)->GetMass();
				vCenter += (vCoordRef + GetSolidBond(vCoordRef, m_pSystemStructure->GetObjectByIndex(part_ID)->GetCoordinates(_dTime), m_pSystemStructure->GetPBC())) * dMass;
				dTotalMass += dMass;
			}
		}
		vCenter = vCenter / dTotalMass;
		m_vSuperAgglomCOM[super_i] = vCenter;
	}
}

void CAgglomeratesAnalyzer::CalculateSuperAgglomsRadiusGyration(double _dTime)
{
	m_vSuperAgglomRadiusGyration.resize(m_vSuperAgglomerates.size(), 0);
	for (size_t super_i = 0; super_i < m_vSuperAgglomerates.size(); ++super_i)
	{
		double tempSuperAgglomRadiusGyration = 0;
		size_t numParts = 0;
		for (size_t agglom_i = 0; agglom_i < m_vSuperAgglomerates[super_i].size(); ++agglom_i)
		{
			size_t agglom_ID = m_vSuperAgglomerates[super_i][agglom_i];
			for (size_t part_i = 0; part_i < m_vAgglomParticles[agglom_ID].size(); part_i++)
			{
				size_t part_ID = m_vAgglomParticles[agglom_ID][part_i];
				tempSuperAgglomRadiusGyration += GetSolidBond(m_pSystemStructure->GetObjectByIndex(part_ID)->GetCoordinates(_dTime), m_vSuperAgglomCOM[super_i], m_pSystemStructure->GetPBC()).SquaredLength();
				numParts++;
				//double tempPartRadiusPart = static_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(part_ID))->GetRadius();
			}
		}
		tempSuperAgglomRadiusGyration /= numParts;
		m_vSuperAgglomRadiusGyration[super_i] = sqrt(tempSuperAgglomRadiusGyration);
	}
}

CVector3 CAgglomeratesAnalyzer::CalculateAgglomVelocity(size_t _iAgglom, double _dTime)
{
	CVector3 vecAmountVelocityAndMass(0);
	double dTotalMass = 0.0;
	for (auto iObj : m_vAgglomParticles[_iAgglom])
	{
		const CPhysicalObject* pObj = m_pSystemStructure->GetObjectByIndex(iObj);
		dTotalMass += pObj->GetMass();
		vecAmountVelocityAndMass += pObj->GetMass() * pObj->GetVelocity(_dTime);
	}
	if (dTotalMass != 0)
		vecAmountVelocityAndMass = vecAmountVelocityAndMass / dTotalMass;
	return vecAmountVelocityAndMass;
}

CVector3 CAgglomeratesAnalyzer::CalculateAgglomOrientation(size_t _iAgglom, double _dTime)
{
	// find pair of points with largest distance between them
	double dMaxDistance = 0;
	CVector3 vOrientation(0);
	for (size_t i = 0; i < m_vAgglomParticles[_iAgglom].size(); ++i)
		for (size_t j = i + 1; j < m_vAgglomParticles[_iAgglom].size(); ++j)
		{
			CVector3 vPos1 = m_pSystemStructure->GetObjectByIndex(m_vAgglomParticles[_iAgglom][j])->GetCoordinates(_dTime);
			CVector3 vPos2 = m_pSystemStructure->GetObjectByIndex(m_vAgglomParticles[_iAgglom][i])->GetCoordinates(_dTime);
			if (Length(vPos1 - vPos2) > dMaxDistance)
			{
				vOrientation = vPos1 - vPos2;
				dMaxDistance = Length(vPos1 - vPos2);
				vOrientation.x = std::fabs(vOrientation.x) / dMaxDistance;
				vOrientation.y = std::fabs(vOrientation.y) / dMaxDistance;
				vOrientation.z = std::fabs(vOrientation.z) / dMaxDistance;
			}
		}
	return vOrientation;
}

double CAgglomeratesAnalyzer::CalculateAgglomDiameter(size_t _iAgglom, double _dTime)
{
	double dVolume = 0;
	for (auto iPart : m_vAgglomParticles[_iAgglom])
		dVolume += dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(iPart))->GetVolume();
	for (auto iBond : m_vAgglomBonds[_iAgglom])
		dVolume += m_pSystemStructure->GetBondVolume(_dTime, iBond);
	return 2 * pow(3 * dVolume / (4 * PI), 1.0 / 3);
}

void CAgglomeratesAnalyzer::ExportCoordinatesAgglomerates(size_t _iAgglom, double _dTime)
{
	const size_t nLatitude = 18;				// Number vertical lines.
	const size_t nLongitude = nLatitude / 2;	// Number horizontal lines.
	SPBC PBC = m_pSystemStructure->GetPBC();
	PBC.UpdatePBC(_dTime);
	const CVector3 vCenterPoint = (PBC.currentDomain.coordBeg + PBC.currentDomain.coordEnd) / 2;
	std::vector<CVector3> vCoords;
	std::vector<double> vRadii;
	// in case of PBC flags indicating that agglomerate cross boundary
	CVector3 vOverXYZ(0);
	for (auto iPart : m_vAgglomParticles[_iAgglom])
	{
		vCoords.push_back(m_pSystemStructure->GetObjectByIndex(iPart)->GetCoordinates(_dTime));
		vRadii.push_back(dynamic_cast<CSphere*>(m_pSystemStructure->GetObjectByIndex(iPart))->GetRadius());
	}

	//consider PBC if some of agglomerate is over PBC - than duplicate it over corresponding PBC
	if (PBC.bEnabled)
	{
		for (auto iBond : m_vAgglomBonds[_iAgglom])
		{
			const auto* pBond = dynamic_cast<CBond*>(m_pSystemStructure->GetObjectByIndex(iBond));
			const CVector3 vPos1 = m_pSystemStructure->GetObjectByIndex(pBond->m_nLeftObjectID)->GetCoordinates(_dTime);
			const CVector3 vPos2 = m_pSystemStructure->GetObjectByIndex(pBond->m_nRightObjectID)->GetCoordinates(_dTime);
			if (std::fabs(vPos1.x - vPos2.x) > PBC.boundaryShift.x / 2 && PBC.bX) vOverXYZ.x = 1;
			if (std::fabs(vPos1.y - vPos2.y) > PBC.boundaryShift.y / 2 && PBC.bY) vOverXYZ.y = 1;
			if (std::fabs(vPos1.z - vPos2.z) > PBC.boundaryShift.z / 2 && PBC.bZ) vOverXYZ.z = 1;
		}
		for (auto& coord : vCoords)
		{
			if (vOverXYZ.x && coord.x > vCenterPoint.x)	coord.x -= PBC.boundaryShift.x;
			if (vOverXYZ.y && coord.y > vCenterPoint.y)	coord.y -= PBC.boundaryShift.y;
			if (vOverXYZ.z && coord.z > vCenterPoint.z)	coord.z -= PBC.boundaryShift.z;
		}

		if (vCoords.size() == 1)
		{
			if (PBC.bX)
			{
				if (vCoords[0].x + vRadii[0] > PBC.currentDomain.coordEnd.x)
				{
					vCoords[0].x -= PBC.boundaryShift.x;
					vOverXYZ.x = 1;
				}
				else if (vCoords[0].x - vRadii[0] < PBC.currentDomain.coordBeg.x)
					vOverXYZ.x = 1;
			}
			if (PBC.bY)
			{
				if (vCoords[0].y + vRadii[0] > PBC.currentDomain.coordEnd.y)
				{
					vCoords[0].y -= PBC.boundaryShift.y;
					vOverXYZ.y = 1;
				}
				else if (vCoords[0].y - vRadii[0] < PBC.currentDomain.coordBeg.y)
					vOverXYZ.y = 1;
			}
			if (PBC.bZ)
			{
				if (vCoords[0].z + vRadii[0] > PBC.currentDomain.coordEnd.z)
				{
					vCoords[0].z -= PBC.boundaryShift.z;
					vOverXYZ.z = 1;
				}
				else if (vCoords[0].z - vRadii[0] < PBC.currentDomain.coordBeg.z)
					vOverXYZ.z = 1;
			}

		}
	}
	std::vector<std::vector<CVector3>> vAllAgglomerates;
	vAllAgglomerates.push_back(vCoords);
	if (PBC.bEnabled)
	{
		if (vOverXYZ.x)                         vAllAgglomerates.push_back(DuplicateAgglomerate(vCoords, CVector3(1, 0, 0), PBC));
		if (vOverXYZ.x&&vOverXYZ.y)             vAllAgglomerates.push_back(DuplicateAgglomerate(vCoords, CVector3(1, 1, 0), PBC));
		if (vOverXYZ.x&&vOverXYZ.y&&vOverXYZ.z) vAllAgglomerates.push_back(DuplicateAgglomerate(vCoords, CVector3(1, 1, 1), PBC));
		if (vOverXYZ.y)                         vAllAgglomerates.push_back(DuplicateAgglomerate(vCoords, CVector3(0, 1, 0), PBC));
		if (vOverXYZ.y&&vOverXYZ.z)             vAllAgglomerates.push_back(DuplicateAgglomerate(vCoords, CVector3(0, 1, 1), PBC));
		if (vOverXYZ.z)                         vAllAgglomerates.push_back(DuplicateAgglomerate(vCoords, CVector3(0, 0, 1), PBC));
	}

	for (auto& coords : vAllAgglomerates)
	{
		for (size_t j = 0; j< coords.size(); ++j)
			MakeStandardSphere(coords[j] * 1000, vRadii[j] * 1000, nLatitude, nLongitude);
		*m_pout << "----------End of agglomerate----------" << std::endl;
	}
}

std::vector<CVector3> CAgglomeratesAnalyzer::DuplicateAgglomerate(const std::vector<CVector3>& _vCoords, const CVector3& _vShiftDirection, const SPBC& _PBC) const
{
	std::vector<CVector3> vResult(_vCoords.size());
	const CVector3 vShift = EntryWiseProduct(_PBC.boundaryShift,_vShiftDirection);
	for (size_t i = 0; i < _vCoords.size(); i++)
		vResult[i] = _vCoords[i] + vShift;
	return vResult;
}

void CAgglomeratesAnalyzer::MakeStandardSphere(const CVector3& _point, double _dRadius, size_t _nLatitude, size_t _nLongitude)
{
	const double DEGS_TO_RAD = PI / 180.0;
	size_t numVertices = 0;
	const size_t nPitch = _nLongitude + 1;
	const double pitchInc = 180. / static_cast<double>(nPitch) * DEGS_TO_RAD;
	const double rotInc = 360. / static_cast<double>(_nLatitude) * DEGS_TO_RAD;

	// PRINT VERTICES
	*m_pout << std::fixed << std::setprecision(15) << _point.x << ' ' << _point.y + _dRadius << ' ' << _point.z << std::endl; // Top vertex
	*m_pout << std::fixed << std::setprecision(15) << _point.x << ' ' << _point.y - _dRadius << ' ' << _point.z << std::endl;	// Bottom vertex
	numVertices += 2;
	for (size_t i = 1; i < nPitch; ++i)	//  Generate all "intermediate vertices":
	{
		const double out = std::fabs(_dRadius * sin(i * pitchInc));
		const double y = _dRadius * std::cos(i * pitchInc);
		for (size_t j = 0; j < _nLatitude; ++j)
		{
			const double x = out * cos(j * rotInc);
			const double z = out * sin(j * rotInc);
			*m_pout << std::fixed << std::setprecision(15) << x + _point.x << ' ' << y + _point.y << ' ' << z + _point.z << std::endl;
			numVertices++;
		}
	}
}
