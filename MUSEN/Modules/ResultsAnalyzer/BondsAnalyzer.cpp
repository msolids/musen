/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "BondsAnalyzer.h"

CBondsAnalyzer::CBondsAnalyzer()
{
	SetPropertyType(CResultsAnalyzer::EPropertyType::BondForce);
}

bool CBondsAnalyzer::Export()
{
	if ( (m_bConcParam) || (m_nResultsType != CResultsAnalyzer::EResultType::Distribution) )
		m_nPropSteps = 1;

	for (size_t iTime = 0; iTime < m_vDistrResults.size(); ++iTime)
	{
		if (CheckTerminationFlag()) return false;

		double dTime = m_vTimePoints[iTime];
		double dTotalForce, dInitLength, dCurrentLength;

		// status description
		m_sStatusDescr = "Time = " + std::to_string(dTime) + " [s]. Applying constraints. ";

		std::vector<size_t> vBonds = m_Constraints.FilteredBonds(dTime);

		// status description
		m_sStatusDescr = "Time = " + std::to_string(dTime) + " [s]. Processing bonds (" + std::to_string(vBonds.size()) + ")";

		for (size_t j = 0; j < vBonds.size(); ++j)
		{
			if (CheckTerminationFlag()) return false;

			CSolidBond* pBond = dynamic_cast<CSolidBond*>(m_pSystemStructure->GetObjectByIndex( vBonds[ j ] ));
			if (!pBond->IsActive(dTime)) continue;
			CPhysicalObject* pSphere1, *pSphere2;
			pSphere1 = m_pSystemStructure->GetObjectByIndex(pBond->m_nLeftObjectID);
			pSphere2 = m_pSystemStructure->GetObjectByIndex(pBond->m_nRightObjectID);
			switch (GetProperty())
			{
			case CResultsAnalyzer::EPropertyType::BondForce:
				dTotalForce = pBond->GetForce(dTime).Length();
				dInitLength = pBond->GetInitLength();
				dCurrentLength = m_pSystemStructure->GetBond(dTime, vBonds[j]).Length();
				if (dCurrentLength > dInitLength) // pulling state
					dTotalForce *= -1;
				WriteValueToResults(dTotalForce, iTime);
				break;
			case CResultsAnalyzer::EPropertyType::Diameter:
				WriteValueToResults(pBond->GetDiameter(), iTime);
				break;
			case CResultsAnalyzer::EPropertyType::ForceTotal:
				WriteComponentToResults(pBond->GetForce(dTime), iTime);
				break;
			case CResultsAnalyzer::EPropertyType::Length:
				WriteValueToResults(m_pSystemStructure->GetBond(dTime, vBonds[j]).Length(), iTime);
				break;
			case CResultsAnalyzer::EPropertyType::Number:
				m_vConcResults[iTime]++;
				break;
			case CResultsAnalyzer::EPropertyType::VelocityTotal:
				WriteComponentToResults(m_pSystemStructure->GetBondVelocity(dTime, vBonds[j]), iTime);
				break;
			case CResultsAnalyzer::EPropertyType::Deformation:
				WriteValueToResults(pBond->GetInitLength() - m_pSystemStructure->GetBond( dTime, vBonds[ j ] ).Length(), iTime );
				break;
			case CResultsAnalyzer::EPropertyType::Strain:
				if (pBond->GetInitLength() != 0)
					WriteValueToResults(100 * (m_pSystemStructure->GetBond(dTime, vBonds[j]).Length()- pBond->GetInitLength()) / pBond->GetInitLength(), iTime);
				break;
			case CResultsAnalyzer::EPropertyType::Stress:
				WriteValueToResults(-1 * DotProduct(pBond->GetForce(dTime), m_pSystemStructure->GetBond(dTime, vBonds[j]).Normalized())/ pBond->m_dCrossCutSurface, iTime);
				break;
			default:
				break;
			}
		}
		m_nProgress = (unsigned)((iTime + 1.) / (double)m_vDistrResults.size() * 100);
	}
	return true;
}
