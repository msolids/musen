/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "CollisionsAnalyzer.h"

CCollisionsAnalyzer::CCollisionsAnalyzer()
{
	SetPropertyType(CResultsAnalyzer::EPropertyType::Coordinate);
	SetRelatonType(CResultsAnalyzer::ERelationType::Appeared);
	SetCollisionType(CResultsAnalyzer::ECollisionType::ParticleParticle);
	m_bAllVolumes = true;
	m_bAllMaterials = true;
	m_bAllDiameters = true;
	m_bAllGeometries = true;
}

CCollisionsAnalyzer::~CCollisionsAnalyzer()
{
	Finalize();
}

void CCollisionsAnalyzer::SetSystemStructure(CSystemStructure* _pSystemStructure)
{
	m_pSystemStructure = _pSystemStructure;
	m_Storage.SetFileName(m_pSystemStructure->GetFileName() + COLL_FILE_EXT);
}

void CCollisionsAnalyzer::ResetAndClear()
{
	m_Storage.SetFileName(m_pSystemStructure->GetFileName() + COLL_FILE_EXT);
	m_Storage.Reset();
}

void CCollisionsAnalyzer::Finalize()
{
	if (m_Storage.IsValid())
		m_Storage.FlushAndCloseFile();
}

void CCollisionsAnalyzer::AddCollisions(const std::vector<SCollision*>& _vCollisionsPP, const std::vector<SCollision*>& _vCollisionsPW)
{
	if (!m_Storage.IsValid()) return;	// nowhere to save
	if (_vCollisionsPP.empty() && _vCollisionsPW.empty()) return;	// nothing to save

	ProtoBlockOfCollisionsInfo descr;
	ProtoBlockOfCollisions data;

	std::vector<SCollision*> vAllCollisions;
	vAllCollisions.reserve(_vCollisionsPP.size() + _vCollisionsPW.size());
	vAllCollisions.insert(vAllCollisions.end(), _vCollisionsPP.begin(), _vCollisionsPP.end());
	vAllCollisions.insert(vAllCollisions.end(), _vCollisionsPW.begin(), _vCollisionsPW.end());

	double dMinTime = vAllCollisions.front()->pSave->dTimeStart;	// minimum start time of collision in current block
	double dMaxTime = vAllCollisions.front()->pSave->dTimeStart;	// maximum end time of collision in current block
	for (size_t i = 0; i < vAllCollisions.size(); ++i)
	{
		if (vAllCollisions[i]->pSave->dTimeStart < dMinTime)
			dMinTime = vAllCollisions[i]->pSave->dTimeStart;
		if (vAllCollisions[i]->pSave->dTimeEnd > dMaxTime)
			dMaxTime = vAllCollisions[i]->pSave->dTimeEnd;

		ProtoCollision *pCurrColl = data.add_collisions();

		pCurrColl->set_src_id(vAllCollisions[i]->nSrcID);
		pCurrColl->set_dst_id(vAllCollisions[i]->nDstID);
		pCurrColl->set_time_start(vAllCollisions[i]->pSave->dTimeStart);
		pCurrColl->set_time_end(vAllCollisions[i]->pSave->dTimeEnd);
		VectorToProtoVector(pCurrColl->mutable_max_total_force(), vAllCollisions[i]->pSave->vMaxTotalForce);
		VectorToProtoVector(pCurrColl->mutable_max_norm_force(), vAllCollisions[i]->pSave->vMaxNormForce);
		VectorToProtoVector(pCurrColl->mutable_max_tang_force(), vAllCollisions[i]->pSave->vMaxTangForce);
		VectorToProtoVector(pCurrColl->mutable_norm_velocity(), vAllCollisions[i]->pSave->vNormVelocity);
		VectorToProtoVector(pCurrColl->mutable_tang_velocity(), vAllCollisions[i]->pSave->vTangVelocity);
		VectorToProtoVector(pCurrColl->mutable_contact_point(), vAllCollisions[i]->pSave->vContactPoint);
	}

	descr.set_time_min(dMinTime);
	descr.set_time_max(dMaxTime);

	// save to file
	m_Storage.SaveBlock(descr, data);
}

bool CCollisionsAnalyzer::Export()
{
	// request proper material database if needed
	if (GetProperty() == CResultsAnalyzer::EPropertyType::Energy)
		if (!CheckMaterialsDatabase())
			return false;

	// Initialize collisions storage
	m_Storage.LoadFromFile(m_pSystemStructure->GetFileName() + COLL_FILE_EXT);
	if (!m_Storage.IsValid())
	{
		m_sStatusDescr = "Unable to load input file. ";
		return false;
	}

	// Get constraints data
	m_bAllVolumes = m_Constraints.IsAllVolumeSelected();
	m_bAllMaterials = m_Constraints.IsAllMaterials2Selected();
	m_bAllGeometries = m_Constraints.IsAllGeometriesSelected();
	m_bAllDiameters = m_Constraints.IsAllDiametersSelected();

	if ((m_bConcParam) || (m_nResultsType != CResultsAnalyzer::EResultType::Distribution))
		m_nPropSteps = 1;

	for (size_t iTime = 0; iTime < m_vTimePoints.size(); ++iTime)	// for all time points from interval
	{
		if (CheckTerminationFlag()) return false;

		double dTime;
		std::vector<ProtoCollision*> vCollisons;

		// take into account collision, which APPEAR on time interval. each collision will be considered only once on time interval of appearance
		if (m_nRelation == CResultsAnalyzer::ERelationType::Appeared)
			if (iTime == m_vTimePoints.size() - 1) break; // one step less as if m_nRelation == Appeared

		dTime = m_vTimePoints[iTime];

		// status description
		m_sMsgTime = "Time = " + std::to_string(dTime) + " [s]. ";

		// take into account collision, which EXIST on time interval. one collision will be considered on each analyzed time point, if it exists
		if (m_nRelation == CResultsAnalyzer::ERelationType::Existing)
			GetCollisionsForTime(vCollisons, dTime);
		else // take into account collision, which APPEAR on time interval. each collision will be considered only once on time interval of appearance
		{
			GetCollisionsForTime(vCollisons, dTime, m_vTimePoints[iTime + 1]);
			dTime = (dTime + m_vTimePoints[iTime + 1]) / 2; // used only for coordinate distribution
		}

		// status description
		m_sStatusDescr = m_sMsgTime + "Processing collisions (" + std::to_string(vCollisons.size()) + ")";

		bool bNewCPFormat = m_Storage.GetFileVersion() >= 100;	// musen v0.9 and earlier do not store contact point

		for (size_t i = 0; i < vCollisons.size(); ++i)	// for all collisions
		{
			if (CheckTerminationFlag()) return false;

			switch (GetProperty())
			{
			case CResultsAnalyzer::EPropertyType::Coordinate:
				if (bNewCPFormat)	// new version collision files
					WriteComponentToResults(ProtoVectorToVector(vCollisons[i]->contact_point()), iTime);
				else // old version collision files
					WriteComponentToResults(CalculateCollisionCoordinate(*vCollisons[i], dTime), iTime);
				break;
			case CResultsAnalyzer::EPropertyType::Distance:
				if (bNewCPFormat)	// new version collision files
				{
					if (m_nDistance == CResultsAnalyzer::EDistanceType::ToPoint)
						WriteComponentToResults(ProtoVectorToVector(vCollisons[i]->contact_point()) - m_Point1, iTime);
					else
						WriteValueToResults(DistanceFromPointToSegment(ProtoVectorToVector(vCollisons[i]->contact_point()), m_Point1, m_Point2), iTime);
				}
				else // old version collision files
				{
					if (m_nDistance == CResultsAnalyzer::EDistanceType::ToPoint)
						WriteComponentToResults(CalculateCollisionCoordinate(*vCollisons[i], dTime) - m_Point1, iTime);
					else
						WriteValueToResults(DistanceFromPointToSegment(CalculateCollisionCoordinate(*vCollisons[i], dTime), m_Point1, m_Point2), iTime);
				}
				break;
			case CResultsAnalyzer::EPropertyType::Duration:
				if (vCollisons[i]->time_end() == -1) continue;	// not finished collision
				WriteValueToResults(vCollisons[i]->time_end() - vCollisons[i]->time_start(), iTime);
				break;
			case CResultsAnalyzer::EPropertyType::Energy:
				WriteValueToResults(CalculateCollisionEnergy(*vCollisons[i]), iTime);
				break;
			case CResultsAnalyzer::EPropertyType::ForceNormal:
				if (vCollisons[i]->time_end() == -1) continue;	// not finished collision
				WriteComponentToResults(ProtoVectorToVector(vCollisons[i]->max_norm_force()), iTime);
				break;
			case CResultsAnalyzer::EPropertyType::ForceTangential:
				if (vCollisons[i]->time_end() == -1) continue;	// not finished collision
				WriteComponentToResults(ProtoVectorToVector(vCollisons[i]->max_tang_force()), iTime);
				break;
			case CResultsAnalyzer::EPropertyType::ForceTotal:
				if (vCollisons[i]->time_end() == -1) continue;	// not finished collision
				WriteComponentToResults(ProtoVectorToVector(vCollisons[i]->max_total_force()), iTime);
				break;
			case CResultsAnalyzer::EPropertyType::Number:
				m_vConcResults[iTime]++;
				break;
			case CResultsAnalyzer::EPropertyType::VelocityNormal:
				WriteComponentToResults(ProtoVectorToVector(vCollisons[i]->norm_velocity()), iTime);
				break;
			case CResultsAnalyzer::EPropertyType::VelocityTangential:
				WriteComponentToResults(ProtoVectorToVector(vCollisons[i]->tang_velocity()), iTime);
				break;
			case CResultsAnalyzer::EPropertyType::VelocityTotal:
				WriteComponentToResults(ProtoVectorToVector(vCollisons[i]->norm_velocity()) + ProtoVectorToVector(vCollisons[i]->tang_velocity()), iTime);
				break;
			default:
				break;
			}
		}

		for (unsigned i = 0; i < vCollisons.size(); ++i)
			delete vCollisons[i];

		m_nProgress = (unsigned)((iTime + 1.) / (double)m_vTimePoints.size() * 100);
	}

	return true;
}

bool CCollisionsAnalyzer::IsParticleInVolume(const std::set<size_t>& _vParticles, size_t _nIndex1, size_t _nIndex2) const
{
	if (_vParticles.find(_nIndex1) != _vParticles.end())
		return true;
	else if (_vParticles.find(_nIndex2) != _vParticles.end())
		return true;
	return false;
}

std::vector<ProtoCollision*> CCollisionsAnalyzer::FilterCollisionsByVolume(const std::vector<ProtoCollision*>& _vCollisions) const
{
	std::set<size_t> vIDs;
	std::vector<CVector3> vCoords(_vCollisions.size());
	for (size_t i = 0; i < _vCollisions.size(); ++i)
	{
		vIDs.insert(i);
		vCoords[i] = ProtoVectorToVector(_vCollisions.at(i)->contact_point());
	}

	// filter by coordinates
	std::set<size_t> vFilteredIDs = m_Constraints.ApplyVolumeFilter(0, vCoords);

	// get IDs for unused
	std::set<size_t> vNotPassedIDs;
	std::set_difference(vIDs.begin(), vIDs.end(), vFilteredIDs.begin(), vFilteredIDs.end(), std::inserter(vNotPassedIDs, vNotPassedIDs.begin()));

	// get passed
	std::vector<ProtoCollision*> vFiltered(vFilteredIDs.size());
	size_t i = 0;
	for (std::set<size_t>::iterator it = vFilteredIDs.begin(); it != vFilteredIDs.end(); ++it)
	{
		vFiltered.at(i) = _vCollisions.at(*it);
		++i;
	}

	// delete unused
	for (std::set<size_t>::iterator it = vNotPassedIDs.begin(); it != vNotPassedIDs.end(); ++it)
		delete _vCollisions.at(*it);

	// return
	return vFiltered;
}

void CCollisionsAnalyzer::GetCollisionsForTime(std::vector<ProtoCollision*>& _vCollisions, double _dT1, double _dT2 /*= -1*/)
{
	_vCollisions.clear();
	double dTStart = _dT1;
	double dTEnd = _dT2 == -1 ? _dT1 : _dT2;
	double dTime = _dT2 == -1 ? _dT1 : ((_dT1 + _dT2) / 2);

	// status description
	m_sStatusDescr = m_sMsgTime + "Applying volume constraints. ";

	bool bNewCPFormat = m_Storage.GetFileVersion() >= 100;	// musen v0.9 and earlier do not store contact point

	std::set<size_t> vParticles;
	if (!m_bAllVolumes)	// get list of particles in constraint volumes for time point
		if (!bNewCPFormat)
			vParticles = m_Constraints.ApplyVolumeFilter(dTime, SPHERE);

	for (unsigned iBlock = 0; iBlock < m_Storage.GetTotalBlocksNumber(); ++iBlock)	// go through all blocks of data
	{
		if (CheckTerminationFlag()) return;

		// status description
		m_sStatusDescr = m_sMsgTime + "Reading data from file. ";

		CCollisionsStorage::SRuntimeBlockInfo *pDescr = m_Storage.GetDesriptor(iBlock);	// get run-time descriptor
		if ((pDescr->dTimeEnd < dTStart) || (pDescr->dTimeStart > dTEnd))	// these collisions does not exist at desired time point or time interval
			continue;

		ProtoBlockOfCollisions *pData = m_Storage.GetData(iBlock);	// read real data from file

		// status description
		m_sStatusDescr = m_sMsgTime + "Filtering collisions (" + std::to_string(pData->collisions_size()) + ")";

		for (int iColl = 0; iColl < pData->collisions_size(); ++iColl)	// for all collisions in block
		{
			if (CheckTerminationFlag()) return;

			ProtoCollision col = pData->collisions(iColl);	// get collision
			double dColStart = col.time_start();								// start of collision
			double dColEnd = _dT2 == -1 ? col.time_end() : col.time_start();	// end of collision
			if ((dColEnd < dTStart) || (dColStart > dTEnd))	// current collision does not exist on time interval or at time point
				continue;

			// get objects' IDs
			size_t nID1 = col.src_id();	// particle or wall
			size_t nID2 = col.dst_id();	// particle

			// for particle-particle collisions
			if(m_nCollisionType == CResultsAnalyzer::ECollisionType::ParticleParticle)
			{
				// check collision type
				if (m_pSystemStructure->GetObjectByIndex(nID1)->GetObjectType() != SPHERE) continue;

				// check materials constraints
				if (!m_bAllMaterials && !m_Constraints.CheckMaterial(nID1, nID2)) continue;

				// check diameter constraints
				if (!m_bAllDiameters && !m_Constraints.CheckDiameter(nID1, nID2)) continue;

				if (!bNewCPFormat) // check volume constraints
					if (!m_bAllVolumes && !IsParticleInVolume(vParticles, nID1, nID2)) continue;
			}
			else // for particle-wall collisions
			{
				// check collision type
				if (m_pSystemStructure->GetObjectByIndex(nID1)->GetObjectType() != TRIANGULAR_WALL) continue;
				if (!m_bAllGeometries && !m_Constraints.CheckGeometry(nID1)) continue;

				// check materials constraints
				if (!m_bAllMaterials && !m_Constraints.CheckMaterial(nID2)) continue;

				// check diameter constraints
				if (!m_bAllDiameters && !m_Constraints.CheckDiameter(nID2)) continue;

				if (!bNewCPFormat) // check volume constraints
					if (!m_bAllVolumes && !IsParticleInVolume(vParticles, nID1, nID2)) continue;
			}

			// add collision to results
			_vCollisions.push_back(new ProtoCollision(pData->collisions(iColl)));
		}

		delete pData;
	}

	if (!m_bAllVolumes)	// get list of contacts in constraint volumes for time point
		if (bNewCPFormat)
			_vCollisions = FilterCollisionsByVolume(_vCollisions);
}

CVector3 CCollisionsAnalyzer::CalculateCollisionCoordinate(ProtoCollision& _collision, double _dTime)
{
	CPhysicalObject *pObj2 = m_pSystemStructure->GetObjectByIndex(_collision.dst_id());
	CVector3 vecDst = pObj2->GetCoordinates(_dTime);
	if (m_nCollisionType == CResultsAnalyzer::ECollisionType::ParticleWall)
		return vecDst;
	else
	{
		CPhysicalObject *pObj1 = m_pSystemStructure->GetObjectByIndex(_collision.src_id());
		CVector3 vecSrc = pObj1->GetCoordinates(_dTime);
		double dRSrc = dynamic_cast<CSphere*>(pObj1)->GetContactRadius();
		double dRDst = dynamic_cast<CSphere*>(pObj2)->GetContactRadius();
		return vecSrc + (vecDst - vecSrc)*dRSrc / (dRSrc + dRDst);
	}
}

double CCollisionsAnalyzer::CalculateCollisionEnergy(ProtoCollision& _collision)
{
	double dM;
	if (m_nCollisionType == CResultsAnalyzer::ECollisionType::ParticleWall)
		dM = m_pSystemStructure->GetObjectByIndex(_collision.dst_id())->GetMass();
	else
	{
		double dMass1 = m_pSystemStructure->GetObjectByIndex(_collision.src_id())->GetMass();
		double dMass2 = m_pSystemStructure->GetObjectByIndex(_collision.dst_id())->GetMass();
		dM = 2 * dMass1*dMass2 / (dMass1 + dMass2);
	}
	double dV = Length(ProtoVectorToVector(_collision.norm_velocity()) + ProtoVectorToVector(_collision.tang_velocity()));
	return 0.5*dM*dV*dV;
}
