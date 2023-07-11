/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "AgglomeratesDatabase.h"
#include "DisableWarningHelper.h"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include "AgglomeratesDatabase.pb.h"
PRAGMA_WARNING_POP

CAgglomeratesDatabase::~CAgglomeratesDatabase()
{
	DeleteAllAgglomerates();
}

void CAgglomeratesDatabase::DeleteAllAgglomerates()
{
	for (unsigned i = 0; i < m_vAgglomerates.size(); i++)
		if (m_vAgglomerates[i] != NULL)
			delete m_vAgglomerates[i];
	m_vAgglomerates.clear();
}

void CAgglomeratesDatabase::AddNewAgglomerate(SAgglomerate& _pAgglomerate)
{
	SAgglomerate* pAgglom = new SAgglomerate;
	*pAgglom = _pAgglomerate;
	pAgglom->dVolume = CalculateVolume(pAgglom);
	m_vAgglomerates.push_back(pAgglom);
}

SAgglomerate* CAgglomeratesDatabase::GetAgglomerate(size_t _nIndex)
{
	if (_nIndex >= m_vAgglomerates.size()) return NULL;
	return m_vAgglomerates[_nIndex];
}

SAgglomerate* CAgglomeratesDatabase::GetAgglomerate(const std::string& _sKey)
{
	for (unsigned i = 0; i < m_vAgglomerates.size(); i++)
		if (m_vAgglomerates[i]->sKey.compare(_sKey) == 0)
			return m_vAgglomerates[i];
	return NULL;
}

double CAgglomeratesDatabase::CalculateVolume(SAgglomerate* _pAgglom)
{
	double dVolume = 0;
	for (unsigned i = 0; i < _pAgglom->vParticles.size(); i++) // all particles
		dVolume += PI * pow(2 * _pAgglom->vParticles[i].dRadius, 3) / 6.0;
	for (unsigned i = 0; i < _pAgglom->vBonds.size(); i++) // all particles
	{
		const SAggloBond& bond = _pAgglom->vBonds[i];
		double dBondLength = CalculateBondLength(Length(_pAgglom->vParticles[bond.nLeftID].vecCoord - _pAgglom->vParticles[bond.nRightID].vecCoord),
			_pAgglom->vParticles[bond.nLeftID].dRadius, _pAgglom->vParticles[bond.nRightID].dRadius, bond.dRadius * 2);
		dVolume += PI * dBondLength*pow(bond.dRadius, 2);
	}
	return dVolume;
}

void CAgglomeratesDatabase::SaveToFile(const std::string& _sFileName)
{
	ProtoAgglomeratesDB protoAgglomeratesDB;

	for (size_t i = 0; i < m_vAgglomerates.size(); i++)
	{
		ProtoAgglomerate* protoAggl = protoAgglomeratesDB.add_agglomerate();
		const SAgglomerate* aggl = m_vAgglomerates[i];
		protoAggl->set_key(aggl->sKey);
		protoAggl->set_name(aggl->sName);
		protoAggl->set_volume(aggl->dVolume);
		protoAggl->set_type(aggl->nType);
		for (size_t j = 0; j < aggl->vParticles.size(); j++)
		{
			ProtoParticle* part = protoAggl->add_particles();
			part->set_radius(aggl->vParticles[j].dRadius);
			part->set_contact_radius(aggl->vParticles[j].dContactRadius);

			part->mutable_coord()->set_x(aggl->vParticles[j].vecCoord.x);
			part->mutable_coord()->set_y(aggl->vParticles[j].vecCoord.y);
			part->mutable_coord()->set_z(aggl->vParticles[j].vecCoord.z);

			part->mutable_quaternion()->set_q0(aggl->vParticles[j].qQuaternion.q0);
			part->mutable_quaternion()->set_q1(aggl->vParticles[j].qQuaternion.q1);
			part->mutable_quaternion()->set_q2(aggl->vParticles[j].qQuaternion.q2);
			part->mutable_quaternion()->set_q3(aggl->vParticles[j].qQuaternion.q3);

			part->set_compound(aggl->vParticles[j].sCompoundAlias);
		}
		for (size_t j = 0; j < aggl->vBonds.size(); j++)
		{
			ProtoBond* bond = protoAggl->add_bonds();
			bond->set_leftid(aggl->vBonds[j].nLeftID);
			bond->set_rightid(aggl->vBonds[j].nRightID);
			bond->set_radius(aggl->vBonds[j].dRadius);
			bond->set_compound(aggl->vBonds[j].sCompoundAlias);
		}
	}

	std::fstream outFile(UnicodePath(_sFileName), std::ios::out | std::ios::trunc | std::ios::binary);
	std::string data;
	// TODO: consider to use SerializeToZeroCopyStream() for performance
	protoAgglomeratesDB.SerializeToString(&data);
	outFile << data;
	outFile.close();
	m_sDatabaseFileName = _sFileName;
}

bool CAgglomeratesDatabase::LoadFromFile(const std::string& _sFileName)
{
	std::fstream inputFile(UnicodePath(_sFileName), std::ios::in | std::ios::binary);
	if (!inputFile)
		return false;

	ProtoAgglomeratesDB protoAgglomeratesDB;
	// TODO: consider to use ParseFromZeroCopyStream() for performance
	if (!protoAgglomeratesDB.ParseFromString(std::string(std::istreambuf_iterator<char>(inputFile), std::istreambuf_iterator<char>())))
		return false;

	bool bIsOldFileFormat = false;
	DeleteAllAgglomerates();
	for (int i = 0; i < protoAgglomeratesDB.agglomerate_size(); i++)
	{
		SAgglomerate* pNewAgglom = new SAgglomerate;
		ProtoAgglomerate* protoAggl = protoAgglomeratesDB.mutable_agglomerate(i);
		pNewAgglom->dVolume = protoAggl->volume();
		pNewAgglom->sKey = protoAggl->key();
		pNewAgglom->sName = protoAggl->name();
		pNewAgglom->nType = protoAggl->type();

		for (int j = 0; j < protoAggl->particles_size(); j++)
		{
			ProtoParticle *pParticle = protoAggl->mutable_particles(j);
			CVector3 tempCoordVector(pParticle->coord().x(), pParticle->coord().y(), pParticle->coord().z());

			if (!pParticle->has_quaternion()) // if it is old file without quaternion field
			{
				bIsOldFileFormat = true;
				CVector3 tempOrientVector(pParticle->orient().x(), pParticle->orient().y(), pParticle->orient().z());

				// set quaternion field
				CQuaternion tempQuaternion(tempOrientVector);
				pParticle->mutable_quaternion()->set_q0(tempQuaternion.q0);
				pParticle->mutable_quaternion()->set_q1(tempQuaternion.q1);
				pParticle->mutable_quaternion()->set_q2(tempQuaternion.q2);
				pParticle->mutable_quaternion()->set_q3(tempQuaternion.q3);

				// clear orientation field
				pParticle->mutable_orient()->clear_x();
				pParticle->mutable_orient()->clear_y();
				pParticle->mutable_orient()->clear_z();
			}
			CQuaternion tempQuaternion(pParticle->quaternion().q0(), pParticle->quaternion().q1(), pParticle->quaternion().q2(), pParticle->quaternion().q3());
			pNewAgglom->vParticles.push_back(SAggloParticle(tempCoordVector, tempQuaternion, pParticle->radius(), pParticle->contact_radius(), pParticle->compound()));
		}
		for (int j = 0; j < protoAggl->bonds_size(); j++)
		{
			ProtoBond *pBond = protoAggl->mutable_bonds(j);
			pNewAgglom->vBonds.push_back(SAggloBond(pBond->radius(), pBond->leftid(), pBond->rightid(), pBond->compound()));
		}
		m_vAgglomerates.push_back(pNewAgglom);
	}
	m_sDatabaseFileName = _sFileName;

	return bIsOldFileFormat;
}

void CAgglomeratesDatabase::DeleteAgglomerate(unsigned _nIndex)
{
	if (_nIndex < m_vAgglomerates.size())
		m_vAgglomerates.erase(m_vAgglomerates.begin() + _nIndex);
}

void CAgglomeratesDatabase::NewDatabase()
{
	DeleteAllAgglomerates();
	m_sDatabaseFileName.clear();
}

int CAgglomeratesDatabase::GetAgglomerateIndex(const std::string& _sKey)
{
	for (size_t i = 0; i < m_vAgglomerates.size(); i++)
		if (m_vAgglomerates[i]->sKey.compare(_sKey) == 0)
			return (int)i;
	return -1;
}

void CAgglomeratesDatabase::UpAgglomerate(unsigned _nIndex)
{
	if (_nIndex < m_vAgglomerates.size() && _nIndex != 0)
		std::iter_swap(m_vAgglomerates.begin() + _nIndex, m_vAgglomerates.begin() + _nIndex - 1);
}

void CAgglomeratesDatabase::DownAgglomerate(unsigned _nIndex)
{
	if ((_nIndex < m_vAgglomerates.size()) && (_nIndex != (m_vAgglomerates.size() - 1)))
		std::iter_swap(m_vAgglomerates.begin() + _nIndex, m_vAgglomerates.begin() + _nIndex + 1);
}
