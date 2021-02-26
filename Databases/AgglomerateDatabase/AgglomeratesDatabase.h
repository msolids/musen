/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include <fstream>
#include "GeometricFunctions.h"
#include "MUSENStringFunctions.h"


#define AGGLOMERATE 0
#define MULTISPHERE 1

struct SAggloParticle
{
	CVector3 vecCoord;
	double dRadius;
	double dContactRadius;
	std::string sCompoundAlias;
	CQuaternion qQuaternion;
	SAggloParticle() :
		vecCoord({ 0 }), qQuaternion({ CQuaternion(0, 1, 0, 0) }), dRadius(0), dContactRadius(0), sCompoundAlias("")
	{
	}
	SAggloParticle(const CVector3& _vecCoord, const CQuaternion& _qQuaternion, double _dRadius, double _dContactRadius, const std::string& _sAlias = ""):
		vecCoord(_vecCoord), qQuaternion(_qQuaternion), dRadius(_dRadius), dContactRadius(_dContactRadius), sCompoundAlias(_sAlias)
	{
	}
};

struct SAggloBond
{
	double dRadius;
	unsigned nLeftID;
	unsigned nRightID;
	std::string sCompoundAlias;
	SAggloBond() :
		dRadius(0), nLeftID(0), nRightID(0), sCompoundAlias("")	{}
	SAggloBond(double _dRadius, unsigned _nLeftID, unsigned _nRightID, const std::string& _sAlias = "") :
		dRadius(_dRadius), nLeftID(_nLeftID), nRightID(_nRightID), sCompoundAlias(_sAlias)	{}
};

struct SAgglomerate
{
	std::string sKey; //unique key of this agglomerate
	std::string sName;
	double dVolume;
	unsigned nType; // MultiSphere or Agglomerate

	// all these vectors are used only in the case when new agglomerate has been created (in other case file will be read)
	std::vector<SAggloParticle> vParticles;
	std::vector<SAggloBond> vBonds;
};


class CAgglomeratesDatabase
{
public:
	~CAgglomeratesDatabase();

	void AddNewAgglomerate(SAgglomerate& _pAgglomerate);
	unsigned GetAgglomNumber() { return (unsigned)m_vAgglomerates.size(); }

	SAgglomerate* GetAgglomerate(size_t _nIndex);
	SAgglomerate* GetAgglomerate(const std::string& _sKey);
	int GetAgglomerateIndex(const std::string& _sKey);

	std::string GetFileName() { return m_sDatabaseFileName; }
	void SaveToFile(const std::string& _sFileName); // save database into the file
	bool LoadFromFile(const std::string& _sFileName);
	void NewDatabase();

	void DeleteAgglomerate(unsigned _nIndex);
	void UpAgglomerate(unsigned _nIndex);
	void DownAgglomerate(unsigned _nIndex);

private:
	std::vector<SAgglomerate*> m_vAgglomerates;
	std::string m_sDatabaseFileName;

	void DeleteAllAgglomerates();
	double CalculateVolume( SAgglomerate* _pAgglom );
};


