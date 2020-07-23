/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "SystemStructure.h"
#include "AgglomeratesDatabase.h"
#include "SimplifiedScene.h"
#include "ThreadPool.h"

class CObjectsGenerator
{

public:
	std::string m_sName;
	std::string m_sVolumeKey;

	CVector3 m_vObjInitVel; // initial object velocity
	double m_dVelMagnitude; // magnitude of random velocity, -1 - not random
	bool m_bRandomVelocity;
	std::string m_sMixtureKey; // key of mixture which will be generated
	std::map<std::string, std::string> m_partMaterials;
	std::map<std::string, std::string> m_bondMaterials;
	bool m_bGenerateMixture; // if false then generate agglomerates

	bool m_bInsideGeometries; // allow to generate objects inside geometries

	std::string m_sAgglomerateKey;
	double m_dAgglomerateScaleFactor;

	double m_dStartGenerationTime;
	double m_dEndGenerationTime;
	double m_dGenerationRate; // [1/s]
	double m_dUpdateStep;

	bool m_bVisible; // visibility in 3D representation
	bool m_bActive;	// activity of this generator

	double m_dLastGenerationTime; // last time point when generation has been done

	CAgglomeratesDatabase* m_pAgglomDB;
	SAgglomerate m_PreLoadedAgglomerate;
	CMaterialsDatabase* m_pMaterialsDB;

public:
	CObjectsGenerator(CAgglomeratesDatabase* _pAgglomD, CMaterialsDatabase* _pMaterialsDB);

	void Initialize();

	// Returns number of objects which have been created.
	size_t Generate(double _dCurrentTime, CSystemStructure* _pSystemStructure, CSimplifiedScene& _Scene);
	// Returns true if some particles must be generated at the specified time point.
	bool IsNeedToBeGenerated(double _dCurrentTime) const;

private:
	// return true if creation was successfully
	void GenerateNewObject( std::vector<CVector3>* _pCoordPart, std::vector<CQuaternion>* _pQuatPart,
		std::vector<double>* _pPartRad, std::vector<double>* _pPartContRad, std::vector<std::string>* _sMaterialsKey, const SVolumeType& _boundBox );

	bool IsOverlapped( const std::vector<CVector3>& _vCoordPart, const std::vector<double>& _vPartContRad,
		const std::vector<unsigned>& _vExistedPartID, const std::vector<unsigned>& _nExistedWallsID, const CSimplifiedScene& _Scene );

	// creates random point in the volume
	void inline CreateRandomPoint( CVector3* _pResult, const SVolumeType& _boundBox );
	void inline CreateRandomAngle( CVector3* _pResult );
};
