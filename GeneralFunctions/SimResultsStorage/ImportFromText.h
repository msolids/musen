/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "SystemStructure.h"

class CPackageGenerator;
class CBondsGenerator;

class CImportFromText
{
public:
	enum class EImportFileResult				// Describes result of import.
	{
		OK,										// Successful import.
		ErrorOpening,							// Input txt file cannot be opened.
		ErrorNoID,								// Object ID is not specified in file.
		ErrorNoType,							// Object type is not specified in file.
		ErrorNoGeometry,						// Object geometry is not specified in file.
	};
	struct SImportFileInfo						// Contains extensive information about import process.
	{
		EImportFileResult importResult;		    // Result of import.
		int nErrorLineNumber = 0;				// Number of line in file where error caused.
		bool bMaterial = false;					// True if objects materials are specified in file.
		bool bActivityInterval = false;			// True if activity intervals are specified in file.
		bool bParticleCoordinates = false;		// True if particle coordinates are specified in file.
	};
	struct SRequiredProps                       // Flags for properties that must be defined for each physical object for correct import.
	{
		bool bObjectID = false;					// Whether ID is defined for object.
		bool bObjectType = false;				// Whether type is defined for object.
		bool bObjectGeometry = false;			// Whether geometric parameters are defined for object.
	};

private:
	CSystemStructure* m_pSystemStructure;		// Pointer to system structure.
	CPackageGenerator* m_packageGenerator{ nullptr };
	CBondsGenerator*   m_bondsGenerator{ nullptr };
	struct STDObjectInfo						// Time-dependent properties for one object.
	{
		std::vector<double> vTime;
		std::vector<CVector3> vVelocity;
		std::vector<CVector3> vCoordinates;
		std::vector<CVector3> vAngleVelocity;
		std::vector<CVector3> vForce;
		std::vector<double> vTotalForce;
		std::vector<CQuaternion> vQuaternion;
		std::vector<CMatrix3> vStressTensor;
		std::vector<double> vTemperature;
	};
	std::set<double> m_allTimePoints;				// Set of all time points specified in file.
	std::vector<STDObjectInfo*> m_vObjects;		// Local storage of all TD properties of all objects.

public:
	CImportFromText(CSystemStructure* _pSystemStructure, CPackageGenerator* _pakageGenerator, CBondsGenerator* _bondsGenerator);

	// Imports data from text file. Returns struct with extensive information about import process.
	SImportFileInfo Import(const std::string& _fileName);

private:
	// Checks existing of required constant properties, depending on current _identifier.
	static EImportFileResult CheckConstantProperties(ETXTCommands _identifier, const SRequiredProps& _props);
	// Sets all time-dependent data from local storage m_vObjects into system structure.
	void SetAllTDdataIntoSystemStructure();
};
