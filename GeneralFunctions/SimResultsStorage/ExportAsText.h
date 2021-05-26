/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "Constraints.h"
#include "SystemStructure.h"
#include "ExportAsTextMacro.h"

// Defines flags and initializes them with true.
#define DEFINE_FLAGS(...) FOR_EACH_VA_DEF(bool,true,__VA_ARGS__)
// Defines flags and a constructor, which puts pointers to them to a std::vector<bool*> flags of SBaseFlags.
#define CREATE_FLAGS(name, ...) DEFINE_FLAGS(__VA_ARGS__) name() { FOR_EACH_VA_REF(flags.push_back,__VA_ARGS__) }

class CPackageGenerator;
class CBondsGenerator;

class CExportAsText
{
	struct SBaseFlags
	{
		std::vector<bool*> flags;
		SBaseFlags& operator=(const SBaseFlags& _other) { return *this;	} // to skip copy of pointers
		bool IsAllOn() const { for (auto f : flags) if (!*f) return false; return true; }
		bool IsAllOff() const { for (auto f : flags) if (*f) return false; return true; }
		void SetAll(bool _value) { for (auto& f : flags) *f = _value; }
		void SetFlags(const std::initializer_list<bool>& _flags)
		{
			assert(_flags.size() == flags.size());
			for (size_t i = 0; i < flags.size(); ++i)
				*flags[i] = *(_flags.begin() + i);
		}
	};
public:
	struct SObjectTypeFlags : SBaseFlags
	{
		CREATE_FLAGS(SObjectTypeFlags, particles, solidBonds, liquidBonds, triangularWalls)
	};
	struct SSceneInfoFlags : SBaseFlags
	{
		CREATE_FLAGS(SSceneInfoFlags, domain, pbc, anisotropy, contactRadius)
	};
	struct SGeometriesFlags : SBaseFlags
	{
		CREATE_FLAGS(SGeometriesFlags, baseInfo, tdProperties, wallsList, analysisVolumes)
	};
	struct SMaterialsFlags : SBaseFlags
	{
		CREATE_FLAGS(SMaterialsFlags, compounds, interactions, mixtures)
	};
	struct SConstPropsFlags : SBaseFlags
	{
		CREATE_FLAGS(SConstPropsFlags, id, type, geometry, material, activityInterval)
	};
	struct STDPropsFlags : SBaseFlags
	{
		CREATE_FLAGS(STDPropsFlags, coordinate, velocity, angularVelocity, totalForce, force, quaternion, stressTensor, totalTorque, tangOverlap, temperature, principalStress)
	};
	struct SGeneratorsFlags : SBaseFlags
	{
		CREATE_FLAGS(SGeneratorsFlags, packageGenerator, bondsGenerator)
	};

private:
	struct SConstantData // Constant data structure for one object.
	{
		size_t objID{};
		unsigned objType{};
		std::string objGeom;
		std::string cmpKey;
		double activeStart{};
		double activeEnd{};
	};

	struct STDData // Time-dependent data structure for one time point.
	{
		double time{};
		CVector3 coord{};
		CVector3 velo{};
		CVector3 angleVel{};
		double totForce{};
		CVector3 force{};
		CQuaternion quaternion{};
		CMatrix3 stressTensor{};
		double temperature{};
	};

	SObjectTypeFlags m_objectTypeFlags;  // Object type flags.
	SSceneInfoFlags m_sceneInfoFlags;    // Scene info flags.
	SConstPropsFlags m_constPropsFlags;  // Constant properties flags.
	STDPropsFlags m_tdPropsFlags;        // TD properties flags.
	SGeometriesFlags m_geometriesFlags;  // Geometry flags.
	SMaterialsFlags m_materialsFlags;    // Materials flags.
	SGeneratorsFlags m_generatorsFlags;  // Generators flags.
	std::vector<SBaseFlags*> m_allFlags; // Vector of pointers to all flags.

	CSystemStructure* m_pSystemStructure;
	CConstraints*	  m_pConstraints;
	CPackageGenerator* m_packageGenerator{ nullptr };
	CBondsGenerator*   m_bondsGenerator{ nullptr };

	std::string m_fileName;			// Name of resulting text file.
	std::vector<double> m_timePoints;  // List of time points, which should be considered.
	std::streamsize m_precision;        // Decimal precision to be used to format floating-point values of time-dependent properties.
	double m_dProgressPercent;			// Percent of progress (is used for progress bar).
	std::string m_sProgressMessage;		// Progress description.
	std::string m_sErrorMessage;		// Error description.
	ERunningStatus m_nCurrentStatus;    // Current status of the exporter: IDLE, RUNNING, etc.

public:
	CExportAsText();

	// Sets SystemStructure and Constraints pointers.
	void SetPointers(CSystemStructure* _pSystemStructure, CConstraints* _pConstaints, CPackageGenerator* _pakageGenerator, CBondsGenerator* _bondsGenerator);

	// Set all flags.
	void SetFlags(const SObjectTypeFlags& _objectTypes, const SSceneInfoFlags& _sceneInfo, const SConstPropsFlags& _constProps, const STDPropsFlags& _tdProps, const SGeometriesFlags& _geometries, const SMaterialsFlags& _materials, const SGeneratorsFlags& _generators);

	// Sets name of text file for data export.
	void SetFileName(const std::string& _sFileName);
	// Sets vector of all time points which should be considered.
	void SetTimePoints(const std::vector<double>& _vTimePoints);
	// Sets decimal precision to be used to format floating-point values of time-dependent properties.
	void SetPrecision(int _nPrecision);

	// Sets current status of data export (IDLE, RUNNING, etc.).
	void SetCurrentStatus(const ERunningStatus& _nNewStatus);
	// Returns current status of data export (IDLE, RUNNING, etc.).
	ERunningStatus GetCurrentStatus() const;

	// Returns decimal precision to be used to format floating-point values of time-dependent properties.
	int GetPrecision() const;
	// Returns progress percent.
	double GetProgressPercent() const;
	// Returns string with progress description.
	const std::string& GetProgressMessage() const;
	// Returns string with error description.
	const std::string& GetErrorMessage() const;
	// Exports data into text file.
	void Export();

private:
	// Returns true if all data should be saved.
	bool IsSaveAll() const;
	// Returns identifiers of all real objects.
	std::set<size_t> GetObjectsIDs() const;
	// Returns constant properties for selected objects. _setObjectsIDs - identifiers of real objects which have to be considered.
	std::vector<SConstantData> GetConstantProperties(const std::set<size_t>& _setObjectsIDs) const;
	// Returns final set of identifiers for selected objects. _setObjectsIDs - identifiers of real objects which have to be considered.
	std::set<size_t> CheckConstraints(const std::set<size_t>& _setObjectsIDs) const;
	// Saves time-dependent data to temporary binary file. Returns true if successful.
	bool SaveTDDataToBinFile(const std::string& _sFileName, const std::set<size_t>& _setObjectsIDs);
};
