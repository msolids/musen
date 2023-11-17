/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "SystemStructure.h"
#include "Constraints.h"

class CResultsAnalyzer
{
public:
	enum class EPropertyType : unsigned
	{
		Coordinate         = 0,
		Diameter           = 1,
		Distance           = 2,
		Duration           = 3,
		ForceNormal        = 4,
		ForceTangential    = 5,
		ForceTotal         = 6,
		Length             = 7,
		Number             = 8,
		TotalVolume        = 9,
		VelocityNormal     = 10,
		VelocityTangential = 11,
		VelocityTotal      = 12,
		VelocityRotational = 13,
		ResidenceTime      = 14,
		BondForce          = 15,
		Energy             = 16,
		KineticEnergy      = 17,
		PotentialEnergy    = 18,
		CoordinationNumber = 19,
		Deformation        = 20,
		Strain             = 21,
		ExportToFile       = 22,
		Orientation        = 23,
		MaxOverlap         = 24,
		Stress			   = 25,
		Temperature		   = 26,
		PrincipalStress	   = 27,
		PartNumber,
		BondNumber
	};
	typedef std::vector<EPropertyType> VPropertyType;
	enum class EDistanceType
	{
		ToPoint = 0,
		ToLine = 1
	};
	enum class EVectorComponent
	{
		Total = 0,
		X = 1,
		Y = 2,
		Z = 3
	};
	enum class ERelationType
	{
		Existing = 0,
		Appeared = 1
	};
	enum class ECollisionType
	{
		ParticleParticle = 0,
		ParticleWall = 1
	};
	enum class EResultType
	{
		Distribution = 0,
		Average = 1,
		Maximum = 2,
		Minimum = 3
	};
	enum class EStatus
	{
		Idle = 0,
		Runned = 1,
		ShouldBeStopped = 2
	};

public:
	EDistanceType m_nDistance;
	EVectorComponent m_nComponent;
	ERelationType m_nRelation;
	ECollisionType m_nCollisionType;
	EResultType m_nResultsType;
	size_t m_nGeometryIndex;
	CVector3 m_Point1, m_Point2;
	double m_dTimeMin, m_dTimeMax, m_dTimeStep; // used for GUI
	bool m_bOnlySavedTP; // used for GUI
	std::vector<double> m_vTimePoints; // list of time points, which should be considered
	double m_dPropMin;
	double m_dPropMax;
	unsigned m_nPropSteps;	// number of property intervals
	std::string m_sOutputFileName;

protected:
	CSystemStructure *m_pSystemStructure;
	CConstraints m_Constraints;
	std::shared_ptr<std::ostream> m_pout;
	double m_dPropStep;
	std::vector<std::vector<size_t>> m_vDistrResults;
	std::vector<std::vector<double>> m_vValueResults;
	std::vector<double> m_vConcResults;
	bool m_bConcParam;
	unsigned m_nProgress; // defines exporting progress for GUI. 0 - 100
	std::string m_sStatusDescr; // text description of current process, for GUI
	EStatus m_CurrentStatus;
	bool m_bCustomFileWriter; // if set to true, WriteResultsToFile() will not be called in CResultsAnalyzer::StartExport()

private:
	VPropertyType m_vProperties;
	bool m_bError;	// error description should be stored in m_sStatusDescr

public:
	CResultsAnalyzer();
	virtual ~CResultsAnalyzer() = 0;
	void UpdateSettings();

	virtual void SetSystemStructure(CSystemStructure* _pSystemStructure);
	const CSystemStructure* GetSystemStructure() const;

	CConstraints* GetConstraintsPtr();

	EPropertyType GetProperty();
	VPropertyType GetProperties() const;

	void SetPropertyType(VPropertyType _property);
	void SetPropertyType(EPropertyType _property);

	void SetOutputStream(std::shared_ptr<std::ostream> _out);

	void SetDistanceType(EDistanceType _distance);
	void SetVectorComponent(EVectorComponent _component);
	void SetRelatonType(ERelationType _relation);
	void SetCollisionType(ECollisionType _type);
	void SetResultsType(EResultType _type);
	void SetGeometryIndex(unsigned _index);
	void SetPoint1(const CVector3& _point);
	void SetPoint2(const CVector3& _point);
	void SetTime(double _timeMin, double _timeMax, double _timeStep, bool _bOnlySaved = false);
	void SetProperty(double _propMin, double _propMax, unsigned _propSteps);
	unsigned GetExportProgress() const;
	std::string GetStatusDescription() const;
	CResultsAnalyzer::EStatus GetCurrentStatus() const;
	bool IsError() const;
	void SetCurrentStatus(CResultsAnalyzer::EStatus _Status);

	void StartExport();
	void StartExport(const std::string& _sFileName);
	virtual bool Export() = 0;

	// Opens file and returns 'true' if it is ready to be written in.
	bool PrepareOutFile(const std::string& _sFileName);
protected:
	// Calculates lengths of time and property steps.
	void CalculateSteps();
	// Resets vectors of results.
	void PrepareResultVectors();
	// Calculates index in vector of properties.
	int CalculatePropIndex(double _dValue) const;
	// Check if time points are set correctly.
	bool CheckTimePoints();
	// Request proper material database: all materials must be defined.
	bool CheckMaterialsDatabase();
	// Returns true if the analysis must be terminated.
	bool CheckTerminationFlag();

	void WriteValueToResults(double _dResult, size_t _nTimeIndex);
	void WriteComponentToResults(const CVector3& _vResult, size_t _nTimeIndex);
	void WriteDistrResultsToFile();
	void WriteConcResultsToFile();
	void WriteResultsToFile();
};