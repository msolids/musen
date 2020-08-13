/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "BaseSimulator.h"
#include "ExportAsText.h"
#include "TriState.h"

struct SJob
{
	enum class EComponent : unsigned
	{
		PACKAGE_GENERATOR  = 0,
		RESULTS_ANALYZER   = 1,
		BONDS_GENERATOR    = 2,
		SIMULATOR          = 3,
		SNAPSHOT_GENERATOR = 4,
		EXPORT_TO_TEXT     = 5,
		IMPORT_FROM_TEXT   = 6,
		COMPARE_FILES      = 7,
		AUTO_TESTING       = 8,
	};

	struct SPackageGenerator
	{
		std::string volume{ "" };
		std::string mixture{ "" };
		double porosity{ 0.0 };
		double overlap{ 0.0 };
		size_t iterations{ 0 };
		CVector3 velocity{ std::numeric_limits<double>::infinity() };
		CTriState inside{ CTriState::EState::UNDEFINED };
	};

	struct SBondGenerator
	{
		std::string material{ "" };
		double minDistance{ std::numeric_limits<double>::infinity() };
		double maxDistance{ std::numeric_limits<double>::infinity() };
		double diameter{ 0.0 };
		CTriState overlay{ CTriState::EState::UNDEFINED };
	};

	struct SMDBMaterialProperties
	{
		ETPPropertyTypes propertyKey;
		std::string compoundKey;
		double value{};
	};

	struct SMDBInteractionProperties
	{
		EIntPropertyTypes propertyKey;
		std::string compoundKey1;
		std::string compoundKey2;
		double value{};
	};

	struct SMDBMixtureProperties
	{
		size_t iMixture;			// Index of the mixture.
		size_t iFraction;			// Index of a fraction within the mixture.
		std::string compoundKey;	// Unique key of compound.
		double diameter;			// Diameter of particles.
		double fraction;			// Number fraction of particles.
	};

	struct SModel
	{
		std::string name;
		std::string parameters;
	};

	std::string sourceFileName;
	std::string resultFileName;
	std::string agglomeratesDBFileName;

	std::map<EMusenModelType, SModel> models{
		{EMusenModelType::PP, {}},
		{EMusenModelType::PW, {}},
		{EMusenModelType::SB, {}},
		{EMusenModelType::LB, {}},
		{EMusenModelType::EF, {}} };

	EComponent component{ EComponent::SIMULATOR };

	ESimulatorType simulatorType{ ESimulatorType::BASE };

	// selective saving
	CTriState selectiveSavingFlag{ CTriState::EState::UNDEFINED };
	SSelectiveSavingFlags selectiveSavingFlags;

	std::vector<std::string> vPostProcessCommands;

	// time
	double dSimulationTimeStep = 0;
	double dSavingTimeStep = 0;
	double dEndSimulationTime = 0;
	double dSnapshotTP = 0;

	CTriState saveCollsionsFlag{ CTriState::EState::UNDEFINED };
	CTriState connectedPPContactFlag{ CTriState::EState::UNDEFINED };	// calculate force between connected particles
	CTriState anisotropyFlag{ CTriState::EState::UNDEFINED };
	CTriState contactRadiusFlag{ CTriState::EState::UNDEFINED };
	CVector3 vExtAccel{ std::numeric_limits<double>::infinity() };
	SVolumeType simulationDomain{ CVector3{ std::numeric_limits<double>::infinity() }, CVector3{ std::numeric_limits<double>::infinity() } };

	// materials
	std::vector<SMDBMaterialProperties> materialProperties;
	std::vector<SMDBInteractionProperties> interactionProperties;
	std::vector<SMDBMixtureProperties> mixtureProperties;

	// verlet list
	CTriState verletAutoFlag{ CTriState::EState::UNDEFINED };
	double verletCoef{ 0 };
	uint32_t iVerletMaxCells{ 0 };

	// variable time step
	CTriState variableTimeStepFlag;
	double maxPartMove{ 0. };
	double stepIncFactor{ 0. };

	// package generator, <index, generator>
	std::map<size_t, SPackageGenerator> packageGenerators;

	// bonds generator, <index, generator>
	std::map<size_t, SBondGenerator> bondGenerators;

	// export as text
	CExportAsText::SObjectTypeFlags txtExportObjects;
	CExportAsText::SSceneInfoFlags txtExportScene;
	CExportAsText::SConstPropsFlags txtExportConst;
	CExportAsText::STDPropsFlags txtExportTD;
	CExportAsText::SGeometriesFlags txtExportGeometries;
	CExportAsText::SMaterialsFlags txtExportMaterials;
	int txtPrecision{ 6 };

	// additional stop criteria
	std::vector<CBaseSimulator::EStopCriteria> stopCriteria;
	CBaseSimulator::SStopValues stopValues;
};
