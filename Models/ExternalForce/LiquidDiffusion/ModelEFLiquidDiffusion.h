/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "AbstractDEMModel.h"
#include <random>
#include <fstream>
#include <curand_kernel.h>

enum class EAnnealing
{
	NONE = 0,
	LIN = 1
};

class CModelEFLiquidDiffusion : public CExternalForceModel
{
	// universal
	bool m_saveFlag = false;
	bool m_checkSeeds = false; // Check seeds for duplicates
	double m_lastSavingTime = -1;
	double m_savingStep = 10E-9;
	std::ofstream m_fileKineticEnergy;
	std::ofstream m_fileTemperature;

	// simulated annealing variables
	EAnnealing m_annealingType = EAnnealing::NONE;
	double m_curTemp{};
	double m_annealingEqTemp{};
	double m_annealingMaxTemp{};
	double m_annealingCooldownTime{};
	double m_annealingPeriod{};
	double m_annealingFinishedTime{};

	// CPU
	uint64_t* m_pSeedsCPU{};
	size_t m_nStatesCPU = 0; // number of generated seeds
	std::mt19937_64* m_pGeneratorsCPU{};
	double* m_pFMagnitudeCPU{};
	double* m_pMMagnitudeCPU{};
	double* m_pDragLinCPU{};
	double* m_pDragRotCPU{};
	double m_temperatureCoefCPU = 1; // coefficient to adapt temperature during the run
	double m_tauMinLin = std::numeric_limits<double>::max();
	double m_tauMinRot = std::numeric_limits<double>::max();

	unsigned m_maxPartIndex = 0;
	unsigned* m_pParticleIndexMapCPU{}; // needed to map index stored in particle property (shifted because of bonds etc) to index used in this model for RNG. Not needed on GPU as all particles are stored together behind each other with consecutive indexes

	// GPU
	curandState* pCurandStatesGPU{};
	unsigned* m_pSeedsGPU{};
	unsigned m_nStatesGPU = 0; // number of generated states (incorporate the seeds)

	double* m_pFMagnitudeGPU{};
	double* m_pMMagnitudeGPU{};
	double* m_pDragLinGPU{};
	double* m_pDragRotGPU{};

public:
	CModelEFLiquidDiffusion();
	void DetectAnnealing();

	//////////////////////////////////////////////////////////////////////////
	/// CPU Implementation.

	void PrecalculateEF(double _time, double _timeStep, SParticleStruct* _particles) override;
	void InitializeDiffusionModelCPU(double _timeStep);
	void UpdateTemperatureCPU(double newTemperature);
	void CreateIndexMapCPU();
	void CalculateDiffusionModelCoefficientsCPU(double _timeStep);
	void InitializeSeedsAndRandomGeneratorsCPU(size_t _desNumStates);
	void CalculateEF(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles) const override;
	void CreateUniqueSeeds64(uint64_t* _pSeeds64, unsigned _desNumStates);

	//////////////////////////////////////////////////////////////////////////
	/// GPU Implementation.

	void SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc) override;
	void InitializeDiffusionModelGPU(double _timeStep, SGPUParticles& _particles);
	void UpdateTemperatureGPU(double newTemperature);
	void CalculateDiffusionModelCoefficientsGPU(double _timeStep, SGPUParticles& _particles);
	void InitializeSeedsAndRandomGeneratorsGPU(size_t _desNumStates);
	void CalculateEFGPU(double _time, double _timeStep, SGPUParticles& _particles) override;
	void GenerateNewCurandStates(size_t _nStates, const unsigned* _seeds, curandState* _states);
	void CreateUniqueSeeds32(uint32_t* pSeeds32, unsigned _desNumStates);
};

