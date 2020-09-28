/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "BaseSimulator.h"
#include "CollisionsCalculator.h"

class CCPUSimulator : public CBaseSimulator
{
	bool m_analyzeCollisions{ false };	// Statistic information about collisions should be saved.

	CCollisionsAnalyzer m_collisionsAnalyzer;
	CCollisionsCalculator m_collisionsCalculator{ m_scene, m_verletList, m_collisionsAnalyzer };

	size_t m_nThreads{ GetThreadsNumber() };	// Number of available CPU threads.
	// They are placed here to avoid memory reallocation.
	std::vector<std::vector<std::vector<SCollision*>>> m_tempCollPPArray{ m_nThreads, std::vector<std::vector<SCollision*>>{ m_nThreads } };
	std::vector<std::vector<std::vector<SCollision*>>> m_tempCollPWArray{ m_nThreads, std::vector<std::vector<SCollision*>>{ m_nThreads } };

public:
	CCPUSimulator() = default;
	CCPUSimulator(const CBaseSimulator& _other);

	void SetSystemStructure(CSystemStructure* _pSystemStructure) override;	// Sets pointer to a system structure.
	void EnableCollisionsAnalysis(bool _bEnable);	// Enables analyzing of collisions.
	bool IsCollisionsAnalysisEnabled() const;		// Returns true if analysis of collisions is currently enabled.

	void Initialize() override;
	void InitializeModels() override;

	void FinalizeSimulation() override;

	void PreCalculationStep() override;
	void UpdateCollisionsStep(double _dTimeStep) override;

	void CalculateForcesStep(double _dTimeStep) override;
	void CalculateForcesPP(double _dTimeStep) override;
	void CalculateForcesPW(double _dTimeStep) override;
	void CalculateForcesSB(double _dTimeStep) override;
	void CalculateForcesLB(double _dTimeStep) override;
	void CalculateForcesEF(double _dTimeStep) override;

	void MoveParticles(bool _bPredictionStep = false) override;
	void MoveWalls(double _dTimeStep) override;

	// Returns all current maximal and average overlap between particles with particle indexes smaller than _nMaxParticleID.
	void GetOverlapsInfo(double& _dMaxOverlap, double& _dAverageOverlap, size_t _nMaxParticleID) override;

	void LoadConfiguration() override; // Uses the same file as system structure to load configuration.
	void SaveConfiguration() override; // Uses the same file as system structure to store configuration.

private:
	void UpdatePBC() override;	// Updates moving PBC.

	void MoveMultispheres(double _dTimeStep, bool _bPredictionStep);

	void PrepareAdditionalSavingData() override;
	void SaveData() override;
	void UpdateVerletLists(double _dTimeStep);
	void CheckParticlesInDomain();	// Check that all particles are remains in simulation domain.

	// Check that all particles have correct coordinates and update coordinates of virtual particles.
	// If some real particles crossed the PBC boundaries, returns true (meaning the need to update verlet lists).
	void MoveParticlesOverPBC();
};

