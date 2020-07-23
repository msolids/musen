/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "BaseSimulator.h"
#include "CollisionsCalculator.h"

class CCPUSimulator : public CBaseSimulator
{
private:
	bool m_bAnalyzeCollisions;	// Statistic information about collisions should be saved.

	CCollisionsAnalyzer m_CollisionsAnalyzer;
	CCollisionsCalculator m_CollCalculator;

	size_t m_nThreadsNumber;	// Number of available CPU threads.
	std::vector<std::vector<std::vector<SCollision*>>> m_vTempCollPPArray; // It is placed here to avoid memory reallocation
	std::vector<std::vector<std::vector<SCollision*>>> m_vTempCollPWArray; // It is placed here to avoid memory reallocation

public:
	CCPUSimulator();
	CCPUSimulator(const CBaseSimulator& _simulator);
	~CCPUSimulator() = default;

	void LoadConfiguration() override; // Uses the same file as system structure to load configuration.
	void SaveConfiguration() override; // Uses the same file as system structure to store configuration.

	void SetSystemStructure(CSystemStructure* _pSystemStructure) override;	// Sets pointer to a system structure.

	void EnableCollisionsAnalysis(bool _bEnable);	// Enables analyzing of collisions.
	bool IsCollisionsAnalysisEnabled() const;		// Returns true if analysis of collisions is currently enabled.

	// Returns all current maximal and average overlap between particles with particle indexes smaller than _nMaxParticleID.
	void GetOverlapsInfo(double& _dMaxOverlap, double& _dAverageOverlap, size_t _nMaxParticleID) override;

	void StartSimulation() override;

	void CalculateForcesPP(double _dTimeStep) override;
	void CalculateForcesPW(double _dTimeStep) override;
	void CalculateForcesSB(double _dTimeStep) override;
	void CalculateForcesLB(double _dTimeStep) override;
	void CalculateForcesEF(double _dTimeStep) override;

private:
	void PerformSimulation();
	void LeapFrogStep(bool _bPredictionStep = false);
	void InitializeStep(double _dTimeStep) override;
	void CalculateForces(double _dTimeStep) override;
	void MoveObjects(double _dTimeStep, bool _bPredictionStep = false) override;

	void MoveParticles(bool _bPredictionStep);
	void MoveWalls(double _dTimeStep);
	void MoveMultispheres(double _dTimeStep, bool _bPredictionStep);

	void InitializeModels();
	size_t GenerateNewObjects();		// Generates new objects.
	void PrepareAdditionalSavingData() override;
	void SaveData() override;
	void UpdateVerletLists(double _dTimeStep);
	void CheckParticlesInDomain();	// Check that all particles are remains in simulation domain.

	void InitializeSimulator();

	// Check that all particles have correct coordinates and update coordinates of virtual particles.
	// If some real particles crossed the PBC boundaries, returns true (meaning the need to update verlet lists).
	void MoveParticlesOverPBC();
};

