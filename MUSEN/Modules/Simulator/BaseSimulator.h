/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GenerationManager.h"
#include "ModelManager.h"
#include "VerletList.h"

class CBaseSimulator : public CMusenComponent
{
	friend class CPackageGenerator;	// Allow generator to work with private data of the class.

public:
	std::ostream* p_out = &std::cout;

	enum class EStopCriteria // Types of additional stop criteria.
	{
		NONE         = 0,
		BROKEN_BONDS = 1,
	};
	struct SStopValues // Values for additional stop criteria.
	{
		size_t maxBrokenBonds{ 0 }; // EStopCriteria::BROKEN_BONDS
	};

private:
	ESimulatorType m_simulatorType;	// Type of current simulator.
	CModelManager* m_pModelManager;

	// selective saving
	bool m_bSelectiveSaving;
	SSelectiveSavingFlags m_SSelectiveSavingFlags;

	// additional saving steps
	std::list<std::function<void()>> m_additionalSavingSteps; // List of additional steps called during saving.

	// additional stop criteria
	std::vector<EStopCriteria> m_stopCriteria;	// Active additional simulation stop criteria.
	SStopValues m_stopValues;					// Values for additional simulation stop criteria.

protected:
	struct SAdditionalSavingDataPart
	{
		CMatrix3 sStressTensor;
		void AddStress(const CVector3& _vContVector, const CVector3& _vForce, double _dVolume )
		{
			sStressTensor.values[0][0] += _vContVector.x*_vForce.x / _dVolume;
			sStressTensor.values[0][1] += _vContVector.x*_vForce.y / _dVolume;
			sStressTensor.values[0][2] += _vContVector.x*_vForce.z / _dVolume;

			sStressTensor.values[1][0] += _vContVector.y*_vForce.x / _dVolume;
			sStressTensor.values[1][1] += _vContVector.y*_vForce.y / _dVolume;
			sStressTensor.values[1][2] += _vContVector.y*_vForce.z / _dVolume;

			sStressTensor.values[2][0] += _vContVector.z*_vForce.x / _dVolume;
			sStressTensor.values[2][1] += _vContVector.z*_vForce.y / _dVolume;
			sStressTensor.values[2][2] += _vContVector.z*_vForce.z / _dVolume;
		}
	};

	double m_dCurrentTime;		 // Current time where simulator is situated.
	double m_dEndTime;			 // Last time point which must be simulated.
	double m_initSimulationStep; // Initial simulation time step.
	double m_currSimulationStep; // Current simulation time step; can change during the simulation.
	double m_dSavingStep;		 // Current saving time interval.
	double m_dLastSavingTime;	 // Last time point when data has been saved.
	bool m_bPredictionStep;		 // Determines whether current simulation step is a prediction step.
	CVector3 m_vecExternalAccel; // Value of external acceleration in the system.

	ERunningStatus m_nCurrentStatus;		// Current Status - IDLE, etc.
	size_t m_nInactiveParticlesNumber;		// Number of particles which lost their activity during simulation.
	unsigned m_nBrockenBondsNumber;			// Number of the broken bonds.
	unsigned m_nBrockenLiquidBondsNumber;	// Number of the ruptured liquid bonds.
	unsigned m_nGeneratedObjects;			// Number of generated objects.
	double m_dMaxParticleVelocity;			// Maximal velocity of particle which is used to calculate verlet list.
	double m_dMaxWallVelocity;				// Maximal velocity of walls.
	bool m_bWallsVelocityChanged;			// Need to recalculate maximal wall velocity.
	uint32_t m_nCellsMax;					// Maximum allowed number of cells in each direction of verlet list.
	double m_dVerletDistanceCoeff;			// A coefficient to calculate verlet distance within a verlet list.
	bool m_bAutoAdjustVerletDistance;		// If set to true - the verlet distance will be automatically adjusted during the simulation.
	bool m_bConsiderAnisotropy;				// Consider anisotropy of non-spherical objects during the simulation.
	bool m_bVariableTimeStep;				// Use variable or constant simulation time step.
	double m_partMoveLimit;					// Max movement of particles over a single time step; is used to calculate flexible time step.
	double m_timeStepFactor;		        // Factor used to increase current simulation time step if flexible time step is used.

	CGenerationManager* m_pGenerationManager;
	CSimplifiedScene m_Scene;	// simplified scene
	CVerletList m_VerletList;	// verlet lists

	CParticleParticleModel* m_pPPModel;
	CParticleWallModel* m_pPWModel;
	CSolidBondModel* m_pSBModel;
	CLiquidBondModel* m_pLBModel;
	CExternalForceModel* m_pEFModel;
	std::vector<CAbstractDEMModel*> m_models;	// All active models.
	std::vector<SAdditionalSavingDataPart> m_vAddSavingDataPart;

	// For performance analysis in console
	std::chrono::steady_clock::time_point m_chronoSimStart;
	std::chrono::steady_clock::time_point m_chronoPauseStart;
	int64_t m_chronoPauseLength{ 0 }; // [ms]

public:
	CBaseSimulator();
	CBaseSimulator(const CBaseSimulator& _simulator);
	virtual ~CBaseSimulator() = default;

	CBaseSimulator& operator =(const CBaseSimulator& _simulator);

	void LoadConfiguration() override; // Uses the same file as system structure to load configuration.
	void SaveConfiguration() override; // Uses the same file as system structure to store configuration.

	const CModelManager* GetModelManager() const;						// Returns pointer to a models manager.
	CModelManager* GetModelManager();									// Returns pointer to a models manager.
	void SetModelManager(CModelManager* _pModelManager);				// Sets pointer to a models manager.
	const CGenerationManager* GetGenerationManager() const;				// Returns pointer to a generation manager.
	void SetGenerationManager(CGenerationManager* _pGenerationManager);	// Sets pointer to a generation manager.
	const CSystemStructure* GetSystemStructure() const;					// Returns pointer to a system structure.

	ESimulatorType GetType() const;								// Returns type of current simulator.
	void SetType(const ESimulatorType& _type);					// Sets type of current simulator.
	ERunningStatus GetCurrentStatus() const;					// Returns current status of simulator.
	void SetCurrentStatus(const ERunningStatus& _nNewStatus);	// Sets new status of simulator.

	double GetCurrentTime() const;
	void SetCurrentTime(double _dTime);
	double GetEndTime() const;
	void SetEndTime(double _dEndTime);
	double GetInitSimulationStep() const;
	void SetInitSimulationStep(double _timeStep);
	double GetCurrSimulationStep() const;
	void SetCurrSimulationStep(double _timeStep);
	double GetSavingStep() const;
	void SetSavingStep(double _dSavingStep);
	uint32_t GetMaxCells() const;
	void SetMaxCells(uint32_t _nMaxCells);
	double GetVerletCoeff() const;
	void SetVerletCoeff(double _dCoeff);
	bool GetAutoAdjustFlag() const;
	void SetAutoAdjustFlag(bool _bFlag);
	bool GetVariableTimeStep() const;
	void SetVariableTimeStep(bool _bFlag);
	double GetPartMoveLimit() const;
	virtual void SetPartMoveLimit(double _dx);
	double GetTimeStepFactor() const;
	virtual void SetTimeStepFactor(double _factor);

	// selective saving
	bool IsSelectiveSavingEnabled() const;
	SSelectiveSavingFlags GetSelectiveSavingFlags() const;

	size_t GetNumberOfInactiveParticles() const;
	size_t GetNumberOfBrockenBonds() const;
	size_t GetNumberOfBrockenLiquidBonds() const;
	size_t GetNumberOfGeneratedObjects() const;
	double GetMaxParticleVelocity() const;
	// Returns all current maximal and average overlap between particles with particle indexes smaller than _nMaxParticleID.
	virtual void GetOverlapsInfo(double& _dMaxOverlap, double& _dAverageOverlap, size_t _nMaxParticleID) {};

	CVector3 GetExternalAccel() const;
	virtual void SetExternalAccel(const CVector3& _accel);

	std::string IsDataCorrect() const;	// Checks data correctness.

	virtual void InitializeModelParameters() {}

	virtual void StartSimulation() {}

	CSimplifiedScene& GetPointerToSimplifiedScene() { return m_Scene; }

	virtual void PrepareAdditionalSavingData()=0;

	void SetSelectiveSaving(bool _bSelectiveSaving);
	void SetSelectiveSavingParameters(const SSelectiveSavingFlags& _SSelectiveSavingFlags);

	std::list<std::function<void()>>::iterator AddSavingStep(const std::function<void()>& _function); // Add arbitrary steps to be called during saving.

	std::vector<EStopCriteria> GetStopCriteria() const;
	SStopValues GetStopValues() const;
	void SetStopCriteria(const std::vector<EStopCriteria>& _criteria);
	void SetStopValues(const SStopValues& _values);

protected:
	virtual void InitializeStep(double _timeStep) {}
	virtual void CalculateForces(double _timeStep) {}
	virtual void MoveObjects(double _timeStep, bool _predictionStep = false) {}

	virtual void CalculateForcesPP(double _timeStep) {}
	virtual void CalculateForcesPW(double _timeStep) {}
	virtual void CalculateForcesSB(double _timeStep) {}
	virtual void CalculateForcesLB(double _timeStep) {}
	virtual void CalculateForcesEF(double _timeStep) {}

	virtual void SaveData() {}

	void p_CopySimulatorData(const CBaseSimulator& _simulator);

	void p_InitializeModels();		// Initializes all selected models.
	size_t p_GenerateNewObjects();	// Generates new objects if necessary, returns number of generated objects.
	void p_SaveData();				// Saves current state of simplified scene into system structure.
	void PrintStatus() const;		// Prints the current simulation status into console.

	bool AdditionalStopCriterionMet(); // Checks whether any additional stop criterion is met.
};
