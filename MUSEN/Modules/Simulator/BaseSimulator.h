/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GenerationManager.h"
#include "ModelManager.h"
#include "VerletList.h"
#include <list>
#include <csignal>

inline volatile sig_atomic_t g_extSignal{ 0 };	// Value of external signal for premature termination in internal loops.

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

protected:
	struct SAdditionalSavingData
	{
		CMatrix3 stressTensor;
		void AddStress(const CVector3& _contVector, const CVector3& _force, double _volume);
	};

	double m_currentTime{ 0.0 };							// Current time where simulator is situated.
	double m_endTime{ DEFAULT_END_TIME };					// Last time point which must be simulated.
	double m_initSimulationStep{ DEFAULT_SIMULATION_STEP }; // Initial simulation time step.
	double m_currSimulationStep{ DEFAULT_SIMULATION_STEP }; // Current simulation time step; can change during the simulation.
	double m_savingStep{ DEFAULT_SAVING_STEP };				// Current saving time interval.
	double m_lastSavingTime{ 0.0 };							// Last time point when data has been saved.
	bool m_isPredictionStep{ true };						// Determines whether current simulation step is a prediction step.

	ERunningStatus m_status{ ERunningStatus::IDLE };			// Current status of the simulator - IDLE, RUNNING, etc.
	CVector3 m_externalAcceleration{ 0, 0, -GRAVITY_CONSTANT };	// Value of external acceleration in the system.

	size_t m_nInactiveParticles{ 0 };								// Number of particles which lost their activity during simulation.
	size_t m_nBrokenBonds{ 0 };									// Number of the broken bonds.
	size_t m_nBrokenLiquidBonds{ 0 };								// Number of the ruptured liquid bonds.
	size_t m_nGeneratedObjects{ 0 };								// Number of generated objects.
	double m_maxParticleVelocity{ 0 };								// Maximal velocity of particle which is used to calculate verlet list.
	double m_maxParticleTemperature{ 0 };							// Maximal temperature of particles.
	double m_maxWallVelocity{ 0 };									// Maximal velocity of walls.
	bool m_wallsVelocityChanged{ true };							// Need to recalculate maximal wall velocity.
	uint32_t m_cellsMax{ DEFAULT_MAX_CELLS };						// Maximum allowed number of cells in each direction of verlet list.
	double m_verletDistanceCoeff{ DEFAULT_VERLET_DISTANCE_COEFF };	// A coefficient to calculate verlet distance within a verlet list.
	bool m_autoAdjustVerletDistance{ true };						// If set to true - the verlet distance will be automatically adjusted during the simulation.
	bool m_considerAnisotropy{ false };								// Consider anisotropy of non-spherical objects during the simulation.
	bool m_variableTimeStep{ false };								// Use variable or constant simulation time step.
	double m_partMoveLimit{ 1e-8 };									// Max movement of particles over a single time step; is used to calculate flexible time step.
	double m_timeStepFactor{ 1.01 };								// Factor used to increase current simulation time step if flexible time step is used.

	CGenerationManager* m_generationManager{ nullptr };
	CSimplifiedScene m_scene;				// simplified scene
	CVerletList m_verletList{ m_scene };	// verlet lists

	std::vector<CParticleParticleModel*> m_PPModels{}; // Active particle-particle models.
	std::vector<CParticleWallModel*> m_PWModels{};     // Active particle-wall models.
	std::vector<CSolidBondModel*> m_SBModels{};        // Active solid bond models.
	std::vector<CLiquidBondModel*> m_LBModels{};       // Active liquid bond models.
	std::vector<CExternalForceModel*> m_EFModels{};    // Active external force models.
	std::vector<CPPHeatTransferModel*> m_PPHTModels;   // Active particle-particle heat transfer models.
	std::vector<CAbstractDEMModel*> m_models;	       // All active models.

	std::vector<SAdditionalSavingData> m_additionalSavingData;

	// For performance analysis in console
	std::chrono::system_clock::time_point m_chronoSimStart;
	std::chrono::system_clock::time_point m_chronoPauseStart;
	int64_t m_chronoPauseLength{ 0 }; // [ms]

private:
	ESimulatorType m_simulatorType{ ESimulatorType::BASE };	// Type of current simulator.
	CModelManager* m_modelManager{ nullptr };

	// selective saving
	bool m_selectiveSaving{ false };
	SSelectiveSavingFlags m_selectiveSavingFlags;

	// additional saving steps
	std::list<std::function<void()>> m_additionalSavingSteps; // List of additional steps called during saving.

	// additional stop criteria
	std::vector<EStopCriteria> m_stopCriteria;	// Active additional simulation stop criteria.
	SStopValues m_stopValues;					// Values for additional simulation stop criteria.

public:
	CBaseSimulator() = default;
	CBaseSimulator(const CBaseSimulator& _other);
	CBaseSimulator(CBaseSimulator&& _other) = delete;
	CBaseSimulator& operator=(CBaseSimulator&& _other) = delete;
	~CBaseSimulator() override = default;

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

	std::chrono::time_point<std::chrono::system_clock> GetStartDateTime() const;	// Returns time point when the simulation has started.
	std::chrono::time_point<std::chrono::system_clock> GetFinishDateTime() const;	// Returns approximated time point when the simulation should finish.

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
	size_t GetNumberOfBrokenBonds() const;
	size_t GetNumberOfBrokenLiquidBonds() const;
	size_t GetNumberOfGeneratedObjects() const;
	double GetMaxParticleVelocity() const;
	// Returns all current maximal and average overlap between particles with particle indexes smaller than _nMaxParticleID.
	virtual void GetOverlapsInfo(double& _dMaxOverlap, double& _dAverageOverlap, size_t _nMaxParticleID) {}

	CVector3 GetExternalAccel() const;
	virtual void SetExternalAccel(const CVector3& _accel);

	std::string IsDataCorrect() const;	// Checks data correctness.

	virtual void Initialize();					// Initializes simulator before start.
	virtual void InitializeModels();			// Initializes all selected models.
	virtual void InitializeModelParameters();	// Initializes parameters used in models.

	virtual void Simulate();
	virtual void StartSimulation();		// Starts the simulation procedure.
	virtual void FinalizeSimulation();	// Is called after the simulation is finished.

	virtual void PreCalculationStep();
	virtual void UpdateCollisionsStep(double _timeStep) {}
	virtual void CalculateForcesStep(double _timeStep) {}
	virtual void CalculateForcesPP(double _timeStep) {}
	virtual void CalculateForcesPW(double _timeStep) {}
	virtual void CalculateForcesSB(double _timeStep) {}
	virtual void CalculateForcesLB(double _timeStep) {}
	virtual void CalculateForcesEF(double _timeStep) {}
	virtual void CalculateHeatTransferPP(double _dTimeStep) {}

	virtual void MoveObjectsStep(double _timeStep, bool _predictionStep = false);
	virtual void MoveParticles(bool _predictionStep = false) {}
	virtual void MoveWalls(double _timeStep) {}
	virtual void UpdateTemperatures(double _timeStep, bool _predictionStep = false) {}

	CSimplifiedScene& GetPointerToSimplifiedScene() { return m_scene; }

	virtual void PrepareAdditionalSavingData() {}

	void SetSelectiveSaving(bool _bSelectiveSaving);
	void SetSelectiveSavingParameters(const SSelectiveSavingFlags& _SSelectiveSavingFlags);

	std::list<std::function<void()>>::iterator AddSavingStep(const std::function<void()>& _function); // Add arbitrary steps to be called during saving.

	std::vector<EStopCriteria> GetStopCriteria() const;
	SStopValues GetStopValues() const;
	void SetStopCriteria(const std::vector<EStopCriteria>& _criteria);
	void SetStopValues(const SStopValues& _values);

protected:
	virtual void SaveData() {}

	virtual size_t GenerateNewObjects();	// Generates new objects if necessary, returns number of generated objects.
	virtual void UpdatePBC() {}				// Updates moving PBC.
	void p_SaveData();						// Saves current state of simplified scene into system structure.
	void PrintStatus() const;				// Prints the current simulation status into console.

	bool AdditionalStopCriterionMet(); // Checks whether any additional stop criterion is met.

private:
	void CopySimulatorData(const CBaseSimulator& _other); // Copies content of the given simulator.
};
