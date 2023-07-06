/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

// TODO: get rid of GetParametersStr()/SetParametersStr()

#include "BasicTypes.h"
#include "SceneTypes.h"
#include "SceneTypesGPU.h"
#include "SceneOptionalVariables.h"

#ifdef _DEBUG
#define MUSEN_CREATE_MODEL_FUN MusenCreateModelV9Debug
#define MUSEN_CREATE_MODEL_FUN_NAME "MusenCreateModelV9Debug"
#else
#define MUSEN_CREATE_MODEL_FUN MusenCreateModelV9
#define MUSEN_CREATE_MODEL_FUN_NAME "MusenCreateModelV9"
#endif

#ifndef DYNAMIC_MODULE
#define DECLDIR
#elif defined DLL_EXPORT
#define DECLDIR __declspec(dllexport)
#else
#define DECLDIR __declspec(dllimport)
#endif


enum class EMusenModelType : unsigned
{
	UNSPECIFIED = 0,	// unknown
	PP          = 1,	// particle-particle
	SB          = 2,	// solid bond
	LB          = 3,	// liquid bond
	PW          = 4,	// particle-wall
	EF          = 5,	// external force
	PPHT        = 6,	// heat transfer between particle-particle
};

struct SModelParameter
{
	double value;
	double defaultValue;
	std::string uniqueName; // name which should not contain spaces
	std::string description;
	SModelParameter(std::string _uniqueName, std::string _description, double _defaultValue) :
		value(_defaultValue), defaultValue(_defaultValue), uniqueName(std::move(_uniqueName)), description(std::move(_description)) {}
};

class CAbstractDEMModel
{
public:
	CAbstractDEMModel();
	virtual ~CAbstractDEMModel() = default;

	EMusenModelType GetType() const;
	std::string GetName() const;
	std::string GetUniqueKey() const;
	std::string GetHelpFileName() const;

	size_t GetParametersNumber() const;								 // Returns the number of currently defined parameters.
	std::vector<SModelParameter> GetAllParameters() const;			 // Returns a set of model parameters.
	std::string GetParametersStr() const;							 // Returns a string representation of all defined parameters with their values.
	void SetParametersStr(const std::string& _parameters);			 // Sets parameters from their string representation.
	double GetParameterValue(const std::string& _name) const;		 // Returns the value of the parameter with the given name.
	void SetParameterValue(const std::string& _name, double _value); // Sets new values to the parameter with the given name.
	void SetDefaultValues();										 // Resets all parameters to their default values.

	void SetPBC(SPBC _pbc);
	SOptionalVariables GetUtilizedVariables() const;

	// Initializes the model with all required data. Called once before the start of the simulation.
	virtual bool Initialize(SParticleStruct* _particles, SWallStruct* _walls, SSolidBondStruct* _solidBinds, SLiquidBondStruct* _liquidBonds, std::vector<SInteractProps>* _interactProps) = 0;
	// Initializes the model with all required data. Called once before the start of the simulation.
	void InitializeGPU(const CCUDADefines* _cudaDefines);

	// Is called each time step before real calculations.
	virtual void Precalculate(double _time, double _timeStep) = 0;

	bool HasGPUSupport() const;

protected:
	EMusenModelType m_type;		// Type of the model.
	std::string m_name;			// Name of the model.
	std::string m_uniqueKey;	// Unique key of the model.
	std::string m_helpFileName;	// Path to the file with documentation.
	std::vector<SModelParameter> m_parameters;	// Model parameters.

	bool m_hasGPUSupport;	// Indicates that this model has GPU support.
	SPBC m_PBC;				// Current PBC.

	const CCUDADefines* m_cudaDefines{ nullptr };	// Needed to call cuda code.
	SOptionalVariables m_requieredVariables;

	// Adds new parameter. The name may not contain spaces, tabs, new lines or other escape characters. Returns true if the addition was successful.
	bool AddParameter(const SModelParameter& _parameter);
	// Adds new parameter. The name may not contain spaces, tabs, new lines or other escape characters. Returns true if the addition was successful.
	bool AddParameter(const std::string& _name, const std::string& _description, double _defaultValue);

	virtual void SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc) {}; // Must be overriden to set model parameters into GPU memory.
};


class CParticleParticleModel : public CAbstractDEMModel
{
	SParticleStruct* m_particles{ nullptr };
	std::vector<SInteractProps>* m_interactProps{ nullptr };

public:
	CParticleParticleModel();

	bool Initialize(SParticleStruct* _particles, SWallStruct* _walls, SSolidBondStruct* _solidBinds, SLiquidBondStruct* _liquidBonds, std::vector<SInteractProps>* _interactProps) override;
	void Precalculate(double _time, double _timeStep) override;
	void Calculate(double _time, double _timeStep, SCollision* _collision) const;
	void ConsolidateSrc(double _time, double _timeStep, SParticleStruct& _particles, const SCollision* _collision) const;
	void ConsolidateDst(double _time, double _timeStep, SParticleStruct& _particles, const SCollision* _collision) const;

	virtual void CalculatePPForceGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions) {}

protected:
	const SParticleStruct& Particles() const { return *m_particles; }
	const SInteractProps& InteractionProperty(const size_t _i) const { return (*m_interactProps)[_i]; }

	virtual void PrecalculatePPModel(double _time, double _timeStep, SParticleStruct* _particles) {}
	virtual void CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const = 0;
	virtual void ConsolidateSrc(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const {}
	virtual void ConsolidateDst(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const {}
};


class CParticleWallModel : public CAbstractDEMModel
{
	SParticleStruct* m_particles{ nullptr };
	SWallStruct* m_walls{ nullptr };
	std::vector<SInteractProps>* m_interactProps{ nullptr };

public:
	CParticleWallModel();

	bool Initialize(SParticleStruct* _particles, SWallStruct* _walls, SSolidBondStruct* _solidBinds, SLiquidBondStruct* _liquidBonds, std::vector<SInteractProps>* _interactProps) override;
	void Precalculate(double _time, double _timeStep) override;
	void Calculate(double _time, double _timeStep, SCollision* _collision) const;
	void ConsolidatePart(double _time, double _timeStep, SParticleStruct& _particles, const SCollision* _collision) const;
	void ConsolidateWall(double _time, double _timeStep, SWallStruct& _walls, const SCollision* _collision) const;

	virtual void CalculatePWForceGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, const SGPUWalls& _walls, SGPUCollisions& _collisions) {}

protected:
	const SParticleStruct& Particles() const { return *m_particles; }
	const SWallStruct& Walls() const { return *m_walls; };
	const SInteractProps& InteractionProperty(const size_t _i) const { return (*m_interactProps)[_i]; }

	virtual void PrecalculatePWModel(double _time, double _timeStep, SParticleStruct* _particles, SWallStruct* _walls) {}
	virtual void CalculatePWForce(double _time, double _timeStep, size_t _iWall, size_t _iPart, const SInteractProps& _interactProp, SCollision* _collision) const = 0;
	virtual void ConsolidatePart(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const {}
	virtual void ConsolidateWall(double _time, double _timeStep, size_t _iWall, SWallStruct& _walls, const SCollision* _collision) const {}
};


class CSolidBondModel : public CAbstractDEMModel
{
	SParticleStruct* m_particles{ nullptr };
	SSolidBondStruct* m_bonds{ nullptr };

public:
	CSolidBondModel();

	bool Initialize(SParticleStruct* _particles, SWallStruct* _walls, SSolidBondStruct* _solidBinds, SLiquidBondStruct* _liquidBonds, std::vector<SInteractProps>* _interactProps) override;
	void Precalculate(double _time, double _timeStep) override;
	void Calculate(double _time, double _timeStep, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _pBrokenBondsNum) const;
	void Consolidate(double _time, double _timeStep, size_t _iBond, size_t _iPart, SParticleStruct& _particles) const;

	virtual void CalculateSBForceGPU(double _time, double _timeStep, const SGPUParticles& _particles, SGPUSolidBonds& _bonds) {}

protected:
	const SParticleStruct& Particles() const { return *m_particles; }
	const SSolidBondStruct& Bonds() const { return *m_bonds; }

	virtual void PrecalculateSBModel(double _time, double _timeStep, SParticleStruct* _particles, SSolidBondStruct* _bonds) {}
	virtual void CalculateSBForce(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _pBrokenBondsNum) const = 0;
	virtual void ConsolidatePart(double _time, double _timeStep, size_t _iBond, size_t _iPart, SParticleStruct& _particles) const {}
};


class CLiquidBondModel : public CAbstractDEMModel
{
	SParticleStruct* m_particles{ nullptr };
	SLiquidBondStruct* m_bonds{ nullptr };

public:
	CLiquidBondModel();

	bool Initialize(SParticleStruct* _particles, SWallStruct* _walls, SSolidBondStruct* _solidBinds, SLiquidBondStruct* _liquidBonds, std::vector<SInteractProps>* _interactProps) override;
	void Precalculate(double _time, double _timeStep) override;
	void Calculate(double _time, double _timeStep, size_t _iBond, SLiquidBondStruct& _bonds, unsigned* _pBrokenBondsNum) const;
	void Consolidate(double _time, double _timeStep, size_t _iBond, SParticleStruct& _particles) const;

protected:
	const SParticleStruct& Particles() const { return *m_particles; }
	const SLiquidBondStruct& Bonds() const { return *m_bonds; }

	virtual void PrecalculateLBModel(double _time, double _timeStep, SParticleStruct* _particles, SLiquidBondStruct* _bonds) {}
	virtual void CalculateLBForce(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SLiquidBondStruct& _bonds, unsigned* _pBrokenBondsNum) const = 0;
	virtual void ConsolidatePart(double _time, double _timeStep, size_t _iBond, SParticleStruct& _particles) const {}
};


class CExternalForceModel :	public CAbstractDEMModel
{
	SParticleStruct* m_particles{ nullptr };

public:
	CExternalForceModel();

	bool Initialize(SParticleStruct* _particles, SWallStruct* _walls, SSolidBondStruct* _solidBinds, SLiquidBondStruct* _liquidBonds, std::vector<SInteractProps>* _interactProps) override;
	void Precalculate(double _time, double _timeStep) override;
	void Calculate(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles) const;

	virtual void CalculateEFForceGPU(double _time, double _timeStep, SGPUParticles& _particles) {}

protected:
	const SParticleStruct& Particles() const { return *m_particles; }

	virtual void PrecalculateEFModel(double _time, double _timeStep, SParticleStruct* _particles) {}
	virtual void CalculateEFForce(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles) const = 0;
};


class CPPHeatTransferModel : public CAbstractDEMModel
{
	SParticleStruct* m_particles{ nullptr };
	std::vector<SInteractProps>* m_interactProps{ nullptr };

public:
	CPPHeatTransferModel();

	bool Initialize(SParticleStruct* _particles, SWallStruct* _walls, SSolidBondStruct* _solidBinds, SLiquidBondStruct* _liquidBonds, std::vector<SInteractProps>* _interactProps) override;
	void Precalculate(double _time, double _timeStep) override;
	void Calculate(double _time, double _timeStep, SCollision* _collision) const;
	void ConsolidateSrc(double _time, double _timeStep, SParticleStruct& _particles, const SCollision* _collision) const;
	void ConsolidateDst(double _time, double _timeStep, SParticleStruct& _particles, const SCollision* _collision) const;

	virtual void CalculatePPHeatTransferGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions) {}

protected:
	const SParticleStruct& Particles() const { return *m_particles; }
	const SInteractProps& InteractionProperty(const size_t _i) const { return (*m_interactProps)[_i]; }

	virtual void PrecalculatePPHTModel(double _time, double _timeStep, SParticleStruct* _particles) const {}
	virtual void CalculatePPHeatTransfer(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const = 0;
	virtual void ConsolidateSrc(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const {}
	virtual void ConsolidateDst(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles, const SCollision* _collision) const {}
};

typedef DECLDIR CAbstractDEMModel* (*CreateModelFunction)();