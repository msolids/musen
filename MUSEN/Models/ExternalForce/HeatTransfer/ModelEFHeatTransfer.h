#pragma once

#include "AbstractDEMModel.h"

class CModelEFHeatTransfer : public CExternalForceModel
{
public:
	CModelEFHeatTransfer();
	void CalculateEFForce(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles) const override;
	void SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc) override;
	void CalculateEFForceGPU(double _time, double _timeStep, SGPUParticles& _particles) override;
};