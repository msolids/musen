/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelEFLiquidDiffusion.cuh"
#include "ModelEFLiquidDiffusion.h"
#include "MUSENDefinitions.h"
#include <iomanip>
#include "DisableWarningHelper.h"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
PRAGMA_WARNING_POP

__constant__ double m_vConstantModelParameters[16];
__constant__ double m_temperatureCoefGPU;

////////////////////////////////////////////////////////
///////////// INITIALIZATION FUNCTIONS
////////////////////////////////////////////////////////

void CModelEFLiquidDiffusion::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
	CUDA_MEMCOPY_TO_SYMBOL(m_temperatureCoefGPU, m_temperatureCoefCPU, sizeof(double));

	// Initialize CUDA Random number generator states
	m_nStatesGPU = 0;

	// set model flags according to parameter values
	if (m_parameters[4].value == 1.0)
		m_checkSeeds = true;

	// Open output files
	if (m_parameters[12].value == 1.0 && !m_saveFlag)
	{
		m_saveFlag = true;
		m_savingStep = m_parameters[13].value;
		if (!m_fileKineticEnergy.is_open())
			m_fileKineticEnergy.open("./Post_KineticEnergy.csv", std::ofstream::out);
		if (!m_fileTemperature.is_open())
			m_fileTemperature.open("./Post_Temperature.csv", std::ofstream::out);
	}

	// Detect annealing
	DetectAnnealing();

	// Check PBC if shear flow is enabled
	if (m_parameters[14].value != 0.0 && _pbc.bZ)
		if (abs(_pbc.currentDomain.coordBeg.z + _pbc.currentDomain.coordEnd.z) > 1e-6 * (_pbc.currentDomain.coordEnd.z - _pbc.currentDomain.coordBeg.z))
			std::cerr << "Error: Using shear flow, but periodic boundaries in Z-direction seem to be non-symmetric. This will cause unphysical results." << std::endl;
}

void CModelEFLiquidDiffusion::InitializeDiffusionModelGPU(double _timeStep, SGPUParticles& _particles)
{
	std::cout << "For using this diffusion model please cite: Depta et al. J. Chem. Inf. Model, 59 (2019) 1, 386-398, DOI: 10.1021/acs.jcim.8b00613" << std::endl;

	// Desired number of states for random numbers - each for every particle
	const size_t desNumStates = _particles.nElements;

	// Calculate diffusion model coefficients (forces, moments, and drag coefficients) for all enzyme kinds
	CalculateDiffusionModelCoefficientsGPU(_timeStep, _particles);

	// Initialize seeds and random number generators for the desired number of states
	InitializeSeedsAndRandomGeneratorsGPU(desNumStates);

	m_nStatesGPU = static_cast<unsigned>(desNumStates); // Set number of states present in memory to desired number of states --> ensures if is only entered at beginning and if particle number changes
}

void CModelEFLiquidDiffusion::UpdateTemperatureGPU(double newTemperature)
{
	m_temperatureCoefCPU = sqrt(newTemperature/m_parameters[0].value);
	CUDA_MEMCOPY_TO_SYMBOL(m_temperatureCoefGPU, m_temperatureCoefCPU, sizeof(double));
}

void CModelEFLiquidDiffusion::CalculateDiffusionModelCoefficientsGPU(double _timeStep, SGPUParticles& _particles)
{
	// Free previous state if required
	if (m_nStatesGPU != 0)
	{
		CUDA_FREE_D(m_pDragLinGPU);
		CUDA_FREE_D(m_pFMagnitudeGPU);
		CUDA_FREE_D(m_pDragRotGPU);
		CUDA_FREE_D(m_pMMagnitudeGPU);
	}
	const size_t bytes = _particles.nElements * sizeof(double);
	CUDA_MALLOC_D(&m_pDragLinGPU, bytes);
	CUDA_MALLOC_D(&m_pFMagnitudeGPU, bytes);
	CUDA_MALLOC_D(&m_pDragRotGPU, bytes);
	CUDA_MALLOC_D(&m_pMMagnitudeGPU, bytes);

	// use thrust to calculate critical time constants
	thrust::device_vector<double> t_TauMinLin(_particles.nElements);
	thrust::device_vector<double> t_TauMinRot(_particles.nElements);
	double* pTauMinLin = thrust::raw_pointer_cast(&t_TauMinLin[0]);
	double* pTauMinRot = thrust::raw_pointer_cast(&t_TauMinRot[0]);

	// Calculate diffusion model coefficients on GPU
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcDifCoeff_LiquidDiffusion_kernel,
		_timeStep,

		static_cast<unsigned>(_particles.nElements),
		_particles.Radii,
		_particles.Masses,
		_particles.InertiaMoments,

		m_pDragLinGPU,
		m_pFMagnitudeGPU,
		m_pDragRotGPU,
		m_pMMagnitudeGPU,

		pTauMinLin,
		pTauMinRot
	);

	// get minimum critical time constants
	m_tauMinLin = t_TauMinLin[thrust::min_element(t_TauMinLin.begin(),t_TauMinLin.end()) - t_TauMinLin.begin()];
	m_tauMinRot = t_TauMinRot[thrust::min_element(t_TauMinRot.begin(),t_TauMinRot.end()) - t_TauMinRot.begin()];
	std::cout << "Minimum critical time constant for diffusion (translation): " << m_tauMinLin << " [s]" << std::endl;
	if (m_parameters[6].value == 1.0)
		std::cout << "Minimum critical time constant for diffusion (rotation): " << m_tauMinRot << " [s]" << std::endl;
}

// Kernel to calculate diffusion coefficients
void __global__ CUDA_CalcDifCoeff_LiquidDiffusion_kernel(
	const double		_timeStep,

	const unsigned		_partNum,
	const double		_partRadii[],
	const double		_partMasses[],
	const double		_intertiaMoments[],

	double				_pDragLinGPU[],
	double				_pFMagnitudeGPU[],
	double				_pDragRotGPU[],
	double				_pMMagnitudeGPU[],

	double				_pTauMinLin[],
	double				_pTauMinRot[]
)
{
	for (unsigned iPart = blockIdx.x * blockDim.x + threadIdx.x; iPart < _partNum; iPart += blockDim.x * gridDim.x)
	{
		const double r_stokes = m_vConstantModelParameters[2] * _partRadii[iPart] + m_vConstantModelParameters[3];
		const double partMass = _partMasses[iPart];
		const double partInerMom = _intertiaMoments[iPart];

		// Calculate translation coefficients
		double temp = 6 * PI * m_vConstantModelParameters[1] * r_stokes * _timeStep / partMass;
		_pDragLinGPU[iPart] = partMass / _timeStep * (1 - exp(-1 * temp));
		_pFMagnitudeGPU[iPart] = sqrt(partMass * BOLTZMANN_CONSTANT * m_vConstantModelParameters[0] * (1 - exp(-2 * temp))) / _timeStep;

		// Calculate rotational coefficients
		temp = 8 * PI * m_vConstantModelParameters[1] * pow(r_stokes, 3.0) * _timeStep / partInerMom;
		_pDragRotGPU[iPart] = partInerMom / _timeStep * (1 - exp(-1 * temp));
		_pMMagnitudeGPU[iPart] = sqrt(partInerMom * BOLTZMANN_CONSTANT * m_vConstantModelParameters[0] * (1 - exp(-2 * temp))) / _timeStep;

		// Critical time constants
		_pTauMinLin[iPart] = partMass / (6 * PI * m_vConstantModelParameters[1] * r_stokes);
		_pTauMinRot[iPart] = partInerMom / (8 * PI * m_vConstantModelParameters[1] * pow(r_stokes, 3.0));

		//printf("%e;%e;%e;%e\n",p_DragLinGPU[iPart],p_FMagnitudeGPU[iPart],p_DragRotGPU[iPart],p_MMagnitudeGPU[iPart]);
	}
}

void CModelEFLiquidDiffusion::InitializeSeedsAndRandomGeneratorsGPU(size_t _desNumStates)
{
	// Free previous state if required
	if (m_nStatesGPU != 0)
	{
		CUDA_FREE_D(pCurandStatesGPU);
	}

	// Determine seeds on CPU
	// 32bit seeds
	auto* pSeedsCPU = new uint32_t[_desNumStates];
	CreateUniqueSeeds32(pSeedsCPU, static_cast<unsigned>(_desNumStates));

	// Copy seeds over to GPU
	const size_t bytes = _desNumStates * sizeof(unsigned);
	CUDA_MALLOC_D(&m_pSeedsGPU, bytes);					// allocate space on GPU
	CUDA_MEMCPY_H2D(m_pSeedsGPU, pSeedsCPU, bytes);		// copy seeds to GPU

	// Generate curand states on GPU
	CUDA_MALLOC_D(&pCurandStatesGPU, _desNumStates * sizeof(curandState));
	GenerateNewCurandStates(_desNumStates, m_pSeedsGPU, pCurandStatesGPU);

	// Remove seed data - not needed any more
	delete[] pSeedsCPU;
	CUDA_FREE_D(m_pSeedsGPU);
}

// Generate new random states (curand states) for diffusion model
void CModelEFLiquidDiffusion::GenerateNewCurandStates(size_t _nStates, const unsigned* _seeds, curandState * _states)
{
	CUDA_KERNEL_ARGS2_DEFAULT(GenerateNewCurandStates_LiquidDiffusion_kernel, _nStates, _seeds, _states);
}

void __global__ GenerateNewCurandStates_LiquidDiffusion_kernel(size_t _nStates, const unsigned* _seeds, curandState* _states)
{
	// Calculation of random states - Seed calculated on CPU - WORKS
	for (size_t iState = blockIdx.x * blockDim.x + threadIdx.x; iState < _nStates; iState += blockDim.x * gridDim.x)
	{
		curand_init(_seeds[iState], 0, 0, &_states[iState]); // Sequence iState is necessary since Seeds might sometimes be the same or similar for large systems
	}
}

////////////////////////////////////////////////////////
///////////// FORCE CALCULATION
////////////////////////////////////////////////////////

// Wrapper to check for changes in system setting (number of particles) and saving switch
// Passes pointer to Curand states to Kernel execution
void CModelEFLiquidDiffusion::CalculateEFForceGPU(double _time, double _timeStep, SGPUParticles& _particles)
{
	// initialize model if necessary
	if (m_nStatesGPU != _particles.nElements) // check if the current number of states in storage is not equal to the particle number (desired number of states)
		InitializeDiffusionModelGPU(_timeStep, _particles);

	// update temperature if necessary
	if (m_annealingType != EAnnealing::NONE)
	{
		if (_time >= m_annealingFinishedTime) // set temperature once annealing is over
		{
			if (_time < m_annealingFinishedTime + 2 * _timeStep)
			{
				m_curTemp = m_annealingEqTemp;
				UpdateTemperatureGPU(m_curTemp);
			}
		}
		else
		{
			if (m_annealingType == EAnnealing::LIN)
			{
				m_curTemp = m_annealingEqTemp;
				const double relAnnealingTime = fmod(_time, m_annealingPeriod);
				if (relAnnealingTime < m_annealingCooldownTime)
					m_curTemp = m_annealingEqTemp + (m_annealingMaxTemp - m_annealingEqTemp) * (1 - relAnnealingTime / m_annealingCooldownTime);
				UpdateTemperatureGPU(m_curTemp);
			}
		}
	}

	// calculate forces
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcExtForce_LiquidDiffusion_kernel,
		static_cast<unsigned>(_particles.nElements),
		_particles.Vels,
		_particles.AnglVels,
		_particles.Forces,
		_particles.Moments,
		_particles.Coords,
		_particles.Radii,

		pCurandStatesGPU,

		m_pDragLinGPU,
		m_pFMagnitudeGPU,
		m_pDragRotGPU,
		m_pMMagnitudeGPU
	);

	// output global data
	if (m_saveFlag)
	{
		if (fabs(_time - m_lastSavingTime) + 0.2*_timeStep > m_savingStep)
		{
			// set saving step
			m_lastSavingTime = _time;
			const unsigned outPerc = 6;

			// global kinetic energy
			static thrust::device_vector<double> tempKineticEnergy;
			if (tempKineticEnergy.size() < _particles.nElements)
				tempKineticEnergy.resize(_particles.nElements);
			CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcKinEnergy_LiquidDiffusion_kernel,
				static_cast<unsigned>(_particles.nElements),
				_particles.Vels,
				_particles.AnglVels,

				_particles.Masses,
				_particles.InertiaMoments,

				tempKineticEnergy.data().get()
			);
			const double kinEnergy = thrust::reduce(tempKineticEnergy.begin(), tempKineticEnergy.begin() + _particles.nElements, static_cast<double>(0), thrust::plus<double>());
			m_fileKineticEnergy << std::setprecision(outPerc) << _time << ";" << kinEnergy << std::endl;

			// temperature
			printf("Temperature at time %6e: %6.2f K\n",_time,m_curTemp);
			m_fileTemperature << std::setprecision(outPerc) << _time << ";" << m_curTemp << std::endl;
		}
	}
}

void __global__ CUDA_CalcExtForce_LiquidDiffusion_kernel(
	unsigned			_partNum,
	const CVector3		_partVels[],
	const CVector3		_partAnglVels[],
	CVector3			_partForces[],
	CVector3			_partMoments[],
	const CVector3		_partCoords[],
	const double		_partRadii[],

	curandState*        _pCurandStatesGPU,

	const double		_pDragLinGPU[],
	const double		_pFMagnitudeGPU[],
	const double		_pDragRotGPU[],
	const double		_pMMagnitudeGPU[]
)
{
	for (unsigned iPart = blockIdx.x * blockDim.x + threadIdx.x; iPart < _partNum; iPart += blockDim.x * gridDim.x)
	{
		curandState localState = _pCurandStatesGPU[iPart];		// save state locally to work in register

		// Draw random numbers for diffusive forces and moments
		const auto ranFx = static_cast<double>(curand_normal(&localState));
		const auto ranFy = static_cast<double>(curand_normal(&localState));
		const auto ranFz = static_cast<double>(curand_normal(&localState));

		// Get local flow velocity
		const double flowVelocityX = m_vConstantModelParameters[14] * abs(_partCoords[iPart].z);

		// Calculate diffusion force in global frame
		CVector3 FDifGlobal;
		FDifGlobal.x = m_temperatureCoefGPU*_pFMagnitudeGPU[iPart]*ranFx - _pDragLinGPU[iPart]*(_partVels[iPart].x - flowVelocityX);
		FDifGlobal.y = m_temperatureCoefGPU*_pFMagnitudeGPU[iPart]*ranFy - _pDragLinGPU[iPart]*_partVels[iPart].y;
		FDifGlobal.z = m_temperatureCoefGPU*_pFMagnitudeGPU[iPart]*ranFz - _pDragLinGPU[iPart]*_partVels[iPart].z;

		// Calculate Saffman lift force, if positive
		const double liftForce = 6.46 * m_vConstantModelParameters[15] * sqrt(m_vConstantModelParameters[1] / m_vConstantModelParameters[15]) * (flowVelocityX - _partVels[iPart].x) * _partRadii[iPart] * _partRadii[iPart] * sqrt(abs(m_vConstantModelParameters[14]));
		if (_partCoords[iPart].z > 0)	// differentiate because shear rate has different directions
			FDifGlobal.z += liftForce;
		else
			FDifGlobal.z -= liftForce;

		// Add to particle force
		_partForces[iPart] += FDifGlobal;

		// Calculate diffusion moment in global frame
		if (m_vConstantModelParameters[6] == 1.0)
		{
			const auto ranMx = static_cast<double>(curand_normal(&localState));
			const auto ranMy = static_cast<double>(curand_normal(&localState));
			const auto ranMz = static_cast<double>(curand_normal(&localState));
			CVector3 MDifGlobal;
			MDifGlobal.x = m_temperatureCoefGPU*_pMMagnitudeGPU[iPart]*ranMx - _pDragRotGPU[iPart]*_partAnglVels[iPart].x;
			MDifGlobal.y = m_temperatureCoefGPU*_pMMagnitudeGPU[iPart]*ranMy - _pDragRotGPU[iPart]*_partAnglVels[iPart].y;
			MDifGlobal.z = m_temperatureCoefGPU*_pMMagnitudeGPU[iPart]*ranMz - _pDragRotGPU[iPart]*_partAnglVels[iPart].z;
			_partMoments[iPart] += MDifGlobal;
		}

		_pCurandStatesGPU[iPart] = localState;
	}
}

void __global__ CUDA_CalcKinEnergy_LiquidDiffusion_kernel(
	unsigned			_partNum,
	const CVector3		_partVels[],
	const CVector3		_partAnglVels[],

	const double		_partMasses[],
	const double		_intertiaMoments[],

	double				_partKinEnergies[]
)
{
	for (unsigned iPart = blockIdx.x * blockDim.x + threadIdx.x; iPart < _partNum; iPart += blockDim.x * gridDim.x)
	{
		_partKinEnergies[iPart] = 0.5 * (_partMasses[iPart] * _partVels[iPart].SquaredLength() + _intertiaMoments[iPart] * _partAnglVels[iPart].SquaredLength());
	}
}