#include "ModelPPSinteringTemperature.cuh"
#include "ModelPPSinteringTemperature.h"
#include "MUSENDefinitions.h"
#include <device_launch_parameters.h>

__constant__ double m_vConstantModelParameters[6];

void CModelPPSinteringTemperature::SetParametersGPU(const std::vector<double>& _parameters, const SPBC& _pbc)
{
	CUDA_MEMCOPY_TO_SYMBOL(m_vConstantModelParameters, *_parameters.data(), sizeof(double) * _parameters.size());
}

void CModelPPSinteringTemperature::CalculatePPForceGPU(double _time, double _timeStep, const SInteractProps _interactProps[], const SGPUParticles& _particles, SGPUCollisions& _collisions)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDA_CalcPPForce_S_Temp_kernel,
		_time,
		_timeStep,
		_interactProps,

		_particles.Vels,
		_particles.Temperatures,
		_particles.Forces,

		_collisions.ActiveCollisionsNum,
		_collisions.ActivityIndices,
		_collisions.InteractPropIDs,
		_collisions.SrcIDs,
		_collisions.DstIDs,
		_collisions.EquivMasses,
		_collisions.EquivRadii,
		_collisions.NormalOverlaps,
		_collisions.ContactVectors,

		_collisions.InitNormalOverlaps,
		_collisions.TangOverlaps
	);
}

void __global__ CUDA_CalcPPForce_S_Temp_kernel(
	const double		 _time,
	const double		 _timeStep,
	const SInteractProps _interactProps[],

	const CVector3	_partVels[],
	const double	_partTemperatures[],

	CVector3	_partForces[],

	const unsigned*	_collActiveCollisionsNum,
	const unsigned	_collActivityIndices[],
	const uint16_t	_collInteractPropIDs[],
	const unsigned	_collSrcIDs[],
	const unsigned	_collDstIDs[],
	const double	_collEquivMasses[],
	const double	_collEquivRadii[],
	const double	_collNormalOverlaps[],
	const CVector3  _collContactVectors[],

	double	 _collInitNormalOverlaps[],
	CVector3 _collTangOverlaps[]
)
{
	for (unsigned iActivColl = blockIdx.x * blockDim.x + threadIdx.x; iActivColl < *_collActiveCollisionsNum; iActivColl += blockDim.x * gridDim.x)
	{
		const unsigned       iColl            = _collActivityIndices[iActivColl];
		const unsigned       iPart1           = _collSrcIDs[iColl];
		const unsigned       iPart2           = _collDstIDs[iColl];
		const SInteractProps prop             = _interactProps[_collInteractPropIDs[iColl]];
		const double         normOverlap      = _collNormalOverlaps[iColl];
		const CVector3       tangOverlapOld   = _collTangOverlaps[iColl];
		const double         equivMass        = _collEquivMasses[iColl];
		const double         equivRadius      = _collEquivRadii[iColl];

		// model parameters
		const double grainBoundTimesDiff = m_vConstantModelParameters[0];
		const double viscousParameter    = m_vConstantModelParameters[1];
		const double atomicVolume        = m_vConstantModelParameters[2];
		const double activationEnergy    = m_vConstantModelParameters[3];
		const double minTemperature      = m_vConstantModelParameters[4];
		const double maxOverlapCoeff     = m_vConstantModelParameters[5];

		const CVector3 normVector = _collContactVectors[iColl].Normalized();
		const double temperatureK   = (_partTemperatures[iPart1] + _partTemperatures[iPart2]) / 2;
		const double maxNormOverlap = maxOverlapCoeff * 2 * equivRadius;

		// normal and tangential relative velocity
		const CVector3 relVel      = _partVels[iPart1] - _partVels[iPart2];
		const double normRelVelLen = DotProduct(normVector, relVel);
		const CVector3 normRelVel  = normVector * normRelVelLen;
		const CVector3 tangRelVel  = relVel - normRelVel;

		// set initial overlap as equilibrium value
		if (_time == 0.0)
		{
			_collInitNormalOverlaps[iColl] = normOverlap;
			_collTangOverlaps[iColl].Init(0);
		}

		CVector3 totalForce;
		// sintering force if temperature is higher than threshold
		if (temperatureK >= minTemperature && normOverlap < maxNormOverlap)
		{
			_collInitNormalOverlaps[iColl] = normOverlap;
			_collTangOverlaps[iColl].Init(0);

			const double diffusionParameter = atomicVolume / (BOLTZMANN_CONSTANT * temperatureK) * grainBoundTimesDiff * exp(-activationEnergy / (GAS_CONSTANT * temperatureK));

			// Bouvard and McMeeking's model
			const double squaredContactRadius = 4 * equivRadius * normOverlap;

			// forces
			const CVector3 sinteringForce = normVector * 1.125 * PI * 2 * equivRadius * prop.dEquivSurfaceEnergy;
			const CVector3 viscousForce   = normRelVel * (-PI * pow(squaredContactRadius, 2.0) / 8 / diffusionParameter);
			const CVector3 tangForce      = tangRelVel * (-viscousParameter * PI * squaredContactRadius * pow(2 * equivRadius, 2.0) / 8 / diffusionParameter);

			totalForce = sinteringForce + viscousForce + tangForce;
		}
		// no sintering force if temperature is lower than threshold - Hertz-Mindlin
		else if (temperatureK <= minTemperature)
		{
			// radius of the contact area
			const double contactAreaRadius = sqrt(_collEquivRadii[iColl] * normOverlap);

			// normal force with damping
			double Kn = 2 * prop.dEquivYoungModulus * contactAreaRadius;
			const double deltaNormOverlap = normOverlap >= maxNormOverlap ? maxNormOverlap : _collInitNormalOverlaps[iColl];
			const double normContactForceLen = -(normOverlap - deltaNormOverlap) * Kn * 2. / 3.;
			const double normDampingForceLen = -_2_SQRT_5_6 * prop.dAlpha * -1 * normRelVelLen * sqrt(Kn * equivMass);
			const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);

			// rotate old tangential overlap
			CVector3 tangOverlapRot = tangOverlapOld - normVector * DotProduct(normVector, tangOverlapOld);
			if (tangOverlapRot.IsSignificant())
				tangOverlapRot *= tangOverlapOld.Length() / tangOverlapRot.Length();
			// calculate new tangential overlap
			const CVector3 tangOverlap = tangOverlapRot + -1 * tangRelVel * _timeStep;

			// tangential force with damping
			const double Kt = 8 * prop.dEquivShearModulus * contactAreaRadius;
			const CVector3 tangShearForce = tangOverlap * Kt;
			const CVector3 tangDampingForce = -1 * tangRelVel * (-_2_SQRT_5_6 * prop.dAlpha * sqrt(Kt * equivMass));
			const CVector3 tangForce = tangShearForce + tangDampingForce;

			// total force
			totalForce = normForce + tangForce;

			// store results in collision
			_collTangOverlaps[iColl] = tangOverlap;
		}
		else
		{
			_collInitNormalOverlaps[iColl] = normOverlap;
			_collTangOverlaps[iColl].Init(0);

			// radius of the contact area
			const double contactAreaRadius = sqrt(_collEquivRadii[iColl] * normOverlap);

			// normal force with damping
			double Kn = 2 * prop.dEquivYoungModulus * contactAreaRadius;
			const double normContactForceLen = -(normOverlap - maxNormOverlap) * Kn * 2. / 3.;
			const double normDampingForceLen = -_2_SQRT_5_6 * prop.dAlpha * -1 * normRelVelLen * std::sqrt(Kn * equivMass);
			const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);

			// total force
			totalForce = normForce;
		}

		// apply forces
		CUDA_VECTOR3_ATOMIC_ADD(_partForces[iPart1], totalForce);
		CUDA_VECTOR3_ATOMIC_SUB(_partForces[iPart2], totalForce);
	}
}