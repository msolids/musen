/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "PackageGeneratorHelperGPU.cuh"

namespace functors
{
	struct mul_04
	{
		__host__ __device__ CVector3 operator()(const CVector3& _v) const {
			return _v * 0.4;
		}
	};

	struct calc_deltas
	{
		const double minR;
		explicit calc_deltas(double _minR) : minR{ _minR } {}
		__host__ __device__ CVector3 operator()(const CVector3& _v, const double& _r) const {
			return _v * (1 - minR / _r);
		}
	};

	struct cals_rel_vel
	{
		__host__ __device__ double operator()(const CVector3& _v, const double& _r) const {
			return _v.SquaredLength() / (_r * _r);
		}
	};
}

CPackageGeneratorHelperGPU::CPackageGeneratorHelperGPU(SGPUParticles* _particles) :
	m_particles{ _particles },
	m_number{ _particles->nElements },
	m_oldVels{ new thrust::device_vector<CVector3>(_particles->nElements, CVector3{ 0.0 }) },
	m_deltaCoord{ new thrust::device_vector<CVector3>(_particles->nElements) },
	m_relVels{ new thrust::device_vector<double>(_particles->nElements) }
{
}

CPackageGeneratorHelperGPU::~CPackageGeneratorHelperGPU()
{
	delete m_oldVels;
	delete m_deltaCoord;
	delete m_relVels;
}

void CPackageGeneratorHelperGPU::LimitVelocities() const
{
	const thrust::device_ptr<CVector3> vels = thrust::device_pointer_cast(m_particles->Vels);

	thrust::copy(m_oldVels->begin(), m_oldVels->end(), vels);
	thrust::transform(vels, vels + m_number, vels, functors::mul_04());
}

void CPackageGeneratorHelperGPU::ScaleVelocitiesToRadius(double _minRadius) const
{
	const thrust::device_ptr<CVector3>     coords = thrust::device_pointer_cast(m_particles->Coords);
	const thrust::device_ptr<CVector3>     vels   = thrust::device_pointer_cast(m_particles->Vels);
	const thrust::device_ptr<const double> radii  = thrust::device_pointer_cast(m_particles->ContactRadii);

	thrust::transform(vels, vels + m_number, radii, m_deltaCoord->begin(), functors::calc_deltas(_minRadius));
	thrust::transform(coords, coords + m_number, m_deltaCoord->begin(), coords, thrust::minus<CVector3>());
}

double CPackageGeneratorHelperGPU::MaxRelativeVelocity() const
{
	const thrust::device_ptr<const CVector3> vels  = thrust::device_pointer_cast(m_particles->Vels);
	const thrust::device_ptr<const double>   radii = thrust::device_pointer_cast(m_particles->ContactRadii);

	thrust::transform(vels, vels + m_number, radii, m_relVels->begin(), functors::cals_rel_vel());
	const double maxRelVel = thrust::reduce(m_relVels->begin(), m_relVels->end(), 0.0, thrust::maximum<double>());
	return sqrt(maxRelVel);
}

void CPackageGeneratorHelperGPU::ResetMovement()
{
	const thrust::device_ptr<CVector3> coords = thrust::device_pointer_cast(m_particles->Coords);
	const thrust::device_ptr<CVector3> vels = thrust::device_pointer_cast(m_particles->Vels);

	thrust::transform(coords, coords + m_number, vels, coords, thrust::minus<CVector3>());
	thrust::fill_n(vels, m_number, CVector3{ 0 });
	thrust::fill(m_oldVels->begin(), m_oldVels->end(), CVector3{ 0 });
}

void CPackageGeneratorHelperGPU::SaveVelocities()
{
	const thrust::device_ptr<const CVector3> vels = thrust::device_pointer_cast(m_particles->Vels);
	thrust::copy(vels, vels + m_number, m_oldVels->begin());
}
