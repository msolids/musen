/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "PackageGeneratorHelperCPU.h"
#include "ThreadPool.h"

CPackageGeneratorHelperCPU::CPackageGeneratorHelperCPU(SParticleStruct* _particles) :
	m_number{ _particles->Size() },
	m_particles{ _particles }
{
	m_oldVels.resize(m_number, CVector3{ 0.0 });
}

void CPackageGeneratorHelperCPU::LimitVelocities() const
{
	for (size_t i = 0; i < m_number; ++i)
		m_particles->Vel(i) = 0.4 * m_oldVels[i];
}

void CPackageGeneratorHelperCPU::ScaleVelocitiesToRadius(double _minRadius) const
{
	ParallelFor(m_number, [&](size_t i)
	{
		m_particles->Coord(i) -= m_particles->Vel(i);
		m_particles->Vel(i)   *= _minRadius / m_particles->ContactRadius(i);
		m_particles->Coord(i) += m_particles->Vel(i);
	});
}

double CPackageGeneratorHelperCPU::MaxRelativeVelocity() const
{
	double maxRelVel;
	for (size_t i = 0; i < m_number; ++i)
		maxRelVel = std::max(maxRelVel, m_particles->Vel(i).SquaredLength() / (m_particles->ContactRadius(i)*m_particles->ContactRadius(i)));
	return sqrt(maxRelVel);
}

void CPackageGeneratorHelperCPU::ResetMovement()
{
	ParallelFor(m_number, [&](size_t i)
	{
		m_particles->Coord(i) -= m_particles->Vel(i);
		m_particles->Vel(i).Init(0);
		m_oldVels[i].Init(0);
	});
}

void CPackageGeneratorHelperCPU::SaveVelocities()
{
	for (size_t i = 0; i < m_number; ++i)
		m_oldVels[i] = m_particles->Vel(i);
}