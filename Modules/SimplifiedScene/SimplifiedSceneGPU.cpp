/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "SimplifiedSceneGPU.h"

CSimplifiedSceneGPU::CSimplifiedSceneGPU(const CCUDADefines* _cudaDefines)
{
	m_gpuScene.SetCudaDefines(_cudaDefines);
}

void CSimplifiedSceneGPU::InitializeScene(CSimplifiedScene& _pScene, CSystemStructure* _pSystemStructure)
{
	// Initialize solid bonds
	SSolidBondStruct& bonds = _pScene.GetRefToSolidBonds();
	m_SolidBonds.Resize(bonds.Size());
	CUDABondsCPU2GPU(_pScene);

	// Initialize particles
	SParticleStruct& particles = _pScene.GetRefToParticles();
	m_Particles.Resize(particles.Size());

	CUDAParticlesCPU2GPU(_pScene);

	// Initialize walls
	SWallStruct& walls = _pScene.GetRefToWalls();
	m_Walls.Resize(walls.Size());
	CUDAWallsCPU2GPU(_pScene);
}

void CSimplifiedSceneGPU::ClearStates() const
{
	CUDA_MEMSET(m_Particles.Forces    , 0, sizeof(std::remove_pointer_t<decltype(m_Particles.Forces    )>) * m_Particles.nElements);
	CUDA_MEMSET(m_Particles.Moments   , 0, sizeof(std::remove_pointer_t<decltype(m_Particles.Moments   )>) * m_Particles.nElements);
	CUDA_MEMSET(m_Particles.HeatFluxes, 0, sizeof(std::remove_pointer_t<decltype(m_Particles.HeatFluxes)>) * m_Particles.nElements);
	CUDA_MEMSET(m_Walls.Forces        , 0, sizeof(std::remove_pointer_t<decltype(m_Walls.Forces        )>) * m_Walls.nElements);
}

void CSimplifiedSceneGPU::GetMaxSquaredPartDist(double* _bufMaxVelocity)
{
	m_gpuScene.GetMaxSquaredPartVerletDistance(m_Particles, _bufMaxVelocity);
}

double CSimplifiedSceneGPU::GetMaxPartVelocity()
{
	return m_gpuScene.GetMaxPartVelocity(m_Particles);
}

double CSimplifiedSceneGPU::GetMaxPartTemperature()
{
	return m_gpuScene.GetMaxPartTemperature(m_Particles);
}

void CSimplifiedSceneGPU::GetMaxWallVelocity(double* _bufMaxVelocity)
{
	m_gpuScene.GetMaxWallVelocity(m_Walls, _bufMaxVelocity);
}

size_t CSimplifiedSceneGPU::GetInactiveBondsNumber() const
{
	return m_gpuScene.GetInactiveBondsNumber(m_SolidBonds);
}

void CSimplifiedSceneGPU::GetActiveBondsNumber(unsigned* _bufNumber) const
{
	m_gpuScene.GetActiveBondsNumber(m_SolidBonds, _bufNumber);
}

void CSimplifiedSceneGPU::CUDASaveVerletCoords() const
{
	using type = std::remove_pointer_t<decltype(m_Particles.Coords)>;
	CUDA_MEMCPY_H2D(m_Particles.CoordsVerlet, m_Particles.Coords, m_Particles.nElements * sizeof(type));
}

void CSimplifiedSceneGPU::CUDABondsCPU2GPU(CSimplifiedScene& _pSceneCPU)
{
	SSolidBondStruct& bondsCPU = _pSceneCPU.GetRefToSolidBonds();
	SGPUSolidBonds bondsHost(bondsCPU.Size(), SBasicGPUStruct::EMemType::HOST);

	ParallelFor(bondsCPU.Size(), [&](size_t i)
	{
		bondsHost.Activities[i] = bondsCPU.Active(i);
		bondsHost.LeftIDs[i] = (unsigned)bondsCPU.LeftID(i);
		bondsHost.RightIDs[i] = (unsigned)bondsCPU.RightID(i);
		bondsHost.CrossCuts[i] = bondsCPU.CrossCut(i);
		bondsHost.Diameters[i] = bondsCPU.Diameter(i);
		bondsHost.InitialLengths[i] = bondsCPU.InitialLength(i);
		bondsHost.NormalStrengths[i] = bondsCPU.NormalStrength(i);
		bondsHost.TangentialStrengths[i] = bondsCPU.TangentialStrength(i);
		bondsHost.InitialIndices[i] = bondsCPU.InitIndex(i);
		bondsHost.NormalMoments[i] = bondsCPU.NormalMoment(i);
		bondsHost.TangentialMoments[i] = bondsCPU.TangentialMoment(i);
		bondsHost.TotalForces[i] = bondsCPU.TotalForce(i);
		bondsHost.AxialMoments[i] = bondsCPU.AxialMoment(i);
		bondsHost.NormalStiffnesses[i] = bondsCPU.NormalStiffness(i);
		bondsHost.TangentialStiffnesses[i] = bondsCPU.TangentialStiffness(i);
		bondsHost.TimeThermExpCoeffs[i] = bondsCPU.TimeThermExpCoeff(i);
		bondsHost.Viscosities[i] = bondsCPU.Viscosity(i);
		bondsHost.YieldStrengths[i] = bondsCPU.YieldStrength(i);
		bondsHost.NormalPlasticStrains[i] = bondsCPU.NormalPlasticStrain(i);
		bondsHost.PrevBonds[i] = bondsCPU.PrevBond(i);
		bondsHost.TangentialOverlaps[i] = bondsCPU.TangentialOverlap(i);
		bondsHost.TangentialPlasticStrains[i] = bondsCPU.TangentialPlasticStrain(i);
		bondsHost.EndActivities[i] = bondsCPU.EndActivity(i);
		if (bondsCPU.ThermalsExist())
		{
			bondsHost.ThermalConductivities[i] = bondsCPU.ThermalConductivity(i);
		}
	});

	m_SolidBonds.CopyFrom(bondsHost);
}

void CSimplifiedSceneGPU::CUDABondsGPU2CPU(CSimplifiedScene& _pSceneCPU)
{
	SSolidBondStruct& bondsCPU = _pSceneCPU.GetRefToSolidBonds();
	SGPUSolidBonds bondsHost(SBasicGPUStruct::EMemType::HOST);
	bondsHost.CopyFrom(m_SolidBonds);

	ParallelFor(bondsCPU.Size(), [&](size_t i)
	{
		bondsCPU.LeftID(i) = bondsHost.LeftIDs[i];
		bondsCPU.RightID(i) = bondsHost.RightIDs[i];
		bondsCPU.Active(i) = bondsHost.Activities[i];
		bondsCPU.CrossCut(i) = bondsHost.CrossCuts[i];
		bondsCPU.Diameter(i) = bondsHost.Diameters[i];
		bondsCPU.InitialLength(i) = bondsHost.InitialLengths[i];
		bondsCPU.NormalStrength(i) = bondsHost.NormalStrengths[i];
		bondsCPU.TangentialStrength(i) = bondsHost.TangentialStrengths[i];
		bondsCPU.InitIndex(i) = bondsHost.InitialIndices[i];
		bondsCPU.NormalMoment(i) = bondsHost.NormalMoments[i];
		bondsCPU.TangentialMoment(i) = bondsHost.TangentialMoments[i];
		bondsCPU.TotalForce(i) = bondsHost.TotalForces[i];
		bondsCPU.AxialMoment(i) = bondsHost.AxialMoments[i];
		bondsCPU.NormalStiffness(i) = bondsHost.NormalStiffnesses[i];
		bondsCPU.TangentialStiffness(i) = bondsHost.TangentialStiffnesses[i];
		bondsCPU.TimeThermExpCoeff(i) = bondsHost.TimeThermExpCoeffs[i];
		bondsCPU.Viscosity(i) = bondsHost.Viscosities[i];
		bondsCPU.YieldStrength(i) = bondsHost.YieldStrengths[i];
		bondsCPU.EndActivity(i) = bondsHost.EndActivities[i];
		bondsCPU.NormalPlasticStrain(i) = bondsHost.NormalPlasticStrains[i];
		bondsCPU.PrevBond(i) = bondsHost.PrevBonds[i];
		bondsCPU.TangentialOverlap(i) = bondsHost.TangentialOverlaps[i];
		bondsCPU.TangentialPlasticStrain(i) = bondsHost.TangentialPlasticStrains[i];
	});
}

void CSimplifiedSceneGPU::CUDABondsGPU2CPUDynamicData(CSimplifiedScene& _sceneCPU) const
{
	SSolidBondStruct& bondsCPU = _sceneCPU.GetRefToSolidBonds();
	const size_t& N = m_SolidBonds.nElements;

	// helper lambda to create a CPU vector and copy GPU array there
	auto CreateAndCopyFromGPU = [N](auto* _gpuPtr)
	{
		using T = std::remove_pointer_t<decltype(_gpuPtr)>;
		std::vector<T> result(N);
		CUDA_MEMCPY_D2H(result.data(), _gpuPtr, sizeof(T) * N);
		return result;
	};

	const auto Activities              = CreateAndCopyFromGPU(m_SolidBonds.Activities);
	const auto NormalMoment            = CreateAndCopyFromGPU(m_SolidBonds.NormalMoments);
	const auto TangentialMoment        = CreateAndCopyFromGPU(m_SolidBonds.TangentialMoments);
	const auto EndActivities           = CreateAndCopyFromGPU(m_SolidBonds.EndActivities);
	const auto NormalPlasticStrain     = CreateAndCopyFromGPU(m_SolidBonds.NormalPlasticStrains);
	const auto PrevBonds               = CreateAndCopyFromGPU(m_SolidBonds.PrevBonds);
	const auto TangentialOverlap       = CreateAndCopyFromGPU(m_SolidBonds.TangentialOverlaps);
	const auto TangentialPlasticStrain = CreateAndCopyFromGPU(m_SolidBonds.TangentialPlasticStrains);

	ParallelFor(bondsCPU.Size(), [&](size_t i)
	{
		bondsCPU.Active(i)                  = Activities[i] != 0;
		bondsCPU.NormalMoment(i)            = NormalMoment[i];
		bondsCPU.TangentialMoment(i)        = TangentialMoment[i];
		bondsCPU.EndActivity(i)             = EndActivities[i];
		bondsCPU.NormalPlasticStrain(i)     = NormalPlasticStrain[i];
		bondsCPU.PrevBond(i)                = PrevBonds[i];
		bondsCPU.TangentialOverlap(i)       = TangentialOverlap[i];
		bondsCPU.TangentialPlasticStrain(i) = TangentialPlasticStrain[i];
	});

}

void CSimplifiedSceneGPU::CUDABondsActivityGPU2CPU(CSimplifiedScene& _pSceneCPU)
{
	SSolidBondStruct& bondsCPU = _pSceneCPU.GetRefToSolidBonds();
	using activity_t = std::remove_pointer_t<decltype(m_SolidBonds.Activities)>;
	activity_t* pActivityHost;
	CUDA_MALLOC_H(&pActivityHost, sizeof(activity_t)*bondsCPU.Size());
	CUDA_MEMCPY_D2H(pActivityHost, m_SolidBonds.Activities, sizeof(activity_t)*bondsCPU.Size());
	ParallelFor(bondsCPU.Size(), [&](size_t i)
	{
		bondsCPU.Active(i) = pActivityHost[i];
	});
	CUDA_FREE_H(pActivityHost);
}

void CSimplifiedSceneGPU::CUDAParticlesCPU2GPU(CSimplifiedScene& _pSceneCPU)
{
	SParticleStruct& particlesCPU = _pSceneCPU.GetRefToParticles();
	SGPUParticles particlesHost(particlesCPU.Size(), SBasicGPUStruct::EMemType::HOST);

	ParallelFor(particlesCPU.Size(), [&](size_t i)
	{
		particlesHost.Coords[i] = particlesCPU.Coord(i);
		particlesHost.Vels[i] = particlesCPU.Vel(i);
		particlesHost.AnglVels[i] = particlesCPU.AnglVel(i);
		if (particlesCPU.QuaternionExist())
			particlesHost.Quaternions[i] = particlesCPU.Quaternion(i);
		particlesHost.Masses[i] = particlesCPU.Mass(i);
		particlesHost.Radii[i] = particlesCPU.Radius(i);
		particlesHost.ContactRadii[i] = particlesCPU.ContactRadius(i);
		particlesHost.InertiaMoments[i] = particlesCPU.InertiaMoment(i);
		particlesHost.CompoundIndices[i] = particlesCPU.CompoundIndex(i);
		particlesHost.Forces[i] = particlesCPU.Force(i);
		particlesHost.Moments[i] = particlesCPU.Moment(i);
		particlesHost.Activities[i] = particlesCPU.Active(i);
		particlesHost.StartActivities[i] = particlesCPU.StartActivity(i);
		particlesHost.EndActivities[i] = particlesCPU.EndActivity(i);
		if (particlesCPU.ThermalsExist())
		{
			particlesHost.HeatCapacities[i] = particlesCPU.HeatCapacity(i);
			particlesHost.Temperatures[i] = particlesCPU.Temperature(i);
		}
	});

	m_Particles.CopyFrom(particlesHost);
}

void CSimplifiedSceneGPU::CUDAParticlesGPU2CPUVerletData(CSimplifiedScene& _pSceneCPU)
{
	SParticleStruct& particlesCPU = _pSceneCPU.GetRefToParticles();
	static std::vector<std::remove_pointer_t<decltype(m_Particles.Activities)>> vActivity; vActivity.resize(m_Particles.nElements);
	static std::vector<std::remove_pointer_t<decltype(m_Particles.Coords    )>> vCoords;   vCoords.resize(m_Particles.nElements);
	CUDA_MEMCPY_D2H(vActivity.data(), m_Particles.Activities, sizeof(std::remove_pointer_t<decltype(m_Particles.Activities)>) * m_Particles.nElements);
	CUDA_MEMCPY_D2H(vCoords  .data(), m_Particles.Coords    , sizeof(std::remove_pointer_t<decltype(m_Particles.Coords    )>) * m_Particles.nElements);
	ParallelFor(particlesCPU.Size(), [&](size_t i)
	{
		particlesCPU.Coord(i) = vCoords[i];
		particlesCPU.Active(i) = vActivity[i] != 0;
	});
}

void CSimplifiedSceneGPU::CUDAParticlesGPU2CPUDynamicData(CSimplifiedScene& _sceneCPU) const
{
	SParticleStruct& particlesCPU = _sceneCPU.GetRefToParticles();
	const size_t& N = m_Particles.nElements;

	// helper lambda to create a CPU vector and copy GPU array there
	auto CreateAndCopyFromGPU = [N](auto* _gpuPtr)
	{
		using T = std::remove_pointer_t<decltype(_gpuPtr)>;
		std::vector<T> result(N);
		CUDA_MEMCPY_D2H(result.data(), _gpuPtr, sizeof(T) * N);
		return result;
	};

	const auto coords        = CreateAndCopyFromGPU(m_Particles.Coords);
	const auto vels          = CreateAndCopyFromGPU(m_Particles.Vels);
	const auto anglVels      = CreateAndCopyFromGPU(m_Particles.AnglVels);
	const auto activities    = CreateAndCopyFromGPU(m_Particles.Activities);
	const auto endActivities = CreateAndCopyFromGPU(m_Particles.EndActivities);

	std::vector<CQuaternion> quaternions(N);
	if (particlesCPU.QuaternionExist())
	{
		quaternions.resize(N);
		CUDA_MEMCPY_D2H(quaternions.data(), m_Particles.Quaternions, sizeof(std::remove_pointer_t<decltype(m_Particles.Quaternions)>) * N);
	}

	std::vector<double> temperature(N);
	if (particlesCPU.ThermalsExist())
	{
		temperature.resize(N);
		CUDA_MEMCPY_D2H(temperature.data(), m_Particles.Temperatures, sizeof(std::remove_pointer_t<decltype(m_Particles.Temperatures)>) * N);
	}

	ParallelFor(particlesCPU.Size(), [&](size_t i)
	{
		particlesCPU.Coord(i) = coords[i];
		particlesCPU.Vel(i) = vels[i];
		particlesCPU.AnglVel(i) = anglVels[i];
		particlesCPU.Active(i) = activities[i] != 0;
		particlesCPU.EndActivity(i) = endActivities[i];
		if (particlesCPU.QuaternionExist())
			particlesCPU.Quaternion(i) = quaternions[i];
		if (particlesCPU.ThermalsExist())
			particlesCPU.Temperature(i) = temperature[i];
	});
}

void CSimplifiedSceneGPU::CUDAParticlesGPU2CPUAllData(CSimplifiedScene& _pSceneCPU)
{
	SParticleStruct& particlesCPU = _pSceneCPU.GetRefToParticles();
	SGPUParticles particlesHost(SBasicGPUStruct::EMemType::HOST);
	particlesHost.CopyFrom(m_Particles);

	ParallelFor(particlesCPU.Size(), [&](size_t i)
	{
		particlesCPU.Coord(i) = particlesHost.Coords[i];
		particlesCPU.Vel(i) = particlesHost.Vels[i];
		if (particlesCPU.QuaternionExist())
			particlesCPU.Quaternion(i) = particlesHost.Quaternions[i];
		particlesCPU.AnglVel(i) = particlesHost.AnglVels[i];
		particlesCPU.Force(i) = particlesHost.Forces[i];
		particlesCPU.Moment(i) = particlesHost.Moments[i];
		particlesCPU.Active(i) = particlesHost.Activities[i] != 0;
		particlesCPU.StartActivity(i) = particlesHost.StartActivities[i];
		particlesCPU.EndActivity(i) = particlesHost.EndActivities[i];
		if (particlesCPU.ThermalsExist())
			particlesCPU.Temperature(i) = particlesHost.Temperatures[i];
	});
}

void CSimplifiedSceneGPU::CUDAWallsCPU2GPU(CSimplifiedScene& _pSceneCPU)
{
	const SWallStruct& wallsCPU = _pSceneCPU.GetRefToWalls();
	SGPUWalls wallsHost(wallsCPU.Size(), SBasicGPUStruct::EMemType::HOST);

	ParallelFor(wallsCPU.Size(), [&](size_t i)
	{
		wallsHost.Vertices1[i] = wallsCPU.Vert1(i);
		wallsHost.Vertices2[i] = wallsCPU.Vert2(i);
		wallsHost.Vertices3[i] = wallsCPU.Vert3(i);
		wallsHost.NormalVectors[i] = wallsCPU.NormalVector(i);
		wallsHost.MinCoords[i] = wallsCPU.MinCoord(i);
		wallsHost.MaxCoords[i] = wallsCPU.MaxCoord(i);
		wallsHost.Vels[i] = wallsCPU.Vel(i);
		wallsHost.Forces[i] = wallsCPU.Force(i);
		wallsHost.RotVels[i] = wallsCPU.RotVel(i);
		wallsHost.RotCenters[i] = wallsCPU.RotCenter(i);
		wallsHost.CompoundIndices[i] = wallsCPU.CompoundIndex(i);
	});

	m_Walls.CopyFrom(wallsHost);
}

void CSimplifiedSceneGPU::CUDAWallsGPU2CPUVerletData(CSimplifiedScene& _pSceneCPU)
{
	SWallStruct& wallsCPU = _pSceneCPU.GetRefToWalls();
	static std::vector<std::remove_pointer_t<decltype(m_Walls.Vertices1)>> vVertex1;			vVertex1.resize(m_Walls.nElements);
	static std::vector<std::remove_pointer_t<decltype(m_Walls.Vertices2)>> vVertex2;			vVertex2.resize(m_Walls.nElements);
	static std::vector<std::remove_pointer_t<decltype(m_Walls.Vertices3)>> vVertex3;			vVertex3.resize(m_Walls.nElements);
	static std::vector<std::remove_pointer_t<decltype(m_Walls.NormalVectors)>> vNormalVector;	vNormalVector.resize(m_Walls.nElements);
	static std::vector<std::remove_pointer_t<decltype(m_Walls.MinCoords)>> vMinCoord;			vMinCoord.resize(m_Walls.nElements);
	static std::vector<std::remove_pointer_t<decltype(m_Walls.MaxCoords)>> vMaxCoord;			vMaxCoord.resize(m_Walls.nElements);
	CUDA_MEMCPY_D2H(vVertex1     .data(), m_Walls.Vertices1    , sizeof(std::remove_pointer_t<decltype(m_Walls.Vertices1    )>) * m_Walls.nElements);
	CUDA_MEMCPY_D2H(vVertex2     .data(), m_Walls.Vertices2    , sizeof(std::remove_pointer_t<decltype(m_Walls.Vertices2    )>) * m_Walls.nElements);
	CUDA_MEMCPY_D2H(vVertex3     .data(), m_Walls.Vertices3    , sizeof(std::remove_pointer_t<decltype(m_Walls.Vertices3    )>) * m_Walls.nElements);
	CUDA_MEMCPY_D2H(vNormalVector.data(), m_Walls.NormalVectors, sizeof(std::remove_pointer_t<decltype(m_Walls.NormalVectors)>) * m_Walls.nElements);
	CUDA_MEMCPY_D2H(vMinCoord    .data(), m_Walls.MinCoords    , sizeof(std::remove_pointer_t<decltype(m_Walls.MinCoords    )>) * m_Walls.nElements);
	CUDA_MEMCPY_D2H(vMaxCoord    .data(), m_Walls.MaxCoords    , sizeof(std::remove_pointer_t<decltype(m_Walls.MaxCoords    )>) * m_Walls.nElements);
	ParallelFor(wallsCPU.Size(), [&](size_t i)
	{
		wallsCPU.Vert1(i) = vVertex1[i];
		wallsCPU.Vert2(i) = vVertex2[i];
		wallsCPU.Vert3(i) = vVertex3[i];
		wallsCPU.NormalVector(i) = vNormalVector[i];
		wallsCPU.MinCoord(i) = vMinCoord[i];
		wallsCPU.MaxCoord(i) = vMaxCoord[i];
	});
}

void CSimplifiedSceneGPU::CUDAWallsGPU2CPUAllData(CSimplifiedScene& _pSceneCPU)
{
	SWallStruct& wallsCPU = _pSceneCPU.GetRefToWalls();
	SGPUWalls wallsHost(SBasicGPUStruct::EMemType::HOST);
	wallsHost.CopyFrom(m_Walls);

	ParallelFor(wallsCPU.Size(), [&](size_t i)
	{
		wallsCPU.Vert1(i) = wallsHost.Vertices1[i];
		wallsCPU.Vert2(i) = wallsHost.Vertices2[i];
		wallsCPU.Vert3(i) = wallsHost.Vertices3[i];
		wallsCPU.NormalVector(i) = wallsHost.NormalVectors[i];
		wallsCPU.MinCoord(i) = wallsHost.MinCoords[i];
		wallsCPU.MaxCoord(i) = wallsHost.MaxCoords[i];
		wallsCPU.Vel(i) = wallsHost.Vels[i];
		wallsCPU.Force(i) = wallsHost.Forces[i];
		wallsCPU.RotVel(i) = wallsHost.RotVels[i];
		wallsCPU.RotCenter(i) = wallsHost.RotCenters[i];
	});
}
