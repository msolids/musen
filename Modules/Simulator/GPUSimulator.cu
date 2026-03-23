/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   Copyright (c) 2026, DyssolTEC GmbH.
   All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
   See LICENSE file for license and warranty information. */

#include "GPUSimulator.cuh"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
PRAGMA_WARNING_POP

CGPU::CGPU(const CCUDADefines* _cudaDefines) :
	m_cudaDefines{ _cudaDefines }
{
	CUDAKernels::SetThreadsNumber(m_cudaDefines->CUDA_THREADS_PER_BLOCK);
}

void CGPU::SetExternalAccel(const CVector3& _acceleration)
{
	CUDAKernels::SetExternalAccel(_acceleration);
}

void CGPU::SetSimulationDomain(const SVolumeType& _domain)
{
	CUDAKernels::SetSimulationDomain(_domain);
}

void CGPU::SetPBC(const SPBC& _PBCInfo)
{
	m_PBCEnabled = _PBCInfo.bEnabled;
	CUDAKernels::SetPBC(_PBCInfo);
}

void CGPU::SetCompoundsNumber(size_t _nCompounds)
{
	CUDAKernels::SetCompoundsNumber(_nCompounds);
}

void CGPU::InitializeWalls(const std::vector<std::vector<unsigned>>& _vvWallsInGeom, const std::vector<std::vector<unsigned>>& _adjacentWalls)
{
	/// set walls in geometries
	m_vvWallsInGeom.resize(_vvWallsInGeom.size());
	for (size_t i = 0; i < _vvWallsInGeom.size(); ++i)
	{
		// NOTE: conventional assignment copy leads to warnings in debug
		m_vvWallsInGeom[i].resize(_vvWallsInGeom[i].size());
		for (size_t j = 0; j < _vvWallsInGeom[i].size(); ++j)
			m_vvWallsInGeom[i][j] = _vvWallsInGeom[i][j];
	}

	/// set adjacent walls
	h_vec_u hostStartIndices;
	h_vec_u hostAdjacentWalls;

	size_t number = 0;	// total number of elements in the matrix
	for (const auto& list : _adjacentWalls)
		number += list.size();

	hostAdjacentWalls.resize(number);
	hostStartIndices.resize(_adjacentWalls.size() + 1);

	if (!hostStartIndices.empty()) hostStartIndices.front() = 0;			// for easier access
	for (size_t i = 1; i < _adjacentWalls.size(); ++i)
		hostStartIndices[i] = hostStartIndices[i - 1] + (unsigned)_adjacentWalls[i - 1].size();
	if (!hostStartIndices.empty()) hostStartIndices.back() = (unsigned)number - 1;	// for easier access

	ParallelFor(_adjacentWalls.size(), [&](size_t i)
	{
		std::copy(_adjacentWalls[i].begin(), _adjacentWalls[i].end(), hostAdjacentWalls.begin() + hostStartIndices[i]);
	});

	m_adjacentWalls.startIndices = hostStartIndices;
	m_adjacentWalls.adjacentWalls = hostAdjacentWalls;
}

void CGPU::InitializeCollisions()
{
	m_CollisionsPP.vVerletDst.clear();
	m_CollisionsPP.vVerletPartInd.clear();
	m_CollisionsPP.vVerletSrc.clear();
	m_CollisionsPP.collisions.Clear();

	m_CollisionsPW.vVerletDst.clear();
	m_CollisionsPW.vVerletPartInd.clear();
	m_CollisionsPW.vVerletSrc.clear();
	m_CollisionsPW.collisions.Clear();
}


void CGPU::Flags2IndicesList(const size_t _size, bool _flags[], d_vec_u& _sequence, d_vec_i8& _storage, unsigned* _listLength, unsigned _list[])
{
	if (!_size)	return;

	// sequence [ 0; _size - 1 ]
	if (_sequence.size() != _size)
	{
		_sequence.resize(_size);
		thrust::sequence(_sequence.begin(), _sequence.end());
	}
	// Determine temporary device storage requirements
	void *pTempStorage = nullptr;
	size_t nTempStorageSize = 0;
	CUDA_CUB_FLAGGED(pTempStorage, nTempStorageSize, _sequence.data().get(), _flags, _list, _listLength, _size);
	// Allocate temporary storage
	if (_storage.size() < nTempStorageSize)
		_storage.resize(nTempStorageSize);
	// Run selection
	CUDA_CUB_FLAGGED(_storage.data().get(), nTempStorageSize, _sequence.data().get(), _flags, _list, _listLength, _size);
}

void CGPU::ApplyExternalAcceleration(SGPUParticles& _particles)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::ApplyExternalAcceleration_kernel, static_cast<unsigned>(_particles.nElements), _particles.Masses, _particles.Forces);
}

double CGPU::CalculateNewTimeStep(double _currTimeStep, double _initTimeStep, double _partMoveLimit, double _timeStepFactor, SGPUParticles& _particles) const
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::GatherForFlexibleTimeStep_kernel, static_cast<unsigned>(_particles.nElements),
		_particles.Masses, _particles.Forces, _particles.TempDouble1);

	static d_vec_d maxTimeStep;
	if (maxTimeStep.empty())
		maxTimeStep.resize(1);
	CUDA_REDUCE_CALLER(CUDAKernels::ReduceMin_kernel, _particles.nElements, _particles.TempDouble1, _particles.TempDouble2, maxTimeStep.data().get());

	double maxStep;
	CUDA_MEMCPY_D2H(&maxStep, maxTimeStep.data().get(), sizeof(double));
	maxStep = std::sqrt(std::sqrt(maxStep) * _partMoveLimit);

	if (_currTimeStep > maxStep)
		return maxStep;
	if (_currTimeStep < _initTimeStep)
		return std::min(_currTimeStep * _timeStepFactor, _initTimeStep);
	return _currTimeStep;
}

void CGPU::MoveParticles(double _timeStep, bool _anisotropy, double _partVelocityLimit, const SGPUParticles& _particles)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CalculateParticlesVelocity_kernel, _timeStep, static_cast<unsigned>(_particles.nElements),
		_particles.Masses, _particles.Forces, _particles.Vels);
	if (_partVelocityLimit > 0)
	{
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::LimitParticlesVelocity_kernel, static_cast<unsigned>(_particles.nElements), _partVelocityLimit, _particles.Vels);
	}
	if (!_anisotropy)
	{
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CalculateParticlesAngVelocity_kernel, _timeStep, static_cast<unsigned>(_particles.nElements),
			_particles.InertiaMoments, _particles.Moments, _particles.AnglVels);
	}
	else
	{
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CalculateParticlesAngVelocityWithAnisotropy_kernel, _timeStep, static_cast<unsigned>(_particles.nElements),
			_particles.InertiaMoments, _particles.Moments, _particles.Quaternions, _particles.AnglVels);
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CalculateParticlesOrientation_kernel, _timeStep, static_cast<unsigned>(_particles.nElements),
			_particles.AnglVels, _particles.Quaternions);
	}
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CalculateParticlesCoordinate_kernel, _timeStep, static_cast<unsigned>(_particles.nElements),
		_particles.Vels, _particles.Coords);
}

void CGPU::MoveParticlesPrediction(double _timeStep, bool _anisotropy, double _partVelocityLimit, const SGPUParticles& _particles)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CalculateParticlesVelocity_kernel, _timeStep, static_cast<unsigned>(_particles.nElements),
		_particles.Masses, _particles.Forces, _particles.Vels);
	if (_partVelocityLimit > 0)
	{
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::LimitParticlesVelocity_kernel, static_cast<unsigned>(_particles.nElements), _partVelocityLimit, _particles.Vels);
	}
	if (!_anisotropy)
	{
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CalculateParticlesAngVelocity_kernel, _timeStep, static_cast<unsigned>(_particles.nElements),
			_particles.InertiaMoments, _particles.Moments, _particles.AnglVels);
	}
	else
	{
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CalculateParticlesAngVelocityWithAnisotropy_kernel, _timeStep, static_cast<unsigned>(_particles.nElements),
			_particles.InertiaMoments, _particles.Moments, _particles.Quaternions, _particles.AnglVels);
	}
}

void CGPU::CalculateTotalForceOnWall(size_t _iGeom, SGPUWalls& _walls, d_vec_v3& _vTotalForce)
{
	static d_vec_v3 forces, temp;
	if (forces.size() != m_vvWallsInGeom[_iGeom].size())
		forces.resize(m_vvWallsInGeom[_iGeom].size());
	if (temp.size() != m_vvWallsInGeom[_iGeom].size())
		temp.resize(m_vvWallsInGeom[_iGeom].size());
	//calculate force
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::GatherForcesFromWalls_kernel, static_cast<unsigned>(m_vvWallsInGeom[_iGeom].size()),
		m_vvWallsInGeom[_iGeom].data().get(), _walls.Forces, forces.data().get());
	CUDA_REDUCE_CALLER(CUDAKernels::ReduceSum_kernel, m_vvWallsInGeom[_iGeom].size(), forces.data().get(), temp.data().get(), _vTotalForce.data().get());
}

CVector3 CGPU::CalculateTotalForceOnWall(size_t _iGeom, SGPUWalls & _walls)
{
	static d_vec_v3 vTotalForce(1);
	CVector3 vResult;
	CalculateTotalForceOnWall(_iGeom, _walls, vTotalForce);
	CUDA_MEMCPY_D2H(&vResult, vTotalForce.data().get(), sizeof(CVector3));
	return vResult;
}

void CGPU::MoveWalls(double _timeStep, size_t _iGeom, const CVector3& _vel, const CVector3& _rotVel, const CVector3& _rotCenter, const CMatrix3& _rotMatrix,
	const CVector3& _freeMotion, bool _isForceDependentMotion, bool _isRotateAroundCenter, double _mass, SGPUWalls& _walls, const CVector3& _externalAccel)
{
	const unsigned wallsInGeom = static_cast<unsigned>(m_vvWallsInGeom[_iGeom].size());

	static d_vec_v3 totalForce(1);
	static d_vec_v3 rotCenter(1); // used in case when rotation around center is defined

	if (_isRotateAroundCenter || _isForceDependentMotion || !_freeMotion.IsZero())
		CalculateTotalForceOnWall(_iGeom, _walls, totalForce);
	if (_isRotateAroundCenter) // precalculate rotation center
	{
		d_vec_d tempAreas(wallsInGeom);
		d_vec_v3 tempWeightedCentroids(wallsInGeom);
		d_vec_d temp_d(wallsInGeom);
		d_vec_v3 temp_v3(wallsInGeom);
		d_vec_d totalArea(1);
		d_vec_v3 totalWeightedCentroid(1);
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::PrecalculateGeometryCenter_kernel, wallsInGeom, m_vvWallsInGeom[_iGeom].data().get(),
			_walls.Vertices1, _walls.Vertices2, _walls.Vertices3, tempAreas.data().get(), tempWeightedCentroids.data().get());
		CUDA_REDUCE_CALLER(CUDAKernels::ReduceSum_kernel, wallsInGeom, tempAreas.data().get(), temp_d.data().get(), totalArea.data().get());
		CUDA_REDUCE_CALLER(CUDAKernels::ReduceSum_kernel, wallsInGeom, tempWeightedCentroids.data().get(), temp_v3.data().get(), totalWeightedCentroid.data().get());
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CalculateGeometryCenter_kernel, totalArea.data().get(), totalWeightedCentroid.data().get(), rotCenter.data().get());
	}
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::MoveWalls_kernel, _timeStep,
		static_cast<unsigned>(m_vvWallsInGeom[_iGeom].size()), _vel, _rotVel, _rotCenter, _rotMatrix,
		_freeMotion, totalForce.data().get(), _mass, _isRotateAroundCenter, _externalAccel,
		rotCenter.data().get(), m_vvWallsInGeom[_iGeom].data().get(),
		_walls.Vertices1, _walls.Vertices2, _walls.Vertices3, _walls.MinCoords,
		_walls.MaxCoords, _walls.NormalVectors, _walls.Vels, _walls.RotCenters, _walls.RotVels);
}

void CGPU::UpdateTemperatures(double _currTimeStep, SGPUParticles& _particles)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::UpdateTemperatures_kernel, _currTimeStep, static_cast<unsigned>(_particles.nElements), _particles.HeatCapacities,
		_particles.Masses, _particles.HeatFluxes, _particles.Temperatures);
}

void CGPU::UpdateVerletLists(bool _bPPVerlet, const SGPUParticles& _particles, const SGPUWalls& _walls, const h_vec_u& _vVerListSrcNew, const h_vec_u& _vVerListDstNew,
	const h_vec_u& _vVerListIndNew, const h_vec_u8& _vVirtShifts, d_vec_u& _vVerListSrcOld, d_vec_u& _vVerListDstOld, d_vec_u& _vVerListIndOld, SGPUCollisions& _collisions) const
{
	const d_vec_u dvVerlSrcNew(_vVerListSrcNew);
	const d_vec_u dvVerlDstNew(_vVerListDstNew);
	const d_vec_u dvVerlIndNew(_vVerListIndNew);
	const size_t collNum = dvVerlDstNew.size();
	static SGPUCollisions newCollisions;
	newCollisions.Resize(collNum);

	CUDA_MEMSET(newCollisions.TangOverlaps,	  0, collNum * sizeof(*newCollisions.TangOverlaps));
	CUDA_MEMSET(newCollisions.TotalForces,	  0, collNum * sizeof(*newCollisions.TotalForces));
	CUDA_MEMSET(newCollisions.NormalOverlaps, 0, collNum * sizeof(*newCollisions.NormalOverlaps));
	CUDA_MEMSET(newCollisions.ActivityFlags,  0, collNum * sizeof(*newCollisions.ActivityFlags));
	CUDA_MEMSET(newCollisions.InitNormalOverlaps, 0, collNum * sizeof(*newCollisions.InitNormalOverlaps));

	if (m_PBCEnabled)
		CUDA_MEMCPY_H2D(newCollisions.VirtualShifts, _vVirtShifts.data(), collNum * sizeof(*newCollisions.VirtualShifts));
	else
		CUDA_MEMSET(newCollisions.VirtualShifts, 0, collNum * sizeof(*newCollisions.VirtualShifts));

	if (_bPPVerlet)
	{
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::InitializePPCollisions_kernel, static_cast<unsigned>(collNum), dvVerlSrcNew.data().get(), dvVerlDstNew.data().get(),
			_particles.ContactRadii, _particles.Masses, _particles.CompoundIndices,
			newCollisions.SrcIDs, newCollisions.DstIDs, newCollisions.EquivMasses, newCollisions.EquivRadii, newCollisions.SumRadii, newCollisions.InteractPropIDs);
		if (!_vVerListDstOld.empty())
			CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CopyCollisionsPP_kernel,
				static_cast<unsigned>(_vVerListDstOld.size()),
				_vVerListSrcOld.data().get(), _vVerListDstOld.data().get(), dvVerlDstNew.data().get(), dvVerlIndNew.data().get(), _collisions.ActivityFlags,
				_collisions.NormalOverlaps, _collisions.InitNormalOverlaps, _collisions.TangOverlaps, _collisions.ContactVectors, _collisions.TotalForces,
				newCollisions.NormalOverlaps, newCollisions.InitNormalOverlaps, newCollisions.TangOverlaps, newCollisions.ContactVectors, newCollisions.TotalForces);
	}
	else
	{
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::InitializePWCollisions_kernel, static_cast<unsigned>(collNum), dvVerlSrcNew.data().get(), dvVerlDstNew.data().get(),
			_particles.CompoundIndices, _walls.CompoundIndices,
			newCollisions.SrcIDs, newCollisions.DstIDs, newCollisions.InteractPropIDs);
		if (!_vVerListDstOld.empty())
			CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CopyCollisionsPW_kernel,
				static_cast<unsigned>(_vVerListDstOld.size()),
				_vVerListSrcOld.data().get(), _vVerListDstOld.data().get(), dvVerlDstNew.data().get(), dvVerlIndNew.data().get(), _collisions.ActivityFlags,
				_collisions.NormalOverlaps, _collisions.TangOverlaps, _collisions.ContactVectors, _collisions.TotalForces,
				newCollisions.ActivityFlags, newCollisions.NormalOverlaps, newCollisions.TangOverlaps, newCollisions.ContactVectors, newCollisions.TotalForces);
	}

	_collisions.CopyFrom(newCollisions);

	_vVerListDstOld = dvVerlDstNew;
	_vVerListIndOld = dvVerlIndNew;
	_vVerListSrcOld = dvVerlSrcNew;
}

void CGPU::SortByDst(unsigned _nPart, const d_vec_u& _vVerListSrc, const d_vec_u& _vVerListDst, d_vec_u& _vVerCollInd_DstSorted, d_vec_u& _vVerPartInd_DstSorted) const
{
	unsigned nCollisions = (unsigned)_vVerListSrc.size();
	static d_vec_u vVerListDstTemp, vTemp;
	vVerListDstTemp = _vVerListDst;
	_vVerCollInd_DstSorted.resize(nCollisions);
	_vVerPartInd_DstSorted.resize(_nPart + 1);
	vTemp.resize(_nPart + 1);
	thrust::fill(vTemp.begin(), vTemp.end(), nCollisions + 1); // fill initially with impossible values to indicate later what was not filled

	thrust::sequence(thrust::device, _vVerCollInd_DstSorted.begin(), _vVerCollInd_DstSorted.end());
	thrust::sort_by_key(thrust::device, vVerListDstTemp.begin(), vVerListDstTemp.end(), _vVerCollInd_DstSorted.begin());

	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::FillUniqueIndexes_kernel, nCollisions, vVerListDstTemp.data().get(), vTemp.data().get());
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::FillNonExistendIndexes_kernel, _nPart, nCollisions, vTemp.data().get(), _vVerPartInd_DstSorted.data().get());
}

void CGPU::UpdateActiveCollisionsPP(const SGPUParticles& _particles)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::UpdateActiveCollisionsPP_kernel, static_cast<unsigned>(m_CollisionsPP.vVerletSrc.size()), m_CollisionsPP.vVerletSrc.data().get(), m_CollisionsPP.vVerletDst.data().get(),
		_particles.Coords, m_CollisionsPP.collisions.VirtualShifts, m_CollisionsPP.collisions.SumRadii, m_CollisionsPP.collisions.ActivityFlags,
		m_CollisionsPP.collisions.NormalOverlaps, m_CollisionsPP.collisions.InitNormalOverlaps,
		m_CollisionsPP.collisions.ContactVectors, m_CollisionsPP.collisions.TangOverlaps);

	static d_vec_u sequence;			// temporal vector for indices needed internally in Flags2IndicesList
	static d_vec_i8 tempStorage;		// temporal storage needed internally in Flags2IndicesList
	Flags2IndicesList(static_cast<unsigned>(m_CollisionsPP.vVerletSrc.size()), m_CollisionsPP.collisions.ActivityFlags, sequence, tempStorage, m_CollisionsPP.collisions.ActiveCollisionsNum, m_CollisionsPP.collisions.ActivityIndices);
}

void CGPU::UpdateActiveCollisionsPW(const SGPUParticles& _particles, const SGPUWalls& _walls)
{
	static d_vec_IT vTempIntersectType;

	static d_vec_b vActivePart;
	static d_vec_u vActivePartIndexes;
	static d_vec_u nActivePartIndexesNumber(1);
	vTempIntersectType.resize(m_CollisionsPW.vVerletDst.size());
	vActivePart.resize(_particles.nElements);
	vActivePartIndexes.resize(_particles.nElements);
	thrust::fill(thrust::device, vActivePart.begin(), vActivePart.end(), false);

	const unsigned nCollisions = static_cast<unsigned>(m_CollisionsPW.vVerletDst.size());

	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::GetIntersectTypePW_kernel, nCollisions,
		m_CollisionsPW.vVerletSrc.data().get(), m_CollisionsPW.vVerletDst.data().get(),
		_particles.ContactRadii, _particles.Coords, _walls.Vertices1, _walls.Vertices2, _walls.Vertices3, _walls.MinCoords, _walls.MaxCoords, _walls.NormalVectors,
		m_CollisionsPW.collisions.VirtualShifts, vTempIntersectType.data().get(), m_CollisionsPW.collisions.ContactVectors, vActivePart.data().get());

	static d_vec_u sequence;
	static d_vec_i8 tempStorage;
	Flags2IndicesList(vActivePart.size(), vActivePart.data().get(), sequence, tempStorage, nActivePartIndexesNumber.data().get(), vActivePartIndexes.data().get());

	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CombineIntersectionsPW_kernel,
		nActivePartIndexesNumber.data().get(), vActivePartIndexes.data().get(),
		static_cast<unsigned>(m_CollisionsPW.vVerletPartInd.size()), nCollisions,
		m_CollisionsPW.vVerletDst.data().get(), m_CollisionsPW.vVerletPartInd.data().get(), _walls.NormalVectors,
		vTempIntersectType.data().get(), m_CollisionsPW.collisions.VirtualShifts);

	// treat contact transition between adjacent triangles
	static d_vec_b collActivated, collDeactivated;
	collActivated.resize(nCollisions);
	collDeactivated.resize(nCollisions);
	thrust::fill(thrust::device, collActivated.begin(), collActivated.end(), false);
	thrust::fill(thrust::device, collDeactivated.begin(), collDeactivated.end(), false);
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::UpdateActiveCollisionsPW_kernel, nCollisions, vTempIntersectType.data().get(),
		m_CollisionsPW.collisions.ActivityFlags, collActivated.data().get(), collDeactivated.data().get());
	static d_vec_u sequence2;
	static d_vec_i8 tempStorage2;
	static d_vec_u activatedCollIndices;
	static d_vec_u nActivatedColls(1);
	activatedCollIndices.resize(nCollisions);
	Flags2IndicesList(nCollisions, collActivated.data().get(), sequence2, tempStorage2, nActivatedColls.data().get(), activatedCollIndices.data().get());
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CopyCollisionsForAdjacentWalls,
		nActivatedColls.data().get(), activatedCollIndices.data().get(), collDeactivated.data().get(),
		m_CollisionsPW.vVerletSrc.data().get(), m_CollisionsPW.vVerletDst.data().get(), m_CollisionsPW.vVerletPartInd.data().get(),
		m_adjacentWalls.adjacentWalls.data().get(), m_adjacentWalls.startIndices.data().get(),
		m_CollisionsPW.collisions.TangOverlaps);

	Flags2IndicesList(nCollisions, m_CollisionsPW.collisions.ActivityFlags, sequence2, tempStorage2, m_CollisionsPW.collisions.ActiveCollisionsNum, m_CollisionsPW.collisions.ActivityIndices);
}

void CGPU::CheckParticlesInDomain(const double _currTime, const SGPUParticles& _particles, unsigned* _bufActivePartsNum) const
{
	if (!_particles.nElements)
	{
		unsigned nTemp = static_cast<unsigned>(_particles.nElements);
		CUDA_MEMCPY_H2D(_bufActivePartsNum, &nTemp, sizeof(unsigned));
		return;
	}

	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CheckParticlesInDomain_kernel, _currTime, static_cast<unsigned>(_particles.nElements),
		_particles.Activities, _particles.EndActivities, _particles.Coords);
	CUDA_REDUCE_CALLER(CUDAKernels::ReduceSum_kernel, _particles.nElements, _particles.Activities, _particles.TempUInt, _bufActivePartsNum);
}

void CGPU::CheckBondsActivity(const double _currTime, const SGPUParticles& _particles, SGPUSolidBonds& _bonds)
{
	if (!_particles.nElements)
		return;

	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CheckBondsActivity_kernel, _currTime, static_cast<unsigned>(_bonds.nElements),
		_particles.Activities, _bonds.Activities, _bonds.LeftIDs, _bonds.RightIDs, _bonds.EndActivities);
}

void CGPU::MoveParticlesOverPBC(const SGPUParticles& _particles)
{
	static d_vec_u8 vCrossingShifts;	// shifts for particles, which crossed PBC boundaries
	static d_vec_b vCrossingFlags;		// indicates that particle crossed PBC boundaries
	vCrossingShifts.resize(_particles.nElements);
	vCrossingFlags.resize(_particles.nElements);
	thrust::fill(thrust::device, vCrossingShifts.begin(), vCrossingShifts.end(), 0);
	thrust::fill(thrust::device, vCrossingFlags.begin(), vCrossingFlags.end(), false);

	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::MoveVirtualParticlesBox, static_cast<unsigned>(_particles.nElements), _particles.Activities,
		_particles.Coords, _particles.CoordsVerlet, vCrossingShifts.data().get(), vCrossingFlags.data().get());

	// turn crossing shifts flags to particles' indices
	static d_vec_u sequence;			// temporal vector for indices needed internally in Flags2IndicesList
	static d_vec_i8 tempStorage;		// temporal storage needed internally in Flags2IndicesList
	static d_vec_u nCrossed(1);			// [0] - number of crossed particles
	static d_vec_u dvCrossedIndices;	// indices of crossed particles
	dvCrossedIndices.resize(_particles.nElements);
	Flags2IndicesList(static_cast<unsigned>(_particles.nElements), vCrossingFlags.data().get(), sequence, tempStorage, nCrossed.data().get(), dvCrossedIndices.data().get());

	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::AddShiftsToCollisions, static_cast<unsigned>(_particles.nElements), nCrossed.data().get(), dvCrossedIndices.data().get(), vCrossingShifts.data().get(),
		m_CollisionsPP.vVerletPartInd.data().get(), m_CollisionsPP.vVerletPartInd_DstSorted.data().get(), m_CollisionsPP.vVerletCollInd_DstSorted.data().get(),
		static_cast<unsigned>(m_CollisionsPP.vVerletSrc.size()), m_CollisionsPP.collisions.SrcIDs, m_CollisionsPP.collisions.DstIDs,		m_CollisionsPP.collisions.VirtualShifts);
}

void CGPU::CopyCollisionsGPU2CPU(SGPUCollisions& _PPCollisionsHost, SGPUCollisions& _PWCollisionsHost) const
{
	_PPCollisionsHost.CopyFrom(m_CollisionsPP.collisions);
	_PWCollisionsHost.CopyFrom(m_CollisionsPW.collisions);
}

void CGPU::GetOverlapsInfo(const SGPUParticles& _particles, size_t _maxParticleID, double& _maxOverlap, double& _avrOverlap) const
{
	const unsigned collNumberPP = (unsigned)m_CollisionsPP.collisions.nElements;
	const unsigned collNumberPW = (unsigned)m_CollisionsPW.collisions.nElements;

	static d_vec_d overlapsPP, overlapsPW, tempPP, tempPW;
	static d_vec_u8 flagsPP, flagsPW;
	overlapsPP.resize(collNumberPP);
	overlapsPW.resize(collNumberPW);
	tempPP.resize(collNumberPP);
	tempPW.resize(collNumberPW);
	flagsPP.resize(collNumberPP);
	flagsPW.resize(collNumberPW);
	thrust::fill(overlapsPP.begin(), overlapsPP.end(), 0.0);
	thrust::fill(overlapsPW.begin(), overlapsPW.end(), 0.0);
	thrust::fill(flagsPP.begin(), flagsPP.end(), 0);
	thrust::fill(flagsPW.begin(), flagsPW.end(), 0);
	d_vec_d res(4, 0); // {maxPP, maxPW, sumPP, sumPW}
	size_t numberPP{ 0 }, numberPW{ 0 };

	// for PP collisions
	if (collNumberPP)
	{
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::GetPPOverlaps_kernel,
			m_CollisionsPP.collisions.ActiveCollisionsNum, m_CollisionsPP.collisions.ActivityIndices, m_CollisionsPP.collisions.SrcIDs, m_CollisionsPP.collisions.DstIDs,
			m_CollisionsPP.collisions.NormalOverlaps, (unsigned)_maxParticleID,
			overlapsPP.data().get(), flagsPP.data().get());
		CUDA_REDUCE_CALLER(CUDAKernels::ReduceMax_kernel, collNumberPP, overlapsPP.data().get(), tempPP.data().get(), thrust::device_pointer_cast(&res[0]).get());
		CUDA_REDUCE_CALLER(CUDAKernels::ReduceSum_kernel, collNumberPP, overlapsPP.data().get(), tempPP.data().get(), thrust::device_pointer_cast(&res[2]).get());
		numberPP = thrust::count(flagsPP.begin(), flagsPP.end(), size_t(1));
	}

	// for PW collisions
	if (collNumberPW)
	{
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::GetPWOverlaps_kernel,
			m_CollisionsPW.collisions.ActiveCollisionsNum, m_CollisionsPW.collisions.ActivityIndices, m_CollisionsPW.collisions.DstIDs,
			m_CollisionsPW.collisions.VirtualShifts, m_CollisionsPW.collisions.ContactVectors,
			_particles.Coords, _particles.ContactRadii,
			(unsigned)_maxParticleID, overlapsPW.data().get(), flagsPW.data().get());

		CUDA_REDUCE_CALLER(CUDAKernels::ReduceMax_kernel, collNumberPW, overlapsPW.data().get(), tempPW.data().get(), thrust::device_pointer_cast(&res[1]).get());
		CUDA_REDUCE_CALLER(CUDAKernels::ReduceSum_kernel, collNumberPW, overlapsPW.data().get(), tempPW.data().get(), thrust::device_pointer_cast(&res[3]).get());
		numberPW = thrust::count(flagsPW.begin(), flagsPW.end(), size_t(1));
	}

	// copy to CPU
	h_vec_d cpu_res = res;
	// calculate results
	_maxOverlap = std::max(cpu_res[0], cpu_res[1]);
	_avrOverlap = numberPP + numberPW ? (cpu_res[2] + cpu_res[3]) / (numberPP + numberPW) : 0;
}

/*
 * ================================================
 * Content of a former GPUSimulator.cpp file,
 * moved to prevent Thrust leaking into .cpp files.
 */

#include "GPUSimulator.h"
#include "SimplifiedSceneGPU.h"

struct CGPUSimulator::Impl
{
	typedef std::vector<std::vector<unsigned>> std_matr_u;
	typedef std::vector<std::vector<uint8_t>> std_matr_u8;

	struct STempStorage
	{
		thrust::host_vector<unsigned> hvVerletPartInd;
		thrust::host_vector<unsigned> hvVerletDst;
		thrust::host_vector<unsigned> hvVerletSrc;
		thrust::host_vector<uint8_t> hvVirtShifts;
	};

	struct SDispatchedResults
	{
		unsigned nActivePartNum;
		unsigned activeBondsNumBeforeBreak;
		unsigned activeBondsNumAfterBreak;
		double dMaxSquaredPartDist;
		double dMaxWallVel;
	};

	CCUDADefines* cudaDefines{ new CCUDADefines{} };
	CGPU gpu{ cudaDefines };
	CSimplifiedSceneGPU sceneGPU{ cudaDefines };
	SInteractProps* pInteractProps{ nullptr };
	SDispatchedResults* pDispatchedResults_d{ nullptr };
	SDispatchedResults* pDispatchedResults_h{ nullptr };
	STempStorage storeGen;             // reused in GenerateNewObjects
	STempStorage storePP, storePW;     // reused in UpdateVerletLists

	~Impl()
	{
		if (pInteractProps)
			CUDA_FREE_D(pInteractProps);
		CUDA_FREE_D(pDispatchedResults_d);
		CUDA_FREE_H(pDispatchedResults_h);
		delete cudaDefines;
	}
};

CGPUSimulator::CGPUSimulator()
	: m_impl{ new Impl }
{
	Construct();
}

CGPUSimulator::CGPUSimulator(const CBaseSimulator& _other) :
	CBaseSimulator{ _other },
	m_impl{ new Impl }
{
	Construct();
}

CGPUSimulator::~CGPUSimulator()
{
	delete m_impl;
}

void CGPUSimulator::Construct()
{
	CGPU::SetExternalAccel(m_externalAcceleration);
	m_impl->gpu.SetPBC(m_scene.m_PBC);

	CUDA_MALLOC_D(&m_impl->pDispatchedResults_d, sizeof(Impl::SDispatchedResults));
	CUDA_MALLOC_H(&m_impl->pDispatchedResults_h, sizeof(Impl::SDispatchedResults));
}

void CGPUSimulator::SetExternalAccel(const CVector3& _accel)
{
	CBaseSimulator::SetExternalAccel(_accel);
	CGPU::SetExternalAccel(m_externalAcceleration);
}

void CGPUSimulator::Initialize()
{
	CBaseSimulator::Initialize();

	CGPU::SetSimulationDomain(m_pSystemStructure->GetSimulationDomain());

	// Initialize scene, PBC, models on GPU
	m_impl->sceneGPU.InitializeScene(m_scene, m_pSystemStructure);

	// store particle coordinates
	m_impl->sceneGPU.CUDASaveVerletCoords();

	m_impl->gpu.SetPBC(m_scene.m_PBC);

	CUDAInitializeWalls();
	m_impl->gpu.InitializeCollisions();
	CUDAInitializeMaterials();
}

void CGPUSimulator::InitializeModelParameters()
{
	CBaseSimulator::InitializeModelParameters();
	for (auto& model : m_models)
		model->InitializeGPU(m_impl->cudaDefines);
}

void CGPUSimulator::UpdateCollisionsStep(double _dTimeStep)
{
	// clear current states of particles and walls on GPU
	m_impl->sceneGPU.ClearStates();

	// check that all particles are remains in simulation domain
	m_impl->gpu.CheckParticlesInDomain(m_currentTime, m_impl->sceneGPU.GetPointerToParticles(), &m_impl->pDispatchedResults_d->nActivePartNum);

	// update statistics
	CUDAUpdateGlobalCPUData();

	// if there is no contact model, then there is no necessity to calculate contacts
	if (!m_PPModels.empty() || !m_PWModels.empty())
	{
		UpdateVerletLists(_dTimeStep); // between PP and PW
		CUDAUpdateActiveCollisions();
	}
}

void CGPUSimulator::CalculateForcesStep(double _dTimeStep)
{
	if (!m_PPModels.empty()) CalculateForcesPP(_dTimeStep);
	if (!m_PWModels.empty()) CalculateForcesPW(_dTimeStep);
	if (!m_SBModels.empty()) CalculateForcesSB(_dTimeStep);
	if (!m_LBModels.empty()) CalculateForcesLB(_dTimeStep);
	if (!m_EFModels.empty()) CalculateForcesEF(_dTimeStep);
}

void CGPUSimulator::CalculateForcesPP(double _dTimeStep)
{
	if (!m_impl->gpu.m_CollisionsPP.collisions.nElements) return;
	for (auto* model : m_PPModels)
		model->CalculatePPGPU(m_currentTime, _dTimeStep, m_impl->pInteractProps, m_impl->sceneGPU.GetPointerToParticles(), m_impl->gpu.m_CollisionsPP.collisions);
}

void CGPUSimulator::CalculateForcesPW(double _dTimeStep)
{
	if (!m_impl->gpu.m_CollisionsPW.collisions.nElements) return;
	for (auto* model : m_PWModels)
		model->CalculatePWGPU(m_currentTime, _dTimeStep, m_impl->pInteractProps, m_impl->sceneGPU.GetPointerToParticles(), m_impl->sceneGPU.GetPointerToWalls(), m_impl->gpu.m_CollisionsPW.collisions);
}

void CGPUSimulator::CalculateForcesSB(double _dTimeStep)
{
	if (m_scene.GetBondsNumber() == 0) return;
	m_impl->sceneGPU.GetActiveBondsNumber(&m_impl->pDispatchedResults_d->activeBondsNumBeforeBreak);
	for (auto* model : m_SBModels)
		model->CalculateSBGPU(m_currentTime, _dTimeStep, m_impl->sceneGPU.GetPointerToParticles(), m_impl->sceneGPU.GetPointerToSolidBonds());
	m_impl->sceneGPU.GetActiveBondsNumber(&m_impl->pDispatchedResults_d->activeBondsNumAfterBreak);
}

void CGPUSimulator::CalculateForcesEF(double _dTimeStep)
{
	for (auto* model : m_EFModels)
		model->CalculateEFGPU(m_currentTime, _dTimeStep, m_impl->sceneGPU.GetPointerToParticles());
}

void CGPUSimulator::MoveParticles(bool _bPredictionStep)
{
	if (!m_externalAcceleration.IsZero())
		m_impl->gpu.ApplyExternalAcceleration(m_impl->sceneGPU.GetPointerToParticles());
	if (!_bPredictionStep)
	{
		if (m_variableTimeStep)
			m_currSimulationStep = m_impl->gpu.CalculateNewTimeStep(m_currSimulationStep, m_initSimulationStep, m_partMoveLimit, m_timeStepFactor, m_impl->sceneGPU.GetPointerToParticles());
		m_impl->gpu.MoveParticles(m_currSimulationStep, m_considerAnisotropy, m_partVelocityLimit.value_or(0.0), m_impl->sceneGPU.GetPointerToParticles());
	}
	else
		m_impl->gpu.MoveParticlesPrediction(m_currSimulationStep / 2., m_considerAnisotropy, m_partVelocityLimit.value_or(0.0), m_impl->sceneGPU.GetPointerToParticles());

	MoveParticlesOverPBC(); // move virtual particles and check boundaries
}

void CGPUSimulator::MoveWalls(double _dTimeStep)
{
	m_wallsVelocityChanged = false;
	// analysis of transition to new interval
	for (unsigned iGeom = 0; iGeom < m_pSystemStructure->GeometriesNumber(); ++iGeom)
	{
		CRealGeometry* pGeom = m_pSystemStructure->Geometry(iGeom);

		if (pGeom->Planes().empty()) continue;
		if ((pGeom->Motion()->MotionType() == CGeometryMotion::EMotionType::FORCE_DEPENDENT) ||
			(pGeom->Motion()->MotionType() == CGeometryMotion::EMotionType::CONSTANT_FORCE)) // force
		{
			const CVector3 vTotalForce = m_impl->gpu.CalculateTotalForceOnWall(iGeom, m_impl->sceneGPU.GetPointerToWalls());
			pGeom->UpdateMotionInfo(vTotalForce.z);
		}
		else
			pGeom->UpdateMotionInfo(m_currentTime); // time

		CVector3 vVel = pGeom->GetCurrentVelocity();
		CVector3 vRotVel = pGeom->GetCurrentRotVelocity();
		CVector3 vRotCenter = pGeom->GetCurrentRotCenter();

		if (m_currentTime == 0 || vVel != pGeom->GetCurrentVelocity() || vRotVel != pGeom->GetCurrentRotVelocity() || vRotCenter != pGeom->GetCurrentRotCenter())
			m_wallsVelocityChanged = true;

		if ( !pGeom->FreeMotion().IsZero() )
			m_wallsVelocityChanged = true;

		if (vRotVel.IsZero() && pGeom->FreeMotion().IsZero() && vVel.IsZero()) continue;
		CMatrix3 RotMatrix;
		if (!vRotVel.IsZero())
			RotMatrix = CQuaternion(vRotVel*_dTimeStep).ToRotmat();

		m_impl->gpu.MoveWalls(_dTimeStep, iGeom, vVel, vRotVel, vRotCenter, RotMatrix, pGeom->FreeMotion(),
			pGeom->Motion()->MotionType() == CGeometryMotion::EMotionType::FORCE_DEPENDENT, pGeom->RotateAroundCenter(), pGeom->Mass(), m_impl->sceneGPU.GetPointerToWalls(), m_externalAcceleration);
	}
}

void CGPUSimulator::UpdateTemperatures(bool _predictionStep)
{
	const double timeStep = !_predictionStep ? m_currSimulationStep : m_currSimulationStep / 2.;
	m_impl->gpu.UpdateTemperatures(timeStep, m_impl->sceneGPU.GetPointerToParticles());
}

void CGPUSimulator::GenerateNewObjects()
{
	if (!m_generationManager->IsNeedToBeGenerated(m_currentTime)) return;

	/* TODO: Current approach to update data on GPU is very slow. Now it takes:
	 * 1. Copy all time-dependent data from GPU to CPU simplified scene.
	 * 2. Generate objects on CPU - add new objects to the CPU simplified scene.
	 * 3. Update Verlet lists on GPU.
	 * 4. Copy all data from GPU to CPU to reflect the new state of the CPU simplified scene - all GPU data structures are reallocated here.
	 * 5. Set Verlet lists to GPU.
	 */

	// copy actual trackable time-dependent data from GPU to CPU
	m_impl->sceneGPU.CUDAParticlesGPU2CPUDynamicData(m_scene);
	m_impl->sceneGPU.CUDABondsGPU2CPUDynamicData(m_scene);
	// copy actual wall coordinates data from GPU to CPU
	m_impl->sceneGPU.CUDAWallsGPU2CPUVerletData(m_scene);

	// generate
	const size_t newObjects = m_generationManager->GenerateObjects(m_currentTime, m_scene, m_generatedObjectsDiff);

	// update CPU scene
	m_verletList.SetSceneInfo(m_pSystemStructure->GetSimulationDomain(), m_scene.GetMinParticleContactRadius(), m_scene.GetMaxParticleContactRadius(), m_cellsMax, m_verletDistanceCoeff, m_autoAdjustVerletDistance);
	m_verletList.ResetCurrentData();
	m_nGeneratedObjects += newObjects;
	m_scene.UpdateParticlesToBonds();

	// update data on device
	m_impl->sceneGPU.CUDAParticlesCPU2GPU(m_scene);
	m_impl->sceneGPU.CUDABondsCPU2GPU(m_scene);

	// update verlet lists
	m_verletList.UpdateList(m_currentTime);
	CUDAUpdateVerletLists(true);
	CUDAUpdateVerletLists(false);
	m_impl->sceneGPU.CUDASaveVerletCoords();
}

void CGPUSimulator::UpdatePBC()
{
	m_scene.m_PBC.UpdatePBC(m_currentTime);
	if (!m_scene.m_PBC.vVel.IsZero())	// if velocity is not zero
	{
		InitializeModelParameters();	// set new PBC to all models
		m_impl->gpu.SetPBC(m_scene.m_PBC);	// set new PBC to GPU scene
	}
}

void CGPUSimulator::PrepareAdditionalSavingData()
{
	SParticleStruct& particles = m_scene.GetRefToParticles();
	SSolidBondStruct& bonds = m_scene.GetRefToSolidBonds();

	static SGPUCollisions PPCollisions(SBasicGPUStruct::EMemType::HOST), PWCollisions(SBasicGPUStruct::EMemType::HOST);
	m_impl->gpu.CopyCollisionsGPU2CPU(PPCollisions, PWCollisions);

	// reset previously calculated stresses
	for (auto& data : m_additionalSavingData)
		data.stressTensor.Init(0);

	// save stresses caused by solid bonds
	for (size_t i = 0; i < bonds.Size(); ++i)
	{
		if (!bonds.Active(i) && !m_pSystemStructure->GetObjectByIndex(bonds.InitIndex(i))->IsActive(m_currentTime)) continue;
		const size_t leftID  = bonds.LeftID(i);
		const size_t rightID = bonds.RightID(i);
		CVector3 connVec = (particles.Coord(leftID) - particles.Coord(rightID)).Normalized();
		m_additionalSavingData[bonds.LeftID(i) ].AddStress(-1 * connVec * particles.Radius(leftID ),      bonds.TotalForce(i), PI * pow(2 * particles.Radius(leftID ), 3.0) / 6);
		m_additionalSavingData[bonds.RightID(i)].AddStress(     connVec * particles.Radius(rightID), -1 * bonds.TotalForce(i), PI * pow(2 * particles.Radius(rightID), 3.0) / 6);
	}

	// save stresses caused by particle-particle contact
	for (size_t i = 0; i < PPCollisions.nElements; ++i)
	{
		if (!PPCollisions.ActivityFlags[i]) continue;
		const size_t srcID = PPCollisions.SrcIDs[i];
		const size_t dstID = PPCollisions.DstIDs[i];
		CVector3 connVec = (particles.Coord(srcID) - particles.Coord(dstID)).Normalized();
		m_additionalSavingData[PPCollisions.SrcIDs[i]].AddStress(-1 * connVec * particles.Radius(srcID),      PPCollisions.TotalForces[i], PI * pow(2 * particles.Radius(srcID), 3.0) / 6);
		m_additionalSavingData[PPCollisions.DstIDs[i]].AddStress(     connVec * particles.Radius(dstID), -1 * PPCollisions.TotalForces[i], PI * pow(2 * particles.Radius(dstID), 3.0) / 6);
	}

	// save stresses caused by particle-wall contacts
	for (size_t i = 0; i < PWCollisions.nElements; ++i)
	{
		if (!PWCollisions.ActivityFlags[i]) continue;
		CVector3 connVec = (PWCollisions.ContactVectors[i] - particles.Coord(PWCollisions.DstIDs[i])).Normalized();
		m_additionalSavingData[PWCollisions.DstIDs[i]].AddStress(connVec * particles.Radius(PWCollisions.DstIDs[i]), PWCollisions.TotalForces[i], PI * pow(2 * particles.Radius(PWCollisions.DstIDs[i]), 3.0) / 6);
	}
}

void CGPUSimulator::SaveData()
{
	clock_t t = clock();
	cudaDeviceSynchronize();
	m_impl->sceneGPU.CUDABondsGPU2CPU( m_scene );
	m_impl->sceneGPU.CUDAParticlesGPU2CPUAllData(m_scene);
	m_impl->sceneGPU.CUDAWallsGPU2CPUAllData(m_scene);
	m_maxParticleVelocity = m_impl->sceneGPU.GetMaxPartVelocity();
	if (m_scene.GetRefToParticles().ThermalsExist())
		m_maxParticleTemperature = m_impl->sceneGPU.GetMaxPartTemperature();
	p_SaveData();

	m_verletList.AddDisregardingTimeInterval(clock() - t);
}

void CGPUSimulator::UpdateVerletLists(double _dTimeStep)
{
	if (m_verletList.IsNeedToBeUpdated(_dTimeStep, sqrt(m_impl->pDispatchedResults_h->dMaxSquaredPartDist) , m_maxWallVelocity))
	{
		m_impl->sceneGPU.CUDAParticlesGPU2CPUVerletData(m_scene);
		m_impl->sceneGPU.CUDAWallsGPU2CPUVerletData(m_scene);
		m_verletList.UpdateList(m_currentTime);

		CUDAUpdateVerletLists(true);
		CUDAUpdateVerletLists(false);
		m_impl->sceneGPU.CUDASaveVerletCoords();
	}
}

void CGPUSimulator::CUDAUpdateGlobalCPUData()
{
	// update max velocities
	m_impl->sceneGPU.GetMaxSquaredPartDist(&m_impl->pDispatchedResults_d->dMaxSquaredPartDist);
	if (m_wallsVelocityChanged)
		m_impl->sceneGPU.GetMaxWallVelocity(&m_impl->pDispatchedResults_d->dMaxWallVel);

	CUDA_MEMCPY_D2H(m_impl->pDispatchedResults_h, m_impl->pDispatchedResults_d, sizeof(std::remove_pointer_t<decltype(m_impl->pDispatchedResults_d)>));

	m_brokenBonds += m_impl->pDispatchedResults_h->activeBondsNumBeforeBreak - m_impl->pDispatchedResults_h->activeBondsNumAfterBreak;

	const bool bNewInactiveParticles = (m_inactiveParticles != m_impl->sceneGPU.GetParticlesNumber() - m_impl->pDispatchedResults_h->nActivePartNum);
	if (bNewInactiveParticles && m_impl->sceneGPU.GetBondsNumber())
	{
		m_impl->gpu.CheckBondsActivity(m_currentTime, m_impl->sceneGPU.GetPointerToParticles(), m_impl->sceneGPU.GetPointerToSolidBonds());
		m_impl->sceneGPU.CUDABondsActivityGPU2CPU(m_scene);
		m_scene.UpdateParticlesToBonds();
		m_inactiveBonds = m_impl->sceneGPU.GetInactiveBondsNumber() - m_brokenBonds;
	}

	m_inactiveParticles = m_impl->sceneGPU.GetParticlesNumber() - m_impl->pDispatchedResults_h->nActivePartNum;
	if (m_wallsVelocityChanged)
		m_maxWallVelocity = m_impl->pDispatchedResults_h->dMaxWallVel;
}

void CGPUSimulator::CUDAUpdateActiveCollisions()
{
	m_impl->gpu.UpdateActiveCollisionsPP(m_impl->sceneGPU.GetPointerToParticles());
	m_impl->gpu.UpdateActiveCollisionsPW(m_impl->sceneGPU.GetPointerToParticles(), m_impl->sceneGPU.GetPointerToWalls());
}

void CGPUSimulator::CUDAUpdateVerletLists(bool _bPPVerlet)
{
	const auto& verletListCPU   = _bPPVerlet ? m_verletList.m_PPList      : m_verletList.m_PWList;
	const auto& verletShiftsCPU = _bPPVerlet ? m_verletList.m_PPVirtShift : m_verletList.m_PWVirtShift;
	auto& collisions            = _bPPVerlet ? m_impl->gpu.m_CollisionsPP : m_impl->gpu.m_CollisionsPW;
	auto& store                 = _bPPVerlet ? m_impl->storePP            : m_impl->storePW;

	const size_t nParticles = m_scene.GetTotalParticlesNumber();
	// calculate total number of possible contacts
	size_t nCollsions = 0;
	for (const auto& colls : verletListCPU)
		nCollsions += colls.size();

	// create verlet lists for GPU
	store.hvVerletPartInd.resize(nParticles + 1);
	store.hvVerletDst.resize(nCollsions);
	store.hvVerletSrc.resize(nCollsions);
	store.hvVirtShifts.resize(nCollsions);

	if (!store.hvVerletPartInd.empty())	store.hvVerletPartInd.front() = 0;			// for easier access
	for (size_t i = 1; i < nParticles; ++i)
		store.hvVerletPartInd[i] = store.hvVerletPartInd[i - 1] + (unsigned)verletListCPU[i-1].size();
	if (!store.hvVerletPartInd.empty())	store.hvVerletPartInd.back() = (unsigned)nCollsions;	// for easier access
	ParallelFor(nParticles, [&](size_t i)
	{
		std::copy(verletListCPU[i].begin(), verletListCPU[i].end(), store.hvVerletDst.begin() + store.hvVerletPartInd[i]);
		std::fill(store.hvVerletSrc.begin() + store.hvVerletPartInd[i], store.hvVerletSrc.begin() + store.hvVerletPartInd[i] + verletListCPU[i].size(), static_cast<unsigned>(i));
		if (m_scene.m_PBC.bEnabled)
			std::copy(verletShiftsCPU[i].begin(), verletShiftsCPU[i].end(), store.hvVirtShifts.begin() + store.hvVerletPartInd[i]);
	});

	// update verlet lists on device
	m_impl->gpu.UpdateVerletLists(_bPPVerlet, m_impl->sceneGPU.GetPointerToParticles(), m_impl->sceneGPU.GetPointerToWalls(), store.hvVerletSrc, store.hvVerletDst, store.hvVerletPartInd,
		store.hvVirtShifts, collisions.vVerletSrc, collisions.vVerletDst, collisions.vVerletPartInd, collisions.collisions);
	m_impl->gpu.SortByDst(_bPPVerlet ? (unsigned)m_scene.GetTotalParticlesNumber() : (unsigned)m_scene.GetWallsNumber(),
		collisions.vVerletSrc, collisions.vVerletDst, collisions.vVerletCollInd_DstSorted, collisions.vVerletPartInd_DstSorted);
}

void CGPUSimulator::CUDAInitializeMaterials()
{
	if (m_impl->pInteractProps)
	{
		CUDA_FREE_D(m_impl->pInteractProps);
		m_impl->pInteractProps = NULL;
	}
	size_t nCompounds = m_scene.GetCompoundsNumber();
	m_impl->gpu.SetCompoundsNumber( nCompounds );
	if (!nCompounds) return;

	using type = std::remove_pointer_t<decltype(m_impl->pInteractProps)>;
	type* pInteractPropsHost;
	CUDA_MALLOC_H(&pInteractPropsHost, sizeof(type)*nCompounds*nCompounds);
	CUDA_MALLOC_D(&m_impl->pInteractProps,   sizeof(type)*nCompounds*nCompounds);

	for (unsigned i = 0; i < nCompounds; ++i)
		for (unsigned j = 0; j < nCompounds; ++j)
			pInteractPropsHost[i*nCompounds + j] = m_scene.GetInteractProp(i*nCompounds + j);

	CUDA_MEMCPY_H2D(m_impl->pInteractProps, pInteractPropsHost, sizeof(type)*nCompounds*nCompounds);
	CUDA_FREE_H(pInteractPropsHost);

}

void CGPUSimulator::CUDAInitializeWalls()
{
	std::vector<std::vector<unsigned>> vvWallsInGeom(m_pSystemStructure->GeometriesNumber());
	for (size_t i = 0; i < m_pSystemStructure->GeometriesNumber(); ++i)
	{
		const CRealGeometry* pGeom = m_pSystemStructure->Geometry(i);
		const auto& planes = pGeom->Planes();
		vvWallsInGeom[i].resize(planes.size());
		for (size_t j = 0; j < planes.size(); ++j)
			vvWallsInGeom[i][j] = static_cast<unsigned>(m_scene.m_vNewIndexes[planes[j]]);
	}
	m_impl->gpu.InitializeWalls(vvWallsInGeom, m_scene.m_adjacentWalls);
}

void CGPUSimulator::MoveParticlesOverPBC()
{
	if (!m_scene.m_PBC.bEnabled) return;
	m_impl->gpu.MoveParticlesOverPBC(m_impl->sceneGPU.GetPointerToParticles());
}

void CGPUSimulator::GetOverlapsInfo(double& _dMaxOverlap, double& _dAverageOverlap, size_t _nMaxParticleID)
{
	m_impl->gpu.GetOverlapsInfo(m_impl->sceneGPU.GetPointerToParticles(), _nMaxParticleID, _dMaxOverlap, _dAverageOverlap);
}

SGPUParticles* CGPUSimulator::GetPointerToParticles()
{
	return &m_impl->sceneGPU.GetPointerToParticles();
}
