/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GPUSimulator.cuh"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include <thrust/count.h>
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

void CGPU::SetAnisotropyFlag(bool _enabled)
{
	CUDAKernels::SetAnisotropyFlag(_enabled);
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
		hostStartIndices[i] = hostStartIndices[i - 1] + _adjacentWalls[i - 1].size();
	if (!hostStartIndices.empty()) hostStartIndices.back() = number - 1;	// for easier access

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

void CGPU::MoveParticles(double& _currTimeStep, double _initTimeStep, SGPUParticles& _particles, bool _bFlexibleTimeStep)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::MoveParticles_kernel, _currTimeStep, static_cast<unsigned>(_particles.nElements), _particles.Masses, _particles.InertiaMoments,
		_particles.Moments, _particles.Forces, _particles.Vels, _particles.AnglVels, _particles.Coords, _particles.Quaternions);
}

void CGPU::MoveParticlesPrediction(double _timeStep, SGPUParticles& _particles)
{
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::MoveParticlesPrediction_kernel, _timeStep, static_cast<unsigned>(_particles.nElements),
		_particles.Masses, _particles.InertiaMoments, _particles.Moments, _particles.Forces, _particles.Vels, _particles.AnglVels, _particles.Quaternions);
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
	const CVector3& _freeMotion, bool _bForceDependentMotion, bool _bRotateAroundCenter, double _dMass, SGPUWalls& _walls, const CVector3& _vExternalAccel)
{
	static d_vec_v3 vTotalForce(1);
	static d_vec_v3 vRotCenter(1); // used in case when rotation around center is defined

	if (_bRotateAroundCenter || _bForceDependentMotion || !_freeMotion.IsZero())
		CalculateTotalForceOnWall(_iGeom, _walls, vTotalForce);
	if (_bRotateAroundCenter) // precalculate rotation center
		CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::CalculateGeometryCenter_kernel, static_cast<unsigned>(m_vvWallsInGeom[_iGeom].size()), m_vvWallsInGeom[_iGeom].data().get(),
			_walls.Vertices1, _walls.Vertices2, _walls.Vertices3, vRotCenter.data().get());
	CUDA_KERNEL_ARGS2_DEFAULT(CUDAKernels::MoveWalls_kernel, _timeStep,
		static_cast<unsigned>(m_vvWallsInGeom[_iGeom].size()), _vel, _rotVel, _rotCenter, _rotMatrix,
		_freeMotion, vTotalForce.data().get(), _dMass, _bRotateAroundCenter, _vExternalAccel,
		vRotCenter.data().get(), m_vvWallsInGeom[_iGeom].data().get(),
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
			m_CollisionsPP.collisions.NormalOverlaps, _maxParticleID,
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
			_maxParticleID, overlapsPW.data().get(), flagsPW.data().get());

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
