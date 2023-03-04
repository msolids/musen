/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "CUDAKernels.cuh"
#include "DisableWarningHelper.h"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
PRAGMA_WARNING_POP

class CGPU
{
	typedef thrust::host_vector<uint8_t> h_vec_u8;
	typedef thrust::host_vector<unsigned> h_vec_u;
	typedef thrust::host_vector<double> h_vec_d;
	typedef thrust::device_vector<bool> d_vec_b;
	typedef thrust::device_vector<uint8_t> d_vec_u8;
	typedef thrust::device_vector<int8_t> d_vec_i8;
	typedef thrust::device_vector<unsigned> d_vec_u;
	typedef thrust::device_vector<double> d_vec_d;
	typedef thrust::device_vector<CVector3> d_vec_v3;
	typedef thrust::device_vector<EIntersectionType> d_vec_IT;

public:
	struct SCollisionsHolder
	{
		// Verlet lists
		d_vec_u vVerletSrc;		// List of src particles according to verlet lists vVerletDst.
		d_vec_u vVerletDst;		// Verlet lists for contacts in array-form sorted by Src particle.
		d_vec_u vVerletPartInd;	// Positions of each new Src particle in verlet list vVerletDst.

		// This two additional arrays are verlet lists sorted by Dst particle
		d_vec_u vVerletCollInd_DstSorted; // Index in collisions of a collision with specific Dst.
		d_vec_u vVerletPartInd_DstSorted; // Position of each new Dst object in array vVerletCollInd_DstSorted.

		SGPUCollisions collisions;	// List of all possible collisions according to the verlet list vVerletDst.
	};

	struct SAdjacentWalls
	{
		d_vec_u startIndices;	// Position of each wall in adjacentWalls.
		d_vec_u adjacentWalls;	// List of adjacent walls as flattened vector of vectors.
	};

private:
	const CCUDADefines* m_cudaDefines{ nullptr };
	std::vector<d_vec_u> m_vvWallsInGeom; // List of walls' indices separately for each geometry.
	bool m_PBCEnabled{ false };

public:
	SCollisionsHolder m_CollisionsPP;	// Particle-particle collisions.
	SCollisionsHolder m_CollisionsPW;	// Particle-wall collisions.
	SAdjacentWalls m_adjacentWalls;		// List of adjacent walls and indices to iterate them.
public:
	//////////////////////////////////////////////////////////////////////////
	/// Setters for constant GPU memory

	CGPU(const CCUDADefines* _cudaDefines);

	static void SetExternalAccel(const CVector3& _acceleration);
	static void SetSimulationDomain(const SVolumeType& _domain);
	void SetPBC(const SPBC& _PBCInfo);
	static void SetCompoundsNumber(size_t _nCompounds);
	static void SetAnisotropyFlag(bool _enabled);

	void InitializeWalls(const std::vector<std::vector<unsigned>>& _vvWallsInGeom, const std::vector<std::vector<unsigned>>& _adjacentWalls);
	void InitializeCollisions();	// create or update information about interaction properties for GPU

	//////////////////////////////////////////////////////////////////////////
	/// Service functions

	void Flags2IndicesList(size_t _size, bool _flags[], d_vec_u& _sequence, d_vec_i8& _storage, unsigned* _listLength, unsigned _list[]);

	//////////////////////////////////////////////////////////////////////////
	/// Wrappers for GPU kernels

	void ApplyExternalAcceleration(SGPUParticles& _particles);
	double CalculateNewTimeStep(double _currTimeStep, double _initTimeStep, double _partMoveLimit, double _timeStepFactor, SGPUParticles& _particles) const;
	void MoveParticles(double& _currTimeStep, double _initTimeStep, SGPUParticles& _particles, bool _bFlexibleTimeStep);
	void MoveParticlesPrediction(double _timeStep, SGPUParticles& _particles);

	void MoveWalls(double _timeStep, size_t _iWallsInGeom, const CVector3& _vel, const CVector3& _rotVel, const CVector3& _rotCenter, const CMatrix3& _rotMatrix,
		const CVector3& _freeMotion, bool _bForceDependentMotion, bool _bRotateAroundCenter, double _dMass, SGPUWalls& _walls, const CVector3& _vExternalAccel);

	void UpdateTemperatures(double _currTimeStep, SGPUParticles& _particles);

	void UpdateVerletLists(bool _bPPVerlet, const SGPUParticles& _particles, const SGPUWalls& _walls, const h_vec_u& _vVerListSrcNew, const h_vec_u& _vVerListDstNew,
		const h_vec_u& _vVerListIndNew, const h_vec_u8& _vVirtShifts, d_vec_u& _vVerListSrcOld, d_vec_u& _vVerListDstOld, d_vec_u& _vVerListIndOld, SGPUCollisions& _collisions) const;

	void UpdateActiveCollisionsPP(const SGPUParticles& _particles);

	void UpdateActiveCollisionsPW(const SGPUParticles& _particles, const SGPUWalls& _walls);

	void SortByDst(unsigned _nPart, const d_vec_u& _vVerListSrc, const d_vec_u& _vVerListDst, d_vec_u& _vVerCollInd_DstSorted, d_vec_u& _vVerPartInd_DstSorted) const;

	void GatherForcesFromPWCollisions(SGPUParticles& _particles, SGPUWalls& _walls) const;

	void CheckParticlesInDomain(double _currTime, const SGPUParticles& _particles, unsigned* _bufActivePartsNum) const;

	// Set inactivity of bonds due to possible inactivity of particles.
	void CheckBondsActivity(double _currTime, const SGPUParticles& _particles, SGPUSolidBonds& _bonds);

	void MoveParticlesOverPBC(const SGPUParticles& _particles);

	void CalculateTotalForceOnWall(size_t iGeom, SGPUWalls& _walls, d_vec_v3& _vTotalForce);

	CVector3 CalculateTotalForceOnWall(size_t iGeom, SGPUWalls& _walls );

	void CopyCollisionsGPU2CPU( SGPUCollisions& _PPCollisionsHost, SGPUCollisions& _PWCollisionsHost ) const;

	// Returns all current maximal and average overlap between particles with particle indexes smaller than _nMaxParticleID.
	void GetOverlapsInfo(const SGPUParticles& _particles, size_t _maxParticleID, double& _maxOverlap, double& _avrOverlap) const;
};
