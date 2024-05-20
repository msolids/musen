/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GeometricFunctions.h"
#include "SystemStructure.h"
#include "SimplifiedSceneGPU.h"
#include "DisableWarningHelper.h"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include <device_launch_parameters.h>
PRAGMA_WARNING_POP

namespace CUDAKernels
{
	struct SGPUGridParams
	{
		int3 vGridSize;
		CVector3 vCellSize;
		CVector3 vCoordBegin;
	};

	//////////////////////////////////////////////////////////////////////////
	/// Setters for constant GPU memory

	void SetThreadsNumber(unsigned _threads);
	void SetExternalAccel(const CVector3& _acceleration);
	void SetSimulationDomain(const SVolumeType& _domain);
	void SetPBC(const SPBC& _PBCInfo);
	void SetCompoundsNumber(size_t _nCompounds);
	void SetAnisotropyFlag(bool _enabled);

	//////////////////////////////////////////////////////////////////////////
	/// GPU kernels

	__global__ void ApplyExternalAcceleration_kernel(unsigned _nParticles, const double* _partMasses, CVector3* _partForces);

	__global__ void GatherForFlexibleTimeStep_kernel(unsigned _nParticles, const double* _partMasses, const CVector3* _partForces, double* _res);

	__global__ void MoveParticles_kernel(double _dTimeStep, unsigned _nParticles,
		const double* _partMasses, const double* _partInertiaMoments, const CVector3* _partMoments,
		CVector3* _partForces, CVector3* _partVels, CVector3* _partAnglVels, CVector3* _partCoords, CQuaternion* _partQuaternions);

	__global__ void MoveParticlesPrediction_kernel(double _dTimeStep, unsigned _nParticles,
		const double* _partMasses, const double* _partInertiaMoments, const CVector3* _partMoments,
		CVector3* _partForces, CVector3* _partVels, CVector3* _partAnglVels, CQuaternion* _partQuaternions);

	__global__ void UpdateTemperatures_kernel(double _dTimeStep, unsigned _nParticles, const double* _partHeatCapacities,
		const double* _partMasses, const double* _partHeatFluxes, double* _partTemperatures);

	// Calculates a vector of _centers for each triangle. To get the actual center, all _centers must be summed up.
	__global__ void PrecalculateGeometryCenter_kernel( unsigned _nWallsInGeom, const unsigned* _wallsInGeom,
		const CVector3* _vertex1, const CVector3* _vertex2, const CVector3* _vertex3, CVector3* _centers);

	__global__ void MoveWalls_kernel(double _timeStep, unsigned _nWallsInGeom, const CVector3 _vel, const CVector3 _rotVel, const CVector3 _definedRotCenter, const CMatrix3 _rotMatrix,
		const CVector3 _freeMotion, const CVector3* _totalForce, double _dMass, bool _bRotateAroundCenter, const CVector3 _vExternalAccel,
		CVector3* _vCalculatedCenter, const unsigned* _wallsInGeom, CVector3* _vertex1, CVector3* _vertex2, CVector3* _vertex3,
		CVector3* _wallMinCoord, CVector3* _wallMaxCoord, CVector3* _wallNormalVector, CVector3* _wallVel, CVector3* _wallRotCenter, CVector3* _wallRotVel);

	__global__ void GatherForcesFromWalls_kernel(unsigned _nWallsInGeom, const unsigned* _wallsInGeom, const CVector3* _wallForce, CVector3* _outForces);

	// Copies old existing new collisions according to new verlet lists.
	__global__ void CopyCollisionsPP_kernel(unsigned _nCollisionsOld,
		const unsigned* _vVerlSrcOld, const unsigned* _vVerlDstOld, const unsigned* _vVerlDstNew, const unsigned* _vPartInd_VerNew, const bool* _oldActiveCollFlags,
		const double* _oldNormalOverlap, const double* _oldInitNormalOverlap, const CVector3* _oldTangOverlap, const CVector3* _oldContactVector, const CVector3* _oldTotalForce,
		double* _newNormalOverlap, double* _newInitNormalOverlap, CVector3* _newTangOverlap, CVector3* _newContactVector, CVector3* _newTotalForce);

	__global__ void CopyCollisionsPW_kernel(unsigned _nCollisionsOld,
		const unsigned* _vVerlSrcOld, const unsigned* _vVerlDstOld, const unsigned* _vVerlDstNew, const unsigned* _vPartInd_VerNew, const bool* _oldActiveCollFlags,
		const double* _oldNormalOverlap, const CVector3* _oldTangOverlap, const CVector3* _oldContactVector, const CVector3* _oldTotalForce,
		bool* _newActiveCollFlags, double* _newNormalOverlap, CVector3* _newTangOverlap, CVector3* _newContactVector, CVector3* _newTotalForce);

	__global__ void InitializePPCollisions_kernel(unsigned _nCollisions, const unsigned* _vVerListSrc, const unsigned* _vVerListDst,
		const double* _partRadii, const double* _partMasses, const unsigned* _partCompoundIndices,
		unsigned* _collSrcID, unsigned* _collDstID, double* _collEquivMass, double* _collEquivRadius, double* _collSumRadii, uint16_t* _collInteractPropID);

	__global__ void InitializePWCollisions_kernel(unsigned _nCollisions, const unsigned* _vVerListSrc, const unsigned* _vVerListDst,
		const unsigned* _partCompoundIndices, const unsigned* _wallCompoundIndices,
		unsigned* _collSrcID, unsigned* _collDstID, uint16_t* _collInteractPropID);

	// Checks activity of available collisions and initializes new ones.
	__global__ void UpdateActiveCollisionsPP_kernel(const unsigned _nCollisions, const unsigned* _vVerListSrc, const unsigned* _vVerListDst, const CVector3* _partCoords,
		const uint8_t* _collVirtShifts, const double* _collSumRadii, bool* _collActiveFlags, double* _collNormalOverlaps, double* _collInitNormalOverlaps,
		CVector3* _collContactVectors, CVector3* _collTangOverlaps);

	__global__ void GetIntersectTypePW_kernel(unsigned _nCollisions, const unsigned* _vVerListSrc, const unsigned* _vVerListDst,
		const double* _partRadii, const CVector3* _partCoords,
		const CVector3* _wallVertex1, const CVector3* _wallVertex2, const CVector3* _wallVertex3, const CVector3* _wallMinCoord, const CVector3* _wallMaxCoord, const CVector3* _wallNormalVector,
		const uint8_t* _collVirtShifts, EIntersectionType* _collTempIntersectionType, CVector3* _collContactPoint, bool* _bActivePart);

	__global__ void CombineIntersectionsPW_kernel(const unsigned* _nActiveParticles, const unsigned* _nActivePartIndexes,
		unsigned _nParticles, unsigned _nCollisions, const unsigned* _vVerList, const unsigned* _vIVerList,
		const CVector3* _wallNormalVector, EIntersectionType* _collTempIntersectionType, const uint8_t* _collVirtShifts);

	__global__ void UpdateActiveCollisionsPW_kernel(unsigned _nCollisions, const EIntersectionType* _collTempIntersectionType,
		bool* _collActiveFlags, bool* _callActivated, bool* _callDeactivated);

	__global__ void CopyCollisionsForAdjacentWalls(const unsigned* _nActivatedColls, const unsigned* _activatedCollIndices, const bool* _callDeactivated,
		const unsigned* _verlSrcIDs, const unsigned* _verlDstIDs, const unsigned* _verlSrcStartIndices,
		const unsigned* _adjWallsIDs, const unsigned* _adjWallsStartIndices,
		CVector3* _tangOverlaps);

	__global__ void CheckParticlesInDomain_kernel(double _currTime, const unsigned _nParticles,
		unsigned* _partActivity, double* _partEndActivity, const CVector3* _partCoords);

	__global__ void CheckBondsActivity_kernel(double _currTime, unsigned _nBonds, const unsigned* _partActivity,
		uint8_t* _bondActive, const unsigned* _bondLeftID, const unsigned* _bondRightID, double* _bondEndActivity);

	__global__ void MoveVirtualParticlesBox(unsigned _nParticles, const unsigned* _partActivity,
		CVector3* _partCoords, CVector3* _partCoordsVerlet, uint8_t* _vCrossingShifts, bool* _vCrossingFlags);

	__global__ void AddShiftsToCollisions(unsigned _nParticles, const unsigned* _nCrossedParticles, const unsigned* _partCrossedIndices, const uint8_t* _vCrossingShifts,
		const unsigned* _vVerListSrcSortInd, const unsigned* _vVerListDstSortInd, const unsigned* _vVerListDstSortColl,
		unsigned _nCollisions, const unsigned* _collSrcID, const unsigned* _collDstID, uint8_t* _collVirtShifts);

	// Fill indexes of unique values in sorted array.
	__global__ void FillUniqueIndexes_kernel(unsigned _num, const unsigned* _idata, unsigned* _odata);

	// Fill total for non-existing elements, i.e. in [X 0 X 2 5 6 x x] --> out [0 0 2 2 5 6 9 9].
	__global__ void FillNonExistendIndexes_kernel(unsigned _nPart, unsigned _nColl, unsigned* _inData, unsigned* _outData);

	__global__ void GatherSquaredPartVerletDistances_kernel(unsigned _nParticles,
		const CVector3* _partCoord, const CVector3* _partCoordVerlet, const unsigned* _partActivity, double* _distances);

	__global__ void GatherWallVelocities_kernel(unsigned _nWalls, const CVector3* _vertex1, const CVector3* _vertex2, const CVector3* _vertex3,
		const CVector3* _vel, const CVector3* _rotCenter, const CVector3* _rotVel, double* _outVelocities);

	__global__ void GetPPOverlaps_kernel(const unsigned* _nActiveCollisions, const unsigned* _collActiveIndices, const unsigned* _collSrcID, const unsigned* _collDstID,
		const double* _collNormalOverlaps, const unsigned _maxParticleID, double* _overlaps, uint8_t* _flags);
	__global__ void GetPWOverlaps_kernel(const unsigned* _nActiveCollisions, const unsigned* _collActiveIndices, const unsigned* _collPartID,
		const uint8_t* _collVirtShifts, const CVector3* _collContactVectors,
		const CVector3* _partCoords, const double* _partRadii, unsigned _maxParticleID, double* _overlaps, uint8_t* _flags);

	__global__ void ReduceSum_kernel(unsigned _num, const unsigned* _idata, unsigned* _odata);
	__global__ void ReduceSum_kernel(unsigned _num, const double* _idata, double* _odata);
	__global__ void ReduceSum_kernel(unsigned _num, const CVector3* _idata, CVector3* _odata);

	__global__ void ReduceMax_kernel(unsigned _num, const double* _idata, double* _odata);
	__global__ void ReduceMax_kernel(unsigned _num, const CVector3* _idata, double* _odata);

	__global__ void ReduceMin_kernel(unsigned _num, const double* _idata, double* _odata);
	__global__ void ReduceMin_kernel(unsigned _num, const CVector3* _idata, double* _odata);
}