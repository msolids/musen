/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "CUDAKernels.cuh"
#include <cfloat>

namespace CUDAKernels
{
	__constant__ CVector3 m_vExternalAccel;			// External acceleration.
	__constant__ SVolumeType m_simulationDomain;	// Current simulation domain.
	__constant__ size_t m_nCompoundsNumber;			// Number of unique compounds on current scene.
	__constant__ bool m_considerAnisotropy;			// Whether to consider anisotropy of particles.
	__constant__ unsigned THREADS_PER_BLOCK;		// Number of threads per block.

	// parameters for PBC
	__constant__ SPBC m_PBCGPU;

	void SetThreadsNumber(unsigned _threads)
	{
		CUDA_MEMCOPY_TO_SYMBOL(THREADS_PER_BLOCK, _threads, sizeof(_threads));
	}

	void SetExternalAccel(const CVector3& _acceleration)
	{
		CUDA_MEMCOPY_TO_SYMBOL(m_vExternalAccel, _acceleration, sizeof(_acceleration));
	}

	void SetSimulationDomain(const SVolumeType& _domain)
	{
		CUDA_MEMCOPY_TO_SYMBOL(m_simulationDomain, _domain, sizeof(_domain));
	}

	void SetPBC(const SPBC& _PBCInfo)
	{
		CUDA_MEMCOPY_TO_SYMBOL(m_PBCGPU, _PBCInfo, sizeof(_PBCInfo));
	}

	void SetCompoundsNumber(size_t _nCompounds)
	{
		CUDA_MEMCOPY_TO_SYMBOL(m_nCompoundsNumber, _nCompounds, sizeof(_nCompounds));
	}

	void SetAnisotropyFlag(bool _enabled)
	{
		CUDA_MEMCOPY_TO_SYMBOL(m_considerAnisotropy, _enabled, sizeof(_enabled));
	}

	__global__ void ApplyExternalAcceleration_kernel(unsigned _nParticles, const double* _partMasses, CVector3* _partForces)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nParticles; i += blockDim.x * gridDim.x)
			_partForces[i] += m_vExternalAccel * _partMasses[i];
	}

	__global__ void GatherForFlexibleTimeStep_kernel(unsigned _nParticles, const double* _partMasses, const CVector3* _partForces, double* _res)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nParticles; i += blockDim.x * gridDim.x)
			if (!_partForces[i].IsZero())
				_res[i] = pow(_partMasses[i], 2.0) / _partForces[i].SquaredLength();
			else
				_res[i] = DBL_MAX;
	}

	__global__ void MoveParticles_kernel(double _dTimeStep, unsigned _nParticles,
		const double* _partMasses, const double* _partInertiaMoments, const CVector3* _partMoments,
		CVector3* _partForces, CVector3* _partVels, CVector3* _partAnglVels, CVector3* _partCoords, CQuaternion* _partQuaternions)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nParticles; i += blockDim.x * gridDim.x)
		{
			_partVels[i] += _partForces[i] * _dTimeStep / _partMasses[i];
			if (m_considerAnisotropy)
			{
				const CMatrix3 rotMatrix = _partQuaternions[i].ToRotmat();
				const CVector3 vTemp = (rotMatrix.Transpose()*_partMoments[i]) / _partInertiaMoments[i];
				_partAnglVels[i] += rotMatrix * vTemp * _dTimeStep;
				CQuaternion quaternTemp;
				quaternTemp.q0 = 0.5*_dTimeStep*(-_partQuaternions[i].q1*_partAnglVels[i].x - _partQuaternions[i].q2*_partAnglVels[i].y - _partQuaternions[i].q3*_partAnglVels[i].z);
				quaternTemp.q1 = 0.5*_dTimeStep*(_partQuaternions[i].q0*_partAnglVels[i].x + _partQuaternions[i].q3*_partAnglVels[i].y - _partQuaternions[i].q2*_partAnglVels[i].z);
				quaternTemp.q2 = 0.5*_dTimeStep*(-_partQuaternions[i].q3*_partAnglVels[i].x + _partQuaternions[i].q0*_partAnglVels[i].y + _partQuaternions[i].q1*_partAnglVels[i].z);
				quaternTemp.q3 = 0.5*_dTimeStep*(_partQuaternions[i].q2*_partAnglVels[i].x - _partQuaternions[i].q1*_partAnglVels[i].y + _partQuaternions[i].q0*_partAnglVels[i].z);
				_partQuaternions[i] += quaternTemp;
				_partQuaternions[i].Normalize();
			}
			else
				_partAnglVels[i] += _partMoments[i] * _dTimeStep / _partInertiaMoments[i];
			_partCoords[i] += _partVels[i] * _dTimeStep;
		}
	}

	__global__ void MoveParticlesPrediction_kernel(double _dTimeStep, unsigned _nParticles,
		const double* _partMasses, const double* _partInertiaMoments, const CVector3* _partMoments,
		CVector3* _partForces, CVector3* _partVels, CVector3* _partAnglVels, CQuaternion* _partQuaternions)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nParticles; i += blockDim.x * gridDim.x)
		{
			_partVels[i] += _partForces[i] / _partMasses[i] * _dTimeStep;
			if (m_considerAnisotropy)
			{
				const CMatrix3 rotMatrix = _partQuaternions[i].ToRotmat();
				const CVector3 vTemp = (rotMatrix.Transpose()*_partMoments[i]) / _partInertiaMoments[i];
				_partAnglVels[i] += rotMatrix * vTemp * _dTimeStep;
			}
			else
				_partAnglVels[i] += _partMoments[i] / _partInertiaMoments[i] * _dTimeStep;
		}
	}

	__global__ void UpdateTemperatures_kernel(double _dTimeStep, unsigned _nParticles, const double* _partHeatCapacities, const double* _partMasses,
		const double* _partHeatFluxes, double* _partTemperatures)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nParticles; i += blockDim.x * gridDim.x)
		{
			_partTemperatures[i] += _partHeatFluxes[i] / (_partHeatCapacities[i] * _partMasses[i]) * _dTimeStep;
			_partTemperatures[i] = _partTemperatures[i] < 0.0 ? 0.0 : _partTemperatures[i];
		}
	}

	__global__ void CalculateGeometryCenter_kernel(unsigned _nWallsInGeom, const unsigned* _wallsInGeom,
		CVector3* _vertex1, CVector3* _vertex2, CVector3* _vertex3, CVector3* _vCenter)
	{
		unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i != 0) return;
		_vCenter->Init(0);
		for (size_t j = 0; j < _nWallsInGeom; j++)
		{
			const unsigned iWall = _wallsInGeom[j];
			_vCenter[0] += (_vertex1[iWall] + _vertex2[iWall] + _vertex3[iWall]) / (3.0*_nWallsInGeom);
		}
	}

	__global__ void MoveWalls_kernel(const double _timeStep, const unsigned _nWallsInGeom, const CVector3 _vel, const CVector3 _rotVel, const CVector3 _definedRotCenter, const CMatrix3 _rotMatrix,
		const CVector3 _freeMotion, const CVector3* _totalForce, double _dMass, bool _bRotateAroundCenter,
		const CVector3 _vExternalAccel, CVector3* _vCalculatedCenter, const unsigned* _wallsInGeom, CVector3* _vertex1, CVector3* _vertex2, CVector3* _vertex3,
		CVector3* _wallMinCoord, CVector3* _wallMaxCoord, CVector3* _wallNormalVector, CVector3* _wallVel, CVector3* _wallRotCenter, CVector3* _wallRotVel)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nWallsInGeom; i += blockDim.x * gridDim.x)
		{
			const unsigned iWall = _wallsInGeom[i];

			CVector3 vRotCenter = _bRotateAroundCenter ? _vCalculatedCenter[0] : _definedRotCenter;
			CVector3 vel = _vel;
			if (!_freeMotion.IsZero())
			{
				if (_freeMotion.x != 0.0) vel.x = (_wallVel[iWall].x + _totalForce->x*_timeStep / _dMass + _vExternalAccel.x*_timeStep);
				if (_freeMotion.y != 0.0) vel.y = (_wallVel[iWall].y + _totalForce->y*_timeStep / _dMass + _vExternalAccel.y*_timeStep);
				if (_freeMotion.z != 0.0) vel.z = (_wallVel[iWall].z + _totalForce->z*_timeStep / _dMass + _vExternalAccel.z*_timeStep);
			}

			_wallVel[iWall] = vel;
			_wallRotVel[iWall] = _rotVel;
			_wallRotCenter[iWall] = vRotCenter;
			if (vel.x != 0.0 || vel.y != 0.0 || vel.z != 0.0)
			{
				_vertex1[iWall] += vel * _timeStep;
				_vertex2[iWall] += vel * _timeStep;
				_vertex3[iWall] += vel * _timeStep;
			}

			if (_rotVel.x != 0.0 || _rotVel.y != 0.0 || _rotVel.z != 0.0)
			{
				_vertex1[iWall] = vRotCenter + _rotMatrix * (_vertex1[iWall] - vRotCenter);
				_vertex2[iWall] = vRotCenter + _rotMatrix * (_vertex2[iWall] - vRotCenter);
				_vertex3[iWall] = vRotCenter + _rotMatrix * (_vertex3[iWall] - vRotCenter);

				_wallNormalVector[iWall] = Normalized((_vertex2[iWall] - _vertex1[iWall])*(_vertex3[iWall] - _vertex1[iWall]));
			}

			// update wall properties
			_wallMinCoord[iWall] = Min(_vertex1[iWall], _vertex2[iWall], _vertex3[iWall]);
			_wallMaxCoord[iWall] = Max(_vertex1[iWall], _vertex2[iWall], _vertex3[iWall]);
		}
	}

	__global__ void GatherForcesFromWalls_kernel(const unsigned _nWallsInGeom, const unsigned* _wallsInGeom, const CVector3* _wallForce, CVector3* _outForces)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nWallsInGeom; i += blockDim.x * gridDim.x)
			_outForces[i] = _wallForce[_wallsInGeom[i]];
	}

	__global__ void CopyCollisionsPP_kernel(unsigned _nCollisionsOld,
		const unsigned* _vVerlSrcOld, const unsigned* _vVerlDstOld, const unsigned* _vVerlDstNew, const unsigned* _vPartInd_VerNew, const bool* _oldActiveCollFlags,
		const double* _oldNormalOverlap, const double* _oldInitNormalOverlap, const CVector3* _oldTangOverlap, const CVector3* _oldContactVector, const CVector3* _oldTotalForce,
		double* _newNormalOverlap, double* _newInitNormalOverlap, CVector3* _newTangOverlap, CVector3* _newContactVector, CVector3* _newTotalForce)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nCollisionsOld; i += blockDim.x * gridDim.x)
		{
			if (!_oldActiveCollFlags[i]) continue;

			const unsigned iSrcOld = _vVerlSrcOld[i];
			const unsigned iDstOld = _vVerlDstOld[i];
			const unsigned iStart = _vPartInd_VerNew[iSrcOld];
			const int iEnd = _vPartInd_VerNew[iSrcOld + 1] - 1;

			for (int k = iStart; k <= iEnd; ++k)
				if (_vVerlDstNew[k] == iDstOld)
				{
					_newNormalOverlap[k] = _oldNormalOverlap[i];
					_newInitNormalOverlap[k] = _oldInitNormalOverlap[i];
					_newTangOverlap[k] = _oldTangOverlap[i];
					_newContactVector[k] = _oldContactVector[i];
					_newTotalForce[k] = _oldTotalForce[i];
					break;
				}
		}
	}

	__global__ void CopyCollisionsPW_kernel(unsigned _nCollisionsOld,
		const unsigned* _vVerlSrcOld, const unsigned* _vVerlDstOld, const unsigned* _vVerlDstNew, const unsigned* _vPartInd_VerNew, const bool* _oldActiveCollFlags,
		const double* _oldNormalOverlap, const CVector3* _oldTangOverlap, const CVector3* _oldContactVector, const CVector3* _oldTotalForce,
		bool* _newActiveCollFlags, double* _newNormalOverlap, CVector3* _newTangOverlap, CVector3* _newContactVector, CVector3* _newTotalForce)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nCollisionsOld; i += blockDim.x * gridDim.x)
		{
			// TODO: parfor by particles: do not run copy, if all contacts for this particle is disabled; copy all, if any is active.
			const unsigned iSrcOld = _vVerlSrcOld[i];
			const unsigned iDstOld = _vVerlDstOld[i];
			const unsigned iStart = _vPartInd_VerNew[iSrcOld];
			const int iEnd = _vPartInd_VerNew[iSrcOld + 1] - 1;

			for (int k = iStart; k <= iEnd; ++k)
				if (_vVerlDstNew[k] == iDstOld)
				{
					_newActiveCollFlags[k] = _oldActiveCollFlags[i]; // is needed for treatment of contact transition between adjacent triangles
					_newNormalOverlap[k]   = _oldNormalOverlap[i];
					_newTangOverlap[k]     = _oldTangOverlap[i];
					_newContactVector[k]   = _oldContactVector[i];
					_newTotalForce[k]      = _oldTotalForce[i];
					break;
				}
		}
	}

	__global__ void InitializePPCollisions_kernel(const unsigned _nCollisions, const unsigned* _vVerListSrc, const unsigned* _vVerListDst,
		const double* _partRadii, const double* _partMasses, const unsigned* _partCompoundIndices,
		unsigned* _collSrcID, unsigned* _collDstID, double* _collEquivMass, double* _collEquivRadius, double* _collSumRadii, uint16_t* _collInteractPropID)
	{
		for (unsigned iColl = blockIdx.x * blockDim.x + threadIdx.x; iColl < _nCollisions; iColl += blockDim.x * gridDim.x)
		{
			const unsigned iSrc = _vVerListSrc[iColl];
			const unsigned iDst = _vVerListDst[iColl];
			_collSrcID[iColl] = iSrc;
			_collDstID[iColl] = iDst;
			_collInteractPropID[iColl] = _partCompoundIndices[iSrc] * m_nCompoundsNumber + _partCompoundIndices[iDst];
			_collEquivRadius[iColl] = _partRadii[iSrc] * _partRadii[iDst] / (_partRadii[iSrc] + _partRadii[iDst]);
			_collEquivMass[iColl] = _partMasses[iSrc] * _partMasses[iDst] / (_partMasses[iSrc] + _partMasses[iDst]);
			_collSumRadii[iColl] = (_partRadii[iSrc] + _partRadii[iDst]);
		}
	}

	__global__ void InitializePWCollisions_kernel(const unsigned _nCollisions, const unsigned* _vVerListSrc, const unsigned* _vVerListDst,
		const unsigned* _partCompoundIndices, const unsigned* _wallCompoundIndices,
		unsigned* _collSrcID, unsigned* _collDstID, uint16_t* _collInteractPropID)
	{
		for (unsigned iColl = blockIdx.x * blockDim.x + threadIdx.x; iColl < _nCollisions; iColl += blockDim.x * gridDim.x)
		{
			const unsigned iPart = _vVerListSrc[iColl];
			const unsigned iWall = _vVerListDst[iColl];
			_collSrcID[iColl] = iWall; // it is assumed that src of collision corresponds to an ID of the wall
			_collDstID[iColl] = iPart;
			_collInteractPropID[iColl] = _wallCompoundIndices[iWall] * m_nCompoundsNumber + _partCompoundIndices[iPart];
		}
	}

	__global__ void UpdateActiveCollisionsPP_kernel(const unsigned _nCollisions, const unsigned* _vVerListSrc, const unsigned* _vVerListDst, const CVector3* _partCoords,
		const uint8_t* _collVirtShifts, const double* _collSumRadii, bool* _collActiveFlags, double* _collNormalOverlaps, double* _collInitNormalOverlaps,
		CVector3* _collContactVectors, CVector3* _collTangOverlaps)
	{
		for (unsigned iColl = blockIdx.x * blockDim.x + threadIdx.x; iColl < _nCollisions; iColl += blockDim.x * gridDim.x)
		{
			const unsigned iSrc = _vVerListSrc[iColl];
			const unsigned iDst = _vVerListDst[iColl];
			const double dSumRadii = _collSumRadii[iColl];
			CVector3 contactVector;
			double dSqrDistance;
			if (m_PBCGPU.bEnabled)
			{
				contactVector = (!_collVirtShifts[iColl] ? _partCoords[iDst] : GetVirtualProperty(_partCoords[iDst], _collVirtShifts[iColl], m_PBCGPU)) - _partCoords[iSrc];
				dSqrDistance = SquaredLength(contactVector);
			}
			else
			{
				contactVector = _partCoords[iDst] - _partCoords[iSrc];
				dSqrDistance = SquaredLength(contactVector);
			}
			const bool isActive = dSumRadii * dSumRadii > dSqrDistance;
			_collActiveFlags[iColl] = isActive;
			if (isActive)
			{
				_collNormalOverlaps[iColl] = dSumRadii - sqrt(dSqrDistance);
				_collContactVectors[iColl] = contactVector;
			}
			else
			{
				_collTangOverlaps[iColl].Init(0);
				_collInitNormalOverlaps[iColl] = 0;
			}
		}
	}

	__global__ void GetIntersectTypePW_kernel(const unsigned _nCollisions, const unsigned* _vVerListSrc, const unsigned* _vVerListDst,
		const double* _partRadii, const CVector3* _partCoords,
		const CVector3* _wallVertex1, const CVector3* _wallVertex2, const CVector3* _wallVertex3, const CVector3* _wallMinCoord, const CVector3* _wallMaxCoord, const CVector3* _wallNormalVector,
		const uint8_t* _collVirtShifts, EIntersectionType* _collTempIntersectionType, CVector3* _collContactPoint, bool* _bActivePart)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nCollisions; i += blockDim.x * gridDim.x)
		{
			const unsigned iWall = _vVerListDst[i];
			const unsigned iPart = _vVerListSrc[i];
			const double radius = _partRadii[iPart];
			const CVector3 partCoord = _partCoords[iPart];

			const CVector3 coord = !_collVirtShifts[i] ? partCoord : GetVirtualProperty(partCoord, _collVirtShifts[i], m_PBCGPU);

			const CVector3 minCoord = _wallMinCoord[iWall];
			const CVector3 maxCoord = _wallMaxCoord[iWall];

			if (coord.x <= minCoord.x - radius) { _collTempIntersectionType[i] = EIntersectionType::NO_CONTACT; continue; }
			if (coord.y <= minCoord.y - radius) { _collTempIntersectionType[i] = EIntersectionType::NO_CONTACT; continue; }
			if (coord.z <= minCoord.z - radius) { _collTempIntersectionType[i] = EIntersectionType::NO_CONTACT; continue; }
			if (coord.x >= maxCoord.x + radius) { _collTempIntersectionType[i] = EIntersectionType::NO_CONTACT; continue; }
			if (coord.y >= maxCoord.y + radius) { _collTempIntersectionType[i] = EIntersectionType::NO_CONTACT; continue; }
			if (coord.z >= maxCoord.z + radius) { _collTempIntersectionType[i] = EIntersectionType::NO_CONTACT; continue; }

			const CVector3 normalVector = _wallNormalVector[iWall];
			const CVector3 vertex1 = _wallVertex1[iWall];
			const CVector3 vertex2 = _wallVertex2[iWall];
			const CVector3 vertex3 = _wallVertex3[iWall];

			double PPD = DotProduct(coord - (vertex1 + vertex2 + vertex3) / 3.0, normalVector); // particle projection point distance
			if (fabs(PPD) >= radius) { _collTempIntersectionType[i] = EIntersectionType::NO_CONTACT; continue; }

			const CVector3 A = coord - normalVector * PPD; // projection point

			const CVector3 edge21 = vertex2 - vertex1;
			const CVector3 edge31 = vertex3 - vertex1;
			const CVector3 W = A - vertex1;

			const double d00 = DotProduct(edge21, edge21);
			const double d01 = DotProduct(edge21, edge31);
			const double d11 = DotProduct(edge31, edge31);
			const double d20 = DotProduct(W, edge21);
			const double d21 = DotProduct(W, edge31);
			const double invDenom = 1.0 / (d00 * d11 - d01 * d01);
			const double gamma = (d11 * d20 - d01 * d21) * invDenom;
			const double betta = (d00 * d21 - d01 * d20) * invDenom;
			const double alpha = 1.0f - gamma - betta;

			if ((gamma > 0 && gamma < 1) && (alpha > 0 && alpha < 1) && (betta > 0 && betta < 1))
			{
				_collContactPoint[i] = A;
				_collTempIntersectionType[i] = EIntersectionType::FACE_CONTACT;
				_bActivePart[iPart] = true;
			}
			else // A is outside polygon
			{
				const CVector3 edge32 = vertex3 - vertex2;
				const CVector3 edge13 = vertex1 - vertex3;
				double lc1 = fmin(fmax(DotProduct(A - vertex1, edge21) / SquaredLength(edge21), 0.), 1.);
				double lc2 = fmin(fmax(DotProduct(A - vertex2, edge32) / SquaredLength(edge32), 0.), 1.);
				double lc3 = fmin(fmax(DotProduct(A - vertex3, edge13) / SquaredLength(edge13), 0.), 1.);
				const bool C1IsVertice = lc1 == 0 || lc1 == 1;
				const bool C2IsVertice = lc2 == 0 || lc2 == 1;
				const bool C3IsVertice = lc3 == 0 || lc3 == 1;
				const CVector3 C1 = vertex1 + lc1 * edge21; // mistake in publication
				const CVector3 C2 = vertex2 + lc2 * edge32;
				const CVector3 C3 = vertex3 + lc3 * edge13;
				const double sqrLength1 = SquaredLength(C1 - coord);
				const double sqrLength2 = SquaredLength(C2 - coord);
				const double sqrLength3 = SquaredLength(C3 - coord);
				if (fmin(fmin(sqrLength1, sqrLength2), sqrLength3) >= radius * radius) { _collTempIntersectionType[i] = EIntersectionType::NO_CONTACT; continue; }
				_bActivePart[iPart] = true;
				if (sqrLength1 <= sqrLength2 && sqrLength1 <= sqrLength3)
				{
					_collContactPoint[i] = C1;
					_collTempIntersectionType[i] = C1IsVertice ? EIntersectionType::VERTEX_CONTACT : EIntersectionType::EDGE_CONTACT;

				}
				else if (sqrLength2 <= sqrLength3)
				{
					_collContactPoint[i] = C2;
					_collTempIntersectionType[i] = C2IsVertice ? EIntersectionType::VERTEX_CONTACT : EIntersectionType::EDGE_CONTACT;
				}
				else
				{
					_collContactPoint[i] = C3;
					_collTempIntersectionType[i] = C3IsVertice ? EIntersectionType::VERTEX_CONTACT : EIntersectionType::EDGE_CONTACT;
				}
			}
		}
	}

	__global__ void CombineIntersectionsPW_kernel(const unsigned* _nActiveParticles, const unsigned* _nActivePartIndexes,
		const unsigned _nParticles, const unsigned _nCollisions, const unsigned* _vVerList, const unsigned* _vIVerList,
		const CVector3* _wallNormalVector, EIntersectionType* _collTempIntersectionType, const uint8_t* _collVirtShifts)
	{
		for (unsigned nInd = blockIdx.x * blockDim.x + threadIdx.x; nInd < *_nActiveParticles; nInd += blockDim.x * gridDim.x)
		{
			const unsigned iPart = _nActivePartIndexes[nInd];
			const int iStart = _vIVerList[iPart];
			const int iEnd = (iPart < _nParticles - 1) ? (_vIVerList[iPart + 1] - 1) : (_nCollisions - 1);

			for (int i = iStart; i <= iEnd - 1; ++i)
				if (_collTempIntersectionType[i] != EIntersectionType::NO_CONTACT)
					for (int j = i + 1; j <= iEnd; ++j)
						if (_collTempIntersectionType[j] != EIntersectionType::NO_CONTACT && SquaredLength(_wallNormalVector[_vVerList[i]] - _wallNormalVector[_vVerList[j]]) < 1e-6)
							switch (_collTempIntersectionType[i])
							{
							case EIntersectionType::FACE_CONTACT:
								if (_collTempIntersectionType[j] == EIntersectionType::EDGE_CONTACT || _collTempIntersectionType[j] == EIntersectionType::VERTEX_CONTACT)
									_collTempIntersectionType[j] = EIntersectionType::NO_CONTACT;
								break;
							case EIntersectionType::EDGE_CONTACT:
								if (_collTempIntersectionType[j] == EIntersectionType::EDGE_CONTACT)
									_collTempIntersectionType[j] = EIntersectionType::NO_CONTACT;
								else if (_collTempIntersectionType[j] == EIntersectionType::FACE_CONTACT)
									_collTempIntersectionType[i] = EIntersectionType::NO_CONTACT;
								else
									_collTempIntersectionType[j] = EIntersectionType::NO_CONTACT;
								break;
							case EIntersectionType::VERTEX_CONTACT:
								_collTempIntersectionType[i] = EIntersectionType::NO_CONTACT;
								break;
							default:
								_collTempIntersectionType[i] = EIntersectionType::NO_CONTACT;
								break;
							}

			for (int i = iStart; i <= iEnd - 1; ++i)
				if (_collVirtShifts[i])
					for (int j = iStart; j <= iEnd - 1; ++j) // additional check that there is no contact between one wall and real and virtual particles
						if (j != i && _collTempIntersectionType[j] != EIntersectionType::NO_CONTACT && _collVirtShifts[j] == 0 && _vVerList[j] == _vVerList[i])
							_collTempIntersectionType[i] = EIntersectionType::NO_CONTACT;
		}
	}

	__global__ void UpdateActiveCollisionsPW_kernel(const unsigned _nCollisions, const EIntersectionType* _collTempIntersectionType,
		bool* _collActiveFlags, bool* _callActivated, bool* _callDeactivated)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nCollisions; i += blockDim.x * gridDim.x)
		{
			const bool active = _collTempIntersectionType[i] != EIntersectionType::NO_CONTACT;
			if (_collActiveFlags[i] ^ active)	// old and new activity flags are different
			{
				_callActivated[i] = active;
				_callDeactivated[i] = !active;
			}
			_collActiveFlags[i] = (_collTempIntersectionType[i] != EIntersectionType::NO_CONTACT);
		}
	}

	__global__ void CopyCollisionsForAdjacentWalls(const unsigned* _nActivatedColls, const unsigned* _activatedCollIndices, const bool* _callDeactivated,
		const unsigned* _verlSrcIDs, const unsigned* _verlDstIDs, const unsigned* _verlSrcStartIndices,
		const unsigned* _adjWallsIDs, const unsigned* _adjWallsStartIndices,
		CVector3* _tangOverlaps)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < *_nActivatedColls; i += blockDim.x * gridDim.x)
		{
			const unsigned iColl      = _activatedCollIndices[i];
			const unsigned partID     = _verlSrcIDs[iColl];
			const unsigned wallID     = _verlDstIDs[iColl];
			const unsigned iStartColl = _verlSrcStartIndices[partID];
			const int iEndColl        = _verlSrcStartIndices[partID + 1] - 1;
			const unsigned iStartWall = _adjWallsStartIndices[wallID];
			const int iEndWall        = _adjWallsStartIndices[wallID + 1] - 1;
			bool found = false;

			for (int j = iStartColl; j <= iEndColl && !found; ++j)	// iterate all collisions for particle 'partID'
				if (_callDeactivated[j])							// found collision, deactivated on the last time step
					for (int k = iStartWall; k <= iEndWall; ++k)	// iterate all walls adjacent to wall wallID
						if (_verlDstIDs[j] == _adjWallsIDs[k])		// ID of the wall in this deactivated collision belongs to the list of adjacent walls
						{
							_tangOverlaps[iColl] = _tangOverlaps[j];
							found = true;
							break;
						}
		}
	}

	__global__ void CheckParticlesInDomain_kernel(double _currTime, const unsigned _nParticles,
		unsigned* _partActivity, double* _partEndActivity, const CVector3* _partCoords)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nParticles; i += blockDim.x * gridDim.x)
			if (_partActivity[i] && !IsPointInDomain(m_simulationDomain, _partCoords[i]))
			{
				_partActivity[i] = 0;
				_partEndActivity[i] = _currTime;
			}
	}

	__global__ void CheckBondsActivity_kernel(double _currTime, const unsigned _nBonds, const unsigned* _partActivity,
		uint8_t* _bondActive, const unsigned* _bondLeftID, const unsigned* _bondRightID, double* _bondEndActivity)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nBonds; i += blockDim.x * gridDim.x)
			if (_bondActive[i] && (!_partActivity[_bondLeftID[i]] || !_partActivity[_bondRightID[i]]))
			{
				_bondActive[i] = 0;
				_bondEndActivity[i] = _currTime;
			}
	}

	__global__ void MoveVirtualParticlesBox(const unsigned _nParticles, const unsigned* _partActivity,
		CVector3* _partCoords, CVector3* _partCoordsVerlet, uint8_t* _vCrossingShifts, bool* _vCrossingFlags)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nParticles; i += blockDim.x * gridDim.x)
		{
			if (!_partActivity[i]) continue;

			// particle crossed left boundary
			if (m_PBCGPU.bX && _partCoords[i].x <= m_PBCGPU.currentDomain.coordBeg.x)  _vCrossingShifts[i] = _vCrossingShifts[i] | 32;
			if (m_PBCGPU.bY && _partCoords[i].y <= m_PBCGPU.currentDomain.coordBeg.y) _vCrossingShifts[i] = _vCrossingShifts[i] | 8;
			if (m_PBCGPU.bZ && _partCoords[i].z <= m_PBCGPU.currentDomain.coordBeg.z) _vCrossingShifts[i] = _vCrossingShifts[i] | 2;
			// particle crossed right boundary
			if (m_PBCGPU.bX && _partCoords[i].x >= m_PBCGPU.currentDomain.coordEnd.x) _vCrossingShifts[i] = _vCrossingShifts[i] | 16;
			if (m_PBCGPU.bY && _partCoords[i].y >= m_PBCGPU.currentDomain.coordEnd.y) _vCrossingShifts[i] = _vCrossingShifts[i] | 4;
			if (m_PBCGPU.bZ && _partCoords[i].z >= m_PBCGPU.currentDomain.coordEnd.z) _vCrossingShifts[i] = _vCrossingShifts[i] | 1;

			if (_vCrossingShifts[i])
			{
				_partCoords[i] += GetVectorFromVirtShift(	_vCrossingShifts[i], m_PBCGPU.boundaryShift );
				_partCoordsVerlet[i] += GetVectorFromVirtShift(_vCrossingShifts[i], m_PBCGPU.boundaryShift);
				_vCrossingFlags[i] = true;
			}
		}
	}

	__global__ void AddShiftsToCollisions(const unsigned _nParticles, const unsigned* _nCrossedParticles, const unsigned* _partCrossedIndices, const uint8_t* _vCrossingShifts,
		const unsigned* _vVerListSrcSortInd, const unsigned* _vVerListDstSortInd, const unsigned* _vVerListDstSortColl,
		const unsigned _nCollisions, const unsigned* _collSrcID, const unsigned* _collDstID, uint8_t* _collVirtShifts)
	{
		for (unsigned iActive = blockIdx.x * blockDim.x + threadIdx.x; iActive < *_nCrossedParticles; iActive += blockDim.x * gridDim.x)
		{
			if (_nCollisions == 0) return;
			const unsigned iPart = _partCrossedIndices[iActive];

			// go through src
			const int iStartSrc = _vVerListSrcSortInd[iPart];
			const int iEndSrc = _vVerListSrcSortInd[iPart + 1] - 1;
			for (int iColl = iStartSrc; iColl <= iEndSrc; ++iColl)
			{
				if (_vCrossingShifts[iPart])								// src particle in this collision crossed boundary
				{
					_collVirtShifts[iColl] = AddVirtShift(_collVirtShifts[iColl], _vCrossingShifts[iPart]);				// add shift from the src particle
					const unsigned iDstPart = _collDstID[iColl];					// index of the dst particle in this collision
					if( _vCrossingShifts[iDstPart] )						// dst particle in this collision also crossed boundary
						_collVirtShifts[iColl] = SubstractVirtShift(_collVirtShifts[iColl], _vCrossingShifts[iDstPart]); // add shift from the dst particle
				}
			}

			// go through dst
			const int iStartDst = _vVerListDstSortInd[iPart];
			const int iEndDst = _vVerListDstSortInd[iPart + 1] - 1;
			for (int i = iStartDst; i <= iEndDst; ++i)
			{
				const unsigned iColl = _vVerListDstSortColl[i];
				if (_vCrossingShifts[_collSrcID[iColl]]) continue;	// src particle already treated during passage by src
				if (_vCrossingShifts[iPart])							// only dst particle in this collision crossed boundary
					_collVirtShifts[iColl] = SubstractVirtShift(_collVirtShifts[iColl], _vCrossingShifts[iPart]); // add shift from the dst particle
			}
		}
	}

	__global__ void FillUniqueIndexes_kernel(unsigned _num, const unsigned* _idata, unsigned* _odata)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _num; i += blockDim.x * gridDim.x)
		{
			if (i == 0)
				_odata[_idata[0]] = 0;
			else if (_idata[i] != _idata[i - 1])
				_odata[_idata[i]] = i;
		}
	}

	__global__ void FillNonExistendIndexes_kernel(unsigned _nPart, unsigned _nColl, unsigned* _inData, unsigned* _outData)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nPart; i += blockDim.x * gridDim.x)
		{
			if (_inData[i] >= _nColl) // this indicates that this element was not found
			{
				// search for next filled element
				unsigned j = i + 1;
				while (j < _nPart - 1 && _inData[j] >= _nColl) ++j;
				// fill data
				if (j >= _nPart)
					_outData[i] = _nColl;
				else
					_outData[i] = _inData[j];
			}
			else
				_outData[i] = _inData[i];
		}
	}

	__global__ void GatherSquaredPartVerletDistances_kernel(const unsigned _nParticles,
		const CVector3* _partCoord, const CVector3* _partCoordVerlet, const unsigned* _partActivity, double* _distances)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nParticles; i += blockDim.x * gridDim.x)
			if (_partActivity[i])
				_distances[i] = SquaredLength(_partCoord[i], _partCoordVerlet[i]);
			else
				_distances[i] = 0;
	}

	__global__ void GatherWallVelocities_kernel(const unsigned _nWalls, const CVector3* _vertex1, const CVector3* _vertex2, const CVector3* _vertex3,
		const CVector3* _vel, const CVector3* _rotCenter, const CVector3* _rotVel, double* _outVelocities)
	{
		for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < _nWalls; i += blockDim.x * gridDim.x)
		{
			const double dSquaredRotVel = _rotVel[i].SquaredLength();
			if (dSquaredRotVel > 0)
				_outVelocities[i] = _vel[i].Length() + sqrt(dSquaredRotVel * fmax(fmax(SquaredLength(_vertex1[i] - _rotCenter[i]), SquaredLength(_vertex2[i] - _rotCenter[i])), SquaredLength(_vertex3[i] - _rotCenter[i])));
			else
				_outVelocities[i] = _vel[i].Length();
		}
	}

	__global__ void GetPPOverlaps_kernel(const unsigned* _nActiveCollisions, const unsigned* _collActiveIndices, const unsigned* _collSrcID, const unsigned* _collDstID,
		const double* _collNormalOverlaps, const unsigned _maxParticleID, double* _overlaps, uint8_t* _flags)
	{
		for (unsigned iActiveColl = blockIdx.x * blockDim.x + threadIdx.x; iActiveColl < *_nActiveCollisions; iActiveColl += blockDim.x * gridDim.x)
		{
			const unsigned iColl = _collActiveIndices[iActiveColl];
			const unsigned iSrc = _collSrcID[iColl];
			const unsigned iDst = _collDstID[iColl];

			const bool consider = iSrc < _maxParticleID || iDst < _maxParticleID;
			_overlaps[iColl] = consider ? _collNormalOverlaps[iColl] : 0;
			_flags[iColl] = consider ? 1 : 0;

		}
	}

	__global__ void GetPWOverlaps_kernel(const unsigned* _nActiveCollisions, const unsigned* _collActiveIndices, const unsigned* _collPartID,
		const uint8_t* _collVirtShifts, const CVector3* _collContactVectors,
		const CVector3* _partCoords, const double* _partRadii, const unsigned _maxParticleID, double* _overlaps, uint8_t* _flags)
	{
		for (unsigned iActiveColl = blockIdx.x * blockDim.x + threadIdx.x; iActiveColl < *_nActiveCollisions; iActiveColl += blockDim.x * gridDim.x)
		{
			const unsigned iColl = _collActiveIndices[iActiveColl];
			const unsigned iPart = _collPartID[iColl];

			const bool consider = iPart < _maxParticleID;
			const CVector3 vRc = _VIRTUAL_COORDINATE(_partCoords[iPart], _collVirtShifts[iColl], m_PBCGPU) - _collContactVectors[iColl];
			_overlaps[iColl] = _partRadii[iPart] - vRc.Length();
			_flags[iColl] = consider ? 1 : 0;
		}
	}

	__global__ void ReduceSum_kernel(const unsigned _num, const unsigned* _idata, unsigned* _odata)
	{
		extern __shared__ volatile unsigned usum[];
		const unsigned iThread = threadIdx.x;
		unsigned i = blockIdx.x * (THREADS_PER_BLOCK * 2) + iThread;
		const unsigned stride = THREADS_PER_BLOCK * 2 * gridDim.x;

		usum[iThread] = 0;
		while (i < _num)
		{
			if (i + THREADS_PER_BLOCK < _num)
				usum[iThread] += _idata[i] + _idata[i + THREADS_PER_BLOCK];
			else
				usum[iThread] += _idata[i];
			i += stride;
		}

		CUDA_SYNCTHREADS;

		if (THREADS_PER_BLOCK >= 512) { if (iThread < 256) { usum[iThread] += usum[iThread + 256]; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 256) { if (iThread < 128) { usum[iThread] += usum[iThread + 128]; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 128) { if (iThread <  64) { usum[iThread] += usum[iThread +  64]; } CUDA_SYNCTHREADS; }
		if (iThread < 32)
		{
			if (THREADS_PER_BLOCK >= 64) { usum[iThread] += usum[iThread + 32]; }
			if (THREADS_PER_BLOCK >= 32) { usum[iThread] += usum[iThread + 16]; }
			if (THREADS_PER_BLOCK >= 16) { usum[iThread] += usum[iThread +  8]; }
			if (THREADS_PER_BLOCK >=  8) { usum[iThread] += usum[iThread +  4]; }
			if (THREADS_PER_BLOCK >=  4) { usum[iThread] += usum[iThread +  2]; }
			if (THREADS_PER_BLOCK >=  2) { usum[iThread] += usum[iThread +  1]; }
		}
		if (iThread == 0) _odata[blockIdx.x] = usum[0];
	}

	__global__ void ReduceSum_kernel(const unsigned _num, const double* _idata, double* _odata)
	{
		extern __shared__ volatile double dsum[];
		const unsigned iThread = threadIdx.x;
		unsigned i = blockIdx.x * (THREADS_PER_BLOCK * 2) + iThread;
		const unsigned stride = THREADS_PER_BLOCK * 2 * gridDim.x;

		dsum[iThread] = 0;
		while (i < _num)
		{
			if (i + THREADS_PER_BLOCK < _num)
				dsum[iThread] += _idata[i] + _idata[i + THREADS_PER_BLOCK];
			else
				dsum[iThread] += _idata[i];
			i += stride;
		}

		CUDA_SYNCTHREADS;

		if (THREADS_PER_BLOCK >= 512) { if (iThread < 256) { dsum[iThread] += dsum[iThread + 256]; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 256) { if (iThread < 128) { dsum[iThread] += dsum[iThread + 128]; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 128) { if (iThread <  64) { dsum[iThread] += dsum[iThread +  64]; } CUDA_SYNCTHREADS; }
		if (iThread < 32)
		{
			if (THREADS_PER_BLOCK >= 64) { dsum[iThread] += dsum[iThread + 32]; }
			if (THREADS_PER_BLOCK >= 32) { dsum[iThread] += dsum[iThread + 16]; }
			if (THREADS_PER_BLOCK >= 16) { dsum[iThread] += dsum[iThread +  8]; }
			if (THREADS_PER_BLOCK >=  8) { dsum[iThread] += dsum[iThread +  4]; }
			if (THREADS_PER_BLOCK >=  4) { dsum[iThread] += dsum[iThread +  2]; }
			if (THREADS_PER_BLOCK >=  2) { dsum[iThread] += dsum[iThread +  1]; }
		}
		if (iThread == 0) _odata[blockIdx.x] = dsum[0];
	}

	__global__ void ReduceSum_kernel(const unsigned _num, const CVector3* _idata, CVector3* _odata)
	{
		extern __shared__ CVector3 vsum[];
		const unsigned iThread = threadIdx.x;
		unsigned i = blockIdx.x * (THREADS_PER_BLOCK * 2) + iThread;
		const unsigned stride = THREADS_PER_BLOCK * 2 * gridDim.x;

		vsum[iThread].Init(0);
		while (i < _num)
		{
			if (i + THREADS_PER_BLOCK < _num)
				vsum[iThread] += _idata[i] + _idata[i + THREADS_PER_BLOCK];
			else
				vsum[iThread] += _idata[i];
			i += stride;
		}

		CUDA_SYNCTHREADS;

		if (THREADS_PER_BLOCK >= 512) { if (iThread < 256) { vsum[iThread] += vsum[iThread + 256]; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 256) { if (iThread < 128) { vsum[iThread] += vsum[iThread + 128]; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 128) { if (iThread <  64) { vsum[iThread] += vsum[iThread +  64]; } CUDA_SYNCTHREADS; }
		if (iThread < 32)
		{
			if (THREADS_PER_BLOCK >= 64) { vsum[iThread] += vsum[iThread + 32]; }
			if (THREADS_PER_BLOCK >= 32) { vsum[iThread] += vsum[iThread + 16]; }
			if (THREADS_PER_BLOCK >= 16) { vsum[iThread] += vsum[iThread +  8]; }
			if (THREADS_PER_BLOCK >=  8) { vsum[iThread] += vsum[iThread +  4]; }
			if (THREADS_PER_BLOCK >=  4) { vsum[iThread] += vsum[iThread +  2]; }
			if (THREADS_PER_BLOCK >=  2) { vsum[iThread] += vsum[iThread +  1]; }
		}
		if (iThread == 0) _odata[blockIdx.x] = vsum[0];
	}

	__global__ void ReduceMax_kernel(unsigned _num, const CVector3* _idata, double* _odata)
	{
		extern __shared__ volatile double vmax[];
		unsigned iThread = threadIdx.x;
		unsigned i = blockIdx.x * (THREADS_PER_BLOCK * 2) + iThread;
		const unsigned stride = THREADS_PER_BLOCK * 2 * gridDim.x;
		double temp;

		vmax[iThread] = -DBL_MAX;
		while (i < _num)
		{
			if (i + THREADS_PER_BLOCK < _num)
				vmax[iThread] = fmax(vmax[iThread], fmax(_idata[i].SquaredLength(), _idata[i + THREADS_PER_BLOCK].SquaredLength()));
			else
				vmax[iThread] = fmax(vmax[iThread], _idata[i].SquaredLength());
			i += stride;
		}

		CUDA_SYNCTHREADS;

		if (THREADS_PER_BLOCK >= 512) { if (iThread < 256) { temp = vmax[iThread + 256]; if (temp > vmax[iThread]) vmax[iThread] = temp; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 256) { if (iThread < 128) { temp = vmax[iThread + 128]; if (temp > vmax[iThread]) vmax[iThread] = temp; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 128) { if (iThread <  64) { temp = vmax[iThread +  64]; if (temp > vmax[iThread]) vmax[iThread] = temp; } CUDA_SYNCTHREADS; }
		if (iThread < 32)
		{
			if (THREADS_PER_BLOCK >= 64) { temp = vmax[iThread + 32]; if (temp > vmax[iThread]) vmax[iThread] = temp; }
			if (THREADS_PER_BLOCK >= 32) { temp = vmax[iThread + 16]; if (temp > vmax[iThread]) vmax[iThread] = temp; }
			if (THREADS_PER_BLOCK >= 16) { temp = vmax[iThread +  8]; if (temp > vmax[iThread]) vmax[iThread] = temp; }
			if (THREADS_PER_BLOCK >=  8) { temp = vmax[iThread +  4]; if (temp > vmax[iThread]) vmax[iThread] = temp; }
			if (THREADS_PER_BLOCK >=  4) { temp = vmax[iThread +  2]; if (temp > vmax[iThread]) vmax[iThread] = temp; }
			if (THREADS_PER_BLOCK >=  2) { temp = vmax[iThread +  1]; if (temp > vmax[iThread]) vmax[iThread] = temp; }
		}
		if (iThread == 0) _odata[blockIdx.x] = vmax[0];
	}

	__global__ void ReduceMax_kernel(unsigned _num, const double* _idata, double* _odata)
	{
		extern __shared__ volatile double dmax[];
		unsigned iThread = threadIdx.x;
		unsigned i = blockIdx.x * (THREADS_PER_BLOCK * 2) + iThread;
		const unsigned stride = THREADS_PER_BLOCK * 2 * gridDim.x;
		double temp;

		dmax[iThread] = -DBL_MAX;
		while (i < _num)
		{
			if (i + THREADS_PER_BLOCK < _num)
				dmax[iThread] = fmax(dmax[iThread], fmax(_idata[i], _idata[i + THREADS_PER_BLOCK]));
			else
				dmax[iThread] = fmax(dmax[iThread], _idata[i]);
			i += stride;
		}

		CUDA_SYNCTHREADS;

		if (THREADS_PER_BLOCK >= 512) { if (iThread < 256) { temp = dmax[iThread + 256]; if (temp > dmax[iThread]) dmax[iThread] = temp; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 256) { if (iThread < 128) { temp = dmax[iThread + 128]; if (temp > dmax[iThread]) dmax[iThread] = temp; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 128) { if (iThread <  64) { temp = dmax[iThread +  64]; if (temp > dmax[iThread]) dmax[iThread] = temp; } CUDA_SYNCTHREADS; }
		if (iThread < 32)
		{
			if (THREADS_PER_BLOCK >= 64) { temp = dmax[iThread + 32]; if (temp > dmax[iThread]) dmax[iThread] = temp; }
			if (THREADS_PER_BLOCK >= 32) { temp = dmax[iThread + 16]; if (temp > dmax[iThread]) dmax[iThread] = temp; }
			if (THREADS_PER_BLOCK >= 16) { temp = dmax[iThread +  8]; if (temp > dmax[iThread]) dmax[iThread] = temp; }
			if (THREADS_PER_BLOCK >=  8) { temp = dmax[iThread +  4]; if (temp > dmax[iThread]) dmax[iThread] = temp; }
			if (THREADS_PER_BLOCK >=  4) { temp = dmax[iThread +  2]; if (temp > dmax[iThread]) dmax[iThread] = temp; }
			if (THREADS_PER_BLOCK >=  2) { temp = dmax[iThread +  1]; if (temp > dmax[iThread]) dmax[iThread] = temp; }
		}
		if (iThread == 0) _odata[blockIdx.x] = dmax[0];
	}

	__global__ void ReduceMin_kernel(unsigned _num, const CVector3* _idata, double* _odata)
	{
		extern __shared__ volatile double vmin[];
		unsigned iThread = threadIdx.x;
		unsigned i = blockIdx.x * (THREADS_PER_BLOCK * 2) + iThread;
		const unsigned stride = THREADS_PER_BLOCK * 2 * gridDim.x;
		double temp;

		vmin[iThread] = DBL_MAX;
		while (i < _num)
		{
			if (i + THREADS_PER_BLOCK < _num)
				vmin[iThread] = fmin(vmin[iThread], fmin(_idata[i].SquaredLength(), _idata[i + THREADS_PER_BLOCK].SquaredLength()));
			else
				vmin[iThread] = fmin(vmin[iThread], _idata[i].SquaredLength());
			i += stride;
		}

		CUDA_SYNCTHREADS;

		if (THREADS_PER_BLOCK >= 512) { if (iThread < 256) { temp = vmin[iThread + 256]; if (temp < vmin[iThread]) vmin[iThread] = temp; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 256) { if (iThread < 128) { temp = vmin[iThread + 128]; if (temp < vmin[iThread]) vmin[iThread] = temp; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 128) { if (iThread <  64) { temp = vmin[iThread +  64]; if (temp < vmin[iThread]) vmin[iThread] = temp; } CUDA_SYNCTHREADS; }
		if (iThread < 32)
		{
			if (THREADS_PER_BLOCK >= 64) { temp = vmin[iThread + 32]; if (temp < vmin[iThread]) vmin[iThread] = temp; }
			if (THREADS_PER_BLOCK >= 32) { temp = vmin[iThread + 16]; if (temp < vmin[iThread]) vmin[iThread] = temp; }
			if (THREADS_PER_BLOCK >= 16) { temp = vmin[iThread +  8]; if (temp < vmin[iThread]) vmin[iThread] = temp; }
			if (THREADS_PER_BLOCK >=  8) { temp = vmin[iThread +  4]; if (temp < vmin[iThread]) vmin[iThread] = temp; }
			if (THREADS_PER_BLOCK >=  4) { temp = vmin[iThread +  2]; if (temp < vmin[iThread]) vmin[iThread] = temp; }
			if (THREADS_PER_BLOCK >=  2) { temp = vmin[iThread +  1]; if (temp < vmin[iThread]) vmin[iThread] = temp; }
		}
		if (iThread == 0) _odata[blockIdx.x] = vmin[0];
	}

	__global__ void ReduceMin_kernel(unsigned _num, const double* _idata, double* _odata)
	{
		extern __shared__ volatile double dmin[];
		unsigned iThread = threadIdx.x;
		unsigned i = blockIdx.x * (THREADS_PER_BLOCK * 2) + iThread;
		const unsigned stride = THREADS_PER_BLOCK * 2 * gridDim.x;
		double temp;

		dmin[iThread] = DBL_MAX;
		while (i < _num)
		{
			if (i + THREADS_PER_BLOCK < _num)
				dmin[iThread] = fmin(dmin[iThread], fmin(_idata[i], _idata[i + THREADS_PER_BLOCK]));
			else
				dmin[iThread] = fmin(dmin[iThread], _idata[i]);
			i += stride;
		}

		CUDA_SYNCTHREADS;

		if (THREADS_PER_BLOCK >= 512) { if (iThread < 256) { temp = dmin[iThread + 256]; if (temp < dmin[iThread]) dmin[iThread] = temp; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 256) { if (iThread < 128) { temp = dmin[iThread + 128]; if (temp < dmin[iThread]) dmin[iThread] = temp; } CUDA_SYNCTHREADS; }
		if (THREADS_PER_BLOCK >= 128) { if (iThread <  64) { temp = dmin[iThread +  64]; if (temp < dmin[iThread]) dmin[iThread] = temp; } CUDA_SYNCTHREADS; }
		if (iThread < 32)
		{
			if (THREADS_PER_BLOCK >= 64) { temp = dmin[iThread + 32]; if (temp < dmin[iThread]) dmin[iThread] = temp; }
			if (THREADS_PER_BLOCK >= 32) { temp = dmin[iThread + 16]; if (temp < dmin[iThread]) dmin[iThread] = temp; }
			if (THREADS_PER_BLOCK >= 16) { temp = dmin[iThread +  8]; if (temp < dmin[iThread]) dmin[iThread] = temp; }
			if (THREADS_PER_BLOCK >=  8) { temp = dmin[iThread +  4]; if (temp < dmin[iThread]) dmin[iThread] = temp; }
			if (THREADS_PER_BLOCK >=  4) { temp = dmin[iThread +  2]; if (temp < dmin[iThread]) dmin[iThread] = temp; }
			if (THREADS_PER_BLOCK >=  2) { temp = dmin[iThread +  1]; if (temp < dmin[iThread]) dmin[iThread] = temp; }
		}
		if (iThread == 0) _odata[blockIdx.x] = dmin[0];
	}
}
