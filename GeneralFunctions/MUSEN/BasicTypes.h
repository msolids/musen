/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "Vector3.h"
#include "Matrix3.h"
#include "Quaternion.h"
#include <vector>
#include <sstream>
#include <numeric>
#include <cuda_runtime.h>
#include <map>

#pragma warning(push)
#pragma warning(disable: 26495)

struct SSelectiveSavingFlags
{
	// particles
	bool bCoordinates   = true;
	bool bVelocity      = true;
	bool bAngVelocity   = true;
	bool bQuaternion    = true;
	bool bForce         = true;
	bool bTensor        = true;
	// solid bonds
	bool bSBForce       = true;
	bool bSBTangOverlap = true;
	bool bSBTotTorque   = true;
	// liquid bonds
	bool bLBForce       = true;
	// triangular walls
	bool bTWPlaneCoord  = true;
	bool bTWForce       = true;
	bool bTWVelocity    = true;

	void SetAll(bool _flag)	{ bCoordinates = bVelocity = bAngVelocity = bQuaternion = bForce = bTensor = bSBForce = bSBTangOverlap = bSBTotTorque = bLBForce = bTWPlaneCoord = bTWForce = bTWVelocity = _flag;	}

	void SetAllParticles(bool _flag) { bCoordinates = bVelocity = bAngVelocity = bQuaternion = bForce = bTensor = _flag; }
	void SetAllSolidBonds(bool _flag) { bSBForce = bSBTangOverlap = bSBTotTorque = _flag; }
	void SetAllLiquidBonds(bool _flag) { bLBForce = _flag; }
	void SetAllWalls(bool _flag) { bTWPlaneCoord = bTWForce = bTWVelocity = _flag; }
};

//type definition for the volume borders coordinates
struct SVolumeType
{
	CVector3 coordBeg;
	CVector3 coordEnd;
	bool IsInf() const { return coordBeg.IsInf() || coordEnd.IsInf(); }
	friend std::istream& operator >> (std::istream& _s, SVolumeType& _v) { _s >> _v.coordBeg >> _v.coordEnd; return _s; }
};

struct SInteractProps
{
	double dRollingFriction;
	double dRestCoeff;
	double dSlidingFriction;
	double dAlpha;
	double dEquivYoungModulus;
	double dEquivShearModulus;
	double dEquivSurfaceTension;
	double dEquivSurfaceEnergy;
	double dEquivThermalConductivity;
};

namespace
{
	// return the length of the cylindrical bond which connects two spheres
	CUDA_DEVICE double CalculateBondLength(double _dDistanceBetweenParticleCenter, double _dR1, double _dR2, double _dBondDiameter)
	{
		double dResult = _dDistanceBetweenParticleCenter;
		if ( (_dBondDiameter < 2*_dR1) && ( _dBondDiameter < 2*_dR2 ))
			dResult -= sqrt(_dR1*_dR1 - _dBondDiameter*_dBondDiameter / 4) + sqrt(_dR2*_dR2 - _dBondDiameter*_dBondDiameter / 4);
		return dResult;
	}
}


//////////////////////////////////////////////////////////////////////////
/// Work with periodic boundary conditions.

// Periodic boundary conditions.
struct SPBC
{
	bool bEnabled;
	bool bX, bY, bZ;			// PBC borders state.
	SVolumeType initDomain;		// PBC domain in the initial time point.
	SVolumeType currentDomain;	// PBC domain in current time point.
	CVector3 boundaryShift;		// Shift to move particles crossing PBC boundaries. Excessive information to accelerate calculations.
	CVector3 vVel;				// velocity of motion of PBC boundaries

	void SetDefaultValues()
	{
		bEnabled = false;
		bX = bY = bZ = false;
		SetDomain(CVector3(-5e-3), CVector3(5e-3));
		vVel.Init(0);
	}

	// Set box domain and calculate corresponding boundaryShift.
	void SetDomain(const CVector3& _beg, const CVector3& _end)
	{
		initDomain = { _beg, _end };
		UpdatePBC(0);
	}

	// Returns true if the specified coordinate is within the PBC volume.
	bool IsCoordInPBC(const CVector3& _coord, const double _dTime)
	{
		UpdatePBC(_dTime);
		const bool xL = bX && (_coord.x <= currentDomain.coordBeg.x);
		const bool yL = bY && (_coord.y <= currentDomain.coordBeg.y);
		const bool zL = bZ && (_coord.z <= currentDomain.coordBeg.z);
		const bool xG = bX && (_coord.x >= currentDomain.coordEnd.x);
		const bool yG = bY && (_coord.y >= currentDomain.coordEnd.y);
		const bool zG = bZ && (_coord.z >= currentDomain.coordEnd.z);
		if (xL || yL || zL || xG || yG || zG)
			return false;
		return true;
	}

	void UpdatePBC(const double _dTime)
	{
		currentDomain.coordBeg = initDomain.coordBeg - vVel * _dTime;
		currentDomain.coordEnd = initDomain.coordEnd + vVel * _dTime;
		boundaryShift = currentDomain.coordEnd - currentDomain.coordBeg;
	}
};

// TODO: remove "VOLUME_"
enum class EVolumeShape : unsigned
{
	VOLUME_SPHERE = 0,
	VOLUME_BOX = 1,
	VOLUME_CYLINDER = 2,
	VOLUME_HOLLOW_SPHERE = 3,
	VOLUME_STL = 5
};

inline std::map<EVolumeShape, std::string> AllStandardVolumeTypes()
{
	return std::map<EVolumeShape, std::string>{
		{ EVolumeShape::VOLUME_BOX,				"Box" },
		{ EVolumeShape::VOLUME_CYLINDER,		"Cylinder" },
		{ EVolumeShape::VOLUME_HOLLOW_SPHERE,	"Hollow sphere" },
		{ EVolumeShape::VOLUME_SPHERE,			"Sphere" },
		{ EVolumeShape::VOLUME_STL,				"STL" } };
}

/// Macros to get virtual properties from models.

#define _VIRTUAL_COORDINATE(coord, shift_info, m_PBC) (((shift_info) != 0) ? GetVirtualProperty(coord, shift_info, m_PBC) : (coord))

#define CPU_GET_VIRTUAL_COORDINATE(coord)	_VIRTUAL_COORDINATE(coord, _pCollision->nVirtShift, m_PBC)
#define GPU_GET_VIRTUAL_COORDINATE(coord)	_VIRTUAL_COORDINATE(coord, _collVirtShifts[iColl], PBC)

namespace
{
	CUDA_HOST_DEVICE CVector3 GetVectorFromVirtShift(const uint8_t _virtShift, const CVector3& _vBoundaryShift)
	{
		return CVector3{
			(_virtShift >> 5 & 1)*_vBoundaryShift.x - (_virtShift >> 4 & 1)*_vBoundaryShift.x,
			(_virtShift >> 3 & 1)*_vBoundaryShift.y - (_virtShift >> 2 & 1)*_vBoundaryShift.y,
			(_virtShift >> 1 & 1)*_vBoundaryShift.z - (_virtShift & 1)*_vBoundaryShift.z };
	}

	CUDA_HOST_DEVICE uint8_t GetVirtShiftFromVector(const CVector3& _vVector )
	{
		uint8_t vVal = 0;
		if (_vVector.z < 0) vVal = vVal | 1;
		if (_vVector.z > 0) vVal = vVal | 2;
		if (_vVector.y < 0) vVal = vVal | 4;
		if (_vVector.y > 0) vVal = vVal | 8;
		if (_vVector.x < 0) vVal = vVal | 16;
		if (_vVector.x > 0) vVal = vVal | 32;
		return vVal;
	}

	CUDA_HOST_DEVICE uint8_t AddVirtShift(uint8_t _old,  uint8_t _new )
	{
		uint8_t temp = _old | _new;
		if (temp & 1 && temp & 2) temp -= 3;
		if (temp & 4 && temp & 8) temp -= 12;
		if (temp & 16 && temp & 32) temp -= 48;
		return temp;
	}
	CUDA_HOST_DEVICE uint8_t InverseVirtShift(uint8_t _new)
	{
		return ((_new << 1) & 42) + ((_new >> 1) & 21);
	}
	CUDA_HOST_DEVICE uint8_t SubstractVirtShift(uint8_t _old, uint8_t _new)
	{
		return AddVirtShift(_old, InverseVirtShift(_new) );
	}


	CUDA_HOST_DEVICE CVector3 GetVirtualProperty(const CVector3& _point, uint8_t _shiftInfo, const SPBC& _pbc)
	{
		if (_pbc.bEnabled)
			return _point + GetVectorFromVirtShift(_shiftInfo, _pbc.boundaryShift);
		else
			return _point;
	}

	CUDA_HOST_DEVICE CVector3 GetSolidBond(const CVector3& _vRCoord, const CVector3& _vLCoord, const SPBC& _pbc )
	{
		CVector3 vSB = _vLCoord - _vRCoord;
		if (_pbc.bEnabled)
		{
			if (_pbc.bX && (fabs(vSB.x) > _pbc.boundaryShift.x / 2))
				(_vRCoord.x > _vLCoord.x) ? vSB.x += _pbc.boundaryShift.x : vSB.x -= _pbc.boundaryShift.x;
			if (_pbc.bY && (fabs(vSB.y) > _pbc.boundaryShift.y / 2))
				(_vRCoord.y > _vLCoord.y) ? vSB.y += _pbc.boundaryShift.y : vSB.y -= _pbc.boundaryShift.y;
			if (_pbc.bZ && (fabs(vSB.z) > _pbc.boundaryShift.z / 2))
				(_vRCoord.z > _vLCoord.z) ? vSB.z += _pbc.boundaryShift.z : vSB.z -= _pbc.boundaryShift.z;
		}
		return vSB;
	}
}

#pragma warning(pop)
