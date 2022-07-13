/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "PhysicalObject.h"
#include "BasicTypes.h"
#include "Triangle.h"

/** This class defines a triangular wall which is described by the three coordinates. */
class CTriangularWall : public CPhysicalObject
{
public:
	CTriangularWall(unsigned _id, CDemStorage *_storage);

	//////////////////////////////////////////////////////////////////////////
	/// Sequential getters of time-dependent data.
	CVector3 GetCoordVertex1(double _time) const { return GetCoordinates(_time); }
	CVector3 GetCoordVertex2(double _time) const { const CQuaternion quant = GetOrientation(_time); return CVector3(quant.q0, quant.q1, quant.q2); }
	CVector3 GetCoordVertex3(double _time) const { return GetAngleVelocity(_time); }
	CTriangle GetPlaneCoords(double _time) const { return {GetCoordVertex1(_time), GetCoordVertex2(_time), GetCoordVertex3(_time)}; }
	CVector3 GetOldCoordVertex2(double _time) const { return GetAngles(_time); }
	CVector3 GetOldCoordVertex3(double _time) const { return GetAngleAcceleration(_time); }
	CVector3 GetNormalVector(double _time) const;

	//////////////////////////////////////////////////////////////////////////
	/// Parallel getters of time-dependent data.
	CVector3 GetCoordVertex1() const { return GetCoordinates(); }
	CVector3 GetCoordVertex2() const { const CQuaternion quant = GetOrientation(); return CVector3(quant.q0, quant.q1, quant.q2); }
	CVector3 GetCoordVertex3() const { return GetAngleVelocity(); }
	CTriangle GetPlaneCoords() const { return { GetCoordVertex1(), GetCoordVertex2(), GetCoordVertex3() }; }
	CVector3 GetOldCoordVertex2() const { return GetAngles(); }
	CVector3 GetOldCoordVertex3() const { return GetAngleAcceleration(); }
	CVector3 GetNormalVector() const;

	//////////////////////////////////////////////////////////////////////////
	/// Sequential setters of time-dependent data.
	void SetPlaneCoord(double _time, const CVector3& _vert1, const CVector3& _vert2, const CVector3& _vert3) const;
	void SetPlaneCoord(double _time, const CTriangle& _triangle) const;

	//////////////////////////////////////////////////////////////////////////
	/// Parallel setters of time-dependent data. To set the time point for parallel access, call CSystemStructure::PrepareTimePointForWrite(time).
	void SetPlaneCoord(const CVector3& _vert1, const CVector3& _vert2, const CVector3& _vert3) const;
	void SetPlaneCoord(const CTriangle& _triangle) const;

	void SetObjectGeometryText(std::stringstream& _inputStream) override {}
	std::string GetObjectGeometryText() const override { return {}; }

	std::vector<uint8_t> GetObjectGeometryBin() const override { return {}; }
	void SetObjectGeometryBin(const std::vector<uint8_t>& _data) override { }

	void UpdateCompoundProperties(const CCompound* _pCompound) override {};

private:
	void UpdatePrecalculatedValues() override {}; // Calculates all constant terms which are time independent.
};


