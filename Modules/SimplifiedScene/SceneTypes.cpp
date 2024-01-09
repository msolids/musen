/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "SceneTypes.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
////////// SGeneralObject

inline void SGeneralObject::AddObject(bool _active, unsigned _initIndex)
{
	active.emplace_back(_active);
	initIndex.emplace_back(_initIndex);
	compoundIndex.emplace_back(0);
	startActivity.emplace_back(0.);
	endActivity.emplace_back(0.);
}

void SGeneralObject::Resize(size_t n)
{
	active.resize(n);
	initIndex.resize(n);
	compoundIndex.resize(n);
	startActivity.resize(n);
	endActivity.resize(n);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////// SBasicParticleStruct

void SBasicParticleStruct::AddBasicParticle(bool _active, CVector3 _coord, double _contactRadius, unsigned _initIndex)
{
	AddObject(_active, _initIndex);

	contactInfo.emplace_back(_coord, _contactRadius, CVector3{ 0 });
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////// SParticleStruct

void SParticleStruct::AddParticle(bool _active, const CVector3& _coord, double _radius, unsigned _initIndex, double _mass, double _inertiaMoment,
	const CVector3& _vel, const CVector3& _anglVel)
{
	AddBasicParticle(_active, _coord, _radius, _initIndex);

	kinematicsInfo.emplace_back(_radius, _mass, _inertiaMoment, _vel, _anglVel, CVector3{ 0 }, CVector3{ 0 });
}

void SParticleStruct::AddContactRadius(double _contactRadius)
{
	contactInfo.back().contactRadius = _contactRadius;
}

void SParticleStruct::AddQuaternion(const CQuaternion& _quaternion)
{
	quaternion.emplace_back(_quaternion);
}

void SParticleStruct::AddMultiSphIndex(int _multiSphIndex)
{
	multiSphIndex.emplace_back(_multiSphIndex);
}

void SParticleStruct::AddThermals(double _temperature, double _heatCapacity)
{
	thermalInfo.emplace_back(_temperature, _heatCapacity);
}

void SParticleStruct::Resize(size_t n)
{
	SGeneralObject::Resize(n);

	contactInfo.resize(n);
	kinematicsInfo.resize(n);
	if (!quaternion.empty())		quaternion.resize(n);
	if (!multiSphIndex.empty())		multiSphIndex.resize(n);
	if (!thermalInfo.empty())		thermalInfo.resize(n);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//////////  SWallStruct

void SWallStruct::AddWall(bool _active, unsigned _initIndex, const CVector3& _vert1, const CVector3& _vert2, const CVector3& _vert3,
	const CVector3& _normalVector, const CVector3& _vel, const CVector3& _rotVel, const CVector3& _rotCenter)
{
	AddObject(_active, _initIndex);

	coordInfo.emplace_back(_vert1, _vert2, _vert3);
	normalVector.emplace_back(_normalVector);
	movementInfo.emplace_back(_vel, _rotVel, _rotCenter);
	force.emplace_back(0);
}

void SWallStruct::Resize(size_t n)
{
	SGeneralObject::Resize(n);

	coordInfo.resize(n);
	normalVector.resize(n);
	movementInfo.resize(n);
	force.resize(n);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////// SBondStruct

void SBondStruct::AddBond(bool _active, unsigned _initIndex, size_t _leftID, size_t _rightID)
{
	AddObject(_active, _initIndex);

	connectionInfo.emplace_back(_leftID, _rightID);
}

void SBondStruct::Resize(size_t n)
{
	SGeneralObject::Resize(n);

	connectionInfo.resize(n);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////// SSolidBondStruct

void SSolidBondStruct::AddSolidBond(bool _active, unsigned _initIndex, size_t _leftID, size_t _rightID, double _diameter, double _crossCut, double _initialLength,
	const CVector3& _vTangOverlap, double _axialMoment, double _normalStiffness, double _tangentialStiffness, double _normalStrength, double _tangentialStrength)
{
	AddBond(_active, _initIndex, _leftID, _rightID);

	baseInfo.emplace_back(_diameter, _crossCut, _initialLength, _axialMoment, _normalStiffness, _tangentialStiffness, _vTangOverlap);
	strengthInfo.emplace_back(_normalStrength, _tangentialStrength);
	kinematicsInfo.emplace_back();
}

void SSolidBondStruct::AddViscosity(double _viscosity)
{
	viscosity.emplace_back(_viscosity);
}

void SSolidBondStruct::AddTimeThermExpCoeff(double _timeThermExpCoeff)
{
	timeThermExpCoeff.emplace_back(_timeThermExpCoeff);
}

void SSolidBondStruct::AddYieldStrength(double _yieldStrength)
{
	yieldStrength.emplace_back(_yieldStrength);
}

void SSolidBondStruct::AddNormalPlasticStrain(double _normalPlasticStrain)
{
	normalPlasticStrain.emplace_back(_normalPlasticStrain);
}

void SSolidBondStruct::AddTangentialPlasticStrain(const CVector3& _tangentialPlasticStrain)
{
	tangentialPlasticStrain.emplace_back(_tangentialPlasticStrain);
}

void SSolidBondStruct::AddThermals(double _thermalConductivity)
{
	thermalInfo.emplace_back(0.0, _thermalConductivity);
}

void SSolidBondStruct::Resize(size_t n)
{
	SBondStruct::Resize(n);

	baseInfo.resize(n);
	strengthInfo.resize(n);
	kinematicsInfo.resize(n);

	if (!viscosity.empty())					viscosity.resize(n);
	if (!timeThermExpCoeff.empty())			timeThermExpCoeff.resize(n);
	if (!yieldStrength.empty())				yieldStrength.resize(n);
	if (!normalPlasticStrain.empty())		normalPlasticStrain.resize(n);
	if (!tangentialPlasticStrain.empty())	tangentialPlasticStrain.resize(n);
	if (!thermalInfo.empty())		        thermalInfo.resize(n);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////// SLiquidBondStruct

void SLiquidBondStruct::AddLiquidBond(bool _active, unsigned _initIndex, size_t _leftID, size_t _rightID, double _volume, double _viscosity, double _surfaceTension)
{
	AddBond(_active, _initIndex, _leftID, _rightID);

	baseInfo.emplace_back(_volume, _viscosity, _surfaceTension);
	kinematicsInfo.emplace_back();
}

void SLiquidBondStruct::Resize(size_t n)
{
	SBondStruct::Resize(n);

	baseInfo.resize(n);
	kinematicsInfo.resize(n);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////// SMultiSphere

void SMultiSphere::AddMultisphere(const std::vector<size_t>& _indexes, const CMatrix3& _lMatrix, const CMatrix3& _inertTensor, const CMatrix3& _invLMatrix,
	const CMatrix3& _invInertTensor, const CVector3& _center, const CVector3& _velocity, const CVector3& _rotVelocity, double _mass)
{
	indices.emplace_back(_indexes);
	matrices.emplace_back(_lMatrix, _inertTensor, _invLMatrix, _invInertTensor);
	props.emplace_back(_center, _velocity, _rotVelocity, _mass);
}

void SMultiSphere::Resize(size_t n)
{
	indices.resize(n);
	matrices.resize(n);
	props.resize(n);
}
