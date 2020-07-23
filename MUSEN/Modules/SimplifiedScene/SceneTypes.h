/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "Quaternion.h"
#include <vector>

#define _EXPAND(x) x
#define _ADD_GET_SET_3(fun_name, path, var_name) \
	  decltype(decltype(path)::value_type::var_name)& fun_name(const size_t i)	     { return path[i].var_name; } \
const decltype(decltype(path)::value_type::var_name)& fun_name(const size_t i) const { return path[i].var_name; }
#define _ADD_GET_SET_2(fun_name, var_name) \
decltype(var_name)::reference		fun_name(const size_t i)       { return var_name[i]; } \
decltype(var_name)::const_reference fun_name(const size_t i) const { return var_name[i]; }
#define _RESOLVE_MACRO(_1,_2,_3,NAME,...) NAME
#define ADD_GET_SET(...) _EXPAND(_RESOLVE_MACRO(__VA_ARGS__, _ADD_GET_SET_3, _ADD_GET_SET_2)(__VA_ARGS__))

struct SGeneralObject
{
protected:
	std::vector<uint8_t>	active;
	std::vector<unsigned>	initIndex;
	std::vector<unsigned>	compoundIndex;
	std::vector<double>		endActivity; // time point when particle is not more active

public:
	ADD_GET_SET(Active,			active)
	ADD_GET_SET(InitIndex,		initIndex) // corresponds to the indexes in the initial array (in the systemStructure) for real objects. for virtual objects it corresponds to index of object in simplified scene
	ADD_GET_SET(CompoundIndex,	compoundIndex)
	ADD_GET_SET(EndActivity,	endActivity)

	bool Empty() const { return active.empty(); }
	size_t Size() const { return active.size(); }

protected:
	inline void AddObject(bool _active, unsigned _initIndex);
	void Resize(size_t n);
};

struct SBasicParticleStruct : SGeneralObject
{
protected:
	struct SContactInformation
	{
		CVector3	coord;
		double		contactRadius{};
		CVector3	coordVerlet;

		SContactInformation() = default;
		SContactInformation(const CVector3& _coord, double _contactRadius, const CVector3& _coordVerlet)
			: coord{ _coord }, contactRadius{ _contactRadius }, coordVerlet{ _coordVerlet } {}
	};
	std::vector<SContactInformation> contactInfo;

public:
	ADD_GET_SET(Coord,			contactInfo, coord)
	ADD_GET_SET(ContactRadius,	contactInfo, contactRadius)
	ADD_GET_SET(CoordVerlet,	contactInfo, coordVerlet)		// coordinates of particles, which was used for last verlet calculation

	void AddBasicParticle(bool _active, CVector3 _coord, double _contactRadius, unsigned _initIndex);
};

struct SParticleStruct : SBasicParticleStruct
{
private:
	struct SKinematics
	{
		double		radius{};
		double		mass{};
		double		inertiaMoment{};
		CVector3	vel;
		CVector3	anglVel;
		CVector3	force;
		CVector3	moment;

		SKinematics() = default;
		SKinematics(double _radius, double _mass, double _inertiaMoment, const CVector3& _vel, const CVector3& _anglVel, const CVector3& _force, const CVector3& _moment)
			: radius{ _radius }, mass{ _mass }, inertiaMoment{ _inertiaMoment }, vel{ _vel }, anglVel{ _anglVel }, force{ _force }, moment{ _moment } {}
	};

	struct SThermals
	{
		double temperature;
		double heatCapacity;

		SThermals() = default;
		SThermals(double _temperature, double _heatCapacity)
			: temperature{ _temperature }, heatCapacity{ _heatCapacity } {}
	};

	// required variables
	std::vector<SKinematics> kinematicsInfo;

	// optional variables
	std::vector<CQuaternion>	quaternion;
	std::vector<int>			multiSphIndex; // index of multi-sphere for this particle
	std::vector<SThermals>		thermalInfo;

public:
	ADD_GET_SET(Radius,			kinematicsInfo, radius)
	ADD_GET_SET(Mass,			kinematicsInfo, mass)
	ADD_GET_SET(InertiaMoment,	kinematicsInfo, inertiaMoment)
	ADD_GET_SET(Vel,			kinematicsInfo, vel)
	ADD_GET_SET(AnglVel,		kinematicsInfo, anglVel)
	ADD_GET_SET(Force,			kinematicsInfo, force)
	ADD_GET_SET(Moment,			kinematicsInfo, moment)

	// Get optional variables. Warning: no bounds check. Caller needs to ensure existence.
	ADD_GET_SET(Quaternion,		quaternion)
	ADD_GET_SET(MultiSphIndex,	multiSphIndex)

	ADD_GET_SET(Temperature,	thermalInfo, temperature)
	ADD_GET_SET(HeatCapacity,	thermalInfo, heatCapacity)

	bool ThermalsExist() const { return thermalInfo.size() == Size() && !Empty(); }
	bool QuaternionExist() const { return quaternion.size() == Size() && !Empty(); }
	bool MultiSphIndexExist() const { return multiSphIndex.size() == Size() && !Empty(); }

	void AddParticle(bool _active, const CVector3& _coord, double _radius, unsigned _initIndex, double _mass, double _inertiaMoment, const CVector3& _vel, const CVector3& _anglVel);
	void AddContactRadius(double _contactRadius);
	void AddQuaternion(const CQuaternion& _quaternion);
	void AddMultiSphIndex(int _multiSphIndex);
	void AddThermals(double _temperature, double _heatCapacity);

	void Resize(size_t n);
};

struct SWallStruct : SGeneralObject
{
public:
	struct SCoordinates
	{
		CVector3 minCoord; // minimal and maximal coordinates of the bounding box
		CVector3 maxCoord;
		CVector3 vert1, vert2, vert3;

		SCoordinates() = default;
		SCoordinates(const CVector3& _vert1, const CVector3& _vert2, const CVector3& _vert3)
			: minCoord{ Min(_vert1, _vert2, _vert3) }, maxCoord{ Max(_vert1, _vert2, _vert3) }, vert1{ _vert1 }, vert2{ _vert2 }, vert3{ _vert3 } {}
	};

private:
	struct SMovement
	{
		CVector3 vel;
		CVector3 rotVel;
		CVector3 rotCenter;

		SMovement() = default;
		SMovement(const CVector3& _vel, const CVector3& _rotVel, const CVector3& _rotCenter)
			: vel{ _vel }, rotVel{ _rotVel }, rotCenter{ _rotCenter } {}
	};

	std::vector<SCoordinates>	coordInfo;
	std::vector<CVector3>		normalVector;   // TODO: maybe put into SCoordinates
	std::vector<SMovement>		movementInfo;
	std::vector<CVector3>		force;

public:
	ADD_GET_SET(Coordinates, coordInfo)

	ADD_GET_SET(MinCoord,	coordInfo, minCoord)
	ADD_GET_SET(MaxCoord,	coordInfo, maxCoord)
	ADD_GET_SET(Vert1,		coordInfo, vert1)
	ADD_GET_SET(Vert2,		coordInfo, vert2)
	ADD_GET_SET(Vert3,		coordInfo, vert3)

	ADD_GET_SET(NormalVector,	normalVector)

	ADD_GET_SET(Vel,		movementInfo, vel)
	ADD_GET_SET(RotVel,		movementInfo, rotVel)
	ADD_GET_SET(RotCenter,	movementInfo, rotCenter)

	ADD_GET_SET(Force, force)

	void AddWall(bool _active, unsigned _initIndex, const CVector3& _vert1, const CVector3& _vert2, const CVector3& _vert3, const CVector3& _normalVector, const CVector3& _vel, const CVector3& _rotVel, const CVector3& _rotCenter);

	void Resize(size_t n);
};

struct SBondStruct : SGeneralObject
{
protected:
	struct SConnection
	{
		size_t leftID;
		size_t rightID;

		SConnection() = default;
		SConnection(size_t _leftID, size_t _rightID) : leftID{ _leftID }, rightID{ _rightID } {}
	};
	std::vector<SConnection> connectionInfo;

public:
	ADD_GET_SET(LeftID,		connectionInfo, leftID)
	ADD_GET_SET(RightID,	connectionInfo, rightID)

	void AddBond(bool _active, unsigned _initIndex, size_t _leftID, size_t _rightID);

	void Resize(size_t n);
};

struct SSolidBondStruct : SBondStruct
{
private:
	struct SBaseInfo
	{
		double		diameter{};
		double		crossCut{};
		double		initialLength{};
		double		axialMoment{};
		double		normalStiffness{};
		double		tangentialStiffness{};
		CVector3	tangentialOverlap{ 0 };
		CVector3	tangentialForce{ 0 };
		CVector3	prevBond{ 0 };

		SBaseInfo() = default;
		SBaseInfo(double _diameter, double _crossCut, double _initialLength, double _axialMoment, double _normalStiffness, double _tangentialStiffness)
			: diameter{ _diameter }, crossCut{ _crossCut }, initialLength{ _initialLength }, axialMoment{ _axialMoment }, normalStiffness{ _normalStiffness }, tangentialStiffness{ _tangentialStiffness } {}
	};

	struct SStrength
	{
		double normalStrength;
		double tangentialStrength;

		SStrength() = default;
		SStrength(double _normalStrength, double _tangentialStrength) : normalStrength{ _normalStrength }, tangentialStrength{ _tangentialStrength } {}
	};

	struct SKinematics
	{
		CVector3 totalForce{ 0 };		// normal + tangential
		CVector3 normalMoment{ 0 };
		CVector3 tangentialMoment{ 0 };
		CVector3 unsymMoment{ 0 };		// unsymmetrical moment
	};

	// required variables
	std::vector<SBaseInfo>		baseInfo;
	std::vector<SStrength>		strengthInfo;
	std::vector<SKinematics>	kinematicsInfo;

	// optional variables
	std::vector<double>		viscosity;
	std::vector<double>		timeThermExpCoeff;
	std::vector<double>		yieldStrength;				// [Pa]
	std::vector<double>		normalPlasticStrain;		// [-]
	std::vector<CVector3>	tangentialPlasticStrain;	// [-,-,-]
	std::vector<double>		thermalConductivity;

public:
	ADD_GET_SET(Diameter,				baseInfo, diameter)
	ADD_GET_SET(CrossCut,				baseInfo, crossCut)
	ADD_GET_SET(InitialLength,			baseInfo, initialLength)
	ADD_GET_SET(AxialMoment,			baseInfo, axialMoment)
	ADD_GET_SET(NormalStiffness,		baseInfo, normalStiffness)
	ADD_GET_SET(TangentialStiffness,	baseInfo, tangentialStiffness)
	ADD_GET_SET(TangentialOverlap,		baseInfo, tangentialOverlap)
	ADD_GET_SET(TangentialForce,		baseInfo, tangentialForce)
	ADD_GET_SET(PrevBond,				baseInfo, prevBond)

	ADD_GET_SET(NormalStrength,		strengthInfo, normalStrength)
	ADD_GET_SET(TangentialStrength, strengthInfo, tangentialStrength)

	ADD_GET_SET(TotalForce,			kinematicsInfo, totalForce)
	ADD_GET_SET(NormalMoment,		kinematicsInfo, normalMoment)
	ADD_GET_SET(TangentialMoment,	kinematicsInfo, tangentialMoment)
	ADD_GET_SET(UnsymMoment,		kinematicsInfo, unsymMoment)

	ADD_GET_SET(Viscosity,					viscosity)
	ADD_GET_SET(TimeThermExpCoeff,			timeThermExpCoeff)
	ADD_GET_SET(YieldStrength,				yieldStrength)
	ADD_GET_SET(NormalPlasticStrain,		normalPlasticStrain)
	ADD_GET_SET(TangentialPlasticStrain,	tangentialPlasticStrain)
	ADD_GET_SET(ThermalConductivity,		thermalConductivity)

	void AddSolidBond(bool _active, unsigned _initIndex, size_t _leftID, size_t _rightID, double _diameter, double _crossCut, double _initialLength, double _axialMoment,
		double _normalStiffness, double _tangentialStiffness, double _normalStrength, double _tangentialStrength);
	void AddViscosity(double _viscosity);
	void AddTimeThermExpCoeff(double _timeThermExpCoeff);
	void AddYieldStrength(double _yieldStrength);
	void AddNormalPlasticStrain(double _normalPlasticStrain);
	void AddTangentialPlasticStrain(const CVector3& _tangentialPlasticStrain);
	void AddThermalConductivity(double _thermalConductivity);

	void Resize(size_t n);
};


struct SLiquidBondStruct : SBondStruct
{
private:
	struct SBaseInfo
	{
		double volume;
		double viscosity;
		double surfaceTension;
	};

	struct SKinematics
	{
		CVector3 normalForce{ 0 };
		CVector3 unsymMoment{ 0 };		// unsymmetrical moment
		CVector3 tangentialForce{ 0 };
	};

	std::vector<SBaseInfo>		baseInfo;
	std::vector<SKinematics>	kinematicsInfo;

public:
	ADD_GET_SET(Viscosity,		baseInfo, viscosity)
	ADD_GET_SET(Volume,			baseInfo, volume)
	ADD_GET_SET(SurfaceTension, baseInfo, surfaceTension)

	ADD_GET_SET(NormalForce,		kinematicsInfo, normalForce)
	ADD_GET_SET(UnsymMoment,		kinematicsInfo, unsymMoment)
	ADD_GET_SET(TangentialForce,	kinematicsInfo, tangentialForce)

	void AddLiquidBond(bool _active, unsigned _initIndex, size_t _leftID, size_t _rightID, double _volume, double _viscosity, double _surfaceTension);

	void Resize(size_t n);
};

struct SMultiSphere
{
private:
	std::vector<std::vector<size_t>> indices; // indexes of particles in simplified scene

	struct SMatrices
	{
		CMatrix3 LMatrix;
		CMatrix3 inertTensor;		// inertial tensor
		CMatrix3 invLMatrix;		// inverse inertial tensor
		CMatrix3 invInertTensor;	// inverse lambda matrix
	};

	struct SProperties
	{
		CVector3 center;		// center of mass of this multi-sphere
		CVector3 velocity;		// velocity of center of mass
		CVector3 rotVelocity;
		double mass{};
	};

	std::vector<SMatrices> matrices;
	std::vector<SProperties> props;

public:
	std::vector<size_t>& Indices(const size_t i) { return indices[i]; };
	CMatrix3& LMatrix(const size_t i) { return matrices[i].LMatrix; };
	CMatrix3& InertTensor(const size_t i) { return matrices[i].inertTensor; };
	CMatrix3& InvLMatrix(const size_t i) { return matrices[i].invLMatrix; };
	CMatrix3& InvInertTensor(const size_t i) { return matrices[i].invInertTensor; };
	CVector3& Center(const size_t i) { return props[i].center; };
	CVector3& Velocity(const size_t i) { return props[i].velocity; };
	CVector3& RotVelocity(const size_t i) { return props[i].rotVelocity; };
	double& Mass(const size_t i) { return props[i].mass; };

	void AddMultisphere(const std::vector<size_t>& _indexes, const CMatrix3& _LMatrix, const CMatrix3& _inertTensor, const CMatrix3& _invLMatrix,
		const CMatrix3& _invInertTensor, const CVector3& _center, const CVector3& _velocity, const CVector3& _rotVelocity, double _mass);

	size_t Size() const { return indices.size(); }
	void Resize(size_t n);
};

struct SCollision;
struct SSavedCollision
{
	unsigned nCnt;			// number of simultaneously existing contacts of a particle and geometry
	int nGeomID;			// identifier of geometry for PW-contact
	double dTimeStart;		// begin of the contact
	double dTimeEnd;		// end of the contact, -1 indicates not finished collision at the simulation end
	CVector3 vMaxTotalForce;// maximum total force during the collision; for simultaneous contact with several walls, sum of forces is analyzed
	CVector3 vMaxNormForce;	// maximum normal force during the collision; for simultaneous contact with several walls, sum of forces is analyzed
	CVector3 vMaxTangForce;	// maximum tangential force during the collision; for simultaneous contact with several walls, sum of forces is analyzed
	CVector3 vNormVelocity;	// relative normal velocity of objects at the first moment of contact; or max value for simultaneous contact with several walls
	CVector3 vTangVelocity;	// relative tangential velocity of objects at the first moment of contact; or max value for simultaneous contact with several walls
	CVector3 vContactPoint;	// point of contact at the dTimeStart; one of them for simultaneous contact with several walls
	std::vector<SCollision*> vPtr;	// list of collisions with length nCnt, contains pointers to simultaneously existing contacts of a particle and geometry
	SSavedCollision()
	{
		vMaxTotalForce.Init(0);
		vMaxNormForce.Init(0);
		vMaxTangForce.Init(0);
	}
};

// Used to describe particle-particle and particle-wall collision.
struct SCollision
{
	bool bContactStillExist; // this flag is used to determine if the contact still exist in compare to previous step
	uint8_t nVirtShift;      // shifts to calculate parameters of virtual particles in case of PBC. For BOX: shift {x, y, z}, for CYLINDER: rotation angle {cos(a), sin (a), 0}.
	uint16_t nInteractProp;  // index of interaction property
	unsigned nSrcID;		 // identifier of first contact partner or wall (nWallID)
	unsigned nDstID;		 // identifier of second contact partner
	double dNormalOverlap;	 //
	double dEquivMass;		 // equivalent mass
	double dEquivRadius;	 // equivalent radius
	CVector3 vTangOverlap;	 // old tangential overlap
	CVector3 vTangForce;	 // total tangential force
	CVector3 vTotalForce;
	CVector3 vResultMoment1; // moment which acts on first particle
	CVector3 vResultMoment2; // moment which acts on first particle
	CVector3 vContactVector; // For PP contact: dstCoord - srcCoord. For PW contact: contact point.
	SSavedCollision *pSave;
	SCollision()
	{
		vTangOverlap.Init(0);
		vTangForce.Init(0);
		vTotalForce.Init(0);
		vResultMoment1.Init(0);
		vResultMoment2.Init(0);
		vContactVector.Init(0);
	}
};

#undef ADD_GET_SET
#undef _RESOLVE_MACRO
#undef _ADD_GET_SET_2
#undef _ADD_GET_SET_3
#undef _EXPAND