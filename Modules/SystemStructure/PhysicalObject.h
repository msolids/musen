/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "Compound.h"
#include "Quaternion.h"
#include "MUSENDefinitions.h"

class CDemStorage;

/* This class defines the common abstract entity for description of every
   physical object in the system. Each physical object consists from a set of
   physical properties, such as mass, compound, and a set of geometrical, such
   as position of a center, object type, rotation angle. */
class CPhysicalObject
{
public:
	unsigned m_lObjectID; // The id of the object.

protected:
	CDemStorage *m_storage;

	double m_dMass; // Object mass [kg].
	std::vector<unsigned> m_ConnectedBondsID; // IDs of bonds which are connected to this object.
	CColor m_ObjectColor;

protected:
	CPhysicalObject(unsigned _id, CDemStorage* _storage);
	virtual ~CPhysicalObject() = default;
	void Save() const; // save geometry
	void Load();

	/// Clones the data from another object of the same type.
	virtual void CloneData(const CPhysicalObject& _other);

	friend class CSystemStructure;

	virtual void UpdateCompoundProperties(const CCompound* _pCompound) = 0;
	virtual void UpdatePrecalculatedValues() = 0; // Calculates all constant terms which are time independent.

public:
	unsigned GetObjectType() const; // Returns the type of the object.

	//////////////////////////////////////////////////////////////////////////
	/// Functions to work with compounds

	std::string GetCompoundKey() const;
	void SetCompound(const CCompound* _pCompound);
	void SetCompoundKey(const std::string& _compoundKey) const;

	//////////////////////////////////////////////////////////////////////////
	/// A set of interface functions to work with time independent data

	inline double GetMass() const { return m_dMass; }
	inline void SetMass(const double _dMass) { m_dMass = _dMass; }

	inline CColor GetColor() const { return m_ObjectColor; }
	inline void SetColor(const CColor& _color) { m_ObjectColor = _color; }

	//////////////////////////////////////////////////////////////////////////
	/// A set of interface functions to work with time dependent data

	//////////////////////////////////////////////////////////////////////////
	/// Sequential getters of time-dependent data.
	CVector3 GetCoordinates(double _time) const;
	CVector3 GetVelocity(double _time) const;
	//CVector3 GetAcceleration(const double _dTime) const;
	CVector3 GetAngles(double _time) const;
	CVector3 GetAngleVelocity(double _time) const;
	CVector3 GetAngleAcceleration(double _time) const;
	CVector3 GetForce(double _time) const;
	double GetTemperature(double _time) const;
	double GetTotalTorque(double _time) const;
	CQuaternion GetOrientation(double _time) const;
	CMatrix3 GetStressTensor(double _time) const;
	CVector3 GetNormalStress(double _time) const; // return diagonal elements of stress tensor

	//////////////////////////////////////////////////////////////////////////
	/// Parallel getters of time-dependent data.
	CVector3 GetCoordinates() const;
	CVector3 GetVelocity() const;
	CVector3 GetAngles() const;
	CVector3 GetAngleVelocity() const;
	CVector3 GetAngleAcceleration() const;
	CVector3 GetForce() const;
	double GetTemperature() const;
	double GetTotalTorque() const;
	CQuaternion GetOrientation() const;
	CMatrix3 GetStressTensor() const;
	CVector3 GetNormalStress() const; // return diagonal elements of stress tensor

	//////////////////////////////////////////////////////////////////////////
	/// Sequential setters of time-dependent data.
	void SetCoordinates(double _time, const CVector3& _coordinates) const;
	void SetVelocity(double _time, const CVector3& _velocity) const;
	void SetAngleVelocity(double _time, const CVector3& _angleVelocity) const;
	void SetForce(double _time, const CVector3& _force) const;
	void SetTemperature(double _time, const double& _temperature) const;
	void SetTotalTorque(double _time, const double& _totalTorque) const;
	void SetOrientation(double _time, const CQuaternion& _orientation) const;
	void SetStressTensor(double _time, const CMatrix3& _stressTensor) const;

	//////////////////////////////////////////////////////////////////////////
	/// Parallel setters of time-dependent data. To set the time point for parallel access, call CSystemStructure::PrepareTimePointForWrite(time).
	void SetCoordinates(const CVector3& _coordinates) const;
	void SetVelocity(const CVector3& _velocity) const;
	void SetAngleVelocity(const CVector3& _angleVelocity) const;
	void SetForce(const CVector3& _force) const;
	void SetTemperature(const double& _temperature) const;
	void SetTotalTorque(const double& _totalTorque) const;
	void SetOrientation(const CQuaternion& _orientation) const;
	void SetStressTensor(const CMatrix3& _stressTensor) const;

	bool IsQuaternionSet(const double _dTime) const;
	void ClearAllTDData(double _dTime);

	//////////////////////////////////////////////////////////////////////////
	/// Get ans sets the information about activity of the object

	bool IsActive(const double& _dTime) const;
	void GetActivityTimeInterval(double* _pStartTime, double* _pEndTime) const;
	[[nodiscard]] std::pair<double, double> GetActivityTimeInterval() const;
	double GetActivityStart() const; // return time point when object has become active
	double GetActivityEnd() const; // return time point when object has become inactive
	void SetStartActivityTime(double _dTime);
	void SetEndActivityTime(double _dTime);
	void SetObjectActivity(double _dTime, bool _bActive);

	//////////////////////////////////////////////////////////////////////////
	/// Functions to work with bonds

	void AddBond(const unsigned& _nBondID); // Adds bond wit specified index.
	void DeleteAllBonds(); // Deletes all bonds by which this object can be connected with other objects.

	//////////////////////////////////////////////////////////////////////////
	/// Functions to work with object geometry

	virtual std::string GetObjectGeometryText() const = 0;
	virtual void SetObjectGeometryText(std::stringstream& _inputStream) = 0;

	virtual std::vector<uint8_t> GetObjectGeometryBin() const = 0;
	virtual void SetObjectGeometryBin(const std::vector<uint8_t>& _data) = 0;
};

