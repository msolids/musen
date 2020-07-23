/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include <vector>
#include <string>

#define UC_TEMPERATURE_UNIT "UC_TEMPERATURE_UNIT"
#define UC_MASS_UNIT "UC_MASS_UNIT"
#define UC_MASS_STREAM_UNIT "UC_MASS_STREAM_UNIT"
#define UC_TIME_UNIT "UC_TIME_UNIT"
#define UC_LENGTH_UNIT "UC_LENGTH_UNIT"
#define UC_PARTICLE_DIAMETER_UNIT "UC_PARTICLE_DIAMETER_UNIT"
#define UC_PRESSURE_UNIT "UC_PRESSURE_UNIT"
#define UC_VELOCITY_UNIT "UC_VELOCITY_UNIT"
#define UC_FORCE_UNIT "UC_FORCE_UNIT"
#define UC_VOLUME_UNIT "UC_VOLUME_UNIT"
#define UC_ANGULAR_VELOCITY_UNIT "UC_ANGULAR_VELOCITY_UNIT"

// various properties
enum class EUnitType : unsigned
{
	NONE		      = 0,
	TEMPERATURE       = 1,
	MASS              = 2,
	MASS_STREAM       = 3,
	TIME              = 4,
	LENGTH            = 5,
	PARTICLE_DIAMETER = 6,
	PRESSURE          = 7,
	VELOCITY          = 8,
	FORCE             = 9,
	VOLUME            = 10,
	SURFACE           = 11,
	PSD_q0            = 12,
	PSD_q3            = 13,
	STRESS            = 14,
	ANGULAR_VELOCITY  = 15
};

class CUnitConvertor
{
private:
	std::vector<EUnitType> m_vProperties;
	std::vector<unsigned> m_vSelectedUnitType;
	std::vector<unsigned> m_vDefaultUnitType;

private:
	int GetPropIndex(EUnitType _nPropertyType) const; // get offset of specified property in the array

	// from SI to current
	double GetTemperature( double _dTemperatureSI ) const;
	double GetMass( double _dMassSI ) const;
	double GetMassStream( double _dMassStreamSI ) const;
	double GetTime( double _dTimeSI ) const;
	double GetLength( double _dLengthSI ) const;
	double GetParticleDiameter( double _dDiameterSI ) const;
	double GetPressure( double _dPressureSI ) const;
	double GetVelocity( double _dVelocitySI ) const;
	double GetForce( double _dForceSI ) const;
	double GetVolume( double _dForceSI ) const;
	double GetSurface( double _dSurfaceSI ) const;
	double GetStress(double _dSurfaceSI) const;
	double GetAngVelocity(double _dVelocitySI) const;

	// from current to SI
	double GetTemperatureSI( double _dValue ) const;
	double GetMassSI( double _dValue ) const;
	double GetMassStreamSI( double _dValue ) const;
	double GetTimeSI( double _dValue ) const;
	double GetLengthSI( double _dValue ) const;
	double GetParticleDiameterSI( double _dValue ) const;
	double GetPressureSI( double _dValue ) const;
	double GetVelocitySI( double _dValue ) const;
	double GetForceSI( double _dValue ) const;
	double GetVolumeSI( double _dValue ) const;
	double GetSurfaceSI( double _dValue ) const;
	double GetStressSI(double _dValue) const;
	double GetAngVelocitySI(double _dValue) const;

public:

	unsigned GetPropertiesNumber() const { return (unsigned)m_vProperties.size(); }

	static std::string GetPropertyName(EUnitType _nPropType);
	std::string GetPropertyNameByIndex( unsigned _nPropertyIndex );
	std::string GetSelectedUnit(EUnitType _nPropertyType);
	std::vector<std::string> GetPossibleUnitsByIndex( unsigned _nPropertyIndex );

	unsigned GetSelectedUnitTypeByIndex( unsigned _nPropertyIndex );
	unsigned GetSelectedUnitType(EUnitType _nPropertyType);
	void SetSelectedUnitTypeByIndex( unsigned _nPropIndex, unsigned _nUnitIndex );
	void SetSelectedUnitType(EUnitType _nPropType, unsigned _nUnitIndex);

	// return value in the modified units ( from the give value in SI units)
	double GetValue(EUnitType _nPropertyType, double _dValueSI) const;
	double GetValueSI(EUnitType _nPropertyType, double _dValue) const; // convert from current units to the SI units

	void RestoreDefaultUnits();

	CUnitConvertor();
};

