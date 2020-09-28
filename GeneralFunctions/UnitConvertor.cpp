/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "UnitConvertor.h"

CUnitConvertor::CUnitConvertor()
{
	m_vProperties.clear();
	m_vProperties.push_back(EUnitType::TEMPERATURE );
	m_vDefaultUnitType.push_back( 1 ); // C

	m_vProperties.push_back(EUnitType::MASS );
	m_vDefaultUnitType.push_back( 1 ); // kg

	m_vProperties.push_back(EUnitType::MASS_STREAM );
	m_vDefaultUnitType.push_back( 1 ); // kg/s

	m_vProperties.push_back(EUnitType::TIME );
	m_vDefaultUnitType.push_back( 1 ); // s

	m_vProperties.push_back(EUnitType::LENGTH );
	m_vDefaultUnitType.push_back( 3 ); // m

	m_vProperties.push_back(EUnitType::PARTICLE_DIAMETER );
	m_vDefaultUnitType.push_back( 1 ); // mm

	m_vProperties.push_back(EUnitType::PRESSURE );
	m_vDefaultUnitType.push_back( 0 ); // mm

	m_vProperties.push_back(EUnitType::VELOCITY );
	m_vDefaultUnitType.push_back( 3 ); // m/s

	m_vProperties.push_back(EUnitType::FORCE );
	m_vDefaultUnitType.push_back( 1 ); // N

	m_vProperties.push_back(EUnitType::VOLUME );
	m_vDefaultUnitType.push_back( 2 ); // m3

	m_vProperties.push_back(EUnitType::SURFACE );
	m_vDefaultUnitType.push_back( 2 ); // m2

	m_vProperties.push_back(EUnitType::STRESS );
	m_vDefaultUnitType.push_back( 3 ); // MPa

	m_vProperties.push_back(EUnitType::ANGULAR_VELOCITY);
	m_vDefaultUnitType.push_back( 0 ); // rad/s

	m_vProperties.push_back(EUnitType::ANGLE);
	m_vDefaultUnitType.push_back( 0 ); // °

	m_vSelectedUnitType.resize( m_vDefaultUnitType.size() );
	for ( unsigned i=0; i < m_vSelectedUnitType.size(); i++ )
		m_vSelectedUnitType[ i ] = m_vDefaultUnitType[ i ];
}

std::string CUnitConvertor::GetPropertyName(EUnitType _nPropType)
{
	switch( _nPropType )
	{
	case EUnitType::NONE             : return "";
	case EUnitType::TEMPERATURE      : return "Temperature";
	case EUnitType::MASS             : return "Mass";
	case EUnitType::MASS_STREAM      : return "Mass stream";
	case EUnitType::TIME             : return "Time";
	case EUnitType::LENGTH           : return "Length";
	case EUnitType::PARTICLE_DIAMETER: return "Particle diameter";
	case EUnitType::PRESSURE         : return "Pressure";
	case EUnitType::VELOCITY         : return "Velocity";
	case EUnitType::FORCE            : return "Force";
	case EUnitType::VOLUME           : return "Volume";
	case EUnitType::SURFACE          : return "Surface";
	case EUnitType::PSD_q0           : return "q0";
	case EUnitType::PSD_q3           : return "q3";
	case EUnitType::STRESS           : return "Stress";
	case EUnitType::ANGULAR_VELOCITY : return "Angular velocity";
	case EUnitType::ANGLE            : return "Angle";
	}
	return "";
}

std::string CUnitConvertor::GetPropertyNameByIndex( unsigned _nIndex )
{
	if ( _nIndex >= m_vProperties.size() )
		return "";
	return GetPropertyName( m_vProperties[ _nIndex ] );
}

unsigned CUnitConvertor::GetSelectedUnitTypeByIndex( unsigned _nPropertyIndex )
{
	if ( _nPropertyIndex >= m_vProperties.size() )
		return 0;
	else
		return m_vSelectedUnitType[ _nPropertyIndex ];
}

unsigned CUnitConvertor::GetSelectedUnitType(EUnitType _nPropType)
{
	return GetSelectedUnitTypeByIndex( GetPropIndex( _nPropType ) );
}

void CUnitConvertor::SetSelectedUnitTypeByIndex( unsigned _nPropIndex, unsigned _nUnitIndex )
{
	if ( _nPropIndex < m_vProperties.size() )
		m_vSelectedUnitType[ _nPropIndex ] = _nUnitIndex;
}

void CUnitConvertor::SetSelectedUnitType(EUnitType _nPropType, unsigned _nUnitIndex)
{
	int nIndex = GetPropIndex( _nPropType );
	if ( nIndex >= 0 )
		SetSelectedUnitTypeByIndex( nIndex, _nUnitIndex );
}

void CUnitConvertor::RestoreDefaultUnits()
{
	m_vSelectedUnitType.resize( m_vDefaultUnitType.size() );
	for ( unsigned i=0; i < m_vSelectedUnitType.size(); i++ )
		m_vSelectedUnitType[ i ] = m_vDefaultUnitType[ i ];
}

std::vector<std::string> CUnitConvertor::GetPossibleUnitsByIndex( unsigned _nPropertyIndex)  const
{
	std::vector<std::string> vResultVector;
	if ( _nPropertyIndex >= m_vProperties.size() )
		return vResultVector;

	switch( m_vProperties[ _nPropertyIndex ] )
	{
	case EUnitType::NONE: break;
	case EUnitType::PSD_q0: break;
	case EUnitType::PSD_q3: break;
	case EUnitType::TEMPERATURE:
		vResultVector.push_back( "K" );
		vResultVector.push_back( "C" ); // default value
		break;
	case EUnitType::MASS:
		vResultVector.push_back( "g" );
		vResultVector.push_back( "kg" ); // default value
		vResultVector.push_back( "t" );
		break;

	case EUnitType::MASS_STREAM:
		vResultVector.push_back( "g/s" );
		vResultVector.push_back( "kg/s" );  // default value
		vResultVector.push_back( "g/min" );
		vResultVector.push_back( "kg/min" );
		vResultVector.push_back( "t/min" );
		vResultVector.push_back( "g/h" );
		vResultVector.push_back( "kg/h" );
		vResultVector.push_back( "t/h" );
		vResultVector.push_back( "kg/day" );
		vResultVector.push_back( "t/day" );
		break;
	case EUnitType::TIME:
		vResultVector.push_back( "ms" );
		vResultVector.push_back( "s" ); // default value
		vResultVector.push_back( "min" );
		vResultVector.push_back( "h" );
		vResultVector.push_back( "day" );
		break;
	case EUnitType::LENGTH:
		vResultVector.push_back( "mkm" );
		vResultVector.push_back( "mm" );
		vResultVector.push_back( "cm" );
		vResultVector.push_back( "m" ); // default value
		break;
	case EUnitType::PARTICLE_DIAMETER:
		vResultVector.push_back( "mkm" );
		vResultVector.push_back( "mm" ); // default value
		vResultVector.push_back( "cm" );
		vResultVector.push_back( "m" );
		break;
	case EUnitType::PRESSURE:
		vResultVector.push_back( "Pa" ); // default value
		vResultVector.push_back( "bar" );
		break;
	case EUnitType::VELOCITY:
		vResultVector.push_back( "mkm/s" );
		vResultVector.push_back( "mm/s" );
		vResultVector.push_back( "cm/s" );
		vResultVector.push_back( "m/s" );  // default value
		vResultVector.push_back( "mkm/min" );
		vResultVector.push_back( "mm/min" );
		vResultVector.push_back( "cm/min" );
		vResultVector.push_back( "m/min" );
		break;
	case EUnitType::FORCE:
		vResultVector.push_back( "mN" );
		vResultVector.push_back( "N" );  // default value
		vResultVector.push_back( "kN" );
		break;
	case EUnitType::VOLUME:
		vResultVector.push_back( "mm3" );
		vResultVector.push_back( "cm3" );
		vResultVector.push_back( "m3" );   // default value
		break;
	case EUnitType::SURFACE:
		vResultVector.push_back( "mm2" );
		vResultVector.push_back( "cm2" );
		vResultVector.push_back( "m2" );   // default value
		break;
	case EUnitType::STRESS:
		vResultVector.push_back("mPa");
		vResultVector.push_back("Pa");
		vResultVector.push_back("KPa");
		vResultVector.push_back("MPa");   // default value
		vResultVector.push_back("GPa");
		break;
	case EUnitType::ANGULAR_VELOCITY:
		vResultVector.push_back("rad/s"); // default value
		vResultVector.push_back("rpm");
		break;
	case EUnitType::ANGLE:
		vResultVector.push_back("°"); // default value
		vResultVector.push_back("rad");
		break;
	}
	return vResultVector;
}

double CUnitConvertor::GetValue(EUnitType _nPropertyType, double _dValueSI) const
{
	switch ( _nPropertyType )
	{
	case EUnitType::TEMPERATURE: return GetTemperature( _dValueSI );
	case EUnitType::MASS: return GetMass( _dValueSI );
	case EUnitType::MASS_STREAM: return GetMassStream( _dValueSI );
	case EUnitType::TIME: return GetTime( _dValueSI );
	case EUnitType::LENGTH: return GetLength( _dValueSI );
	case EUnitType::PARTICLE_DIAMETER: return GetParticleDiameter( _dValueSI );
	case EUnitType::PRESSURE: return GetPressure( _dValueSI );
	case EUnitType::VELOCITY: return GetVelocity( _dValueSI );
	case EUnitType::FORCE: return GetForce( _dValueSI );
	case EUnitType::VOLUME: return GetVolume( _dValueSI );
	case EUnitType::SURFACE: return GetSurface( _dValueSI );
	case EUnitType::PSD_q0: return _dValueSI/GetParticleDiameter( 1 );
	case EUnitType::PSD_q3: return _dValueSI/GetParticleDiameter( 1 );
	case EUnitType::STRESS: return GetStress( _dValueSI );
	case EUnitType::ANGULAR_VELOCITY: return GetAngVelocity( _dValueSI );
	case EUnitType::ANGLE: return GetAngle( _dValueSI );
	case EUnitType::NONE: return _dValueSI;
	}
	return 0;
}

double CUnitConvertor::GetValueSI(EUnitType _nPropertyType, double _dValue) const
{
	switch ( _nPropertyType )
	{
	case EUnitType::TEMPERATURE: return GetTemperatureSI(_dValue);
	case EUnitType::MASS: return GetMassSI(_dValue);
	case EUnitType::MASS_STREAM: return GetMassStreamSI(_dValue);
	case EUnitType::TIME: return GetTimeSI(_dValue);
	case EUnitType::LENGTH: return GetLengthSI(_dValue);
	case EUnitType::PARTICLE_DIAMETER: return GetParticleDiameterSI(_dValue);
	case EUnitType::PRESSURE: return GetPressureSI(_dValue);
	case EUnitType::VELOCITY: return GetVelocitySI(_dValue);
	case EUnitType::FORCE: return GetForceSI(_dValue);
	case EUnitType::VOLUME: return GetVolumeSI(_dValue);
	case EUnitType::SURFACE: return GetSurfaceSI(_dValue);
	case EUnitType::PSD_q0: return _dValue * GetParticleDiameter(1);
	case EUnitType::PSD_q3: return _dValue * GetParticleDiameter(1);
	case EUnitType::STRESS: return GetStressSI(_dValue);
	case EUnitType::ANGULAR_VELOCITY: return GetAngVelocitySI(_dValue);
	case EUnitType::ANGLE: return GetAngleSI(_dValue);
	case EUnitType::NONE: return _dValue;
	}
	return 0;
}

std::string CUnitConvertor::GetSelectedUnit(EUnitType _nPropertyType) const
{
	std::string sTemp = "1/";
	if (_nPropertyType == EUnitType::PSD_q0 || _nPropertyType == EUnitType::PSD_q3)
	{
		sTemp = sTemp.append( GetPossibleUnitsByIndex( GetPropIndex(EUnitType::PARTICLE_DIAMETER ) )[ m_vSelectedUnitType[ GetPropIndex(EUnitType::PARTICLE_DIAMETER ) ] ] );
		return sTemp;
	}

	int nIndex = GetPropIndex( _nPropertyType );
	if ( nIndex < 0 ) return "-";
	return GetPossibleUnitsByIndex( nIndex )[ m_vSelectedUnitType[ nIndex ] ];
}

int CUnitConvertor::GetPropIndex(EUnitType _nPropertyType) const
{
	for ( unsigned i=0; i < m_vProperties.size(); i++ )
		if ( m_vProperties[ i ] == _nPropertyType )
			return i;
	return -1;
}

double CUnitConvertor::GetTemperature( double _dTemperatureSI ) const
{
	int nIndex = GetPropIndex(EUnitType::TEMPERATURE );
	if ( nIndex < 0 ) return 0;
	switch( m_vSelectedUnitType[ nIndex ] )
	{
	case 0:		return _dTemperatureSI; // [K]
	case 1:		return _dTemperatureSI-273.15; //[C]
	}
	return 0;
}

double CUnitConvertor::GetMassStream( double _dMassStreamSI ) const
{
	int nIndex = GetPropIndex(EUnitType::MASS_STREAM );
	if ( nIndex < 0 ) return 0;
	switch( m_vSelectedUnitType[ nIndex ] )
	{
	case 0:		return _dMassStreamSI * 1000; // [g/s]
	case 1:		return _dMassStreamSI; //[kg/s]
	case 2:		return _dMassStreamSI * 1000 * 60.0; // [g/min]
	case 3:		return _dMassStreamSI * 60.0; // [kg/min]
	case 4:		return _dMassStreamSI / 1000 * 60.0; // [t/min]
	case 5:		return _dMassStreamSI * 1000 * 3600.0; // [g/h]
	case 6:		return _dMassStreamSI * 3600.0; // [kg/h]
	case 7:		return _dMassStreamSI / 1000 * 3600.0; // [t/h]
	case 8:		return _dMassStreamSI * 24; // [kg/day]
	case 9:		return _dMassStreamSI / 1000 * 24; // [t/day]
	}
	return 0;
}

double CUnitConvertor::GetMass( double _dMassSI ) const
{
	int nIndex = GetPropIndex(EUnitType::MASS);
	if (nIndex < 0) return 0;
	switch (m_vSelectedUnitType[nIndex])
	{
	case 0:		return _dMassSI * 1000; // [g]
	case 1:		return _dMassSI; //[kg]
	case 2:		return _dMassSI / 1000; // [t]
	}
	return 0;
}

double CUnitConvertor::GetTime(double _dTimeSI) const
{
	int nIndex = GetPropIndex(EUnitType::TIME);
	if (nIndex < 0) return 0;
	switch (m_vSelectedUnitType[nIndex])
	{
	case 0:	return _dTimeSI * 1000; // [ms]
	case 1:	return _dTimeSI; // [s]
	case 2:	return _dTimeSI / 60.0; //[min]
	case 3:	return _dTimeSI / 3600.0; // [h]
	case 4:	return _dTimeSI / 3600.0 / 24; // [d]
	}
	return 0;
}

double CUnitConvertor::GetLength(double _dLengthSI) const
{
	int nIndex = GetPropIndex(EUnitType::LENGTH);
	if (nIndex < 0) return 0;
	switch (m_vSelectedUnitType[nIndex])
	{
	case 0:		return _dLengthSI * 1000 * 1000; // [mkm]
	case 1:		return _dLengthSI * 1000; // [mm]
	case 2:		return _dLengthSI * 100; //[cm]
	case 3:		return _dLengthSI; // [m]
	}
	return 0;
}

double CUnitConvertor::GetParticleDiameter(double _dDiameterSI) const
{
	int nIndex = GetPropIndex(EUnitType::PARTICLE_DIAMETER);
	if (nIndex < 0) return 0;
	switch (m_vSelectedUnitType[nIndex])
	{
	case 0:		return _dDiameterSI * 1000 * 1000; // [mkm]
	case 1:		return _dDiameterSI * 1000; //[mm]
	case 2:		return _dDiameterSI * 100; // [cm]
	case 3:		return _dDiameterSI; // [m]
	}
	return 0;
}

double CUnitConvertor::GetPressure(double _dPressureSI) const
{
	int nIndex = GetPropIndex(EUnitType::PRESSURE);
	if (nIndex < 0) return 0;
	switch (m_vSelectedUnitType[nIndex])
	{
	case 0:		return _dPressureSI; // [Pa]
	case 1:		return _dPressureSI / 100000; //[bar]
	}
	return 0;
}

double CUnitConvertor::GetVelocity(double _dVelocitySI) const
{
	int nIndex = GetPropIndex(EUnitType::VELOCITY);
	if (nIndex < 0) return 0;
	switch (m_vSelectedUnitType[nIndex])
	{
	case 0:		return _dVelocitySI * 1e+6; // [mkm/s]
	case 1:		return _dVelocitySI * 1e+3; //[mm/s]
	case 2:		return _dVelocitySI * 100; //[cm/s]
	case 3:		return _dVelocitySI; //[m/s]
	case 4:		return _dVelocitySI * 1e+6 * 60; // [mkm/min]
	case 5:		return _dVelocitySI * 1e+3 * 60; //[mm/min]
	case 6:		return _dVelocitySI * 100 * 60; //[cm/min]
	case 7:		return _dVelocitySI * 60; //[m/min]
	}
	return 0;
}

double CUnitConvertor::GetAngVelocity(double _dVelocitySI) const
{
	int nIndex = GetPropIndex(EUnitType::ANGULAR_VELOCITY);
	if (nIndex < 0) return 0;
	switch (m_vSelectedUnitType[nIndex])
	{
	case 0:		return _dVelocitySI; // [rad/s]
	case 1:		return _dVelocitySI * 9.54929658551369; // [rpm]
	}
	return 0;
}

double CUnitConvertor::GetAngle(double _dAngleSI) const
{
	int nIndex = GetPropIndex(EUnitType::ANGLE);
	if (nIndex < 0) return 0;
	switch (m_vSelectedUnitType[nIndex])
	{
	case 0:		return _dAngleSI; // [°]
	case 1:		return _dAngleSI * 0.01745329251994; // [rad]
	}
	return 0;
}

double CUnitConvertor::GetForce(double _dForceSI) const
{
	int nIndex = GetPropIndex(EUnitType::FORCE);
	if (nIndex < 0) return 0;
	switch (m_vSelectedUnitType[nIndex])
	{
	case 0:		return _dForceSI * 1e+3; // [mN]
	case 1:		return _dForceSI; //[N]
	case 2:		return _dForceSI / 1000; //[kN]
	}
	return 0;
}

double CUnitConvertor::GetVolume(double _dVolumeSI) const
{
	int nIndex = GetPropIndex(EUnitType::VOLUME);
	if (nIndex < 0) return 0;
	switch (m_vSelectedUnitType[nIndex])
	{
	case 0:		return _dVolumeSI * 1e+3*1e+3*1e+3; // [mm3]
	case 1:		return _dVolumeSI * 100 * 100 * 100; //[cm3]
	case 2:		return _dVolumeSI; //[m3]
	}
	return 0;
}

double CUnitConvertor::GetSurface(double _dSurfaceSI) const
{
	int nIndex = GetPropIndex(EUnitType::SURFACE);
	if (nIndex < 0) return 0;
	switch (m_vSelectedUnitType[nIndex])
	{
	case 0:		return _dSurfaceSI * 1e+3*1e+3; // [mm2]
	case 1:		return _dSurfaceSI * 100 * 100; //[cm2]
	case 2:		return _dSurfaceSI; //[m2]
	}
	return 0;
}

double CUnitConvertor::GetStress(double _dStressSI) const
{
	int nIndex = GetPropIndex(EUnitType::STRESS);
	if (nIndex < 0) return 0;
	switch (m_vSelectedUnitType[nIndex])
	{
	case 0:		return _dStressSI * 1e+3; // [mPa]
	case 1:		return _dStressSI; //[Pa]
	case 2:		return _dStressSI / 1000; //[KPa]
	case 3:		return _dStressSI / 1e+6; //[MPa]
	case 4:		return _dStressSI / 1e+9; //[GPa]
	}
	return 0;
}

double CUnitConvertor::GetTemperatureSI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::TEMPERATURE)])
	{
	case 0:	return _dValue;
	case 1: return _dValue + 273.15; // from celcius to kelvin
	}
	return 0;
}


double CUnitConvertor::GetMassSI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::MASS)])
	{
	case 0:	return _dValue / 1000;
	case 1: return _dValue;
	case 2: return _dValue * 1000;
	}
	return 0;
}


double CUnitConvertor::GetMassStreamSI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::MASS_STREAM)])
	{
	case 0:	return _dValue / 1000;
	case 1: return _dValue;
	case 2: return _dValue / 1000.0 / 60;
	case 3: return _dValue / 60;
	case 4: return _dValue * 1000 / 60;
	case 5: return _dValue / 1000 / 3600;
	case 6: return _dValue / 3600;
	case 7: return _dValue * 1000 / 3600;
	case 8: return _dValue / 24;
	case 9: return _dValue * 1000 / 24;

	}
	return 0;

}

double CUnitConvertor::GetTimeSI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::TIME)])
	{
	case 0:	return _dValue / 1000;
	case 1:	return _dValue;
	case 2: return _dValue * 60;
	case 3: return _dValue * 3600;
	case 4: return _dValue * 3600 * 24;
	}
	return 0;

}

double CUnitConvertor::GetLengthSI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::LENGTH)])
	{
	case 0: return _dValue / 1000.0 / 1000.0;
	case 1:	return _dValue / 1000.0;
	case 2: return _dValue / 100.0;
	case 3: return _dValue;
	}
	return 0;
}

double CUnitConvertor::GetParticleDiameterSI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::PARTICLE_DIAMETER)])
	{
	case 0:	return _dValue / 1000.0 / 1000.0;
	case 1: return _dValue / 1000.0;
	case 2: return _dValue / 100.0;
	case 3: return _dValue;
	}
	return 0;
}

double CUnitConvertor::GetPressureSI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::PRESSURE)])
	{
	case 0:	return _dValue;
	case 1: return _dValue * 100000;
	}
	return 0;
}

double CUnitConvertor::GetVelocitySI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::VELOCITY)])
	{
	case 0:		return _dValue / 1e+6; // [mkm/s]
	case 1:		return _dValue / 1e+3; //[mm/s]
	case 2:		return _dValue / 100; //[cm/s]
	case 3:		return _dValue; //[m/s]
	case 4:		return _dValue / 1e+6 / 60; // [mkm/min]
	case 5:		return _dValue / 1e+3 / 60; //[mm/min]
	case 6:		return _dValue / 100 / 60; //[cm/min]
	case 7:		return _dValue / 60; //[m/min]
	}
	return 0;
}

double CUnitConvertor::GetAngVelocitySI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::ANGULAR_VELOCITY)])
	{
	case 0:		return _dValue; // [rad/s]
	case 1:		return _dValue / 9.54929658551369; // [rpm]
	}
	return 0;
}

double CUnitConvertor::GetAngleSI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::ANGLE)])
	{
	case 0:		return _dValue; // [°]
	case 1:		return _dValue / 0.01745329251994; // [rad]
	}
	return 0;
}

double CUnitConvertor::GetForceSI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::FORCE)])
	{
	case 0:		return _dValue / 1e+3;
	case 1:		return _dValue;
	case 2:		return _dValue * 1000;
	}
	return 0;
}

double CUnitConvertor::GetVolumeSI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::VOLUME)])
	{
	case 0:		return _dValue / 1e+3 / 1e+3 / 1e+3;
	case 1:		return _dValue / 100 / 100 / 100;
	case 2:		return _dValue;
	}
	return 0;
}

double CUnitConvertor::GetSurfaceSI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::SURFACE)])
	{
	case 0:		return _dValue / 1e+3 / 1e+3;
	case 1:		return _dValue / 100 / 100;
	case 2:		return _dValue;
	}
	return 0;
}

double CUnitConvertor::GetStressSI(double _dValue) const
{
	switch (m_vSelectedUnitType[GetPropIndex(EUnitType::STRESS)])
	{
	case 0:		return _dValue / 1e+3;
	case 1:		return _dValue;
	case 2:		return _dValue * 1e+3;
	case 3:		return _dValue * 1e+6;
	case 4:		return _dValue * 1e+9;
	}
	return 0;
}