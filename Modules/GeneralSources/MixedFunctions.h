/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <iomanip>
#include <sstream>
#include <vector>
#include <climits>
#include <cmath>

class CColor
{
public:
#pragma pack (push, 1)
	float r, g, b, a;
#pragma pack (pop)
	CColor() : r{ 0.f }, g{ 0.f }, b{ 0.f }, a{ 1.f }
	{
	}

	CColor(float _r, float _g, float _b)
	{
		r = _r; g = _g; b = _b; a = 1.0f;
	}

	CColor(float _r, float _g, float _b, float _a)
	{
		r = _r; g = _g; b = _b; a = _a;
	}

	CColor(const CColor& _color, float _a)
	{
		r = _color.r; g = _color.g; b = _color.b; a = _a;
	}

	void SetColor(float _r, float _g, float _b, float _a)
	{
		r = _r; g = _g; b = _b; a = _a;
	}

	bool operator==(const CColor& _color) const
	{
		return r == _color.r && g == _color.g && b == _color.b && a == _color.a;
	}

	bool operator!=(const CColor& _color) const
	{
		return !operator==(_color);
	}

	friend std::ostream& operator<<(std::ostream& _s, const CColor& _c)
	{
		return _s << _c.r << " " << _c.g << " " << _c.b << " " << _c.a;
	}

	friend std::istream& operator>>(std::istream& _s, CColor& _c)
	{
		_s >> _c.r >> _c.g >> _c.b >> _c.a;
		return _s;
	}

	static CColor DefaultParticleColor()			{ return CColor{ 1.0f, 1.0f, 1.0f, 1.0f }; }
	static CColor DefaultBondColor()			    { return CColor{ 0.7f, 0.7f, 0.7f, 1.0f }; }
	static CColor DefaultRealGeometryColor()		{ return CColor{ 0.5f, 0.5f, 1.0f, 1.0f }; }
	static CColor DefaultAnalysisVolumeColor()		{ return CColor{ 0.6f, 1.0f, 0.6f, 1.0f }; }
	static CColor DefaultSampleAnalyzerColor()		{ return CColor{ 0.4f, 0.4f, 0.4f, 0.5f }; }
	static CColor DefaultSimulationDomainColor()	{ return CColor{ 0.2f, 0.2f, 1.0f, 0.2f }; }
};

double inline ClampFunction( double _dValue, double _dMin, double _dMax )
{
	if ( _dValue < _dMin )
		return _dMin;
	if ( _dValue > _dMax )
		return _dMax;
	return _dValue;
}

// Determines whether the value is in range [min..max].
template<typename T>
bool IsInRange(T _value, T _min, T _max)
{
	return (_min <= _value && _value <= _max);
}

bool inline IsNaN(double _dVal)
{
	return (_dVal != _dVal);
}

void inline RandomSeed()
{
	unsigned seed = (unsigned)time(0);
	switch (seed % 5)
	{
	case 4: seed *= 2; break;
	case 3: seed *= seed; break;
	case 2: seed += seed*seed; break;
	case 1: seed = seed / 2; break;
	case 0: break;
	}
	srand(seed);
}

std::string inline Double2Percent(double _v)
{
	std::stringstream ss;
	ss << std::fixed << std::setprecision(2) << _v * 100. << "%";
	return ss.str();
}

// Rounds the value to the selected number of digits
double inline RoundToDecimalPlaces(double _value, size_t _places)
{
	const double factor = std::pow(10.0, _places - std::ceil(std::log10(std::fabs(_value))));
	return std::round(_value * factor) / factor;
}


std::string inline MsToTimeSpan(int64_t _ms)
{
	int64_t seconds = _ms / 1000;
	int64_t minutes = seconds / 60;
	int64_t hours = minutes / 60;
	int64_t days = hours / 24;
	seconds -= minutes * 60;
	minutes -= hours * 60;
	hours -= days * 24;
	std::stringstream ss;
	ss << std::setw(2) << std::setfill('0') << days    << ":"
	   << std::setw(2) << std::setfill('0') << hours   << ":"
	   << std::setw(2) << std::setfill('0') << minutes << ":"
	   << std::setw(2) << std::setfill('0') << seconds;
	return ss.str();
}

struct SDate
{
	unsigned nYear;
	unsigned nMonth;
	unsigned nDay;
	SDate() : nYear(1900), nMonth(1), nDay(1) {}
	SDate(unsigned _nY, unsigned _nM, unsigned _nD) : nYear(_nY), nMonth(_nM), nDay(_nD) {}
	void SetDate(unsigned _nY, unsigned _nM, unsigned _nD) { nYear = _nY; nMonth = _nM; nDay = _nD; }
};

inline SDate CurrentDate()
{
	time_t t = time(0);   // get time now
	struct tm now;
#ifdef PATH_CONFIGURED // for Linux
	localtime_r(&t, &now);
#else
	localtime_s(&now, &t);
#endif
	return SDate(now.tm_year + 1900, now.tm_mon + 1, now.tm_mday);
}

// Converts enumerator value to its underlying integral type.
template<typename E> constexpr typename std::underlying_type<E>::type E2I(E e)
{
	return static_cast<typename std::underlying_type<E>::type>(e);
}

// Converts vector of enumerators to the vector of its underlying integral type.
template <typename E> std::vector<typename std::underlying_type<E>::type> E2I(const std::vector<E>& _enums)
{
	using integral_type = typename std::underlying_type<E>::type;
	std::vector<integral_type> res;
	res.reserve(_enums.size());
	for (const auto& e : _enums)
		res.push_back(static_cast<integral_type>(e));
	return res;
}

// Converts double value to enumerator.
template<typename E>
constexpr E D2E(double d)
{
	return static_cast<E>(static_cast<typename std::underlying_type<E>::type>(d));
}

// casts vector of TI to a vector of TO
template<typename TO, typename TI> constexpr auto vector_cast(const std::vector<TI>& _vector)
{
	std::vector<TO> res;
	res.reserve(_vector.size());
	for (const auto& val : _vector)
		res.push_back(static_cast<TO>(val));
	return res;
}

enum class ESimulatorType : unsigned { BASE = 0, CPU = 1, GPU = 2 };

inline uint64_t MortonEncode(size_t _x, size_t _y, size_t _z)
{
	uint64_t answer = 0;
	for (uint64_t i = 0; i < sizeof(uint64_t) * CHAR_BIT / 3; ++i)
	{
		const uint64_t shifted = static_cast<uint64_t>(1) << i;
		answer |= (_x & shifted) << (2 * i) | (_y & shifted) << (2 * i + 1) | (_z & shifted) << (2 * i + 2);
	}
	return answer;
}