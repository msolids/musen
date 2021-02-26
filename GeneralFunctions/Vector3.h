/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <iostream>
#include <math.h>
#include "BasicGPUFunctions.cuh"

template<class T> class MinValueHelper
{
public:	static constexpr T min_value() { return T{0}; }
};

template<> class MinValueHelper<double>
{
public: static constexpr double min_value() { return 1e-100; }
};

template<> class MinValueHelper<float>
{
public: static constexpr float min_value() { return 1e-19f; }
};

template<class T>
class CBasicVector3
{
	static constexpr T MIN_SIGNIFICANT_VALUE = MinValueHelper<T>::min_value();

public:
	T x, y, z;

	CUDA_HOST_DEVICE CBasicVector3() {}

	CUDA_HOST_DEVICE explicit CBasicVector3(const T& _d) : x(_d), y(_d), z(_d) {}
	CUDA_HOST_DEVICE explicit CBasicVector3(const T& _dx, const T& _dy, const T& _dz) : x(_dx), y(_dy), z(_dz) {}

	CUDA_HOST_DEVICE void Init(const T& _d) { x = _d; y = _d; z = _d; }
	CUDA_HOST_DEVICE void Init(const T& _dx, const T& _dy, const T& _dz) { x = _dx; y = _dy; z = _dz; }

	CUDA_HOST_DEVICE size_t Size() const { return 3; }
	CUDA_HOST_DEVICE bool IsZero() const { return x == 0 && y == 0 && z == 0; }
	CUDA_HOST_DEVICE bool IsSignificant() const { return fabs(x) > MIN_SIGNIFICANT_VALUE || fabs(y) > MIN_SIGNIFICANT_VALUE || fabs(z) > MIN_SIGNIFICANT_VALUE; }
	CUDA_HOST_DEVICE bool IsInf() const { return isinf(x) || isinf(y) || isinf(z); }

	CUDA_HOST_DEVICE T Length() const { return sqrt(x*x + y*y + z*z); }
	CUDA_HOST_DEVICE friend T Length(const CBasicVector3& _v) { return sqrt(_v.x*_v.x + _v.y*_v.y + _v.z*_v.z); }
	CUDA_HOST_DEVICE friend T Length(const CBasicVector3& _v1, const CBasicVector3& _v2) { return sqrt((_v1.x - _v2.x)*(_v1.x - _v2.x) + (_v1.y - _v2.y)*(_v1.y - _v2.y) + (_v1.z - _v2.z)*(_v1.z - _v2.z)); }

	CUDA_HOST_DEVICE T SquaredLength() const { return x*x + y*y + z*z; }
	CUDA_HOST_DEVICE friend T SquaredLength(const CBasicVector3& _v) { return _v.x*_v.x + _v.y*_v.y + _v.z*_v.z; }
	CUDA_HOST_DEVICE friend T SquaredLength(const CBasicVector3& _v1, const CBasicVector3& _v2) { return (_v1.x - _v2.x)*(_v1.x - _v2.x) + (_v1.y - _v2.y)*(_v1.y - _v2.y) + (_v1.z - _v2.z)*(_v1.z - _v2.z); }

	CUDA_HOST_DEVICE friend T DotProduct(const CBasicVector3& _v1, const CBasicVector3& _v2) { return _v1.x * _v2.x + _v1.y * _v2.y + _v1.z * _v2.z; }
	CUDA_HOST_DEVICE friend CBasicVector3 EntryWiseProduct(const CBasicVector3& _v1, const CBasicVector3& _v2) { return CBasicVector3(_v1.x * _v2.x, _v1.y * _v2.y, _v1.z * _v2.z); }
	CUDA_HOST_DEVICE friend T ScalarTripleProduct(const CBasicVector3& _v1, const CBasicVector3& _v2, const CBasicVector3& _v3) { return DotProduct(_v1 * _v2, _v3); }

	CUDA_HOST_DEVICE CBasicVector3 Normalized() const { const T len = sqrt(x*x + y*y + z*z); return len != 0 ? *this / len : CBasicVector3(0); }
	CUDA_HOST_DEVICE friend CBasicVector3 Normalized(const CBasicVector3& _v) { const T len = sqrt(_v.x*_v.x + _v.y*_v.y + _v.z*_v.z); return len != 0 ? _v / len : CBasicVector3(0); }

	CUDA_HOST_DEVICE friend CBasicVector3 Min(const CBasicVector3& _v1, const CBasicVector3& _v2) { return CBasicVector3(_v2.x < _v1.x ? _v2.x : _v1.x, _v2.y < _v1.y ? _v2.y : _v1.y, _v2.z < _v1.z ? _v2.z : _v1.z); }
	CUDA_HOST_DEVICE friend CBasicVector3 Min(const CBasicVector3& _v1, const CBasicVector3& _v2, const CBasicVector3& _v3) { return Min(Min(_v1, _v2), _v3); }
	CUDA_HOST_DEVICE friend CBasicVector3 Min(const CBasicVector3& _v1, const CBasicVector3& _v2, const CBasicVector3& _v3, const CBasicVector3& _v4) { return Min(Min(_v1, _v2), Min(_v3, _v4)); }

	CUDA_HOST_DEVICE friend CBasicVector3 Max(const CBasicVector3& _v1, const CBasicVector3& _v2) { return CBasicVector3(_v2.x > _v1.x ? _v2.x : _v1.x, _v2.y > _v1.y ? _v2.y : _v1.y, _v2.z > _v1.z ? _v2.z : _v1.z); }
	CUDA_HOST_DEVICE friend CBasicVector3 Max(const CBasicVector3& _v1, const CBasicVector3& _v2, const CBasicVector3& _v3) { return Max(Max(_v1, _v2), _v3); }
	CUDA_HOST_DEVICE friend CBasicVector3 Max(const CBasicVector3& _v1, const CBasicVector3& _v2, const CBasicVector3& _v3, const CBasicVector3& _v4) { return Max(Max(_v1, _v2), Max(_v3, _v4)); }

	CUDA_HOST_DEVICE friend CBasicVector3 MaxLength(const CBasicVector3& _v1, const CBasicVector3& _v2) { return _v1.SquaredLength() > _v2.SquaredLength() ? _v1 : _v2; }

	CUDA_HOST_DEVICE CBasicVector3& operator+=(const CBasicVector3& _v) { x += _v.x; y += _v.y; z += _v.z; return *this; }
	CUDA_HOST_DEVICE CBasicVector3& operator+=(const T& _d) { x += _d; y += _d; z += _d; return *this; }
	CUDA_HOST_DEVICE CBasicVector3& operator-=(const CBasicVector3& _v) { x -= _v.x; y -= _v.y; z -= _v.z; return *this; }
	CUDA_HOST_DEVICE CBasicVector3& operator-=(const T& _d) { x -= _d; y -= _d; z -= _d; return *this; }
	CUDA_HOST_DEVICE CBasicVector3& operator*=(const T& _d) { x *= _d; y *= _d; z *= _d; return *this; }
	CUDA_HOST_DEVICE CBasicVector3& operator/=(const T& _d) { x /= _d; y /= _d; z /= _d; return *this; }

	CUDA_HOST_DEVICE CBasicVector3 operator+(const T& _d) const { return CBasicVector3(x + _d, y + _d, z + _d); }
	CUDA_HOST_DEVICE CBasicVector3 operator+(const CBasicVector3& _v) const { return CBasicVector3(x + _v.x, y + _v.y, z + _v.z); }
	CUDA_HOST_DEVICE CBasicVector3 operator-(const T& _d) const { return CBasicVector3(x - _d, y - _d, z - _d); }
	CUDA_HOST_DEVICE CBasicVector3 operator-(const CBasicVector3& _v) const { return CBasicVector3(x - _v.x, y - _v.y, z - _v.z); }
	CUDA_HOST_DEVICE CBasicVector3 operator*(const T& _d) const { return CBasicVector3(x * _d, y * _d, z * _d); }
	CUDA_HOST_DEVICE CBasicVector3 operator*(const CBasicVector3& _v) const { return CBasicVector3(y*_v.z - z*_v.y, z*_v.x - x*_v.z, x*_v.y - y*_v.x); }
	CUDA_HOST_DEVICE friend CBasicVector3 operator*(const T& _d, const CBasicVector3& _v) { return CBasicVector3(_v.x * _d, _v.y * _d, _v.z * _d); }
	CUDA_HOST_DEVICE CBasicVector3 operator/(const T& _d) const { return CBasicVector3(x / _d, y / _d, z / _d); }
	CUDA_HOST_DEVICE CBasicVector3 operator/(const CBasicVector3& _v) const { return CBasicVector3(x / _v.x, y / _v.y, z / _v.z); }
	CUDA_HOST_DEVICE T operator[](size_t _i) const { switch (_i) { case 0: return x; case 1: return y; case 2: return z; default: return {}; } }
	CUDA_HOST_DEVICE T& operator[](size_t _i) { switch (_i) { case 0: return x; case 1: return y; case 2: return z; default: throw std::out_of_range("CBasicVector3<T>::operator[size_t] : index is out of range"); } }

	CUDA_HOST_DEVICE bool operator==(const CBasicVector3& _v) const { return (x == _v.x && y == _v.y && z == _v.z); }
	CUDA_HOST_DEVICE bool operator!=(const CBasicVector3& _v) const { return !(x == _v.x && y == _v.y && z == _v.z); }
	CUDA_HOST_DEVICE bool operator>(const CBasicVector3& _v) const { return SquaredLength() > _v.SquaredLength(); }
	CUDA_HOST_DEVICE bool operator<(const CBasicVector3& _v) const { return SquaredLength() < _v.SquaredLength(); }

	friend std::ostream& operator<<(std::ostream& _s, const CBasicVector3& _v) { return _s << _v.x << " " << _v.y << " " << _v.z; }
	friend std::istream& operator >> (std::istream& _s, CBasicVector3& _v) { _s >> _v.x >> _v.y >> _v.z; return _s; }

	template<typename T2>
	CUDA_HOST_DEVICE operator CBasicVector3<T2>() const { return CBasicVector3<T2>(static_cast<T2>(x), static_cast<T2>(y), static_cast<T2>(z)); }
};

using CVector3  = CBasicVector3<double>;
using CVector3d = CBasicVector3<double>;
using CVector3f = CBasicVector3<float>;
using CVector3b = CBasicVector3<bool>;