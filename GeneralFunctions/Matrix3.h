/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "Vector3.h"
#include "MUSENDefinitions.h"

template<typename T>
class CBasicMatrix3
{
public:
	T values[3][3];

	CUDA_HOST_DEVICE CBasicMatrix3() {}
	CUDA_HOST_DEVICE CBasicMatrix3(const T& _d) : values{ { _d, _d, _d },{ _d, _d, _d },{ _d, _d, _d } } {}
	CUDA_HOST_DEVICE CBasicMatrix3(
		const T& _d00, const T& _d01, const T& _d02,
		const T& _d10, const T& _d11, const T& _d12,
		const T& _d20, const T& _d21, const T& _d22) :
		values{	{ _d00, _d01, _d02 },
				{ _d10, _d11, _d12 },
				{ _d20, _d21, _d22 } } {}

	CUDA_HOST_DEVICE void Init(const T& _d)
	{
		values[0][0] = values[0][1] = values[0][2] = _d;
		values[1][0] = values[1][1] = values[1][2] = _d;
		values[2][0] = values[2][1] = values[2][2] = _d;
	}

	// Returns diagonal matrix.
	static CUDA_HOST_DEVICE CBasicMatrix3 Diagonal()
	{
		return {
			1, 0, 0,
			0, 1, 0,
			0, 0, 1
		};
	}

	static CUDA_HOST_DEVICE CBasicMatrix3 GetFromVecMult(const CBasicVector3<T>& _v)
	{
		return {
			_v.x*_v.x, _v.x*_v.y, _v.x*_v.z,
			_v.y*_v.x, _v.y*_v.y, _v.y*_v.z,
			_v.z*_v.x, _v.z*_v.y, _v.z*_v.z
		};
	}


	CUDA_HOST_DEVICE CBasicMatrix3 Inverse() const
	{
		const T det = values[0][0] * (values[1][1] * values[2][2] - values[1][2] * values[2][1]) -
					  values[0][1] * (values[1][0] * values[2][2] - values[1][2] * values[2][0]) +
					  values[0][2] * (values[1][0] * values[2][1] - values[1][1] * values[2][0]);

		if (det == 0) return {};

		const T invDet = 1. / det;
		return {
			(values[1][1] * values[2][2] - values[2][1] * values[1][2])*invDet,
			(values[0][2] * values[2][1] - values[0][1] * values[2][2])*invDet,
			(values[0][1] * values[1][2] - values[0][2] * values[1][1])*invDet,
			(values[1][2] * values[2][0] - values[1][0] * values[2][2])*invDet,
			(values[0][0] * values[2][2] - values[0][2] * values[2][0])*invDet,
			(values[1][0] * values[0][2] - values[0][0] * values[1][2])*invDet,
			(values[1][0] * values[2][1] - values[2][0] * values[1][1])*invDet,
			(values[2][0] * values[0][1] - values[0][0] * values[2][1])*invDet,
			(values[0][0] * values[1][1] - values[1][0] * values[0][1])*invDet
		};
	}

	CUDA_HOST_DEVICE CBasicMatrix3 Transpose() const
	{
		return {
			values[0][0], values[1][0], values[2][0],
			values[0][1], values[1][1], values[2][1],
			values[0][2], values[1][2], values[2][2]
		};
	}

	CUDA_HOST_DEVICE CVector3 GetPrincipalStresses() const
	{
		CVector3 res;
		const double I1 = values[0][0] + values[1][1] + values[2][2];
		const double I2 = values[0][0] * values[1][1] + values[1][1] * values[2][2] + values[2][2] * values[0][0]
			- pow(values[0][1], 2.) - pow(values[0][2], 2.) - pow(values[1][2], 2.);
		const double I3 = values[0][0] * values[1][1] * values[2][2]
			- values[0][0] * pow(values[1][2], 2.) - values[1][1] * pow(values[0][2], 2.) - values[2][2] * pow(values[0][1], 2.)
			+ 2 * values[0][1] * values[0][2] * values[1][2];
		const double Q = (3 * I2 - pow(I1, 2.)) / 9.0;
		const double R = (2 * pow(I1, 3.) - 9 * I1 * I2 + 27 * I3) / 54;
		const double Te = Q < 0.0 ? acos(R / sqrt(pow(-Q, 3.))) : 0.0;
		res.x = 2 * sqrt(-Q) * cos(Te / 3) + 1.0 / 3 * I1;
		res.y = 2 * sqrt(-Q) * cos((Te + 2 * PI) / 3) + 1.0 / 3 * I1;
		res.z = 2 * sqrt(-Q) * cos((Te + 4 * PI) / 3) + 1.0 / 3 * I1;
		return res;
	}

	CUDA_HOST_DEVICE CBasicMatrix3 operator+(const CBasicMatrix3& _m) const
	{
		return {
			values[0][0] + _m.values[0][0], values[0][1] + _m.values[0][1], values[0][2] + _m.values[0][2],
			values[1][0] + _m.values[1][0], values[1][1] + _m.values[1][1], values[1][2] + _m.values[1][2],
			values[2][0] + _m.values[2][0], values[2][1] + _m.values[2][1], values[2][2] + _m.values[2][2]
		};
	}

	CUDA_HOST_DEVICE CBasicMatrix3 operator-(const CBasicMatrix3& _m) const
	{
		return {
			values[0][0] - _m.values[0][0], values[0][1] - _m.values[0][1], values[0][2] - _m.values[0][2],
			values[1][0] - _m.values[1][0], values[1][1] - _m.values[1][1], values[1][2] - _m.values[1][2],
			values[2][0] - _m.values[2][0], values[2][1] - _m.values[2][1], values[2][2] - _m.values[2][2]
		};
	}

	CUDA_HOST_DEVICE CBasicMatrix3 operator*(const T& _d) const
	{
		return {
			values[0][0] * _d, values[0][1] * _d, values[0][2] * _d,
			values[1][0] * _d, values[1][1] * _d, values[1][2] * _d,
			values[2][0] * _d, values[2][1] * _d, values[2][2] * _d
		};
	}

	CUDA_HOST_DEVICE friend CBasicMatrix3 operator*(const T& _d, const CBasicMatrix3& _m)
	{
		return _m * _d;
	}

	CUDA_HOST_DEVICE CBasicVector3<T> operator*(const CBasicVector3<T>& _v) const
	{
		return CBasicVector3<T> {
			values[0][0] * _v.x + values[0][1] * _v.y + values[0][2] * _v.z,
			values[1][0] * _v.x + values[1][1] * _v.y + values[1][2] * _v.z,
			values[2][0] * _v.x + values[2][1] * _v.y + values[2][2] * _v.z
		};
	}

	CUDA_HOST_DEVICE CBasicMatrix3 operator*(const CBasicMatrix3& _m) const
	{
		return {
			values[0][0] * _m.values[0][0] + values[0][1] * _m.values[1][0] + values[0][2] * _m.values[2][0],
			values[0][0] * _m.values[0][1] + values[0][1] * _m.values[1][1] + values[0][2] * _m.values[2][1],
			values[0][0] * _m.values[0][2] + values[0][1] * _m.values[1][2] + values[0][2] * _m.values[2][2],
			values[1][0] * _m.values[0][0] + values[1][1] * _m.values[1][0] + values[1][2] * _m.values[2][0],
			values[1][0] * _m.values[0][1] + values[1][1] * _m.values[1][1] + values[1][2] * _m.values[2][1],
			values[1][0] * _m.values[0][2] + values[1][1] * _m.values[1][2] + values[1][2] * _m.values[2][2],
			values[2][0] * _m.values[0][0] + values[2][1] * _m.values[1][0] + values[2][2] * _m.values[2][0],
			values[2][0] * _m.values[0][1] + values[2][1] * _m.values[1][1] + values[2][2] * _m.values[2][1],
			values[2][0] * _m.values[0][2] + values[2][1] * _m.values[1][2] + values[2][2] * _m.values[2][2]
		};
	}

	CUDA_HOST_DEVICE CBasicMatrix3 operator/(const T& _d) const
	{
		return {
			values[0][0] / _d, values[0][1] / _d, values[0][2] / _d,
			values[1][0] / _d, values[1][1] / _d, values[1][2] / _d,
			values[2][0] / _d, values[2][1] / _d, values[2][2] / _d
		};
	}

	friend std::ostream& operator << (std::ostream& _s, const CBasicMatrix3& _m)
	{
		return _s << _m.values[0][0] << " " << _m.values[0][1] << " " << _m.values[0][2] << " " << _m.values[1][0] << " " << _m.values[1][1] << " " << _m.values[1][2] << " " << _m.values[2][0] << " " << _m.values[2][1] << " " << _m.values[2][2];
	}

	friend std::istream& operator >> (std::istream& _s, CBasicMatrix3& _m)
	{
		_s >> _m.values[0][0] >> _m.values[0][1] >> _m.values[0][2] >> _m.values[1][0] >> _m.values[1][1] >> _m.values[1][2] >> _m.values[2][0] >> _m.values[2][1] >> _m.values[2][2];
		return _s;
	}

	template<typename D>
	CUDA_HOST_DEVICE operator CBasicMatrix3<D>() const { return CBasicMatrix3<D>((D)values[0][0], (D)values[0][1], (D)values[0][2], (D)values[1][0], (D)values[1][1], (D)values[1][2], (D)values[2][0], (D)values[2][1], (D)values[2][2]); }
};

using CMatrix3  = CBasicMatrix3<double>;
using CMatrix3d = CBasicMatrix3<double>;
using CMatrix3f = CBasicMatrix3<float>;