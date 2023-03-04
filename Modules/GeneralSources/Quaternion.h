/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <cmath>
#include "BasicTypes.h"
#include "BasicGPUFunctions.cuh"
#include "Vector3.h"
#include "Matrix3.h"

template<typename T>
class CBasicQuaternion
{
public:
	T q0, q1, q2, q3;

	CBasicQuaternion() = default;
	CUDA_HOST_DEVICE explicit CBasicQuaternion(const T& _q0, const T& _q1, const T& _q2, const T& _q3) : q0(_q0), q1(_q1), q2(_q2), q3(_q3) {}
	CUDA_HOST_DEVICE CBasicQuaternion(const CBasicVector3<T>& _vec) { SetFromEulerAnglesXYZ(_vec); }
	CUDA_HOST_DEVICE CBasicQuaternion(const CBasicMatrix3<T>& _M) { FromRotmat(_M); }

	CUDA_HOST_DEVICE void Init(const T& _q0, const T& _q1, const T& _q2, const T& _q3) { q0 = _q0; q1 = _q1; q2 = _q2; q3 = _q3; }

	CUDA_HOST_DEVICE void Normalize()
	{
		T dTempSum = sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
		if (q0 < 0)			// ensure that first element is always positive (convention, doesn't change rotation)
			dTempSum *= -1;

		q0 /= dTempSum; q1 /= dTempSum; q2 /= dTempSum; q3 /= dTempSum;
	}

	CUDA_HOST_DEVICE CBasicQuaternion Normalized()
	{
		Normalize();
		return *this;
	}

	CUDA_HOST_DEVICE void RandomGenerator()
	{
		*this = Random();
	}

	CUDA_HOST_DEVICE static CBasicQuaternion Random()
	{
		// From http://planning.cs.uiuc.edu/node198.html. See K. Shoemake - Uniform random rotations - In D. Kirk, editor, Graphics Gems III, pages 124-132. Academic, New York, 1992.

		const T u1 = static_cast<T>(rand()) / RAND_MAX;
		const T u2 = static_cast<T>(rand()) / RAND_MAX;
		const T u3 = static_cast<T>(rand()) / RAND_MAX;
		CBasicQuaternion quaternion{
			sqrt(1 - u1) * sin(2 * PI * u2),
			sqrt(1 - u1) * cos(2 * PI * u2),
			sqrt(u1) * sin(2 * PI * u3),
			sqrt(u1) * cos(2 * PI * u3) };
		quaternion.Normalize();
		return quaternion;
	}

	// Calculate Euler angles (order first X, then Y, then Z) from quaternion, see https://github.com/mrdoob/three.js/blob/dev/src/math/Euler.js (ZYX)
	CUDA_HOST_DEVICE CBasicVector3<T> ToEulerAnglesXYZ()
	{
		Normalize();

		CBasicVector3<T> vResult;
		// y-axis rotation
		T t2 = 2 * (q0 * q2 - q1 * q3);
		t2 = t2 > 1.0 ? 1.0 : t2;
		t2 = t2 < -1.0 ? -1.0 : t2;
		vResult.y = asin(t2);
		// x- and z-axis rotation
		if (abs(t2) < 0.99999)
		{
			vResult.x = atan2(2 * (q2*q3 + q0 * q1), q0*q0 - q1 * q1 - q2 * q2 + q3 * q3);
			vResult.z = atan2(2 * (q1*q2 + q0 * q3), q0*q0 + q1 * q1 - q2 * q2 - q3 * q3);
		}
		else
		{
			vResult.x = 0;
			vResult.z = atan2(2 * (q0*q3 - q1 * q2), q0*q0 - q1 * q1 + q2 * q2 - q3 * q3);
		}

		return vResult;
	}

	CUDA_HOST_DEVICE CBasicVector3<T> ToEulerAnglesXYZNoNormalization()
	{
		CBasicVector3<T> vResult;
		// y-axis rotation
		T t2 = 2 * (q0 * q2 - q1 * q3);
		t2 = t2 > 1.0 ? 1.0 : t2;
		t2 = t2 < -1.0 ? -1.0 : t2;
		vResult.y = asin(t2);
		// x- and z-axis rotation
		if (abs(t2) < 0.99999)
		{
			vResult.x = atan2(2 * (q2*q3 + q0 * q1), q0*q0 - q1 * q1 - q2 * q2 + q3 * q3);
			vResult.z = atan2(2 * (q1*q2 + q0 * q3), q0*q0 + q1 * q1 - q2 * q2 - q3 * q3);
		}
		else
		{
			vResult.x = 0;
			vResult.z = atan2(2 * (q0*q3 - q1 * q2), q0*q0 - q1 * q1 + q2 * q2 - q3 * q3);
		}

		return vResult;
	}

	//  Calculate quaternion from Euler angles (order first X, then Y, then Z)
	CUDA_HOST_DEVICE void SetFromEulerAnglesXYZ(const CBasicVector3<T>& _vec)
	{
		const T c1 = cos(_vec.z*0.5);
		const T s1 = sin(_vec.z*0.5);
		const T c2 = cos(_vec.y*0.5);
		const T s2 = sin(_vec.y*0.5);
		const T c3 = cos(_vec.x*0.5);
		const T s3 = sin(_vec.x*0.5);

		q0 = c1 * c2*c3 + s1 * s2*s3;
		q1 = c1 * c2*s3 - s1 * s2*c3;
		q2 = c1 * s2*c3 + s1 * c2*s3;
		q3 = s1 * c2*c3 - c1 * s2*s3;

		Normalize();
	}

	// Calculate rotation matrix from quaternion
	CUDA_HOST_DEVICE CBasicMatrix3<T> ToRotmat()
	{
		Normalize();

		const T w1 = q0 * q0;
		const T w2 = q1 * q1;
		const T w3 = q2 * q2;
		const T w4 = q3 * q3;
		const T ij = 2 * q1*q2;
		const T ik = 2 * q1*q3;
		const T jk = 2 * q2*q3;
		const T iw = 2 * q1*q0;
		const T jw = 2 * q2*q0;
		const T kw = 2 * q3*q0;

		return CBasicMatrix3<T> {
			w1 + w2 - w3 - w4	, ij - kw				, jw + ik ,
			ij + kw				, w1 - w2 + w3 - w4		, jk - iw ,
			ik - jw				, jk + iw				, w1 - w2 - w3 + w4 };
	}

	CUDA_HOST_DEVICE CBasicMatrix3<T> ToRotmatNoNormalization() const
	{
		const T w1 = q0 * q0;
		const T w2 = q1 * q1;
		const T w3 = q2 * q2;
		const T w4 = q3 * q3;
		const T ij = 2 * q1*q2;
		const T ik = 2 * q1*q3;
		const T jk = 2 * q2*q3;
		const T iw = 2 * q1*q0;
		const T jw = 2 * q2*q0;
		const T kw = 2 * q3*q0;

		return CBasicMatrix3<T> {
			w1 + w2 - w3 - w4, ij - kw,           jw + ik,
			ij + kw,           w1 - w2 + w3 - w4, jk - iw,
			ik - jw,           jk + iw,           w1 - w2 - w3 + w4};
	}

	// from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
	CUDA_HOST_DEVICE void FromRotmat(const CBasicMatrix3<T>& _M)
	{
		T trace = _M.values[0][0] + _M.values[1][1] + _M.values[2][2];
		if (trace > 0)
		{
			T s = 0.5 / sqrt(trace + 1.0);
			q0 = 0.25 / s;
			q1 = (_M.values[2][1] - _M.values[1][2]) * s;
			q2 = (_M.values[0][2] - _M.values[2][0]) * s;
			q3 = (_M.values[1][0] - _M.values[0][1]) * s;
		}
		else
		{
			if (_M.values[0][0] > _M.values[1][1] && _M.values[0][0] > _M.values[2][2])
			{
				T s = 2.0 * sqrt(1.0 + _M.values[0][0] - _M.values[1][1] - _M.values[2][2]);
				q0 = (_M.values[2][1] - _M.values[1][2]) / s;
				q1 = 0.25 * s;
				q2 = (_M.values[0][1] + _M.values[1][0]) / s;
				q3 = (_M.values[0][2] + _M.values[2][0]) / s;
			}
			else if (_M.values[1][1] > _M.values[2][2])
			{
				T s = 2.0 * sqrt(1.0 + _M.values[1][1] - _M.values[0][0] - _M.values[2][2]);
				q0 = (_M.values[0][2] - _M.values[2][0]) / s;
				q1 = (_M.values[0][1] + _M.values[1][0]) / s;
				q2 = 0.25 * s;
				q3 = (_M.values[1][2] + _M.values[2][1]) / s;
			}
			else
			{
				T s = 2.0 * sqrt(1.0 + _M.values[2][2] - _M.values[0][0] - _M.values[1][1]);
				q0 = (_M.values[1][0] - _M.values[0][1]) / s;
				q1 = (_M.values[0][2] + _M.values[2][0]) / s;
				q2 = (_M.values[1][2] + _M.values[2][1]) / s;
				q3 = 0.25 * s;
			}
		}
		Normalize();
	}

	CUDA_HOST_DEVICE CBasicQuaternion Inverse() const
	{
		const T dTempSum = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
		return CBasicQuaternion(q0 / dTempSum, -q1 / dTempSum, -q2 / dTempSum, -q3 / dTempSum);
	}

	CUDA_HOST_DEVICE CBasicQuaternion& operator+=(const CBasicQuaternion& _q) { q0 += _q.q0; q1 += _q.q1; q2 += _q.q2; q3 += _q.q3; return *this; }

	CUDA_HOST_DEVICE const CBasicQuaternion operator-(const CBasicQuaternion& _q) const { return CBasicQuaternion(q0 - _q.q0, q1 - _q.q1, q2 - _q.q2, q3 - _q.q3); }
	CUDA_HOST_DEVICE const CBasicQuaternion operator+(const CBasicQuaternion& _q) const { return CBasicQuaternion(q0 + _q.q0, q1 + _q.q1, q2 + _q.q2, q3 + _q.q3); }
	CUDA_HOST_DEVICE const CBasicQuaternion operator/(const T& _d) const { return CBasicQuaternion(q0 / _d, q1 / _d, q2 / _d, q3 / _d); }
	CUDA_HOST_DEVICE const CBasicQuaternion operator*(const T& _d) const { return CBasicQuaternion(q0 * _d, q1 * _d, q2 * _d, q3 * _d); }
	CUDA_HOST_DEVICE friend const CBasicQuaternion operator*(const T& _d, const CBasicQuaternion& _q) { return CBasicQuaternion(_q.q0 * _d, _q.q1 * _d, _q.q2 * _d, _q.q3 * _d); }
	CUDA_HOST_DEVICE const CBasicQuaternion operator*(const CBasicQuaternion & _q) const
	{
		return CBasicQuaternion{
			q0*_q.q0 - q1 * _q.q1 - q2 * _q.q2 - q3 * _q.q3,
			q1*_q.q0 + q0 * _q.q1 - q3 * _q.q2 + q2 * _q.q3,
			q2*_q.q0 + q3 * _q.q1 + q0 * _q.q2 - q1 * _q.q3,
			q3*_q.q0 - q2 * _q.q1 + q1 * _q.q2 + q0 * _q.q3 };
	}

	friend std::ostream& operator<<(std::ostream& _s, const CBasicQuaternion& _q) { return _s << _q.q0 << " " << _q.q1 << " " << _q.q2 << " " << _q.q3; }
	friend std::istream& operator>>(std::istream& _s, CBasicQuaternion& _q) { _s >> _q.q0 >> _q.q1 >> _q.q2 >> _q.q3; return _s; }

	CUDA_HOST_DEVICE friend CBasicQuaternion QuaternionAverage(const CBasicQuaternion& _q1, const CBasicQuaternion& _q2)
	{
		// Method from Markley - "Quaternion Averaging"
		CBasicQuaternion output;
		T q1Tq2 = _q1.q0*_q2.q0 + _q1.q1*_q2.q1 + _q1.q2*_q2.q2 + _q1.q3*_q2.q3;

		if (q1Tq2 != 0)
		{
			output = _q1 * std::abs(q1Tq2) + _q2 * q1Tq2;
		}
		else
		{
			// near singularity
			T w1 = 0.5;
			T w2 = 0.6;
			T z = sqrt((w1 - w2)*(w1 - w2) + 4 * w1*w2*q1Tq2*q1Tq2);
			output = _q1 * 2 * w1 * q1Tq2 + _q2 * (w2 - w1 + z);
		}
		output.Normalize();
		return output;
	}

	CUDA_HOST_DEVICE friend CBasicQuaternion QuatRelAtoB(const CBasicQuaternion& _q1, const CBasicQuaternion& _q2) { CBasicQuaternion quatTemp = _q2 * _q1.Inverse(); return quatTemp.Normalized(); }
	CUDA_HOST_DEVICE friend CBasicQuaternion QuatRelAtoBNoNormalization(const CBasicQuaternion& _q1, const CBasicQuaternion& _q2) { CBasicQuaternion quatTemp = _q2 * _q1.Inverse(); return quatTemp; }

	CUDA_HOST_DEVICE CBasicQuaternion Quat2AxAng() const
	{
		// Elements: angle, x, y, z
		CBasicQuaternion input(*this);
		input.Normalize();
		const T angle = 2 * std::acos(input.q0);
		CBasicVector3<T> axis(input.q1, input.q2, input.q3);
		axis = axis.Normalized();
		return CBasicQuaternion(angle, axis.x, axis.y, axis.z);
	}

	CUDA_HOST_DEVICE friend CBasicVector3<T> QuatRotateVector(const CBasicQuaternion& _quatInput, const CBasicVector3<T>& _v)
	{
		CBasicQuaternion _q = _quatInput;
		_q.Normalize();
		return _q.ToRotmat() * _v;
	}

	CUDA_HOST_DEVICE CBasicQuaternion NewRefFrame(CBasicQuaternion NewRefFrame) const
	{
		// Express quaternion in new reference frame specified by NewRefFrame - this is equivalent to rotating axis vector specified by last three elements
		CBasicQuaternion input(*this);
		input.Normalize();
		CBasicVector3<T> axis = NewRefFrame.ToRotmat().Transpose() * CBasicVector3<T>(input.q1, input.q2, input.q3);
		return CBasicQuaternion(input.q0, axis.x, axis.y, axis.z);
	}

	CUDA_HOST_DEVICE CBasicQuaternion NewRefFrameNoNormalization(CBasicQuaternion NewRefFrame) const
	{
		// Express quaternion in new reference frame specified by NewRefFrame - this is equivalent to rotating axis vector specified by last three elements
		CBasicQuaternion input(*this);
		CBasicVector3<T> axis = NewRefFrame.ToRotmat().Transpose() * CBasicVector3<T>(input.q1, input.q2, input.q3);
		return CBasicQuaternion(input.q0, axis.x, axis.y, axis.z);
	}

	CUDA_HOST_DEVICE CBasicQuaternion NewRefFrame(CBasicMatrix3<T> NewRefFrameRotmatTranspose) const
	{
		// Express quaternion in new reference frame specified by NewRefFrame - this is equivalent to rotating axis vector specified by last three elements
		CBasicQuaternion input(*this);
		input.Normalize();
		CBasicVector3<T> axis = NewRefFrameRotmatTranspose * CBasicVector3<T>(input.q1, input.q2, input.q3);
		return CBasicQuaternion(input.q0, axis.x, axis.y, axis.z);
	}

	CUDA_HOST_DEVICE CBasicQuaternion NewRefFrameNoNormalization(CBasicMatrix3<T> NewRefFrameRotmatTranspose) const
	{
		// Express quaternion in new reference frame specified by NewRefFrame - this is equivalent to rotating axis vector specified by last three elements
		CBasicQuaternion input(*this);
		CBasicVector3<T> axis = NewRefFrameRotmatTranspose * CBasicVector3<T>(input.q1, input.q2, input.q3);
		return CBasicQuaternion(input.q0, axis.x, axis.y, axis.z);
	}

	template<typename T2>
	CUDA_HOST_DEVICE operator CBasicQuaternion<T2>() const { return CBasicQuaternion<T2>(static_cast<T2>(q0), static_cast<T2>(q1), static_cast<T2>(q2), static_cast<T2>(q3)); }
};

using CQuaternion  = CBasicQuaternion<double>;
using CQuaterniond = CBasicQuaternion<double>;
using CQuaternionf = CBasicQuaternion<float>;