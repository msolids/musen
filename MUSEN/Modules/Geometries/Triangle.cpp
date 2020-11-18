/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "Triangle.h"

CTriangle::CTriangle(double _val) :
	p1{ _val }, p2{ _val }, p3{ _val }
{
}

CTriangle::CTriangle(const CVector3& _v1, const CVector3& _v2, const CVector3& _v3) :
	p1{ _v1 }, p2{ _v2 }, p3{ _v3 }
{
}

void CTriangle::Shift(const CVector3& _offs)
{
	p1 += _offs;
	p2 += _offs;
	p3 += _offs;
}

CTriangle CTriangle::Shifted(const CVector3& _offs) const
{
	return { p1 + _offs, p2 + _offs, p3 + _offs };
}

void CTriangle::Scale(const CVector3& _center, double _factor)
{
	*this = Scaled(_center, _factor);
}

void CTriangle::Scale(const CVector3& _center, const CVector3& _factors)
{
	*this = Scaled(_center, _factors);
}

CTriangle CTriangle::Scaled(const CVector3& _center, double _factor) const
{
	return Scaled(_center, CVector3{ _factor });
}

CTriangle CTriangle::Scaled(const CVector3& _center, const CVector3& _factors) const
{
	return {
		_center + EntryWiseProduct(p1 - _center, _factors),
		_center + EntryWiseProduct(p2 - _center, _factors),
		_center + EntryWiseProduct(p3 - _center, _factors)
	};
}

void CTriangle::Rotate(const CVector3& _center, const CMatrix3& _rot)
{
	p1 = _center + _rot * (p1 - _center);
	p2 = _center + _rot * (p2 - _center);
	p3 = _center + _rot * (p3 - _center);
}

CTriangle CTriangle::Rotated(const CVector3& _center, const CMatrix3& _rot) const
{
	return {
		_center + _rot * (p1 - _center),
		_center + _rot * (p2 - _center),
		_center + _rot * (p3 - _center)
	};
}

void CTriangle::InvertOrientation()
{
	std::swap(p2, p3);
}

CVector3 CTriangle::Normal() const
{
	return Normalized((p2 - p1) * (p3 - p1));
}