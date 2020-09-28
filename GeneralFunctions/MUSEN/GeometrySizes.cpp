/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GeometrySizes.h"

CGeometrySizes::CGeometrySizes(const CGeometrySizes& _other) :
	m_width      { _other.m_width      },
	m_depth      { _other.m_depth      },
	m_height     { _other.m_height     },
	m_radius     { _other.m_radius     },
	m_innerRadius{ _other.m_innerRadius}
{
}

CGeometrySizes::CGeometrySizes(CGeometrySizes&& _other) noexcept :
	m_width      { std::exchange(_other.m_width      , {}) },
	m_depth      { std::exchange(_other.m_depth      , {}) },
	m_height     { std::exchange(_other.m_height     , {}) },
	m_radius     { std::exchange(_other.m_radius     , {}) },
	m_innerRadius{ std::exchange(_other.m_innerRadius, {}) }
{
	_other.m_sizes.clear();
}

CGeometrySizes& CGeometrySizes::operator=(const CGeometrySizes& _other)
{
	for (size_t i = 0; i < _other.m_sizes.size(); ++i)
		*m_sizes[i] = *_other.m_sizes[i];
	return *this;
}

CGeometrySizes& CGeometrySizes::operator=(CGeometrySizes&& _other) noexcept
{
	for (size_t i = 0; i < _other.m_sizes.size(); ++i)
		*m_sizes[i] = std::exchange(*_other.m_sizes[i], {});
	return *this;
}

void CGeometrySizes::Scale(double _factor)
{
	for (auto& s : m_sizes)
		*s *= _factor;
}

void CGeometrySizes::ResetToDefaults(double _reference/* = 0.0*/)
{
	if (_reference == 0.0)
		_reference = 0.1;
	m_width = _reference / 4;
	m_depth = _reference / 5;
	m_height = _reference / 6;
	m_radius = _reference / 4;
	m_innerRadius = _reference / 5;
}

CGeometrySizes CGeometrySizes::Defaults(double _reference)
{
	CGeometrySizes sizes;
	sizes.ResetToDefaults(_reference);
	return sizes;
}

std::vector<double> CGeometrySizes::RelevantSizes(EVolumeShape _shape) const
{
	switch (_shape)
	{
	case EVolumeShape::VOLUME_SPHERE:        return { m_radius };
	case EVolumeShape::VOLUME_BOX:           return { m_width, m_depth, m_height };
	case EVolumeShape::VOLUME_CYLINDER:      return { m_radius, m_height };
	case EVolumeShape::VOLUME_HOLLOW_SPHERE: return { m_radius, m_innerRadius };
	case EVolumeShape::VOLUME_STL:           return {};
	}

	return {};
}

void CGeometrySizes::SetRelevantSizes(const std::vector<double>& _sizes, EVolumeShape _shape)
{
	switch (_shape)
	{
	case EVolumeShape::VOLUME_SPHERE:		 { if (_sizes.size() >= 1) { m_radius = _sizes[0];                                                  } break; }
	case EVolumeShape::VOLUME_BOX:			 { if (_sizes.size() >= 3) { m_width  = _sizes[0]; m_depth       = _sizes[1]; m_height = _sizes[2]; } break; }
	case EVolumeShape::VOLUME_CYLINDER:		 { if (_sizes.size() >= 2) { m_radius = _sizes[0]; m_height      = _sizes[1];                       } break; }
	case EVolumeShape::VOLUME_HOLLOW_SPHERE: { if (_sizes.size() >= 2) { m_radius = _sizes[0]; m_innerRadius = _sizes[1];                       } break; }
	case EVolumeShape::VOLUME_STL: break;
	}
}

bool operator==(const CGeometrySizes& _lhs, const CGeometrySizes& _rhs)
{
	for (size_t i = 0; i < _lhs.m_sizes.size(); ++i)
		if (*_lhs.m_sizes[i] != *_rhs.m_sizes[i])
			return false;
	return true;
}

bool operator!=(const CGeometrySizes& _lhs, const CGeometrySizes& _rhs)
{
	return !(_lhs == _rhs);
}
