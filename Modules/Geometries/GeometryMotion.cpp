/* Copyright (c) 2013-2020, MUSEN Development Team.
   Copyright (c) 2026, DyssolTEC GmbH.
   All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
   See LICENSE file for license and warranty information. */

#include "GeometryMotion.h"

#include "MixedFunctions.h"
#include "MUSENStringFunctions.h"
#include "ProtoFunctions.h"

#include <cmath>

// TODO: sort time-dependent motion intervals

CGeometryMotion::EMotionType CGeometryMotion::MotionType() const
{
	return m_motionType;
}

void CGeometryMotion::SetMotionType(EMotionType _type)
{
	m_motionType = _type;
}

bool CGeometryMotion::IsForceDriven() const
{
	return m_motionType == EMotionType::FORCE_DEPENDENT || m_motionType == EMotionType::CONSTANT_FORCE || m_motionType == EMotionType::CYCLIC_FORCE;
}

void CGeometryMotion::AddInterval()
{
	switch (m_motionType)
	{
	case EMotionType::NONE:									break;
	case EMotionType::TIME_DEPENDENT:	AddTimeInterval();	break;
	case EMotionType::FORCE_DEPENDENT:	AddForceInterval();	break;
	case EMotionType::CONSTANT_FORCE:
	case EMotionType::CYCLIC_FORCE:		if (m_intervalsForce.empty()) AddForceInterval();	break;

	}
}

void CGeometryMotion::AddTimeInterval()
{
	if (m_intervalsTime.empty())
		AddTimeInterval({ 0.0, 1.0, SMotionInfo{} });
	else
		AddTimeInterval({ m_intervalsTime.back().timeEnd, m_intervalsTime.back().timeEnd + 1.0, SMotionInfo{} });
}

void CGeometryMotion::AddTimeInterval(const STimeMotionInterval& _interval)
{
	m_intervalsTime.push_back(_interval);
}

void CGeometryMotion::ChangeTimeInterval(size_t _index, const STimeMotionInterval& _interval)
{
	if (_index < m_intervalsTime.size())
		m_intervalsTime[_index] = _interval;
}

CGeometryMotion::STimeMotionInterval CGeometryMotion::GetTimeInterval(size_t _index) const
{
	if (_index < m_intervalsTime.size())
		return m_intervalsTime[_index];
	return {};
}

std::vector<CGeometryMotion::STimeMotionInterval> CGeometryMotion::GetTimeIntervals() const
{
	return m_intervalsTime;
}

void CGeometryMotion::AddForceInterval()
{
	if (m_intervalsForce.empty())
		AddForceInterval({ 1.0, SForceMotionInterval::ELimitType::MAX, SMotionInfo{} });
	else
		AddForceInterval({ m_intervalsForce.back().forceLimit, m_intervalsForce.back().limitType, SMotionInfo{} });
}

void CGeometryMotion::AddForceInterval(const SForceMotionInterval& _interval)
{
	m_intervalsForce.push_back(_interval);
}

void CGeometryMotion::ChangeForceInterval(size_t _index, const SForceMotionInterval& _interval)
{
	if (_index < m_intervalsForce.size())
		m_intervalsForce[_index] = _interval;
}

CGeometryMotion::SForceMotionInterval CGeometryMotion::GetForceInterval(size_t _index) const
{
	if (_index < m_intervalsForce.size())
		return m_intervalsForce[_index];
	return {};
}

std::vector<CGeometryMotion::SForceMotionInterval> CGeometryMotion::GetForceIntervals() const
{
	return m_intervalsForce;
}

const CVector3& CGeometryMotion::GetForceDirection() const
{
	return m_forceDirection;
}

void CGeometryMotion::SetForceDirection(const CVector3& _dir)
{
	const CVector3 normalized = Normalized(_dir);
	const bool valid = !normalized.IsZero() && std::isfinite(normalized.x) && std::isfinite(normalized.y) && std::isfinite(normalized.z);
	m_forceDirection = valid ? normalized : CVector3{ 0.0, 0.0, 1.0 };
}

double CGeometryMotion::SensedForce(const CVector3& _totalForce) const
{
	return DotProduct(_totalForce, m_forceDirection);
}

double CGeometryMotion::GetStrokeLength() const
{
	return m_strokeLength;
}

void CGeometryMotion::SetStrokeLength(double _length)
{
	m_strokeLength = _length;
}

CVector3 CGeometryMotion::StrokeResetShift() const
{
	if (m_strokeLength <= 0.0 || m_intervalsForce.empty())
		return CVector3{ 0.0 };
	// the stroke is measured along the direction in which the geometry presses, which is opposite to the force it senses there
	const CVector3 pressDirection = Normalized(m_intervalsForce.front().motion.velocity);
	if (DotProduct(m_accumulatedShift, pressDirection) < m_strokeLength)
		return CVector3{ 0.0 };
	return -1 * m_accumulatedShift;
}

void CGeometryMotion::DeleteInterval(size_t _index)
{
	switch (m_motionType)
	{
	case EMotionType::TIME_DEPENDENT:
		if (_index < m_intervalsTime.size())
			m_intervalsTime.erase(m_intervalsTime.begin() + _index);
		break;
	case EMotionType::FORCE_DEPENDENT:
	case EMotionType::CONSTANT_FORCE:
	case EMotionType::CYCLIC_FORCE:
		if (_index < m_intervalsForce.size())
			m_intervalsForce.erase(m_intervalsForce.begin() + _index);
		break;
	case EMotionType::NONE: break;
	}
}

void CGeometryMotion::MoveIntervalUp(size_t _index)
{
	switch (m_motionType)
	{
	case EMotionType::TIME_DEPENDENT:
		if (_index < m_intervalsTime.size() && _index != 0)
			std::iter_swap(m_intervalsTime.begin() + _index, m_intervalsTime.begin() + _index - 1);
		break;
	case EMotionType::FORCE_DEPENDENT:
		if (_index < m_intervalsForce.size() && _index != 0)
			std::iter_swap(m_intervalsForce.begin() + _index, m_intervalsForce.begin() + _index - 1);
		break;
	case EMotionType::CONSTANT_FORCE:
	case EMotionType::CYCLIC_FORCE:
	case EMotionType::NONE: break;
	}
}

void CGeometryMotion::MoveIntervalDown(size_t _index)
{
	switch (m_motionType)
	{
	case EMotionType::TIME_DEPENDENT:
		if (_index < m_intervalsTime.size() && _index != m_intervalsTime.size() - 1)
			std::iter_swap(m_intervalsTime.begin() + _index, m_intervalsTime.begin() + _index + 1);
		break;
	case EMotionType::FORCE_DEPENDENT:
		if (_index < m_intervalsForce.size() && _index != m_intervalsForce.size() - 1)
			std::iter_swap(m_intervalsForce.begin() + _index, m_intervalsForce.begin() + _index + 1);
		break;
	case EMotionType::CONSTANT_FORCE:
	case EMotionType::CYCLIC_FORCE:
	case EMotionType::NONE: break;
	}
}

bool CGeometryMotion::HasMotion() const
{
	return !m_intervalsTime.empty() || !m_intervalsForce.empty();
}

void CGeometryMotion::Clear()
{
	m_intervalsTime.clear();
	m_intervalsForce.clear();
	m_forceDirection.Init(0.0, 0.0, 1.0);
	m_strokeLength = 0.0;
	m_accumulatedShift.Init(0.0);
}

bool CGeometryMotion::IsValid() const
{
	switch (m_motionType)
	{
	case EMotionType::TIME_DEPENDENT:
		if (m_intervalsTime.empty())
		{
			m_errorMessage = "Time-dependent movement is selected, but time intervals are not specified.";
			return false;
		}
		break;
	case EMotionType::FORCE_DEPENDENT:
	case EMotionType::CONSTANT_FORCE:
	case EMotionType::CYCLIC_FORCE:
		if (m_intervalsForce.empty())
		{
			m_errorMessage = "Force-dependent movement is selected, but force intervals are not specified.";
			return false;
		}
		if (m_motionType == EMotionType::CYCLIC_FORCE)
		{
			if (m_strokeLength <= 0.0)
			{
				m_errorMessage = "Cyclic force movement is selected, but the stroke length is not positive.";
				return false;
			}
			if (m_intervalsForce.front().motion.velocity.IsZero())
			{
				m_errorMessage = "Cyclic force movement is selected, but the velocity is zero, so a stroke can never be completed.";
				return false;
			}
			if (!m_intervalsForce.front().motion.rotationVelocity.IsZero())
			{
				m_errorMessage = "Cyclic force movement is selected, but a rotational velocity is specified. Only the translation motion can be used.";
				return false;
			}
		}
		break;
	case EMotionType::NONE:	break;
	}

	m_errorMessage.clear();
	return true;
}

std::string CGeometryMotion::ErrorMessage() const
{
	return m_errorMessage;
}

void CGeometryMotion::UpdateMotionInfo(double _dependentValue, double _timeStep)
{
	switch (m_motionType)
	{
	case EMotionType::TIME_DEPENDENT:
	{
		bool found = false;				// is needed to accelerate updating
		const size_t iStart = m_iMotion == static_cast<size_t>(-1) ? 0 : m_iMotion;
		for (size_t i = iStart; i < m_intervalsTime.size() && !found; ++i)										// search starting from the current
			if (m_intervalsTime[i].timeBeg <= _dependentValue && _dependentValue <= m_intervalsTime[i].timeEnd)	// the value is in interval
			{
				found = true;
				if (m_iMotion != i)		// it is a new interval - update current values
				{
					m_iMotion = i;
					m_currentMotion = m_intervalsTime[i].motion;
				}
			}
		if (!found)						// such interval does not exist
			m_currentMotion.Clear();	// set current velocities to zero
		break;
	}
	case EMotionType::FORCE_DEPENDENT:
	{
		if (m_iMotion == static_cast<size_t>(-1))	// initialize
		{
			m_iMotion = 0;
			m_currentMotion = m_intervalsForce[m_iMotion].motion;
		}
		if (m_iMotion >= m_intervalsForce.size())	// no intervals defined
		{
			m_currentMotion.Clear();				// set current velocities to zero
			break;
		}

		bool updated = false;	// is needed to accelerate updating
		switch (m_intervalsForce[m_iMotion].limitType)
		{
		case SForceMotionInterval::ELimitType::MIN:	if (_dependentValue < m_intervalsForce[m_iMotion].forceLimit) { ++m_iMotion; updated = true; } break;
		case SForceMotionInterval::ELimitType::MAX:	if (_dependentValue > m_intervalsForce[m_iMotion].forceLimit) { ++m_iMotion; updated = true; } break;
		}

		if (updated)			// it is a new interval - update current values
		{
			if (m_iMotion >= m_intervalsForce.size())
				m_currentMotion.Clear();
			else
				m_currentMotion = m_intervalsForce[m_iMotion].motion;
		}
		break;
	}
	case EMotionType::CONSTANT_FORCE:
	case EMotionType::CYCLIC_FORCE:
	{
		bool bReverseDirection=false;
		m_iMotion = 0; // only first interval is used
		if (m_iMotion >= m_intervalsForce.size())	// no intervals defined
		{
			m_currentMotion.Clear();				// set current velocities to zero
			break;
		}
		m_currentMotion = m_intervalsForce[m_iMotion].motion;
		switch (m_intervalsForce[m_iMotion].limitType)
		{
		case SForceMotionInterval::ELimitType::MIN:	if (_dependentValue < m_intervalsForce[m_iMotion].forceLimit) bReverseDirection = true; break;
		case SForceMotionInterval::ELimitType::MAX:	if (_dependentValue > m_intervalsForce[m_iMotion].forceLimit) bReverseDirection = true; break;
		}
		if (bReverseDirection) // make motion in reverse direction
		{
			m_currentMotion.rotationVelocity *= -1;
			m_currentMotion.velocity *= -1;
		}
		if (m_motionType == EMotionType::CYCLIC_FORCE)	// advance the current stroke
		{
			if (!StrokeResetShift().IsZero())	// the previous stroke is finished and its reset is already applied
				m_accumulatedShift.Init(0.0);
			m_accumulatedShift += m_currentMotion.velocity * _timeStep;
		}
		break;
	}
	case EMotionType::NONE: break;
	}
}

void CGeometryMotion::ResetMotionInfo()
{
	m_iMotion = -1;
	m_currentMotion.Clear();
	m_accumulatedShift.Init(0.0);
}

CGeometryMotion::SMotionInfo CGeometryMotion::GetCurrentMotion() const
{
	return m_currentMotion;
}

CVector3 CGeometryMotion::TimeDependentShift(double _time) const
{
	if (m_motionType != EMotionType::TIME_DEPENDENT) return CVector3{ 0.0 };

	CVector3 shift{ 0 };
	for (const auto& interval : m_intervalsTime)
		if (_time > interval.timeBeg)
			shift += (std::min(interval.timeEnd, _time) - interval.timeBeg) * interval.motion.velocity;
	return shift;
}

void CGeometryMotion::LoadFromProto(const ProtoGeometryMotion& _proto)
{
	m_motionType = static_cast<EMotionType>(_proto.type());
	SetForceDirection(Proto2Val(_proto.force_direction()));
	m_strokeLength = _proto.stroke_length();
	switch (m_motionType)
	{
	case EMotionType::TIME_DEPENDENT:
		for (const auto& interval : _proto.intervals())
			AddTimeInterval({ interval.limit1(), interval.limit2(),
				SMotionInfo{Proto2Val(interval.velocity()), Proto2Val(interval.rot_velocity()), Proto2Val(interval.rot_center())} });
		break;
	case EMotionType::FORCE_DEPENDENT:
	case EMotionType::CONSTANT_FORCE:
	case EMotionType::CYCLIC_FORCE:
		for (const auto& interval : _proto.intervals())
			AddForceInterval({ interval.limit1(), static_cast<SForceMotionInterval::ELimitType>(interval.limit_type()),
				SMotionInfo{Proto2Val(interval.velocity()), Proto2Val(interval.rot_velocity()), Proto2Val(interval.rot_center())} });
		break;
	case EMotionType::NONE: break;
	}
}

void CGeometryMotion::SaveToProto(ProtoGeometryMotion& _proto) const
{
	_proto.set_version(0);
	_proto.set_type(E2I(m_motionType));
	Val2Proto(_proto.mutable_force_direction(), m_forceDirection);
	_proto.set_stroke_length(m_strokeLength);
	_proto.clear_intervals();
	switch (m_motionType)
	{
	case EMotionType::TIME_DEPENDENT:
		for (const auto& interval : m_intervalsTime)
		{
			auto* protoInterval = _proto.add_intervals();
			protoInterval->set_limit1(interval.timeBeg);
			protoInterval->set_limit2(interval.timeEnd);
			Val2Proto(protoInterval->mutable_velocity(), interval.motion.velocity);
			Val2Proto(protoInterval->mutable_rot_velocity(), interval.motion.rotationVelocity);
			Val2Proto(protoInterval->mutable_rot_center(), interval.motion.rotationCenter);
		}
		break;
	case EMotionType::FORCE_DEPENDENT:
	case EMotionType::CONSTANT_FORCE:
	case EMotionType::CYCLIC_FORCE:
		for (const auto& interval : m_intervalsForce)
		{
			auto* protoInterval = _proto.add_intervals();
			protoInterval->set_limit1(interval.forceLimit);
			protoInterval->set_limit_type(E2I(interval.limitType));
			Val2Proto(protoInterval->mutable_velocity(), interval.motion.velocity);
			Val2Proto(protoInterval->mutable_rot_velocity(), interval.motion.rotationVelocity);
			Val2Proto(protoInterval->mutable_rot_center(), interval.motion.rotationCenter);
		}
		break;
	case EMotionType::NONE: break;
	}
}

std::ostream& operator<<(std::ostream& _s, const CGeometryMotion& _obj)
{
	_s << E2I(_obj.m_motionType) << " ";
	switch (_obj.m_motionType)
	{
	case CGeometryMotion::EMotionType::TIME_DEPENDENT:
	{
		_s << _obj.GetTimeIntervals().size() << " ";
		for (const auto& interval : _obj.GetTimeIntervals())
			_s << interval << " ";
		break;
	}
	case CGeometryMotion::EMotionType::FORCE_DEPENDENT:
	case CGeometryMotion::EMotionType::CONSTANT_FORCE:
	case CGeometryMotion::EMotionType::CYCLIC_FORCE:
	{
		_s << _obj.GetForceIntervals().size() << " ";
		for (const auto& interval : _obj.GetForceIntervals())
			_s << interval << " ";
		_s << _obj.m_forceDirection << " ";
		if (_obj.m_motionType == CGeometryMotion::EMotionType::CYCLIC_FORCE)
			_s << _obj.m_strokeLength << " ";
		break;
	}
	case CGeometryMotion::EMotionType::NONE:
		_s << 0 << " ";
		break;
	}
	return _s;
}

std::istream& operator>>(std::istream& _s, CGeometryMotion& _obj)
{
	_obj.Clear();
	_obj.m_motionType = GetEnumFromStream<CGeometryMotion::EMotionType>(_s);
	const auto intervals = GetValueFromStream<size_t>(&_s);
	switch (_obj.m_motionType)
	{
	case CGeometryMotion::EMotionType::TIME_DEPENDENT:
	{
		for (size_t i = 0; i < intervals; ++i)
			_obj.AddTimeInterval(GetValueFromStream<CGeometryMotion::STimeMotionInterval>(&_s));
		break;
	}
	case CGeometryMotion::EMotionType::FORCE_DEPENDENT:
	case CGeometryMotion::EMotionType::CONSTANT_FORCE:
	case CGeometryMotion::EMotionType::CYCLIC_FORCE:
	{
		for (size_t i = 0; i < intervals; ++i)
			_obj.AddForceInterval(GetValueFromStream<CGeometryMotion::SForceMotionInterval>(&_s));
		const CVector3 direction = GetValueFromStream<CVector3>(&_s);
		if (_s)
			_obj.SetForceDirection(direction);
		else if (_s.eof()) // the record ends here, keep the default {0,0,1}
			_s.clear();
		if (_obj.m_motionType == CGeometryMotion::EMotionType::CYCLIC_FORCE)
		{
			const double stroke = GetValueFromStream<double>(&_s);
			if (_s)
				_obj.m_strokeLength = stroke;
			else if (_s.eof()) // the record ends here, keep the default 0.0
				_s.clear();
		}
		break;
	}
	case CGeometryMotion::EMotionType::NONE:
		break;
	}
	return _s;
}
