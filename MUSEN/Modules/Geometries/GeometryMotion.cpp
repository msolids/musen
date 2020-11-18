/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GeometryMotion.h"
#include "MixedFunctions.h"
#include "ProtoFunctions.h"

// TODO: sort time-dependent motion intervals

CGeometryMotion::EMotionType CGeometryMotion::MotionType() const
{
	return m_motionType;
}

void CGeometryMotion::SetMotionType(EMotionType _type)
{
	m_motionType = _type;
}

void CGeometryMotion::AddInterval()
{
	switch (m_motionType)
	{
	case EMotionType::NONE:									break;
	case EMotionType::TIME_DEPENDENT:	AddTimeInterval();	break;
	case EMotionType::FORCE_DEPENDENT:	AddForceInterval();	break;
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

void CGeometryMotion::DeleteInterval(size_t _index)
{
	switch (m_motionType)
	{
	case EMotionType::TIME_DEPENDENT:
		if (_index < m_intervalsTime.size())
			m_intervalsTime.erase(m_intervalsTime.begin() + _index);
		break;
	case EMotionType::FORCE_DEPENDENT:
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
		if (m_intervalsForce.empty())
		{
			m_errorMessage = "Force-dependent movement is selected, but force intervals are not specified.";
			return false;
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

void CGeometryMotion::UpdateMotionInfo(double _dependentValue)
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
			m_currentMotion = m_intervalsForce[m_iMotion].motion;

		break;
	}
	case EMotionType::NONE: break;
	}
}

void CGeometryMotion::ResetMotionInfo()
{
	m_iMotion = -1;
	m_currentMotion.Clear();
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
	switch (m_motionType)
	{
	case EMotionType::TIME_DEPENDENT:
		for (const auto& interval : _proto.intervals())
			AddTimeInterval({ interval.limit1(), interval.limit2(),
				SMotionInfo{Proto2Val(interval.velocity()), Proto2Val(interval.rot_velocity()), Proto2Val(interval.rot_center())} });
		break;
	case EMotionType::FORCE_DEPENDENT:
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
