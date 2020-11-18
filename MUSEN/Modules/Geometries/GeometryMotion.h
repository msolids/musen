/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <vector>
#include "Vector3.h"

class ProtoGeometryMotion;

class CGeometryMotion
{
public:
	// Moving type of a geometry.
	enum class EMotionType : uint32_t
	{
		NONE            = 0,
		TIME_DEPENDENT  = 1,
		FORCE_DEPENDENT = 2
	};

	// Information about movement characteristics.
	struct SMotionInfo
	{
		CVector3 velocity{ 0 };			// Translational velocity.
		CVector3 rotationVelocity{ 0 };	// Rotational velocity.
		CVector3 rotationCenter{ 0 };	// Center of rotation.
		void Clear()
		{
			velocity.Init(0);
			rotationVelocity.Init(0);
			rotationCenter.Init(0);
		}
	};

	// Description of a time-dependent motion interval.
	struct STimeMotionInterval
	{
		double timeBeg{};	// Interval start time.
		double timeEnd{};	// Interval end time.
		SMotionInfo motion;	// Movement characteristics for this interval.
	};

	// Description of a force-dependent motion interval.
	struct SForceMotionInterval
	{
		enum class ELimitType { MIN, MAX };
		double forceLimit{};						// Force limit to switch to the next motion interval.
		ELimitType limitType{ ELimitType::MAX };	// Whether the force limit is a MAX or MIN value of the interval.
		SMotionInfo motion;							// Movement characteristics for this interval.
	};

private:
	mutable std::string m_errorMessage;		// Description of the last occurred error.

	EMotionType m_motionType{ EMotionType::NONE };		// Type of geometry's motion.
	std::vector<STimeMotionInterval>  m_intervalsTime;	// Time-dependent motion of this geometry. Is used if (m_motionType == TIME_DEPENDENT).
	std::vector<SForceMotionInterval> m_intervalsForce;	// Force-dependent motion of this geometry. Is used if (m_motionType == FORCE_DEPENDENT).

	size_t m_iMotion{ static_cast<size_t>(-1) };	// Index of currently acting motion characteristics.
	SMotionInfo m_currentMotion;					// Currently acting motion characteristics.

public:
	EMotionType MotionType() const;			// Returns current motion type.
	void SetMotionType(EMotionType _type);	// Sets current motion type.

	void AddInterval();		// Adds a new motion interval of the currently selected type.

	void AddTimeInterval();															// Adds new time-dependent motion interval.
	void AddTimeInterval(const STimeMotionInterval& _interval);						// Adds new time-dependent motion interval.
	void ChangeTimeInterval(size_t _index, const STimeMotionInterval& _interval);	// Changes existing time-dependent motion interval.
	STimeMotionInterval GetTimeInterval(size_t _index) const;						// Returns selected time-dependent motion interval.
	std::vector<STimeMotionInterval> GetTimeIntervals() const;						// Returns all defined time-dependent motion intervals.

	void AddForceInterval();														// Adds new force-dependent motion interval.
	void AddForceInterval(const SForceMotionInterval& _interval);					// Adds new force-dependent motion interval.
	void ChangeForceInterval(size_t _index, const SForceMotionInterval& _interval);	// Changes existing force-dependent motion interval.
	SForceMotionInterval GetForceInterval(size_t _index) const;						// Returns selected force-dependent motion interval.
	std::vector<SForceMotionInterval> GetForceIntervals() const;					// Returns all defined force-dependent motion intervals.

	void DeleteInterval(size_t _index);		// Removes motion interval of the currently selected type.
	void MoveIntervalUp(size_t _index);		// Moves motion interval of the currently selected type upwards in the list.
	void MoveIntervalDown(size_t _index);	// Moves motion interval of the currently selected type downwards in the list.

	bool HasMotion() const;				// Returns true if any motion interval is defied.
	void Clear();						// Clears all specified motion information of all types.
	bool IsValid() const;				// Checks whether the selected settings are valid.
	std::string ErrorMessage() const;	// Returns description of the last occurred error.

	void UpdateMotionInfo(double _dependentValue);	// Updates current motion characteristics according to the current time or force.
	void ResetMotionInfo();							// Resets current motion characteristics to the initial state.
	SMotionInfo GetCurrentMotion() const;			// Returns current motion characteristics.

	CVector3 TimeDependentShift(double _time) const;	// Returns time-dependent translational shift.

	void LoadFromProto(const ProtoGeometryMotion& _proto);	// Load from protobuf message.
	void SaveToProto(ProtoGeometryMotion& _proto) const;	// Save to protobuf message.
};


