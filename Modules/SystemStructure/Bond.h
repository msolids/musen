/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "PhysicalObject.h"

class CBond : public CPhysicalObject
{
public:
	unsigned int m_nLeftObjectID;	// The id of the first connected body.
	unsigned int m_nRightObjectID;	// The id of the second connected body.

	CBond(unsigned _id, CDemStorage* _storage);

	std::string GetObjectGeometryText() const override;
	void SetObjectGeometryText(std::stringstream& _inputStream) override;

	std::vector<uint8_t> GetObjectGeometryBin() const override;
	void SetObjectGeometryBin(const std::vector<uint8_t>& _data) override;

	/// Clones the data from another object of the same type.
	void CloneData(const CPhysicalObject& _other) override;

	inline double GetDiameter() const { return m_dDiameter; }
	inline double GetInitLength() const { return m_dInitialLength; }
	inline double GetViscosity() const { return m_dViscosity; }
	inline void SetDiameter(const double _diameter) { m_dDiameter = _diameter; UpdatePrecalculatedValues(); }
	inline void SetInitialLength(const double _dLength) { m_dInitialLength = _dLength; }

protected:
	double m_dDiameter; // the diameter of the bond [m]
	double m_dViscosity;
	double m_dInitialLength;		// Initial length of the bond [m].

private:
	// Is needed to convert data for saving/loading.
	union USaveHelper
	{
		#pragma pack(push, 1)	// Is important for cross-compiler compatibility.
		struct SSaveData		// A structure with data for saving/loading. The order matters!
		{
			uint32_t idLeft;
			uint32_t idRight;
			double diameter;
			double initialLength;
		} data;
		#pragma pack(pop)
		uint8_t binary[sizeof(SSaveData)];	// A binary form of the data.
		USaveHelper(const SSaveData& _data) : data{ _data } {}														// Initialize data.
		explicit USaveHelper(const std::vector<uint8_t>& _data) { std::copy(_data.begin(), _data.end(), binary); }	// Initialize binary.
	};
};