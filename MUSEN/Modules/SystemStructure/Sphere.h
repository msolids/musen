/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "PhysicalObject.h"

class CSphere : public CPhysicalObject
{
	// Is needed to convert data for saving/loading.
	union USaveHelper
	{
		#pragma pack(push, 1)	// Is important for cross-compiler compatibility.
		struct SSaveData		// A structure with data for saving/loading. The order matters!
		{
			double radius;
			double contactRadius;
		} data;
		#pragma pack(pop)
		uint8_t binary[sizeof(SSaveData)];	// A binary form of the data.
		USaveHelper(const SSaveData& _data) : data{ _data } {}														// Initialize data.
		explicit USaveHelper(const std::vector<uint8_t>& _data) { std::copy(_data.begin(), _data.end(), binary); }	// Initialize binary.
	};

	double m_dRadius;			// Radius in [m].
	double m_dContactRadius;	// Contact radius of particle, used in some models. If it is not set explicitly, it is equal to radius.
	double m_dRadiusSqrt;		// The square root of radius.
	double m_dInertiaMoment;

public:
	CSphere(unsigned _id, CDemStorage* _storage);

	std::string GetObjectGeometryText() const override;
	void SetObjectGeometryText(std::stringstream& _inputStream) override;

	std::vector<uint8_t> GetObjectGeometryBin() const override;
	void SetObjectGeometryBin(const std::vector<uint8_t>& _data) override;

	inline double GetRadius() const { return m_dRadius; }
	inline double GetContactRadius() const { return m_dContactRadius; }
	inline double GetSqrtRadius() const { return m_dRadiusSqrt; }
	inline double GetInertiaMoment() const { return m_dInertiaMoment; }
	inline double GetVolume() const { return 4 / 3.0*PI*m_dRadius*m_dRadius*m_dRadius; }

	void SetRadius(const double& _radius);
	void SetContactRadius(const double& _radius);

	void UpdateCompoundProperties(const CCompound* _pCompound) override;

private:
	void UpdatePrecalculatedValues() override; // Calculates all constant terms which are time independent.
};

