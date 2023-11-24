/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#define DLL_EXPORT
#define DYNAMIC_MODULE

#include "ModelSB.h"

extern "C" DECLDIR CAbstractDEMModel* MUSEN_CREATE_MODEL_FUN()
{
	return new CModelSB();
}

//////////////////////////////////////////////////////////////////////////
// Do not change above this line!

CModelSB::CModelSB()
{
	/// TODO: Set the name of your model
	m_name = "Model name";
	/// TODO: Set alphanumeric identifier of your model, which must be unique among all existing models
	m_uniqueKey = "UNIQUE_IDENTIFIER";

	/// TODO: Setup parameters of your model here if any
	AddParameter("PARAM_NAME_WITHOUT_SPACES", "User friendly parameter description", 13);

	// TODO: Set to 'true' if GPU version is implemented
	m_hasGPUSupport = false;
}

//////////////////////////////////////////////////////////////////////////
/// CPU Implementation
//////////////////////////////////////////////////////////////////////////

/// TODO: This function can be removed if not used.
void CModelSB::PrecalculateSB(double _time, double _timeStep, SParticleStruct* _particles, SSolidBondStruct* _bonds)
{
	// TODO: Write your pre-calculation step here.
}

void CModelSB::CalculateSB(double _time, double _timeStep, size_t _iLeft, size_t _iRight, size_t _iBond, SSolidBondStruct& _bonds, unsigned* _pBrokenBondsNum) const
{
	// TODO: Write your model here.
}

void CModelSB::ConsolidatePart(double _time, double _timeStep, size_t _iBond, size_t _iPart, SParticleStruct& _particles) const
{
	// TODO: Write your consolidation step here.
	// This example is for calculation of forces and moments.
	if (Bonds().LeftID(_iBond) == _iPart)
	{
		_particles.Force(_iPart) += Bonds().TotalForce(_iBond);
		_particles.Moment(_iPart) += Bonds().NormalMoment(_iBond) + Bonds().TangentialMoment(_iBond) - Bonds().UnsymMoment(_iBond);
	}
	else if (Bonds().RightID(_iBond) == _iPart)
	{
		_particles.Force(_iPart) -= Bonds().TotalForce(_iBond);
		_particles.Moment(_iPart) -= Bonds().NormalMoment(_iBond) + Bonds().TangentialMoment(_iBond) + Bonds().UnsymMoment(_iBond);
	}
}
