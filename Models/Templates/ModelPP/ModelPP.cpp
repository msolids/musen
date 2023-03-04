/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#define DLL_EXPORT
#define DYNAMIC_MODULE

#include "ModelPP.h"

extern "C" DECLDIR CAbstractDEMModel* MUSEN_CREATE_MODEL_FUN()
{
	return new CModelPP();
}

//////////////////////////////////////////////////////////////////////////
// Do not change above this line!

CModelPP::CModelPP()
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
void CModelPP::PrecalculatePPModel(double _time, double _timeStep, SParticleStruct* _particles)
{
	// TODO: Write your pre-calculation step here.
}

void CModelPP::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const
{
	// TODO: Write your model here.
}
