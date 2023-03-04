#include "ModelPPSinteringTemperature.h"
#include "MUSENDefinitions.h"

CModelPPSinteringTemperature::CModelPPSinteringTemperature()
{
	m_name                         = "Sintering considering temperature";
	m_uniqueKey                    = "A41AB40D3A074FE9AAE0B0ADB27XBFBA";
	m_requieredVariables.bThermals = true;
	m_hasGPUSupport                = true;

	/* 0*/ AddParameter("GRAINBOUND_DIFFUSION", "Grain-boundary thickness times diffusion coefficient [??]", 1.3e-8  );
	/* 1*/ AddParameter("VISCOUS_PARAMETER"   , "Viscous parameter for tangential force (eta) [-]"         , 0.01    );
	/* 2*/ AddParameter("ATOMIC_VOLUME"       , "Atomic volume [m3]"                                       , 8.47e-30);
	/* 3*/ AddParameter("ACTIVATION_ENERGY"   , "Activation energy [J/mol]"                                , 475000  );
	/* 4*/ AddParameter("MINIMAL_TEMPERATURE" , "Minimal temperature [K]"                                  , 1000    ); // if temperature less than this threshold no sintering takes place
	/* 5*/ AddParameter("MAX_OVERLAP_COEFF"   , "Max overlap coefficient for stop criterion [-]"           , 0.2784  );
}

void CModelPPSinteringTemperature::CalculatePPForce(double _time, double _timeStep, size_t _iSrc, size_t _iDst, const SInteractProps& _interactProp, SCollision* _collision) const
{
	// model parameters
	const double grainBoundTimesDiff = m_parameters[0].value;
	const double viscousParameter    = m_parameters[1].value;
	const double atomicVolume        = m_parameters[2].value;
	const double activationEnergy    = m_parameters[3].value;
	const double minTemperature      = m_parameters[4].value;
	const double maxOverlapCoeff     = m_parameters[5].value;

	const CVector3 normVector = _collision->vContactVector.Normalized();
	const double temperatureK   = (Particles().Temperature(_iSrc) + Particles().Temperature(_iDst)) / 2;
	const double maxNormOverlap = maxOverlapCoeff * 2 * _collision->dEquivRadius;

	// normal and tangential relative velocity
	// TODO: switch and remove -1?
	const CVector3 relVel        = Particles().Vel(_iSrc) - Particles().Vel(_iDst);
	const double   normRelVelLen = DotProduct(normVector, relVel);
	const CVector3 normRelVel    = normRelVelLen * normVector;
	const CVector3 tangRelVel    = relVel - normRelVel;

	// set initial overlap as equilibrium value
	if (_time == 0.0)
	{
		_collision->vTangOverlap.Init(0);
		_collision->dInitNormalOverlap = _collision->dNormalOverlap;
	}

	CVector3 totalForce;
	// sintering force if temperature is higher than threshold
	if (temperatureK >= minTemperature && _collision->dNormalOverlap < maxNormOverlap)
	{
		// reset normal and tangential overlaps
		_collision->vTangOverlap.Init(0);
		_collision->dInitNormalOverlap = _collision->dNormalOverlap;

		const double diffusionParameter = atomicVolume / (BOLTZMANN_CONSTANT * temperatureK) * grainBoundTimesDiff * std::exp(-activationEnergy / (GAS_CONSTANT * temperatureK));

		// Bouvard and McMeeking's model
		const double squaredContactRadius = 4 * _collision->dEquivRadius * _collision->dNormalOverlap;

		// forces
		const CVector3 sinteringForce = normVector * 1.125 * PI * 2 * _collision->dEquivRadius * _interactProp.dEquivSurfaceEnergy;
		const CVector3 viscousForce   = normRelVel * (-PI * std::pow(squaredContactRadius, 2) / 8 / diffusionParameter);
		const CVector3 tangForce      = tangRelVel * (-viscousParameter * PI * squaredContactRadius * std::pow(2 * _collision->dEquivRadius, 2) / 8 / diffusionParameter);

		// total force
		totalForce = sinteringForce + viscousForce + tangForce;
	}
	// no sintering force if temperature is lower than threshold - Hertz-Mindlin
	else if (temperatureK <= minTemperature)
	{
		// radius of the contact area
		const double contactAreaRadius = std::sqrt(_collision->dEquivRadius * _collision->dNormalOverlap);

		// normal force with damping
		const double Kn = 2 * _interactProp.dEquivYoungModulus * contactAreaRadius;
		const double deltaNormOverlap = _collision->dNormalOverlap >= maxNormOverlap ? maxNormOverlap : _collision->dInitNormalOverlap;
		const double normContactForceLen = -(_collision->dNormalOverlap - deltaNormOverlap) * Kn * 2. / 3.;
		// TODO: why -1?
		const double normDampingForceLen = -_2_SQRT_5_6 * _interactProp.dAlpha * -1 * normRelVelLen * std::sqrt(Kn * _collision->dEquivMass);
		const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);

		// rotate old tangential overlap
		CVector3 tangOverlapRot = _collision->vTangOverlap - normVector * DotProduct(normVector, _collision->vTangOverlap);
		if (tangOverlapRot.IsSignificant())
			tangOverlapRot *= _collision->vTangOverlap.Length() / tangOverlapRot.Length();
		// calculate new tangential overlap
		// TODO: why -1?
		const CVector3 tangOverlap = tangOverlapRot + -1 * tangRelVel * _timeStep;

		// tangential force with damping
		const double Kt = 8 * _interactProp.dEquivShearModulus * contactAreaRadius;
		const CVector3 tangShearForce = tangOverlap * Kt;
		// TODO: why -1?
		const CVector3 tangDampingForce = -1 * tangRelVel * (-_2_SQRT_5_6 * _interactProp.dAlpha * std::sqrt(Kt * _collision->dEquivMass));
		const CVector3 tangForce = tangShearForce + tangDampingForce;

		// total force
		totalForce = normForce + tangForce;

		// store results in collision
		_collision->vTangOverlap = tangOverlap;
	}
	else
	{
		// reset normal and tangential overlaps
		_collision->vTangOverlap.Init(0);
		_collision->dInitNormalOverlap = _collision->dNormalOverlap;

		// radius of the contact area
		const double contactAreaRadius = std::sqrt(_collision->dEquivRadius * _collision->dNormalOverlap);

		// normal force with damping
		const double Kn = 2 * _interactProp.dEquivYoungModulus * contactAreaRadius;
		const double normContactForceLen = -(_collision->dNormalOverlap - maxNormOverlap) * Kn * 2. / 3.;
		// TODO: why -1?
		const double normDampingForceLen = -_2_SQRT_5_6 * _interactProp.dAlpha * -1 * normRelVelLen * std::sqrt(Kn * _collision->dEquivMass);
		const CVector3 normForce = normVector * (normContactForceLen + normDampingForceLen);

		// total force
		totalForce = normForce;
	}

	// store results in collision
	_collision->vTotalForce = totalForce;
}
