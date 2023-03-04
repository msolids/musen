/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "CPUSimulator.h"

CCPUSimulator::CCPUSimulator(const CBaseSimulator& _other) :
	CBaseSimulator{ _other }
{
}

void CCPUSimulator::SetSystemStructure(CSystemStructure* _pSystemStructure)
{
	m_collisionsCalculator.SetSystemStructure(_pSystemStructure);
	m_collisionsAnalyzer.SetSystemStructure(_pSystemStructure);
	m_pSystemStructure = _pSystemStructure;
}

void CCPUSimulator::EnableCollisionsAnalysis(bool _bEnable)
{
	if (m_status != ERunningStatus::IDLE) return;

	m_analyzeCollisions = _bEnable;
	m_collisionsCalculator.EnableCollisionsAnalysis(m_analyzeCollisions);
}

bool CCPUSimulator::IsCollisionsAnalysisEnabled() const
{
	return m_analyzeCollisions;
}

void CCPUSimulator::Initialize()
{
	CBaseSimulator::Initialize();

	// delete collisions data
	m_collisionsCalculator.ClearCollMatrixes();
	m_collisionsCalculator.ClearFinishedCollisionMatrixes();
	if (m_analyzeCollisions)
		m_collisionsAnalyzer.ResetAndClear();

	// store particle coordinates
	m_scene.SaveVerletCoords();
}

void CCPUSimulator::InitializeModels()
{
	CBaseSimulator::InitializeModels();

	for (auto& model : m_models)
		model->Initialize(
			m_scene.GetPointerToParticles().get(),
			m_scene.GetPointerToWalls().get(),
			m_scene.GetPointerToSolidBonds().get(),
			m_scene.GetPointerToLiquidBonds().get(),
			m_scene.GetPointerToInteractProperties().get());
}

void CCPUSimulator::FinalizeSimulation()
{
	CBaseSimulator::FinalizeSimulation();

	if (m_analyzeCollisions)
		m_collisionsCalculator.SaveRestCollisions();
}

void CCPUSimulator::PreCalculationStep()
{
	CBaseSimulator::PreCalculationStep();

	if (m_analyzeCollisions)
		m_collisionsCalculator.SaveCollisions();
}

void CCPUSimulator::UpdateCollisionsStep(double _dTimeStep)
{
	m_scene.ClearAllForcesAndMoments();
	CheckParticlesInDomain();

	// if there is no contact model, then there is no necessity to calculate contacts
	if (m_pPPModel || m_pPWModel)
	{
		UpdateVerletLists(_dTimeStep); // between PP and PW
		m_collisionsCalculator.UpdateCollisionMatrixes(_dTimeStep, m_currentTime);
	}

	if (m_pPPHTModel)
		m_scene.ClearHeatFluxes();
}

void CCPUSimulator::CalculateForcesStep(double _dTimeStep)
{
	if (m_pEFModel)   CalculateForcesEF(_dTimeStep);
	if (m_pPPModel)   CalculateForcesPP(_dTimeStep);
	if (m_pPWModel)   CalculateForcesPW(_dTimeStep);
	if (m_pSBModel)   CalculateForcesSB(_dTimeStep);
	if (m_pLBModel)   CalculateForcesLB(_dTimeStep);
	if (m_pPPHTModel) CalculateHeatTransferPP(_dTimeStep);
	m_collisionsCalculator.CalculateTotalStatisticsInfo();
}

void CCPUSimulator::CalculateForcesPP(double _dTimeStep)
{
	m_pPPModel->Precalculate(m_currentTime, _dTimeStep);

	SParticleStruct& particles = m_scene.GetRefToParticles();
	for (auto& vCollisions : m_tempCollPPArray)
		for (auto& coll : vCollisions)
			coll.clear();

	ParallelFor(m_collisionsCalculator.m_vCollMatrixPP.size(), [&](size_t i)
	{
		const size_t nIndex = i%m_nThreads;
		for (auto& pColl : m_collisionsCalculator.m_vCollMatrixPP[i])
		{
			m_pPPModel->Calculate(m_currentTime, _dTimeStep, pColl);
			particles.Force(i) += pColl->vTotalForce;
			particles.Moment(i) += pColl->vResultMoment1;

			m_tempCollPPArray[nIndex][pColl->nDstID % m_nThreads].push_back(pColl);
		}
	});

	ParallelFor([&](size_t i)
	{
		for (size_t j = 0; j < m_nThreads; ++j)
			for (auto pColl : m_tempCollPPArray[j][i])
			{
				particles.Force(pColl->nDstID) -= pColl->vTotalForce;
				particles.Moment(pColl->nDstID) += pColl->vResultMoment2;
			}
	});
}

void CCPUSimulator::CalculateForcesPW(double _dTimeStep)
{
	m_pPWModel->Precalculate(m_currentTime, _dTimeStep);

	SParticleStruct& particles = m_scene.GetRefToParticles();
	SWallStruct& walls = m_scene.GetRefToWalls();

	for (auto& vCollisions : m_tempCollPWArray)
		for (auto& coll : vCollisions)
			coll.clear();

	ParallelFor(m_collisionsCalculator.m_vCollMatrixPW.size(), [&](size_t i)
	{
		const size_t nIndex = i%m_nThreads;
		for (auto& pColl : m_collisionsCalculator.m_vCollMatrixPW[i])
		{
			m_pPWModel->Calculate(m_currentTime, _dTimeStep, pColl);
			particles.Force(i) += pColl->vTotalForce;
			particles.Moment(i) += pColl->vResultMoment1;
			m_tempCollPWArray[nIndex][pColl->nSrcID % m_nThreads].push_back(pColl);
		}
	});

	ParallelFor([&](size_t i)
	{
		for (size_t j = 0; j < m_nThreads; ++j)
			for (auto& pColl : m_tempCollPWArray[j][i])
				walls.Force(pColl->nSrcID) -= pColl->vTotalForce;
	});
}

void CCPUSimulator::CalculateForcesSB(double _dTimeStep)
{
	if (m_scene.GetBondsNumber() == 0) return;

	m_pSBModel->Precalculate(m_currentTime, _dTimeStep);

	SParticleStruct& particles = m_scene.GetRefToParticles();
	SSolidBondStruct& bonds = m_scene.GetRefToSolidBonds();
	std::vector<std::vector<unsigned>>& partToSolidBonds = *m_scene.GetPointerToPartToSolidBonds();
	std::vector<unsigned> vBrokenBonds(m_nThreads, 0);

	ParallelFor(m_scene.GetBondsNumber(), [&](size_t i)
	{
		if (bonds.Active(i))
			m_pSBModel->Calculate(m_currentTime, _dTimeStep, i, bonds, &vBrokenBonds[i% m_nThreads]);
	});
	m_nBrokenBonds += VectorSum(vBrokenBonds);

	ParallelFor(partToSolidBonds.size(), [&](size_t i)
	{
		for (size_t j = 0; j < partToSolidBonds[i].size(); j++)
		{
			unsigned nBond = partToSolidBonds[i][j];
			if (!bonds.Active(nBond)) continue;
			if (bonds.LeftID(nBond) == i)
			{
				particles.Force(i) += bonds.TotalForce(nBond);
				particles.Moment(i) += bonds.NormalMoment(nBond) + bonds.TangentialMoment(nBond) - bonds.UnsymMoment(nBond);
			}
			else if (bonds.RightID(nBond) == i)
			{
				particles.Force(i) -= bonds.TotalForce(nBond);
				particles.Moment(i) -= bonds.NormalMoment(nBond) + bonds.TangentialMoment(nBond) + bonds.UnsymMoment(nBond);
			}
		}
	});
}

void CCPUSimulator::CalculateForcesLB(double _dTimeStep)
{
	if (m_scene.GetLiquidBondsNumber() == 0) return;

	m_pLBModel->Precalculate(m_currentTime, _dTimeStep);

	SLiquidBondStruct& bonds = m_scene.GetRefToLiquidBonds();
	SParticleStruct& particles = m_scene.GetRefToParticles();
	std::vector<unsigned> vBrokenBonds(m_nThreads, 0);

	ParallelFor(m_scene.GetLiquidBondsNumber(), [&](size_t i)
	{
		if (bonds.Active(i))
			m_pLBModel->Calculate(m_currentTime, _dTimeStep,i, bonds, &vBrokenBonds[i%m_nThreads]);
	});
	m_nBrokenLiquidBonds += VectorSum(vBrokenBonds);

	for (size_t i = 0; i < m_scene.GetLiquidBondsNumber(); ++i)
		if (bonds.Active(i))
		{
			particles.Force(bonds.LeftID(i)) += bonds.NormalForce(i) + bonds.TangentialForce(i);
			particles.Force(bonds.RightID(i)) -= bonds.NormalForce(i) + bonds.TangentialForce(i);
			particles.Moment(bonds.LeftID(i)) -= bonds.UnsymMoment(i);
			particles.Moment(bonds.RightID(i)) -= bonds.UnsymMoment(i);
		}
}

void CCPUSimulator::CalculateForcesEF(double _dTimeStep)
{
	m_pEFModel->Precalculate(m_currentTime, _dTimeStep);

	SParticleStruct& particles = m_scene.GetRefToParticles();

	ParallelFor(particles.Size(), [&](size_t i)
	{
		if (particles.Active(i))
			m_pEFModel->Calculate(m_currentTime, _dTimeStep,i, particles);
	});
}

void CCPUSimulator::CalculateHeatTransferPP(double _dTimeStep)
{
	SParticleStruct& particles = m_scene.GetRefToParticles();

	m_pPPHTModel->Precalculate(m_currentTime, _dTimeStep);

	for (auto& vCollisions : m_tempCollPPArray)
		for (auto& coll : vCollisions)
			coll.clear();

	ParallelFor(m_collisionsCalculator.m_vCollMatrixPP.size(), [&](size_t i)
	{
		const size_t nIndex = i % m_nThreads;
		for (auto& pColl : m_collisionsCalculator.m_vCollMatrixPP[i])
		{
			m_pPPHTModel->Calculate(m_currentTime, _dTimeStep, pColl);
			particles.HeatFlux(i) += pColl->dHeatFlux;
			m_tempCollPPArray[nIndex][pColl->nDstID % m_nThreads].push_back(pColl);
		}
	});

	ParallelFor([&](size_t i)
	{
		for (size_t j = 0; j < m_nThreads; ++j)
			for (auto* pColl : m_tempCollPPArray[j][i])
				particles.HeatFlux(pColl->nDstID) -= pColl->dHeatFlux;
	});
}

void CCPUSimulator::MoveParticles(bool _bPredictionStep)
{
	SParticleStruct& particles = m_scene.GetRefToParticles();

	// apply external acceleration
	ParallelFor(m_scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		if (!particles.Active(i)) return;
		particles.Force(i) += m_externalAcceleration * particles.Mass(i);
	});

	// change current simulation time step
	if (m_variableTimeStep)
	{
		double maxStep = std::numeric_limits<double>::max();
		for (size_t i = 0; i < particles.Size(); i++)
		{
			if (!particles.Force(i).IsZero())
				maxStep = std::min(maxStep, std::pow(particles.Mass(i), 2.) / particles.Force(i).SquaredLength());
		}
		maxStep = std::sqrt(std::sqrt(maxStep) * m_partMoveLimit);
		if (m_currSimulationStep > maxStep)
			m_currSimulationStep = maxStep;
		else if (m_currSimulationStep < m_initSimulationStep)
			m_currSimulationStep = std::min(m_currSimulationStep * m_timeStepFactor, m_initSimulationStep);
	}
	const double dTimeStep = !_bPredictionStep ? m_currSimulationStep : m_currSimulationStep / 2.;

	// move particles
	ParallelFor(m_scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		if (!particles.Active(i)) return;

		particles.Vel(i) += particles.Force(i) / particles.Mass(i) * dTimeStep;

		if (m_considerAnisotropy)
		{
			const CMatrix3 rotMatrix = particles.Quaternion(i).ToRotmat();
			CVector3 vTemp = (rotMatrix.Transpose()*particles.Moment(i));
			vTemp.x = vTemp.x / particles.InertiaMoment(i);
			vTemp.y = vTemp.y / particles.InertiaMoment(i);
			vTemp.z = vTemp.z / particles.InertiaMoment(i);
			particles.AnglVel(i) += rotMatrix * vTemp * dTimeStep;
			if (!_bPredictionStep)
			{
				const CVector3& angVel = particles.AnglVel(i);
				CQuaternion& quart = particles.Quaternion(i);
				CQuaternion quaternTemp;
				quaternTemp.q0 = 0.5*dTimeStep*(-quart.q1*angVel.x - quart.q2*angVel.y - quart.q3*angVel.z);
				quaternTemp.q1 = 0.5*dTimeStep*(quart.q0*angVel.x + quart.q3*angVel.y - quart.q2*angVel.z);
				quaternTemp.q2 = 0.5*dTimeStep*(-quart.q3*angVel.x + quart.q0*angVel.y + quart.q1*angVel.z);
				quaternTemp.q3 = 0.5*dTimeStep*(quart.q2*angVel.x - quart.q1*angVel.y + quart.q0*angVel.z);
				quart += quaternTemp;
				quart.Normalize();
			}
		}
		else
			particles.AnglVel(i) += particles.Moment(i) / particles.InertiaMoment(i) * dTimeStep;
		if (!_bPredictionStep)
			particles.Coord(i) += particles.Vel(i)*dTimeStep;
	});

	MoveParticlesOverPBC(); // move virtual particles and check boundaries
}

void CCPUSimulator::MoveWalls(double _dTimeStep)
{
	SWallStruct& pWalls = m_scene.GetRefToWalls();
	m_wallsVelocityChanged = false;
	for (size_t i = 0; i < m_pSystemStructure->GeometriesNumber(); i++)
	{
		CRealGeometry* pGeom = m_pSystemStructure->Geometry(i);
		const auto& planes = pGeom->Planes();
		if (planes.empty()) continue;

		if ((pGeom->Motion()->MotionType() == CGeometryMotion::EMotionType::FORCE_DEPENDENT) ||  // force
			(pGeom->Motion()->MotionType() == CGeometryMotion::EMotionType::CONSTANT_FORCE))
		{
			double dTotalForceZ = 0;
			for (const auto& plane : planes)
			{
				size_t nIndex = m_scene.m_vNewIndexes[plane];
				dTotalForceZ += (pWalls.Force(nIndex).z);
			}
			pGeom->UpdateMotionInfo(dTotalForceZ);
		}
		else
			pGeom->UpdateMotionInfo(m_currentTime); // time
		CVector3 vVel = pGeom->GetCurrentVelocity();
		CVector3 vRotVel = pGeom->GetCurrentRotVelocity();
		CVector3 vRotCenter;
		if (pGeom->RotateAroundCenter())
		{
			vRotCenter.Init(0);
			for (const auto& plane : planes)
			{
				size_t nIndex = m_scene.m_vNewIndexes[plane];
				vRotCenter += (pWalls.Vert1(nIndex) + pWalls.Vert2(nIndex) + pWalls.Vert3(nIndex)) / (3.0*planes.size());
			}
		}
		else
			vRotCenter = pGeom->GetCurrentRotCenter();

		if (m_currentTime == 0)
			m_wallsVelocityChanged = true;
		else
		{
			if (!(pGeom->GetCurrentVelocity() - pGeom->GetCurrentVelocity()).IsZero())
				m_wallsVelocityChanged = true;
			else if (!(pGeom->GetCurrentRotVelocity() - pGeom->GetCurrentRotVelocity()).IsZero())
				m_wallsVelocityChanged = true;
			else if (!(pGeom->GetCurrentRotCenter() - pGeom->GetCurrentRotCenter()).IsZero())
				m_wallsVelocityChanged = true;
		}

		if (!pGeom->FreeMotion().IsZero() && pGeom->Mass())// solve newtons motion for wall
		{
			CVector3 vTotalForce = m_externalAcceleration * pGeom->Mass();
			CVector3 vTotalAverVel(0);
			// calculate total force acting on wall
			for (const auto& plane : planes)
			{
				vTotalForce += pWalls.Force(m_scene.m_vNewIndexes[plane]);
				vTotalAverVel += pWalls.Vel(m_scene.m_vNewIndexes[plane]) / static_cast<double>(planes.size());
			}
			if (pGeom->FreeMotion().x)
				vVel.x = vTotalAverVel.x + _dTimeStep * vTotalForce.x / pGeom->Mass();
			if (pGeom->FreeMotion().y)
				vVel.y = vTotalAverVel.y + _dTimeStep * vTotalForce.y / pGeom->Mass();
			if (pGeom->FreeMotion().z)
				vVel.z = vTotalAverVel.z + _dTimeStep * vTotalForce.z / pGeom->Mass();
			m_wallsVelocityChanged = true;
		}

		if (vVel.IsZero() && vRotVel.IsZero()) continue;
		CMatrix3 RotMatrix;
		if (!vRotVel.IsZero())
			RotMatrix = CQuaternion(vRotVel*_dTimeStep).ToRotmat();

		ParallelFor(planes.size(), [&](size_t j)
		{
			size_t nIndex = m_scene.m_vNewIndexes[planes[j]];
			pWalls.Vel(nIndex) = vVel;
			pWalls.RotVel(nIndex) = vRotVel;
			pWalls.RotCenter(nIndex) = vRotCenter;
			if (!vVel.IsZero())
			{
				pWalls.Vert1(nIndex) += vVel * _dTimeStep;
				pWalls.Vert2(nIndex) += vVel * _dTimeStep;
				pWalls.Vert3(nIndex) += vVel * _dTimeStep;
			}

			if (!vRotVel.IsZero())
			{
				pWalls.Vert1(nIndex) = vRotCenter + RotMatrix * (pWalls.Vert1(nIndex) - vRotCenter);
				pWalls.Vert2(nIndex) = vRotCenter + RotMatrix * (pWalls.Vert2(nIndex) - vRotCenter);
				pWalls.Vert3(nIndex) = vRotCenter + RotMatrix * (pWalls.Vert3(nIndex) - vRotCenter);
			}
			// update wall properties
			pWalls.MinCoord(nIndex) = Min(pWalls.Vert1(nIndex), pWalls.Vert2(nIndex), pWalls.Vert3(nIndex));
			pWalls.MaxCoord(nIndex) = Max(pWalls.Vert1(nIndex), pWalls.Vert2(nIndex), pWalls.Vert3(nIndex));

			if (!vRotVel.IsZero())
				pWalls.NormalVector(nIndex) = Normalized((pWalls.Vert2(nIndex) - pWalls.Vert1(nIndex))*(pWalls.Vert3(nIndex) - pWalls.Vert1(nIndex)));
		});
	}
}

void CCPUSimulator::UpdateTemperatures(double _timeStep, bool _predictionStep)
{
	SParticleStruct& particles = m_scene.GetRefToParticles();
	const double timeStep = !_predictionStep ? m_currSimulationStep : m_currSimulationStep / 2.;
	ParallelFor(m_scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		const double tempCelcius = particles.Temperature(i) - 273.15;
		const double heatCapacity = 1117 + 0.14*tempCelcius - 411 * exp(-0.006*tempCelcius);
		particles.Temperature(i) += particles.HeatFlux(i) / (heatCapacity*particles.Mass(i)) * timeStep;
	});
}

void CCPUSimulator::MoveMultispheres(double _dTimeStep, bool _bPredictionStep)
{
	SMultiSphere& pMultispheres = m_scene.GetRefToMultispheres();
	SParticleStruct& pParticles = m_scene.GetRefToParticles();

	// move particles which does not correlated to any multisphere
	ParallelFor(pMultispheres.Size(), [&](size_t i)
	{
		if (pParticles.MultiSphIndex(i) == -1)
		{
			pParticles.Force(i) += m_externalAcceleration * pParticles.Mass(i);
			pParticles.Vel(i) += pParticles.Force(i) / pParticles.Mass(i) * _dTimeStep;
			pParticles.AnglVel(i) += pParticles.Moment(i) / pParticles.InertiaMoment(i) * _dTimeStep;
			if (!_bPredictionStep)
				pParticles.Coord(i) += pParticles.Vel(i)*_dTimeStep;
		}
	});

	// move all multispheres
	ParallelFor(pMultispheres.Size(), [&](size_t i)
	{
		CVector3 vTotalForce(0), vTotalMoment(0);

		for (unsigned j = 0; j < pMultispheres.Indices(i).size(); j++)
		{
			vTotalForce += pParticles.Force(pMultispheres.Indices(i)[j]);
			vTotalMoment += (pParticles.Coord(pMultispheres.Indices(i)[j]) - pMultispheres.Center(i))*pParticles.Force(pMultispheres.Indices(i)[j]);
		}
		vTotalForce += m_externalAcceleration * pMultispheres.Mass(i);

		pMultispheres.Velocity(i) += vTotalForce / pMultispheres.Mass(i)*_dTimeStep;
		CVector3 vAngle = pMultispheres.InvLMatrix(i)*pMultispheres.RotVelocity(i)*_dTimeStep;
		double dAngleLSquared = vAngle.SquaredLength();
		CMatrix3 newLMatrix, newInvLMatrix, deltaL;
		if (dAngleLSquared > 0)
		{
			double dAngleL = sqrt(dAngleLSquared);
			double dCosPhi = cos(dAngleL);
			double dSinPhi = sin(dAngleL);
			deltaL.values[0][0] = vAngle.x*vAngle.x / dAngleLSquared + dCosPhi * (1 - vAngle.x*vAngle.x / dAngleLSquared);
			deltaL.values[0][1] = vAngle.x*vAngle.y / dAngleLSquared * (1 - dCosPhi) - vAngle.z*dSinPhi / dAngleL;
			deltaL.values[0][2] = vAngle.x*vAngle.z / dAngleLSquared * (1 - dCosPhi) + vAngle.y*dSinPhi / dAngleL;

			deltaL.values[1][0] = vAngle.y*vAngle.x / dAngleLSquared * (1 - dCosPhi) + vAngle.z*dSinPhi / dAngleL;
			deltaL.values[1][1] = vAngle.y*vAngle.y / dAngleLSquared + dCosPhi * (1 - vAngle.y*vAngle.y / dAngleLSquared);
			deltaL.values[1][2] = vAngle.y*vAngle.z / dAngleLSquared * (1 - dCosPhi) - vAngle.x*dSinPhi / dAngleL;

			deltaL.values[2][0] = vAngle.z*vAngle.x / dAngleLSquared * (1 - dCosPhi) - vAngle.y*dSinPhi / dAngleL;
			deltaL.values[2][1] = vAngle.z*vAngle.y / dAngleLSquared * (1 - dCosPhi) + vAngle.x*dSinPhi / dAngleL;
			deltaL.values[2][2] = vAngle.z*vAngle.z / dAngleLSquared + dCosPhi * (1 - vAngle.z*vAngle.z / dAngleLSquared);

			newLMatrix = deltaL * pMultispheres.LMatrix(i);
			newInvLMatrix = newLMatrix.Inverse();
			pMultispheres.RotVelocity(i) = (newLMatrix*pMultispheres.InvInertTensor(i)*newInvLMatrix)*
				(pMultispheres.LMatrix(i)*pMultispheres.InertTensor(i)*pMultispheres.InvLMatrix(i)*pMultispheres.RotVelocity(i) + vTotalMoment * _dTimeStep);

			pMultispheres.LMatrix(i) = newLMatrix;
			pMultispheres.InvLMatrix(i) = newInvLMatrix;
		}
		else
			pMultispheres.RotVelocity(i) = (pMultispheres.LMatrix(i)*pMultispheres.InvInertTensor(i)*pMultispheres.InvLMatrix(i))*(vTotalMoment*_dTimeStep);

		if (!_bPredictionStep)
		{
			CVector3 vOldCoord = pMultispheres.Center(i);
			pMultispheres.Center(i) += pMultispheres.Velocity(i)*_dTimeStep;
			for (unsigned j = 0; j < pMultispheres.Indices(i).size(); j++)
			{
				if (dAngleLSquared > 0)
					pParticles.Coord(pMultispheres.Indices(i)[j]) = pMultispheres.Center(i) +
					deltaL * (pParticles.Coord(pMultispheres.Indices(i)[j]) - vOldCoord);
				else
					pParticles.Coord(pMultispheres.Indices(i)[j]) = pMultispheres.Center(i) + pParticles.Coord(pMultispheres.Indices(i)[j]) - vOldCoord;

				pParticles.Vel(pMultispheres.Indices(i)[j]) = pMultispheres.Velocity(i) +
					pMultispheres.RotVelocity(i)*(pParticles.Coord(pMultispheres.Indices(i)[j]) - pMultispheres.Center(i));
				pParticles.AnglVel(pMultispheres.Indices(i)[j]) = pMultispheres.RotVelocity(i);
			}
		}
	});
}

void CCPUSimulator::PrepareAdditionalSavingData()
{
	SParticleStruct& particles = m_scene.GetRefToParticles();

	// reset previously calculated stresses
	for (auto& data : m_additionalSavingData)
		data.stressTensor.Init(0);

	// save stresses caused by solid bonds
	SSolidBondStruct& solidBonds = m_scene.GetRefToSolidBonds();
	for (size_t i = 0; i < solidBonds.Size(); ++i)
	{
		if (!solidBonds.Active(i) && !m_pSystemStructure->GetObjectByIndex(solidBonds.InitIndex(i))->IsActive(m_currentTime)) continue;
		const size_t leftID  = solidBonds.LeftID(i);
		const size_t rightID = solidBonds.RightID(i);
		CVector3 connVec = (particles.Coord(leftID) - particles.Coord(rightID)).Normalized();
		m_additionalSavingData[leftID ].AddStress(-1 * connVec * particles.Radius(leftID ),      solidBonds.TotalForce(i), PI * pow(2 * particles.Radius(leftID ), 3) / 6);
		m_additionalSavingData[rightID].AddStress(     connVec * particles.Radius(rightID), -1 * solidBonds.TotalForce(i), PI * pow(2 * particles.Radius(rightID), 3) / 6);
	}

	// save stresses caused by particle-particle contact
	for (auto& collisions : m_collisionsCalculator.m_vCollMatrixPP)
		for (auto& collision : collisions)
		{
			const size_t srcID = collision->nSrcID;
			const size_t dstID = collision->nDstID;
			CVector3 connVec = (particles.Coord(srcID) - particles.Coord(dstID)).Normalized();
			const double srcRadius = particles.Radius(srcID);
			const double dstRadius = particles.Radius(dstID);
			m_additionalSavingData[collision->nSrcID].AddStress(-1 * connVec * srcRadius,      collision->vTotalForce, PI * pow(2 * srcRadius, 3) / 6);
			m_additionalSavingData[collision->nDstID].AddStress(     connVec * dstRadius, -1 * collision->vTotalForce, PI * pow(2 * dstRadius, 3) / 6);
		}

	// save stresses caused by particle-wall contacts
	for (auto& collisions : m_collisionsCalculator.m_vCollMatrixPW)
		for (auto& collision : collisions)
		{
			CVector3 connVec = (collision->vContactVector - particles.Coord(collision->nDstID)).Normalized();
			m_additionalSavingData[collision->nDstID].AddStress(connVec * particles.Radius(collision->nDstID), collision->vTotalForce, PI * pow(2 * particles.Radius(collision->nDstID), 3) / 6);
		}
}

void CCPUSimulator::SaveData()
{
	const clock_t t = clock();
	if (m_scene.GetRefToParticles().ThermalsExist())
		m_maxParticleTemperature = m_scene.GetMaxParticleTemperature();
	p_SaveData();
	m_verletList.AddDisregardingTimeInterval(clock() - t);
}

void CCPUSimulator::UpdateVerletLists(double _dTimeStep)
{
	// update max velocity
	m_maxParticleVelocity = m_scene.GetMaxParticleVelocity();
	if (m_wallsVelocityChanged)
		m_maxWallVelocity = m_scene.GetMaxWallVelocity();
	if (m_verletList.IsNeedToBeUpdated(_dTimeStep, m_scene.GetMaxPartVerletDistance(), m_maxWallVelocity))
	{
		m_verletList.UpdateList(m_currentTime);
		m_scene.SaveVerletCoords();
	}
}

void CCPUSimulator::CheckParticlesInDomain()
{
	SVolumeType simDomain = m_pSystemStructure->GetSimulationDomain();
	SParticleStruct& particles = m_scene.GetRefToParticles();
	SSolidBondStruct& solidBonds = m_scene.GetRefToSolidBonds();
	SLiquidBondStruct& liquidBonds = m_scene.GetRefToLiquidBonds();
	std::vector<size_t> vInactiveParticlesNum(m_scene.GetTotalParticlesNumber());
	ParallelFor(m_scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		if ((particles.Active(i)) && (!IsPointInDomain(simDomain, particles.Coord(i)))) // remove particles situated not in the domain
		{
			particles.Active(i) = false;
			particles.EndActivity(i) = m_currentTime;
			vInactiveParticlesNum[i]++;
			for (size_t j = 0; j < solidBonds.Size(); ++j) // delete all bonds that connected to this particle
				if ((solidBonds.Active(j)) && ((solidBonds.LeftID(j) == i) || (solidBonds.RightID(j) == i)))
				{
					solidBonds.Active(j) = false;
					solidBonds.EndActivity(j) = m_currentTime;
				}
			for (size_t j = 0; j < liquidBonds.Size(); ++j) // delete all bonds that connected to this particle
				if ((liquidBonds.Active(j)) && ((liquidBonds.LeftID(j) == i) || (liquidBonds.RightID(j) == i)))
				{
					liquidBonds.Active(j) = false;
					liquidBonds.EndActivity(j) = m_currentTime;
				}
		}
	});

	const size_t nNewInactiveParticles = VectorSum(vInactiveParticlesNum);
	m_nInactiveParticles += nNewInactiveParticles;
	if (nNewInactiveParticles > 0)
		m_scene.UpdateParticlesToBonds();
}

void CCPUSimulator::MoveParticlesOverPBC()
{
	const SPBC& pbc = m_scene.m_PBC;
	if (!pbc.bEnabled) return;
	SParticleStruct& particles = m_scene.GetRefToParticles();

	//use <unsigned char> in order to avoid problems with simultaneous writing from several threads
	//std::vector<unsigned char> vBoundaryCrossed(m_scene.GetTotalParticlesNumber());	// whether the particle crossed the PBC boundary
	std::vector<uint8_t> vShifts(m_scene.GetTotalParticlesNumber(), 0);	// whether the particle crossed the PBC boundary

	// TODO: It is unnecessary to analyze all particles for all crosses. Only those should be analyzed, who can cross boundary during current verlet step.
	// shift particles if they crossed boundary
	ParallelFor(m_scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		CVector3& vCoord = particles.Coord(i);
		// particle crossed left boundary
		if (pbc.bX && vCoord.x <= pbc.currentDomain.coordBeg.x)  vShifts[i] = vShifts[i] | 32;
		if (pbc.bY && vCoord.y <= pbc.currentDomain.coordBeg.y) vShifts[i] = vShifts[i] | 8;
		if (pbc.bZ && vCoord.z <= pbc.currentDomain.coordBeg.z) vShifts[i] = vShifts[i] | 2;
		// particle crossed right boundary
		if (pbc.bX && vCoord.x >= pbc.currentDomain.coordEnd.x) vShifts[i] = vShifts[i] | 16;
		if (pbc.bY && vCoord.y >= pbc.currentDomain.coordEnd.y) vShifts[i] = vShifts[i] | 4;
		if (pbc.bZ && vCoord.z >= pbc.currentDomain.coordEnd.z) vShifts[i] = vShifts[i] | 1;

		if (vShifts[i])
		{
			vCoord += GetVectorFromVirtShift(vShifts[i], m_scene.m_PBC.boundaryShift);
			particles.CoordVerlet(i) += GetVectorFromVirtShift(vShifts[i], m_scene.m_PBC.boundaryShift);
		}
	});

	// this can be in case when all contact models are turned off
	if (m_verletList.m_PPList.empty() && m_verletList.m_PWList.empty()) return;

	ParallelFor(m_scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		// modify shift in possible particle-particle contacts
		for (size_t j = 0; j < m_verletList.m_PPList[i].size(); j++)
		{
			const size_t srcID = i;
			const size_t dstID = m_verletList.m_PPList[i][j];
			if (vShifts[srcID])
				m_verletList.m_PPVirtShift[i][j] = AddVirtShift(m_verletList.m_PPVirtShift[i][j], vShifts[srcID]);
			if (vShifts[dstID])
				m_verletList.m_PPVirtShift[i][j] = SubstractVirtShift(m_verletList.m_PPVirtShift[i][j], vShifts[dstID]);
		}

		// modify shift in existing particle-particle  collisions
		for (size_t j = 0; j < m_collisionsCalculator.m_vCollMatrixPP[i].size(); j++)
		{
			const unsigned srcID = m_collisionsCalculator.m_vCollMatrixPP[i][j]->nSrcID;
			const unsigned dstID = m_collisionsCalculator.m_vCollMatrixPP[i][j]->nDstID;
			if (vShifts[srcID])
				m_collisionsCalculator.m_vCollMatrixPP[i][j]->nVirtShift = AddVirtShift(m_collisionsCalculator.m_vCollMatrixPP[i][j]->nVirtShift, vShifts[srcID]);
			if (vShifts[dstID])
				m_collisionsCalculator.m_vCollMatrixPP[i][j]->nVirtShift = SubstractVirtShift(m_collisionsCalculator.m_vCollMatrixPP[i][j]->nVirtShift, vShifts[dstID]);
		}

		// modify shift in possible particle-wall contacts
		for (size_t j = 0; j < m_verletList.m_PWList[i].size(); j++)
			if (vShifts[i])
				m_verletList.m_PWVirtShift[i][j] = SubstractVirtShift(m_verletList.m_PWVirtShift[i][j], vShifts[i]);

		// modify shift in existing particle-wall collisions
		for (size_t j = 0; j < m_collisionsCalculator.m_vCollMatrixPW[i].size(); j++)
			if (vShifts[i])
				m_collisionsCalculator.m_vCollMatrixPW[i][j]->nVirtShift = SubstractVirtShift(m_collisionsCalculator.m_vCollMatrixPW[i][j]->nVirtShift, vShifts[i]);
	});
}

void CCPUSimulator::UpdatePBC()
{
	m_scene.m_PBC.UpdatePBC(m_currentTime);
	if (!m_scene.m_PBC.vVel.IsZero())	// if velocity is not zero
		InitializeModelParameters();	// set new PBC to all models
}

void CCPUSimulator::LoadConfiguration()
{
	CBaseSimulator::LoadConfiguration();

	const ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	if (!protoMessage.has_simulator()) return;
	const ProtoModuleSimulator& sim = protoMessage.simulator();
	EnableCollisionsAnalysis(sim.save_collisions());
}

void CCPUSimulator::SaveConfiguration()
{
	CBaseSimulator::SaveConfiguration();

	ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	ProtoModuleSimulator* pSim = protoMessage.mutable_simulator();
	pSim->set_save_collisions(m_analyzeCollisions);
}

void CCPUSimulator::GetOverlapsInfo(double& _dMaxOverlap, double& _dAverageOverlap, size_t _nMaxParticleID)
{
	_dMaxOverlap = 0;
	_dAverageOverlap = 0;
	size_t nCollNumber = 0;
	for (const auto& collisions : m_collisionsCalculator.m_vCollMatrixPP)
		for (const auto coll : collisions)
		{
			if ((coll->nSrcID < _nMaxParticleID) || (coll->nDstID < _nMaxParticleID))
			{
				_dMaxOverlap = std::max(_dMaxOverlap, coll->dNormalOverlap);
				_dAverageOverlap += coll->dNormalOverlap;
				nCollNumber++;
			}
		}

	const SParticleStruct& particles = m_scene.GetRefToParticles();
	for (const auto& collisions : m_collisionsCalculator.m_vCollMatrixPW)
		for (const auto coll : collisions)
		{
			if (coll->nDstID < _nMaxParticleID)
			{
				const CVector3 vRc = _VIRTUAL_COORDINATE(particles.Coord(coll->nDstID), coll->nVirtShift, m_scene.m_PBC) - coll->vContactVector;
				const double dOverlap = particles.ContactRadius(coll->nDstID) - vRc.Length();
				_dMaxOverlap = std::max(_dMaxOverlap, dOverlap);
				_dAverageOverlap += dOverlap;
				nCollNumber++;
			}
		}

	if (nCollNumber)
		_dAverageOverlap = _dAverageOverlap / nCollNumber;
}
