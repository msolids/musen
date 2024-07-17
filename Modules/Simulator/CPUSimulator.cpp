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
	m_scene.ClearState();
	CheckParticlesInDomain();

	// if there is no contact model, then there is no necessity to calculate contacts
	if (!m_PPModels.empty() || !m_PWModels.empty())
	{
		UpdateVerletLists(_dTimeStep); // between PP and PW
		m_collisionsCalculator.UpdateCollisionMatrixes(_dTimeStep, m_currentTime);
	}
}

void CCPUSimulator::CalculateForcesStep(double _dTimeStep)
{
	if (!m_EFModels.empty()) CalculateForcesEF(_dTimeStep);
	if (!m_PPModels.empty()) CalculateForcesPP(_dTimeStep);
	if (!m_PWModels.empty()) CalculateForcesPW(_dTimeStep);
	if (!m_SBModels.empty()) CalculateForcesSB(_dTimeStep);
	if (!m_LBModels.empty()) CalculateForcesLB(_dTimeStep);
	m_collisionsCalculator.CalculateTotalStatisticsInfo();
}

void CCPUSimulator::CalculateForcesPP(double _timeStep)
{
	SParticleStruct& particles = m_scene.GetRefToParticles();

	for (auto* model : m_PPModels)
	{
		model->Precalculate(m_currentTime, _timeStep);

		for (auto& collisions : m_tempCollPPArray)
			for (auto& coll : collisions)
				coll.clear();

		ParallelFor(m_collisionsCalculator.m_vCollMatrixPP.size(), [&](size_t i)
		{
			const size_t index = i % m_nThreads;
			for (auto& coll : m_collisionsCalculator.m_vCollMatrixPP[i])
			{
				model->Calculate(m_currentTime, _timeStep, coll);
				model->ConsolidateSrc(m_currentTime, _timeStep, particles, coll);

				m_tempCollPPArray[index][coll->nDstID % m_nThreads].push_back(coll);
			}
		});

		ParallelFor([&](size_t i)
		{
			for (size_t j = 0; j < m_nThreads; ++j)
				for (const auto& coll : m_tempCollPPArray[j][i])
					model->ConsolidateDst(m_currentTime, _timeStep, particles, coll);
		});
	}
}

void CCPUSimulator::CalculateForcesPW(double _timeStep)
{
	SParticleStruct& particles = m_scene.GetRefToParticles();
	SWallStruct& walls = m_scene.GetRefToWalls();

	for (auto* model : m_PWModels)
	{
		model->Precalculate(m_currentTime, _timeStep);

		for (auto& collisions : m_tempCollPWArray)
			for (auto& coll : collisions)
				coll.clear();

		ParallelFor(m_collisionsCalculator.m_vCollMatrixPW.size(), [&](size_t i)
		{
			const size_t index = i % m_nThreads;
			for (auto& coll : m_collisionsCalculator.m_vCollMatrixPW[i])
			{
				model->Calculate(m_currentTime, _timeStep, coll);
				model->ConsolidatePart(m_currentTime, _timeStep, particles, coll);

				m_tempCollPWArray[index][coll->nSrcID % m_nThreads].push_back(coll);
			}
		});

		ParallelFor([&](size_t i)
		{
			for (size_t j = 0; j < m_nThreads; ++j)
				for (const auto& coll : m_tempCollPWArray[j][i])
					model->ConsolidateWall(m_currentTime, _timeStep, walls, coll);
		});
	}
}

void CCPUSimulator::CalculateForcesSB(double _timeStep)
{
	if (m_scene.GetBondsNumber() == 0) return;

	SParticleStruct& particles = m_scene.GetRefToParticles();
	SSolidBondStruct& bonds = m_scene.GetRefToSolidBonds();
	const auto& partToSolidBonds = *m_scene.GetPointerToPartToSolidBonds();

	for (auto* model : m_SBModels)
	{
		model->Precalculate(m_currentTime, _timeStep);

		std::vector<unsigned> brokenBonds(m_nThreads, 0);

		ParallelFor(m_scene.GetBondsNumber(), [&](size_t iBond)
		{
			if (bonds.Active(iBond))
				model->Calculate(m_currentTime, _timeStep, iBond, bonds, &brokenBonds[iBond % m_nThreads]);
		});
		m_nBrokenBonds += VectorSum(brokenBonds);

		ParallelFor(partToSolidBonds.size(), [&](size_t iPart)
		{
			for (size_t j = 0; j < partToSolidBonds[iPart].size(); ++j)
			{
				const unsigned iBond = partToSolidBonds[iPart][j];
				if (!bonds.Active(iBond)) continue;
				model->Consolidate(m_currentTime, _timeStep, iBond, iPart, particles);
			}
		});
	}
}

void CCPUSimulator::CalculateForcesLB(double _timeStep)
{
	if (m_scene.GetLiquidBondsNumber() == 0) return;

	SLiquidBondStruct& bonds = m_scene.GetRefToLiquidBonds();
	SParticleStruct& particles = m_scene.GetRefToParticles();

	for (auto* model : m_LBModels)
	{
		model->Precalculate(m_currentTime, _timeStep);

		std::vector<unsigned> brokenBonds(m_nThreads, 0);

		ParallelFor(m_scene.GetLiquidBondsNumber(), [&](size_t i)
		{
			if (bonds.Active(i))
				model->Calculate(m_currentTime, _timeStep, i, bonds, &brokenBonds[i % m_nThreads]);
		});
		m_nBrokenLiquidBonds += VectorSum(brokenBonds);

		for (size_t iBond = 0; iBond < m_scene.GetLiquidBondsNumber(); ++iBond)
			if (bonds.Active(iBond))
				model->Consolidate(m_currentTime, _timeStep, iBond, particles);
	}
}

void CCPUSimulator::CalculateForcesEF(double _timeStep)
{
	SParticleStruct& particles = m_scene.GetRefToParticles();

	for (auto* model : m_EFModels)
	{
		model->Precalculate(m_currentTime, _timeStep);

		ParallelFor(particles.Size(), [&](size_t iPart)
		{
			if (particles.Active(iPart))
				model->Calculate(m_currentTime, _timeStep, iPart, particles);
		});
	}
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
			vTemp /= particles.InertiaMoment(i);
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

void CCPUSimulator::MoveWalls(double _timeStep)
{
	SWallStruct& walls = m_scene.GetRefToWalls();
	m_wallsVelocityChanged = false;
	for (size_t i = 0; i < m_pSystemStructure->GeometriesNumber(); ++i)
	{
		CRealGeometry* geom = m_pSystemStructure->Geometry(i);
		const auto& planes = geom->Planes();
		if (planes.empty()) continue;

		if (geom->Motion()->MotionType() == CGeometryMotion::EMotionType::FORCE_DEPENDENT ||  // force
			geom->Motion()->MotionType() == CGeometryMotion::EMotionType::CONSTANT_FORCE)
		{
			double totalForceZ = 0;
			for (const auto& plane : planes)
			{
				const size_t iWall = m_scene.m_vNewIndexes[plane];
				totalForceZ += walls.Force(iWall).z;
			}
			geom->UpdateMotionInfo(totalForceZ);
		}
		else
			geom->UpdateMotionInfo(m_currentTime); // time
		CVector3 vel = geom->GetCurrentVelocity();
		CVector3 rotVel = geom->GetCurrentRotVelocity();
		CVector3 rotCenter;
		if (geom->RotateAroundCenter())
		{
			double totalArea{ 0.0 };
			CVector3 totalWeightedCentroid{ 0.0 };
			rotCenter.Init(0);
			for (const auto& plane : planes)
			{
				const size_t iWall = m_scene.m_vNewIndexes[plane];
				const double area = 0.5 * Length((walls.Vert2(iWall) - walls.Vert1(iWall)) * (walls.Vert3(iWall) - walls.Vert1(iWall)));
				const CVector3 centroid = (walls.Vert1(iWall) + walls.Vert2(iWall) + walls.Vert3(iWall)) / 3.;
				totalArea += area;
				totalWeightedCentroid += area * centroid;
			}
			if (totalArea != 0.0)
				rotCenter = totalWeightedCentroid / totalArea;
		}
		else
			rotCenter = geom->GetCurrentRotCenter();

		if (m_currentTime == 0.0)
			m_wallsVelocityChanged = true;
		else
		{
			if (!(geom->GetCurrentVelocity() - geom->GetCurrentVelocity()).IsZero())
				m_wallsVelocityChanged = true;
			else if (!(geom->GetCurrentRotVelocity() - geom->GetCurrentRotVelocity()).IsZero())
				m_wallsVelocityChanged = true;
			else if (!(geom->GetCurrentRotCenter() - geom->GetCurrentRotCenter()).IsZero())
				m_wallsVelocityChanged = true;
		}

		if (!geom->FreeMotion().IsZero() && geom->Mass() != 0.0)// solve newtons motion for wall
		{
			CVector3 totalForce = m_externalAcceleration * geom->Mass();
			CVector3 totalAverVel(0);
			// calculate total force acting on wall
			for (const auto& plane : planes)
			{
				const size_t iWall = m_scene.m_vNewIndexes[plane];
				totalForce += walls.Force(iWall);
				totalAverVel += walls.Vel(iWall) / static_cast<double>(planes.size());
			}
			if (geom->FreeMotion().x)
				vel.x = totalAverVel.x + _timeStep * totalForce.x / geom->Mass();
			if (geom->FreeMotion().y)
				vel.y = totalAverVel.y + _timeStep * totalForce.y / geom->Mass();
			if (geom->FreeMotion().z)
				vel.z = totalAverVel.z + _timeStep * totalForce.z / geom->Mass();
			m_wallsVelocityChanged = true;
		}

		if (vel.IsZero() && rotVel.IsZero()) continue;
		CMatrix3 rotMatrix;
		if (!rotVel.IsZero())
			rotMatrix = CQuaternion(rotVel*_timeStep).ToRotmat();

		ParallelFor(planes.size(), [&](size_t j)
		{
			const size_t iWall = m_scene.m_vNewIndexes[planes[j]];
			walls.Vel(iWall) = vel;
			walls.RotVel(iWall) = rotVel;
			walls.RotCenter(iWall) = rotCenter;
			// If the rotation is done around the calculated center, it is important to first rotate the geometry and only then move it.
			if (!rotVel.IsZero())
			{
				walls.Vert1(iWall) = rotCenter + rotMatrix * (walls.Vert1(iWall) - rotCenter);
				walls.Vert2(iWall) = rotCenter + rotMatrix * (walls.Vert2(iWall) - rotCenter);
				walls.Vert3(iWall) = rotCenter + rotMatrix * (walls.Vert3(iWall) - rotCenter);
			}
			if (!vel.IsZero())
			{
				walls.Vert1(iWall) += vel * _timeStep;
				walls.Vert2(iWall) += vel * _timeStep;
				walls.Vert3(iWall) += vel * _timeStep;
			}

			// update wall properties
			walls.MinCoord(iWall) = Min(walls.Vert1(iWall), walls.Vert2(iWall), walls.Vert3(iWall));
			walls.MaxCoord(iWall) = Max(walls.Vert1(iWall), walls.Vert2(iWall), walls.Vert3(iWall));

			if (!rotVel.IsZero())
				walls.NormalVector(iWall) = Normalized((walls.Vert2(iWall) - walls.Vert1(iWall))*(walls.Vert3(iWall) - walls.Vert1(iWall)));
		});
	}
}

void CCPUSimulator::UpdateTemperatures(bool _predictionStep)
{
	SParticleStruct& particles = m_scene.GetRefToParticles();
	const double timeStep = !_predictionStep ? m_currSimulationStep : m_currSimulationStep / 2.;
	ParallelFor(m_scene.GetTotalParticlesNumber(), [&](size_t i)
	{
		particles.Temperature(i) += particles.HeatFlux(i) / (particles.HeatCapacity(i) * particles.Mass(i)) * timeStep;
		particles.Temperature(i) = particles.Temperature(i) < 0.0 ? 0.0 : particles.Temperature(i);
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

void CCPUSimulator::GenerateNewObjects()
{
	const size_t nNewParticles = m_generationManager->GenerateObjects(m_currentTime, m_scene, m_generatedObjectsDiff);
	if (nNewParticles > 0)
	{
		m_verletList.SetSceneInfo(m_pSystemStructure->GetSimulationDomain(), m_scene.GetMinParticleContactRadius(), m_scene.GetMaxParticleContactRadius(), m_cellsMax, m_verletDistanceCoeff, m_autoAdjustVerletDistance);
		m_verletList.ResetCurrentData();
		m_nGeneratedObjects += nNewParticles;
		m_scene.UpdateParticlesToBonds();
	}
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
