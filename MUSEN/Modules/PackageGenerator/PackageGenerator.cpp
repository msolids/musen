/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "PackageGenerator.h"
#include "PackageGeneratorHelperCPU.h"
#include "PackageGeneratorHelperGPU.cuh"
#include "CPUSimulator.h"
#include "GPUSimulator.h"
#include "ParticleFilter.h"
#include "MUSENFileFunctions.h"
#include <random>

#undef GetCurrentTime

CPackageGenerator::~CPackageGenerator()
{
	Clear();
}

void CPackageGenerator::SetSimulatorType(ESimulatorType _type)
{
	m_simulatorType = _type;
}

ESimulatorType CPackageGenerator::GetSimulatorType() const
{
	return m_simulatorType;
}

void CPackageGenerator::SetVerletCoefficient(double _coeff)
{
	m_verletCoeff = _coeff;
}

double CPackageGenerator::GetVerletCoefficient() const
{
	return m_verletCoeff;
}

bool CPackageGenerator::IsDataCorrect() const
{
	if (m_generators.empty())
	{
		m_errorMessage = "No generators defined";
		return false;
	}

	for (const auto& g : m_generators)
	{
		if (!g.active) continue;

		// check generation volume
		const CAnalysisVolume* volume = m_pSystemStructure->AnalysisVolume(g.volumeKey);
		if (!volume)
		{
			m_errorMessage = "Error in generator '" + g.name + "': generation volume not available or not specified";
			return false;
		}
		if (!volume->Mesh(0).IsFaceNormalsConsistent())
		{
			m_errorMessage = "Error in generator '" + g.name + "': generation volume is corrupted";
			return false;
		}

		// check mixture
		const std::string mixtureError = m_pSystemStructure->m_MaterialDatabase.IsMixtureCorrect(g.mixtureKey);
		if (!mixtureError.empty())
		{
			m_errorMessage = "Error in generator '" + g.name + "': " + mixtureError;
			return false;
		}
		const CMixture* mixture = m_pSystemStructure->m_MaterialDatabase.GetMixture(g.mixtureKey);

		// check generation settings
		const double maxGenerDiameter = mixture->GetMaxFractionContactDiameter();
		const double maxAllowDiameter = volume->MaxInscribedDiameter();
		if (maxAllowDiameter*0.95 < maxGenerDiameter)
		{
			m_errorMessage = "Error in generator '" + g.name + "': particle diameter (" + std::to_string(maxGenerDiameter) + " m) is too large";
			return false;
		}
	}

	m_errorMessage.clear();
	return true;
}

void CPackageGenerator::StartGeneration()
{
	m_status = ERunningStatus::RUNNING;

	for (auto& generator : m_generators)
	{
		if (m_status == ERunningStatus::TO_BE_STOPPED) break;
		if (!generator.active) continue;

		// initialize generator and simulator and place particles on the scene
		Initialize(generator);

		// prepare some variables
		auto simulator = generator.simulator;
		double normForceCoeff = 0.001;		// coefficient for normal force
		const double multiplier = 1.2;		// multiplier for the normal force coefficient
		const double timeStep = 1.0;		// simulation time step
		const double minRadius = simulator->GetSystemStructure()->GetMinParticleDiameter() / 2;	// min radius of particles
		const double maxRadius = simulator->GetSystemStructure()->GetMaxParticleDiameter() / 2;	// max radius of particles

		// make prediction step and the first simulation step
		simulator->Simulate();
		UpdateReachedOverlap(generator);

		// data helper to support simulation
		IPackageGeneratorHelper* helper{ nullptr };
		switch (m_simulatorType)
		{
		case ESimulatorType::BASE:
		case ESimulatorType::CPU:	helper = new CPackageGeneratorHelperCPU{ &dynamic_cast<CCPUSimulator*>(simulator)->GetPointerToSimplifiedScene().GetRefToParticles() };	break;
		case ESimulatorType::GPU:	helper = new CPackageGeneratorHelperGPU{ &dynamic_cast<CGPUSimulator*>(simulator)->GetPointerToSceneGPU().GetPointerToParticles() };		break;
		}

		// perform generation
		size_t iteration = 0;
		simulator->GetModelManager()->SetModelParameters(EMusenModelType::PW, "NORMAL_FORCE_COEFF 0.9");
		while (generator.maxReachedOverlap > generator.targetMaxOverlap && iteration < generator.maxIterations && m_status != ERunningStatus::TO_BE_STOPPED)
		{
			simulator->GetModelManager()->SetModelParameters(EMusenModelType::PP, "NORMAL_FORCE_COEFF " + Double2String(normForceCoeff));
			simulator->InitializeModelParameters();
			simulator->UpdateCollisionsStep(timeStep);

			if (iteration % 10 == 0)
			{
				UpdateReachedOverlap(generator);
				if (iteration % 1000 == 0)
					std::cout << "Reached overlap: " << generator.maxReachedOverlap << "/" << generator.targetMaxOverlap << std::endl;
			}

			helper->LimitVelocities();

			simulator->CalculateForcesPP(timeStep);
			simulator->MoveObjectsStep(timeStep);

			// reset old velocities
			if (minRadius != maxRadius) // often it is a case
				helper->ScaleVelocitiesToRadius(minRadius);

			const double maxRelVel = helper->MaxRelativeVelocity();
			if (maxRelVel > 0.1) // check that particle cannot move more than 0.1 of its radius per step
			{
				helper->ResetMovement();
				normForceCoeff /= multiplier;
			}
			else
			{
				helper->SaveVelocities();

				simulator->SetCurrentTime(simulator->GetCurrentTime() + timeStep);
				simulator->UpdateCollisionsStep(timeStep);
				simulator->CalculateForcesPW(timeStep);
				simulator->MoveObjectsStep(timeStep);
				simulator->SetCurrentTime(simulator->GetCurrentTime() + timeStep);

				if (maxRelVel < 0.01 && normForceCoeff < 0.05) 	// set new force coefficients
					normForceCoeff *= multiplier;
			}

			generator.completness = ++iteration * 100.0 / static_cast<double>(generator.maxIterations);
		}
		generator.completness = 100;

		// copy generated particles into the system structure
		SaveGeneratedObjects(generator);

		// clear
		ClearSimulator(generator.simulator);
		delete helper;
	}

	m_status = ERunningStatus::IDLE;
}

void CPackageGenerator::Initialize(SPackage& _generator) const
{
	// initialize generator's parameters
	_generator.generatedParticles = 0;
	_generator.maxReachedOverlap  = 0;
	_generator.avrReachedOverlap  = 0;
	_generator.completness        = 0;

	// create a new system structure for the generator
	auto* scene = new CSystemStructure();
	scene->SaveToFile(m_pSystemStructure->GetFileName() + ".pg");

	// calculate and set simulation domain
	SetupSimulationDomain(_generator, *scene);

	// setup additional parameters
	scene->SetPBC(m_pSystemStructure->GetPBC());
	scene->EnableAnisotropy(m_pSystemStructure->IsAnisotropyEnabled());
	scene->EnableContactRadius(m_pSystemStructure->IsContactRadiusEnabled());

	// place initial particles for generation
	PlaceParticlesInitially(_generator, *scene);

	// add particles already existing in generation volume
	AddOverlayedParticles(_generator, *scene);

	// add generation volume and real geometries
	AddGeometries(_generator, *scene);

	// initialize simulator
	SetupSimulator(_generator, *scene);
}

void CPackageGenerator::SetupSimulationDomain(SPackage& _generator, CSystemStructure& _scene) const
{
	SVolumeType domain = m_pSystemStructure->AnalysisVolume(_generator.volumeKey)->BoundingBox(0);

	// increase simulation domain to cover all existing geometries
	if (!_generator.insideGeometry)
		for (const auto& g : m_pSystemStructure->AllGeometries())
			for (const auto& wall : g->Walls())
			{
				domain.coordBeg = Min(domain.coordBeg, wall->GetCoordVertex1(0), wall->GetCoordVertex2(0), wall->GetCoordVertex3(0));
				domain.coordEnd = Max(domain.coordEnd, wall->GetCoordVertex1(0), wall->GetCoordVertex2(0), wall->GetCoordVertex3(0));
			}

	// increase simulation domain to consider possible virtual particles
	if (m_pSystemStructure->GetPBC().bEnabled)
	{
		const SPBC pbc = m_pSystemStructure->GetPBC();
		const CMixture* mixture = m_pSystemStructure->m_MaterialDatabase.GetMixture(_generator.mixtureKey);
		const double sizeInc = mixture->GetMaxFractionContactDiameter() + mixture->GetMinFractionContactDiameter();
		if (pbc.bX) domain.coordBeg.x = std::min(domain.coordBeg.x, pbc.initDomain.coordBeg.x - sizeInc);
		if (pbc.bY) domain.coordBeg.y = std::min(domain.coordBeg.y, pbc.initDomain.coordBeg.y - sizeInc);
		if (pbc.bZ) domain.coordBeg.z = std::min(domain.coordBeg.z, pbc.initDomain.coordBeg.z - sizeInc);
		if (pbc.bX) domain.coordEnd.x = std::max(domain.coordEnd.x, pbc.initDomain.coordEnd.x + sizeInc);
		if (pbc.bY) domain.coordEnd.y = std::max(domain.coordEnd.y, pbc.initDomain.coordEnd.y + sizeInc);
		if (pbc.bZ) domain.coordEnd.z = std::max(domain.coordEnd.z, pbc.initDomain.coordEnd.z + sizeInc);
	}

	// increase size of the simulation domain by 1/8
	const CVector3 inc = (domain.coordEnd - domain.coordBeg) / 8;
	domain = { domain.coordBeg - inc, domain.coordEnd + inc };

	// set simulation domain
	_scene.SetSimulationDomain(domain);
}

void CPackageGenerator::PlaceParticlesInitially(SPackage& _generator, CSystemStructure& _scene) const
{
	RandomSeed();
	const CMixture* mixture = m_pSystemStructure->m_MaterialDatabase.GetMixture(_generator.mixtureKey);

	// prepare some parameters
	const double totalMass = 1;											// mass of all generating particles
	std::vector<size_t> partPerFrac = ParticlesPerFraction(_generator);	// number of particles to be generated for each fraction
	const size_t totalNumber = std::accumulate(partPerFrac.begin(), partPerFrac.end(), size_t(0));	// total number of particles to be generated

	// setup compounds
	std::vector<CCompound*> compounds;		// list of compounds for each fraction
	for (size_t i = 0; i < mixture->FractionsNumber(); ++i)
	{
		CCompound* compound = _scene.m_MaterialDatabase.AddCompound(std::to_string(i));				// add compound for specific fraction. this key will be used afterwards to determine a real compound
		const double density = 6 * totalMass / (PI * std::pow(mixture->GetFractionDiameter(i), 3));	// density to equalize masses of particles from different fractions
		compound->SetPropertyValue(ETPPropertyTypes::PROPERTY_DENSITY, density);
		compounds.push_back(compound);
	}

	// setup volume filter
	CParticleFilter filter{ *m_pSystemStructure , _generator };

	// setup random generator
	SVolumeType bbox = m_pSystemStructure->AnalysisVolume(_generator.volumeKey)->BoundingBox(0);	// bounding box of the generation volume
	std::mt19937_64 randGen{ std::random_device{}() };
	std::uniform_real_distribution<double> randDistrX(bbox.coordBeg.x, bbox.coordEnd.x);
	std::uniform_real_distribution<double> randDistrY(bbox.coordBeg.y, bbox.coordEnd.y);
	std::uniform_real_distribution<double> randDistrZ(bbox.coordBeg.z, bbox.coordEnd.z);

	// place particles
	while (_generator.generatedParticles != totalNumber)
	{
		if (m_status == ERunningStatus::TO_BE_STOPPED) return;

		// place new particles randomly
		SParticles particles; // information about new generated particles
		for (size_t iFrac = 0; iFrac < partPerFrac.size(); ++iFrac)		// for each fraction in mixture
			for (size_t iPart = 0; iPart < partPerFrac[iFrac]; ++iPart)	// for each particle
				particles.Add(
					CVector3{ randDistrX(randGen), randDistrY(randGen), randDistrZ(randGen) },
					mixture->GetFractionDiameter(iFrac) / 2,
					m_pSystemStructure->IsContactRadiusEnabled() ? mixture->GetFractionContactDiameter(iFrac) / 2 : mixture->GetFractionDiameter(iFrac) / 2,
					CQuaternion::Random(),
					compounds[iFrac],
					iFrac);

		// get particles, which hit the volume
		std::vector<size_t> insideIDs = filter.Filter(particles.coords, particles.contRadii);

		// add particles to the scene
		AddParticles(_scene, insideIDs, particles);

		// reduce number of particles in specific fraction still to be generated
		for (auto id : insideIDs)
			partPerFrac[particles.fractions[id]]--;

		// remember amount of already generated particles
		_generator.generatedParticles += insideIDs.size();
	}
}

void CPackageGenerator::AddParticles(CSystemStructure& _scene, std::vector<size_t> _indices, const SParticles& _particles)
{
	std::vector<CPhysicalObject*> newObjects = _scene.AddSeveralObjects(SPHERE, _indices.size());
	for (size_t i = 0; i < _indices.size(); ++i)	// for all hit particles
	{
		const size_t id = _indices[i];
		auto* part = dynamic_cast<CSphere*>(newObjects[i]);
		part->SetRadius(_particles.radii[id]);
		part->SetContactRadius(_particles.contRadii[id]);
		part->SetCoordinates(0, _particles.coords[id]);
		part->SetCompound(_particles.compounds[id]);
		part->SetOrientation(0, _particles.orients[id]);
	}
}

void CPackageGenerator::AddOverlayedParticles(SPackage& _generator, CSystemStructure& _scene) const
{
	const std::vector<const CSphere*> existingParticles = m_pSystemStructure->AnalysisVolume(_generator.volumeKey)->GetParticlesInside(0, false);
	if (existingParticles.empty()) return;

	CCompound* compound = _scene.m_MaterialDatabase.AddCompound("Overlap"); // create compound for overlapping particles
	compound->SetPropertyValue(ETPPropertyTypes::PROPERTY_DENSITY, 1e+99);	// set large density
	// set restitution coefficient between overlaid and generated particles
	for (size_t i = 0; i < m_pSystemStructure->m_MaterialDatabase.GetMixture(_generator.mixtureKey)->FractionsNumber(); ++i)
		_scene.m_MaterialDatabase.SetInteractionValue(std::to_string(i), "Overlap", PROPERTY_RESTITUTION_COEFFICIENT, 0.2);

	// add overlapping particles
	for (const auto& oldParticle : existingParticles)
	{
		auto* newParticle = dynamic_cast<CSphere*>(_scene.AddObject(SPHERE));
		newParticle->SetRadius(oldParticle->GetRadius());
		newParticle->SetContactRadius(oldParticle->GetContactRadius());
		newParticle->SetCoordinates(0, oldParticle->GetCoordinates(0));
		newParticle->SetCompound(compound);
		newParticle->SetOrientation(0, oldParticle->GetOrientation(0));
	}
}

void CPackageGenerator::AddGeometries(SPackage& _generator, CSystemStructure& _scene) const
{
	// add generation volume as a real geometry
	const CAnalysisVolume* volume = m_pSystemStructure->AnalysisVolume(_generator.volumeKey);
	CRealGeometry* geometry = _scene.AddGeometry(volume->Mesh(0).CreateInvertedMesh());	// add corresponding real volume
	CCompound* wallCompound = _scene.m_MaterialDatabase.AddCompound("Wall");			// create compound for walls
	geometry->SetMaterial(wallCompound->GetKey());										// set material

	// add existing geometries
	if (!_generator.insideGeometry)
		for (const auto& g : m_pSystemStructure->AllGeometries())
		{
			std::vector<CTriangle> triangles;
			for (const auto& wall : g->Walls())
				triangles.push_back(wall->GetPlaneCoords(0));
			geometry = _scene.AddGeometry(CTriangularMesh{ g->Name(), triangles });
			geometry->SetMaterial(wallCompound->GetKey());
		}
}

void CPackageGenerator::SetupSimulator(SPackage& _generator, CSystemStructure& _scene) const
{
	// create and initialize model manager
	auto* modelManager = new CModelManager();
	modelManager->SetModelPath(EMusenModelType::PP, "ModelPPSimpleViscoElastic");
	modelManager->SetModelPath(EMusenModelType::PW, "ModelPWSimpleViscoElastic");

	// setup initial model parameters
	modelManager->SetModelParameters(EMusenModelType::PP, "NORMAL_FORCE_COEFF 0");
	modelManager->SetModelParameters(EMusenModelType::PW, "NORMAL_FORCE_COEFF 0");

	// create and initialize simulator
	switch (m_simulatorType)
	{
	case ESimulatorType::BASE:
	case ESimulatorType::CPU:	_generator.simulator = new CCPUSimulator();		break;
	case ESimulatorType::GPU:	_generator.simulator = new CGPUSimulator();		break;
	}

	// setup simulator
	_generator.simulator->SetSystemStructure(&_scene);						// system structure
	_generator.simulator->SetModelManager(modelManager);					// model manager
	_generator.simulator->SetGenerationManager(new CGenerationManager());	// empty dynamic generator
	_generator.simulator->SetExternalAccel(CVector3{ 0 });					// disable gravity force
	_generator.simulator->SetInitSimulationStep(1.0);						// simulation time step
	_generator.simulator->SetEndTime(1.0);									// simulation time
	_generator.simulator->SetVerletCoeff(m_verletCoeff);					// verlet coefficient
	_generator.simulator->SetAutoAdjustFlag(false);							// auto adjust verlet distance flag
}

void CPackageGenerator::SaveGeneratedObjects(SPackage& _generator) const
{
	// copy generated particles into the system structure
	_generator.simulator->SaveData(); // save all generated data into system structure
	std::vector<size_t> indexes;
	for (size_t i = 0; i < _generator.generatedParticles; ++i)
		if (_generator.simulator->GetSystemStructure()->GetObjectByIndex(i)->IsActive(0))
			indexes.push_back(i);
	std::vector<CPhysicalObject*> objects = m_pSystemStructure->AddSeveralObjects(SPHERE, indexes.size());
	for (size_t i = 0; i < indexes.size(); ++i)
	{
		const auto* generated = dynamic_cast<const CSphere*>(_generator.simulator->GetSystemStructure()->GetObjectByIndex(indexes[i])); // first generatedParticles particles are our generated ones
		auto* sphere = dynamic_cast<CSphere*>(objects[i]);
		sphere->SetRadius(generated->GetRadius());
		sphere->SetContactRadius(generated->GetContactRadius());
		sphere->SetCoordinates(0, generated->GetCoordinates(_generator.simulator->GetSystemStructure()->GetAllTimePoints().back()));
		sphere->SetOrientation(0, generated->GetOrientation(_generator.simulator->GetSystemStructure()->GetAllTimePoints().back()));
		sphere->SetAngleVelocity(0, CVector3{ 0 });
		sphere->SetVelocity(0, _generator.initVelocity);

		// get and set proper compound
		const int iFraction = std::stoi(generated->GetCompoundKey()); // convert compound key to fraction index
		const CMixture* mixture = m_pSystemStructure->m_MaterialDatabase.GetMixture(_generator.mixtureKey);
		const std::string compoundKey = mixture->GetFractionCompound(iFraction);
		const CCompound* compound = m_pSystemStructure->m_MaterialDatabase.GetCompound(compoundKey);
		sphere->SetCompound(compound);
	}
}

std::vector<size_t> CPackageGenerator::ParticlesPerFraction(const SPackage& _generator) const
{
	const CMixture* mixture = m_pSystemStructure->m_MaterialDatabase.GetMixture(_generator.mixtureKey);
	std::vector<size_t> res(mixture->FractionsNumber());
	const double volumeToFill = VolumeToFill(_generator);
	const double targetSolidVolume = (1.0 - _generator.targetPorosity) * volumeToFill;
	double volume = 0.0;
	for (size_t i = 0; i < mixture->FractionsNumber(); ++i)
		volume += mixture->GetFractionValue(i) * PI / 6.0 * std::pow(mixture->GetFractionDiameter(i), 3);
	for (size_t i = 0; i < mixture->FractionsNumber(); ++i) // to consider correct number of discrete particles to be generated
		res[i] = static_cast<size_t>(targetSolidVolume / volume * mixture->GetFractionValue(i));
	return res;
}

double CPackageGenerator::VolumeToFill(const SPackage& _generator) const
{
	const auto pbc = m_pSystemStructure->GetPBC();
	const auto* geometry = m_pSystemStructure->AnalysisVolume(_generator.volumeKey);

	// if PBC is disabled, the required volume is just a volume of the generation geometry.
	if (!pbc.bEnabled)
		return geometry->Volume();

	// if PBC is enabled, shrink generation geometry to the size of the PBC
	// all points of the geometry that lay outside of the PBC are projected on the planes of the PBC,
	// the obtained figure represents the final generation volume

	// activity of each PBC plane
	const CVector3b activePBCPlanes{ pbc.bX, pbc.bY, pbc.bZ };
	// coordinates of each PBC plane
	const CVector3 planeCoordsBeg = pbc.initDomain.coordBeg;
	const CVector3 planeCoordsEnd = pbc.initDomain.coordEnd;
	// for each plane, a point laying on that plane
	const std::vector<CVector3> pointsOnPlaneBeg{ CVector3{ planeCoordsBeg[0], 0.0, 0.0 } , CVector3{ 0.0, planeCoordsBeg[1], 0.0 }, CVector3{ 0.0, 0.0, planeCoordsBeg[2] } };
	const std::vector<CVector3> pointsOnPlaneEnd{ CVector3{ planeCoordsEnd[0], 0.0, 0.0 } , CVector3{ 0.0, planeCoordsEnd[1], 0.0 }, CVector3{ 0.0, 0.0, planeCoordsEnd[2] } };
	// for each plane, its normal vector
	const std::vector<CVector3> planeNormsBeg{ CVector3{ -1.0, 0.0, 0.0 } , CVector3{ 0.0, -1.0, 0.0 }, CVector3{ 0.0, 0.0, -1.0 } };
	const std::vector<CVector3> planeNormsEnd{ CVector3{  1.0, 0.0, 0.0 } , CVector3{ 0.0,  1.0, 0.0 }, CVector3{ 0.0, 0.0,  1.0 } };

	// project points
	auto triangles = geometry->Mesh().Triangles();
	for (auto& t : triangles) {												// for each triangle
		for (size_t iP = 0; iP < t.Size(); ++iP) {							// for each point of the triangle
			for (size_t iXYZ = 0; iXYZ < activePBCPlanes.Size(); ++iXYZ) {	// for each coordinate X/Y/Z of the point
				if (activePBCPlanes[iXYZ])
				{
					if (t[iP][iXYZ] < planeCoordsBeg[iXYZ])
						t[iP] = ProjectionPointToPlane(t[iP], pointsOnPlaneBeg[iXYZ], planeNormsBeg[iXYZ]);
					if (t[iP][iXYZ] > planeCoordsEnd[iXYZ])
						t[iP] = ProjectionPointToPlane(t[iP], pointsOnPlaneEnd[iXYZ], planeNormsEnd[iXYZ]);
				}
			}
		}
	}

	// calculate volume of the obtained figure
	return CTriangularMesh{ "", triangles }.Volume();
}

void CPackageGenerator::UpdateReachedOverlap(SPackage& _generator)
{
	_generator.simulator->GetOverlapsInfo(_generator.maxReachedOverlap, _generator.avrReachedOverlap, _generator.generatedParticles);
}

void CPackageGenerator::Clear()
{
	for (auto& g : m_generators)
		ClearSimulator(g.simulator);
	m_generators.clear();
}

void CPackageGenerator::ClearSimulator(CBaseSimulator*& _simulator)
{
	if (!_simulator) return;
	const std::string fileName = _simulator->GetSystemStructure()->GetFileName();
	delete _simulator->GetGenerationManager();
	delete _simulator->GetModelManager();
	delete _simulator->GetSystemStructure();
	delete _simulator;
	_simulator = nullptr;
	MUSENFileFunctions::removeFile(fileName);
}

void CPackageGenerator::LoadConfiguration()
{
	Clear();
	const ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	m_simulatorType = static_cast<ESimulatorType>(protoMessage.package_generator().simulator_type());
	m_verletCoeff = protoMessage.package_generator().verlet_coeff() != 0 ? protoMessage.package_generator().verlet_coeff() : m_verletCoeff;
	for (int i = 0; i < protoMessage.package_generator().generators_size(); ++i)
	{
		const ProtoPackageGenerator& gen = protoMessage.package_generator().generators(i);
		m_generators.emplace_back();
		SPackage& pack = m_generators.back();
		pack.active           = gen.volume_active();
		pack.name             = gen.volume_name();
		pack.mixtureKey       = gen.mixture_key();
		pack.volumeKey        = gen.volume_key();
		pack.targetPorosity   = gen.porosity();
		pack.targetMaxOverlap = gen.max_overlap();
		pack.maxIterations    = gen.max_iterations();
		pack.insideGeometry   = gen.inside_geometry();
		pack.initVelocity     = Proto2Val(gen.init_velocity());
	}
}

void CPackageGenerator::SaveConfiguration()
{
	ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	protoMessage.mutable_package_generator()->clear_generators();
	protoMessage.mutable_package_generator()->set_simulator_type(E2I(m_simulatorType));
	protoMessage.mutable_package_generator()->set_verlet_coeff(m_verletCoeff);
	for (const auto& generator : m_generators)
	{
		ProtoPackageGenerator* pack = protoMessage.mutable_package_generator()->add_generators();
		pack->set_volume_active(generator.active);
		pack->set_volume_name(generator.name);
		pack->set_mixture_key(generator.mixtureKey);
		pack->set_volume_key(generator.volumeKey);
		pack->set_porosity(generator.targetPorosity);
		pack->set_max_overlap(generator.targetMaxOverlap);
		pack->set_max_iterations(generator.maxIterations);
		pack->set_inside_geometry(generator.insideGeometry);
		Val2Proto(pack->mutable_init_velocity(), generator.initVelocity);
	}
}

size_t CPackageGenerator::ParticlesToGenerate(size_t _index) const
{
	const SPackage* gen = Generator(_index);
	if (!gen) return 0;
	if (!m_pSystemStructure->m_MaterialDatabase.GetMixture(gen->mixtureKey)) return 0;
	if (!m_pSystemStructure->AnalysisVolume(gen->volumeKey)) return 0;
	std::vector<size_t> numbers = ParticlesPerFraction(*gen);
	return std::accumulate(numbers.begin(), numbers.end(), static_cast<size_t>(0));
}

std::ostream& operator<<(std::ostream& _s, const CPackageGenerator& _obj)
{
	return _s << E2I(_obj.m_simulatorType) << " " << _obj.m_verletCoeff;
}

std::istream& operator>>(std::istream& _s, CPackageGenerator& _obj)
{
	return _s >> S2E(_obj.m_simulatorType) >> _obj.m_verletCoeff;
}
