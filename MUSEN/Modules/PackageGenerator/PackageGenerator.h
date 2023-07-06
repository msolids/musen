/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BaseSimulator.h"
#include "GeneratorComponent.h"

// Information about generating package.
struct SPackage
{
	bool active{ true };				// Activity flag, if not set - the package will not be generated.
	std::string name{ "Generator" };	// Name of the generator.
	std::string mixtureKey{ "" };		// Unique key of the generating mixture.
	std::string volumeKey{ "" };		// Unique key of the volume, where the package will be generated.
	double targetPorosity{ 0.5 };		// Porosity, which must be reached during generation.
	double targetMaxOverlap{ 1e-7 };	// Maximum allowed overlap between each two generating particles.
	unsigned maxIterations{ 100000 };	// Maximum allowed number of iterations of the generation algorithm, after which generation will be stopped.
	CVector3 initVelocity{ 0 };			// Initial velocity of generated particles.
	bool insideGeometry{ false };		// Whether to generate objects inside real geometrical objects.

	size_t generatedParticles{ 0 };	// Number of generated particles at the current time of generation.
	double maxReachedOverlap{ 0 };	// Maximum overlap between particles at the current time of generation.
	double avrReachedOverlap{ 0 };	// Average overlap between particles at the current time of generation.
	double completness{ 0 };		// The ratio of the number of performed iterations to the number of maximum allowed iterations.

	CBaseSimulator* simulator{ nullptr };	// Pointer to a simulator used to generate particles.

	friend std::ostream& operator<<(std::ostream& _s, const SPackage& _obj)
	{
		return _s << MakeSingleString(static_cast<const std::string&>(_obj.name)) << " " << _obj.active << " "
			<< _obj.volumeKey << " " << _obj.mixtureKey << " " << _obj.targetPorosity << " " << _obj.targetMaxOverlap << " " << _obj.maxIterations << " "
			<< _obj.initVelocity << " " << _obj.insideGeometry;
	}

	friend std::istream& operator>>(std::istream& _s, SPackage& _obj)
	{
		return _s >> _obj.name >> _obj.active
			>> _obj.volumeKey >> _obj.mixtureKey >> _obj.targetPorosity >> _obj.targetMaxOverlap >> _obj.maxIterations
			>> _obj.initVelocity >> _obj.insideGeometry;
	}
};

// Generates packages from mixtures.
class CPackageGenerator final : public CMusenComponent, public IGenerator<SPackage>
{
	struct SParticles
	{
		std::vector<CVector3> coords;		// positions of particles
		std::vector<double> radii;			// radii of particles
		std::vector<double> contRadii;		// contact radii of particles
		std::vector<CQuaternion> orients;	// orientations of particles
		std::vector<const CCompound*> compounds;	// pointers to compounds of particles
		std::vector<size_t> fractions;		// fractions of particles

		void Add(const CVector3& _coordinate, double _radius, double _contRadius, const CQuaternion& _orientation, const CCompound* _compound, size_t _fraction)
		{
			coords.push_back(_coordinate);
			radii.push_back(_radius);
			contRadii.push_back(_contRadius);
			orients.push_back(_orientation);
			compounds.push_back(_compound);
			fractions.push_back(_fraction);
		}
	};

	inline static const std::string c_PPModelName = "ModelPPSimpleViscoElastic";
	inline static const std::string c_PWModelName = "ModelPWSimpleViscoElastic";

	ESimulatorType m_simulatorType{ ESimulatorType::CPU };	// Simulator type used to generate packages.
	double m_verletCoeff{ 2.0 };							// Verlet coefficient used in simulator.

public:
	CPackageGenerator() = default;
	CPackageGenerator(const CPackageGenerator& _other)                = delete;
	CPackageGenerator(CPackageGenerator&& _other) noexcept            = delete;
	CPackageGenerator& operator=(const CPackageGenerator& _other)     = delete;
	CPackageGenerator& operator=(CPackageGenerator&& _other) noexcept = delete;
	~CPackageGenerator();

	void SetSimulatorType(ESimulatorType _type);	// Sets type of the simulator used for generation.
	ESimulatorType GetSimulatorType() const;		// Returns type of the simulator used for generation.
	void SetVerletCoefficient(double _coeff);		// Sets Verlet coefficient used in simulator.
	double GetVerletCoefficient() const;			// Returns Verlet coefficient used in simulator.

	bool IsDataCorrect() const;			// Checks correctness of data in all generators.

	void StartGeneration();				// Starts generation process.

	void LoadConfiguration() override;	// Uses the same file as system structure to load configuration.
	void SaveConfiguration() override;	// Uses the same file as system structure to store configuration.

	size_t ParticlesToGenerate(size_t _index) const;	// Returns number of particles to be generated in the specified generator.
	void Clear();										// Clears and removes all generators.

	friend std::ostream& operator<<(std::ostream& _s, const CPackageGenerator& _obj);
	friend std::istream& operator>>(std::istream& _s, CPackageGenerator& _obj);

private:
	void Initialize(SPackage& _generator) const;																	// Initializes all data in the generator.
	void SetupSimulationDomain(SPackage& _generator, CSystemStructure& _scene) const;								// Calculates and set simulation domain.
	void PlaceParticlesInitially(SPackage& _generator, CSystemStructure& _scene) const;								// Randomly places particles in the specified volume.
	static void AddParticles(CSystemStructure& _scene, std::vector<size_t> _indices, const SParticles& _particles);	// Adds generated particles to the scene.
	void AddOverlayedParticles(SPackage& _generator, CSystemStructure& _scene) const;								// Adds already created particles which are situated in the generation volume.
	void AddGeometries(SPackage& _generator, CSystemStructure& _scene) const;										// Adds all required geometrical objects.
	void SetupSimulator(SPackage& _generator, CSystemStructure& _scene) const;										// Initializes all data in the simulator.
	void SaveGeneratedObjects(SPackage& _generator) const;		// Saves generated objects into the main system structure.

	std::vector<size_t> ParticlesPerFraction(const SPackage& _generator) const;	// Returns number of particles to be generated for each fraction.
	double VolumeToFill(const SPackage& _generator) const;						// Returns volume that must be filled with particles.
	static void UpdateReachedOverlap(SPackage& _generator);						// Calculated maximum and average reached overlaps.

	static void ClearSimulator(CBaseSimulator*& _simulator);	// Clears all pointers in the specified simulator.
};

