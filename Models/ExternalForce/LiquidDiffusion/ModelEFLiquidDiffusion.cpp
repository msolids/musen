/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelEFLiquidDiffusion.h"
#include "MixedFunctions.h"
#include "ThreadPool.h"
#include "MUSENDefinitions.h"

CModelEFLiquidDiffusion::CModelEFLiquidDiffusion()
{
	m_name = "Liquid Diffusion";
	m_uniqueKey = "57b926898e00d0c4350a9097fdf9cccdcd57be54";
	m_hasGPUSupport = true;

	/*  0 */ AddParameter("TEMPERATURE", "Temperature [K]", 300);
	/*  1 */ AddParameter("DYN_VISCOSITY_OF_LIQUID", "Dynamic viscosity of liquid [N*s/m^2]", 8.54163E-04);
	/*  2 */ AddParameter("STOKES_FACTOR", "Stokes factor r_s / r = XX", 1);
	/*  3 */ AddParameter("HYDR_SHELL", "Shell of hydration (added after Stokes factor", 0);
	/*  4 */ AddParameter("SEED_CHECK", "Perform seed uniqueness check Yes=1/No=0", 0);
	/*  5 */ AddParameter("FIXED_SEED", "Use same seed for every simulation Yes=1/No=0", 0);
	/*  6 */ AddParameter("ORIENT_SWITCH", "Orientation diffusion on Yes=1/No=0", 0);

	// Simulated annealing parameters
	/*  7 */ AddParameter("ANNEALING_TYPE", "No=0, Lin=1", 0);
	/*  8 */ AddParameter("ANNEALING_MAX_TEMP", "Maximum temperature for annealing [K]", 1000);
	/*  9 */ AddParameter("ANNEALING_COOLDOWN_TIME", "Cooldown time for one annealing procedure [s]", 8e-7);
	/* 10 */ AddParameter("ANNEALING_PERIOD", "Period of annealing procedures (time between repetitions) [s]", 1e-6);
	/* 11 */ AddParameter("ANNEALING_FINISHED_TIME", "Maximum/final time of annealing procedure [s]", 1e-3);

	// Saving flag
	/* 12 */ AddParameter("SAVE_FLAG", "Saving of kinetic energy and temperature. Yes=1/No=0", 0);
	/* 13 */ AddParameter("SAVE_STEP", "Saving step for kinetic energy and temperature [s]", 10E-9);

	// Shear flow
	/* 14 */ AddParameter("SHEAR_RATE", "Shear rate [1/s]. Flow in X, increasing shear in Z from Z=0.", 0);
	/* 15 */ AddParameter("DENSITY_LIQUID", "Density of liquid [kg/m^3]", 1000);
}

void CModelEFLiquidDiffusion::DetectAnnealing()
{
	// set annealing procedure
	m_annealingType = D2E<EAnnealing>(std::round(m_parameters[7].value));
	std::cout << "Detected annealing type: " << E2I(m_annealingType) << std::endl;
	switch (m_annealingType)
	{
	case EAnnealing::NONE:
		m_curTemp               = m_parameters[0].value;
		m_annealingEqTemp       = m_parameters[0].value;
		m_annealingMaxTemp      = m_parameters[0].value;
		m_annealingCooldownTime = 0;
		m_annealingPeriod       = 0;
		m_annealingFinishedTime = 0;
		break;
	case EAnnealing::LIN:
		m_curTemp               = m_parameters[0].value;
		m_annealingEqTemp       = m_parameters[0].value;
		m_annealingMaxTemp      = m_parameters[8].value;
		m_annealingCooldownTime = m_parameters[9].value;
		m_annealingPeriod       = m_parameters[10].value;
		m_annealingFinishedTime = m_parameters[11].value;
		break;
	default:
		std::cerr << "Annealing type not recognized. Found value " << E2I(m_annealingType) << " converted from input " << m_parameters[7].value << std::endl;
		m_annealingType         = EAnnealing::NONE;
		m_curTemp               = m_parameters[0].value;
		m_annealingEqTemp       = m_parameters[0].value;
		m_annealingMaxTemp      = m_parameters[0].value;
		m_annealingCooldownTime = 0;
		m_annealingPeriod       = 0;
		m_annealingFinishedTime = 0;
	}

	if (m_annealingType != EAnnealing::NONE)
	{
		std::cout << std::endl;
		std::cout << "Simulated annealing procedure found. Properties:"         << std::endl;
		std::cout << "Equilibrium temperature [K]: " << m_annealingEqTemp       << std::endl;
		std::cout << "Maximum temperature [K]:     " << m_annealingMaxTemp      << std::endl;
		std::cout << "Cooldown time [s]:           " << m_annealingCooldownTime << std::endl;
		std::cout << "Annealing period [s]:        " << m_annealingPeriod       << std::endl;
		std::cout << "Annealing final time [K]:    " << m_annealingFinishedTime << std::endl;
		std::cout << std::endl;
	}
}

//////////////////////////////////////////////////////////////////////////
/// CPU Implementation
//////////////////////////////////////////////////////////////////////////

void CModelEFLiquidDiffusion::PrecalculateEF(double _time, double _timeStep, SParticleStruct* _particles)
{
	if (m_nStatesCPU != _particles->Size()) // check if the current number of states in storage is not equal to the particle number (desired number of states)
	{
		DetectAnnealing();
		InitializeDiffusionModelCPU(_timeStep);
	}

	// update temperature if necessary
	if (m_annealingType != EAnnealing::NONE)
	{
		if (_time >= m_annealingFinishedTime) // set temperature once annealing is over
		{
			if (_time < m_annealingFinishedTime + 2 * _timeStep)
			{
				m_curTemp = m_annealingEqTemp;
				UpdateTemperatureCPU(m_curTemp);
			}
		}
		else
		{
			if (m_annealingType == EAnnealing::LIN)
			{
				m_curTemp = m_annealingEqTemp;
				const double relAnnealingTime = std::fmod(_time, m_annealingPeriod);
				if (relAnnealingTime < m_annealingCooldownTime)
					m_curTemp = m_annealingEqTemp + (m_annealingMaxTemp - m_annealingEqTemp) * (1 - relAnnealingTime / m_annealingCooldownTime);
				UpdateTemperatureCPU(m_curTemp);
			}
		}
	}

	// output global data
	if (m_saveFlag)
	{
		if (std::fabs(_time - m_lastSavingTime) + 0.2*_timeStep > m_savingStep)
		{
			// set saving step
			const size_t partNum = Particles().Size();
			m_lastSavingTime = _time;
			const unsigned outPerc = 6;

			// global kinetic energy
			size_t threadsNumber = GetThreadsNumber();
			std::vector<double> kinEnergies(threadsNumber, 0);
			// std::vector<double> vTransEnergies(threadsNumber, 0);
			// std::vector<double> vAngEnergies(threadsNumber, 0);
			ParallelFor(threadsNumber, [&](size_t t)
			{
				const size_t iBeg = t * (partNum/threadsNumber + 1);
				const size_t iEnd = iBeg + std::min(partNum/threadsNumber + 1, partNum - iBeg);
				for (size_t i = iBeg; i < iEnd; i++)
				{
					kinEnergies[t] += 0.5 * (Particles().Mass(i) * Particles().Vel(i).SquaredLength() + Particles().InertiaMoment(i) * Particles().AnglVel(i).SquaredLength());
					// vTransEnergies[t] += 0.5 * Particles().Mass(i) * Particles().Vel(i).SquaredLength();
					// vAngEnergies[t] += 0.5 * Particles().InertiaMoment(i) * Particles().AnglVel(i).SquaredLength();
				}
			});
			double kinEnergy = 0;
			// double trans_energy = 0;
			// double ang_energy = 0;
			for (size_t i = 0; i < threadsNumber; ++i)
			{
				kinEnergy += kinEnergies[i];
				// trans_energy += vTransEnergies[i];
				// ang_energy += vAngEnergies[i];
			}
			// std::cout << "Translational energy: " << trans_energy << std::endl;
			// std::cout << "Angular energy: " << ang_energy << std::endl;
			m_fileKineticEnergy << std::setprecision(outPerc) << _time << ";" << kinEnergy << std::endl;

			// temperature
			printf("Temperature at time %6e: %6.2f K\n", _time, m_curTemp);
			m_fileTemperature << std::setprecision(outPerc) << _time << ";" << m_curTemp << std::endl;
		}
	}
}

void CModelEFLiquidDiffusion::CalculateEF(double _time, double _timeStep, size_t _iPart, SParticleStruct& _particles) const
{
	// Diffusion model. See Depta et al. J. Chem. Inf. Model, 59 (2019) 1, 386-398, DOI: 10.1021/acs.jcim.8b00613
	const unsigned partIndex = m_pParticleIndexMapCPU[Particles().InitIndex(_iPart)];

	//// Random Number Generation: Seed is only calculated at beginning for every particle
	std::normal_distribution<double> normalDist(0, 1);
	const double ranFx = normalDist(m_pGeneratorsCPU[partIndex]);
	const double ranFy = normalDist(m_pGeneratorsCPU[partIndex]);
	const double ranFz = normalDist(m_pGeneratorsCPU[partIndex]);

	// Get local flow velocity
	const double flowVelocityX = m_parameters[14].value * abs(Particles().Coord(_iPart).z);

	// Calculate diffusion force and moment in body frame
	const CVector3 vGlobal = Particles().Vel(_iPart);
	CVector3 FDifGlobal;
	FDifGlobal.x = m_temperatureCoefCPU*m_pFMagnitudeCPU[partIndex]*ranFx - m_pDragLinCPU[partIndex]*(Particles().Vel(_iPart).x - flowVelocityX);
	FDifGlobal.y = m_temperatureCoefCPU*m_pFMagnitudeCPU[partIndex]*ranFy - m_pDragLinCPU[partIndex]*vGlobal.y;
	FDifGlobal.z = m_temperatureCoefCPU*m_pFMagnitudeCPU[partIndex]*ranFz - m_pDragLinCPU[partIndex]*vGlobal.z;

	// Calculate Saffman lift force, if positive
	const double liftForce = 6.46 * m_parameters[15].value * sqrt(m_parameters[1].value / m_parameters[15].value) * (flowVelocityX - Particles().Vel(_iPart).x) * Particles().Radius(_iPart) * Particles().Radius(_iPart) * sqrt(abs(m_parameters[14].value));
	if (Particles().Coord(_iPart).z > 0)	// differentiate because shear rate has different directions
		FDifGlobal.z += liftForce;
	else
		FDifGlobal.z -= liftForce;

	// Add to particle force
	_particles.Force(_iPart) += FDifGlobal;

	// moment
	if (m_parameters[6].value == 1)
	{
		const double ranMx = normalDist(m_pGeneratorsCPU[partIndex]);
		const double ranMy = normalDist(m_pGeneratorsCPU[partIndex]);
		const double ranMz = normalDist(m_pGeneratorsCPU[partIndex]);
		const CVector3 wGlobal = Particles().AnglVel(_iPart);
		CVector3 MDifGlobal;
		MDifGlobal.x = m_temperatureCoefCPU*m_pMMagnitudeCPU[partIndex]*ranMx - m_pDragRotCPU[partIndex]*wGlobal.x;
		MDifGlobal.y = m_temperatureCoefCPU*m_pMMagnitudeCPU[partIndex]*ranMy - m_pDragRotCPU[partIndex]*wGlobal.y;
		MDifGlobal.z = m_temperatureCoefCPU*m_pMMagnitudeCPU[partIndex]*ranMz - m_pDragRotCPU[partIndex]*wGlobal.z;
		_particles.Moment(_iPart) += MDifGlobal;
	}
}


//////////////////////////////////////////////////////////////////////////
/// SUB-FUNCTIONS
//////////////////////////////////////////////////////////////////////////
void CModelEFLiquidDiffusion::InitializeDiffusionModelCPU(double _timeStep)
{
	std::cout << "For using this diffusion model please cite: Depta et al. J. Chem. Inf. Model, 59 (2019) 1, 386-398, DOI: 10.1021/acs.jcim.8b00613" << std::endl;

	// Open output files
	if (m_parameters[12].value == 1.0 && !m_saveFlag)
	{
		m_saveFlag = true;
		m_savingStep = m_parameters[13].value;
		if (!m_fileKineticEnergy.is_open())
			m_fileKineticEnergy.open("./Post_KineticEnergy.csv", std::ofstream::out);
		if (!m_fileTemperature.is_open())
			m_fileTemperature.open("./Post_Temperature.csv", std::ofstream::out);
	}

	// Check input parameters
	if (m_parameters[4].value == 1.0)
		m_checkSeeds = true;

	// Check PBC if shear flow is enabled
	if (m_parameters[14].value != 0.0 && m_PBC.bZ)
		if (abs(m_PBC.currentDomain.coordBeg.z + m_PBC.currentDomain.coordEnd.z) > 1e-6 * (m_PBC.currentDomain.coordEnd.z - m_PBC.currentDomain.coordBeg.z))
			std::cerr << "Error: Using shear flow, but periodic boundaries in Z-direction seem to be non-symmetric. This will cause unphysical results." << std::endl;

	// Desired number of states for random numbers - each for every particle
	const size_t desNumStates = Particles().Size();

	// Create index map for CPU indexes (not necessary on GPU as all particles are in consecutive order)
	CreateIndexMapCPU();

	// Calculate diffusion model coefficients (forces, moments, and drag coefficients) for all enzyme kinds
	CalculateDiffusionModelCoefficientsCPU(_timeStep);

	// Initialize seeds and random number generators for the desired number of states
	InitializeSeedsAndRandomGeneratorsCPU(desNumStates);

	m_nStatesCPU = desNumStates; // Set number of states present in memory to desired number of states --> ensures if is only entered at beginning and if particle number changes
}

void CModelEFLiquidDiffusion::UpdateTemperatureCPU(double newTemperature)
{
	m_temperatureCoefCPU = std::sqrt(newTemperature/m_parameters[0].value);
}

void CModelEFLiquidDiffusion::CreateIndexMapCPU()
{
	// Function creates mapping from nInitialIndex in particle struct to index in this model
	const size_t numPart = Particles().Size();
	if (m_nStatesCPU != 0)
		delete[] m_pParticleIndexMapCPU; // free if previous data has been allocated

	// Determine maximal nInitialIndex for array
	m_maxPartIndex = static_cast<unsigned>(numPart); // assign to this length at beginning to not have to write at every iteration
	for (size_t i = 0; i < numPart; i++)
	{
		if (Particles().InitIndex(i) > m_maxPartIndex)
			m_maxPartIndex = Particles().InitIndex(i);
	}
	m_maxPartIndex++; // has to be increased by one to allow storage of the last one
	std::cout << "m_maxPartIndex: " << m_maxPartIndex << std::endl;
	m_pParticleIndexMapCPU = new unsigned[m_maxPartIndex]; // allocate array for maximal number of indexes
	std::fill_n(m_pParticleIndexMapCPU, m_maxPartIndex, UINT_MAX);

	// Create map
	for (unsigned iPart = 0; iPart < numPart; iPart++)
		m_pParticleIndexMapCPU[Particles().InitIndex(iPart)] = iPart; // map nInitialIndex -> iPart
}

void CModelEFLiquidDiffusion::CalculateDiffusionModelCoefficientsCPU(double _timeStep)
{
	// Diffusion model coefficients. Calculate diffusive variables (forces, moments, and drag coefficients) for all particles
	std::cout << "Calculating diffusion coefficients ..." << std::endl;

	// Input parameters
	const double temperature      = m_parameters[0].value;	// User input of temperature
	const double dynamicViscosity = m_parameters[1].value;	// User input of dynamic viscosity of liquid
	const double stokesFactor     = m_parameters[2].value;	// User input of dynamic viscosity of liquid
	const double hydrShell        = m_parameters[3].value;	// User input of dynamic viscosity of liquid

	// Determine maximal nInitialIndex for array
	m_maxPartIndex = static_cast<unsigned>(Particles().Size()); // assign to this length at beginning to not have to write at every iteration
	for (size_t i = 0; i < Particles().Size(); i++)
	{
		if (Particles().InitIndex(i) > m_maxPartIndex)
			m_maxPartIndex = Particles().InitIndex(i);
	}
	m_maxPartIndex++; // has to be increased by one to allow storage of the last one

	// allocate memory
	if (m_nStatesCPU != 0) // free if previous data has been allocated
	{
		delete[] m_pDragLinCPU;
		delete[] m_pFMagnitudeCPU;
		delete[] m_pDragRotCPU;
		delete[] m_pMMagnitudeCPU;
	}
	m_pDragLinCPU    = new double[m_maxPartIndex];
	m_pFMagnitudeCPU = new double[m_maxPartIndex];
	m_pDragRotCPU    = new double[m_maxPartIndex];
	m_pMMagnitudeCPU = new double[m_maxPartIndex];
	auto* pTauMinLin = new double[m_maxPartIndex];
	auto* pTauMinRot = new double[m_maxPartIndex];

	// Calculate all coefficients
	ParallelFor(Particles().Size(), [&](size_t iPart)
	{
		const size_t i = Particles().InitIndex(iPart);		// initial index for accessing from index map
		const double r_stokes = stokesFactor * Particles().Radius(iPart) + hydrShell;
		const double partMass = Particles().Mass(iPart);
		const double partInerMom = Particles().InertiaMoment(iPart);

		// Calculate translation coefficients
		double temp = 6 * PI * dynamicViscosity * r_stokes * _timeStep / partMass;
		m_pDragLinCPU[i] = partMass / _timeStep * (1 - std::exp(-1 * temp));
		m_pFMagnitudeCPU[i] = std::sqrt(partMass * BOLTZMANN_CONSTANT * temperature * (1 - std::exp(-2 * temp))) / _timeStep;

		// Calculate rotational coefficients
		temp = 8 * PI * dynamicViscosity * pow(r_stokes,3) * _timeStep / partInerMom;
		m_pDragRotCPU[i] = partInerMom / _timeStep * (1 - std::exp(-1 * temp));
		m_pMMagnitudeCPU[i] = std::sqrt(partInerMom * BOLTZMANN_CONSTANT * temperature * (1 - std::exp(-2 * temp))) / _timeStep;

		// Critical time constants
		pTauMinLin[i] = partMass / (6 * PI * dynamicViscosity * r_stokes);
		pTauMinRot[i] = partInerMom / (8 * PI * dynamicViscosity * pow(r_stokes,3));

		//printf("%e;%e;%e;%e\n",m_pDragLinCPU[i],m_pFMagnitudeCPU[i],m_pDragRotCPU[i],m_pMMagnitudeCPU[i]);
	});

	// Get minimum critical time constant
	m_tauMinLin = std::numeric_limits<double>::max();
	m_tauMinRot = std::numeric_limits<double>::max();
	for (size_t iPart = 0; iPart < Particles().Size(); iPart++)
	{
		const size_t i = Particles().InitIndex(iPart); // initial index for accessing from index map
		if (m_tauMinLin > pTauMinLin[i])
			m_tauMinLin = pTauMinLin[i];
		if (m_tauMinRot > pTauMinRot[i])
			m_tauMinRot = pTauMinRot[i];
	}
	std::cout << "Minimum critical time constant for diffusion (translation): " << m_tauMinLin << " [s]" << std::endl;
	if (m_parameters[6].value == 1)
		std::cout << "Minimum critical time constant for diffusion (rotation): " << m_tauMinRot << " [s]" << std::endl;
	delete[] pTauMinLin;
	delete[] pTauMinRot;
}

void CModelEFLiquidDiffusion::InitializeSeedsAndRandomGeneratorsCPU(size_t _desNumStates)
{
	// Free previous states of random number generators
	if (m_nStatesCPU != 0)
		delete[] m_pGeneratorsCPU;

	// 64bit seeds
	m_pSeedsCPU = new uint64_t[_desNumStates];
	CreateUniqueSeeds64(m_pSeedsCPU, static_cast<unsigned>(_desNumStates));

	// Create generators from seeds
	m_pGeneratorsCPU = new std::mt19937_64[_desNumStates];
	for (size_t i = 0; i < _desNumStates; ++i)
	{
		std::mt19937_64 generator;
		generator.seed(m_pSeedsCPU[i]);
		m_pGeneratorsCPU[i] = generator;
	}

	// free seed memory - not needed any more as generators exist now
	delete[] m_pSeedsCPU;
}

void CModelEFLiquidDiffusion::CreateUniqueSeeds64(uint64_t* _pSeeds64, unsigned _desNumStates)
{
	// 32 bit
	std::random_device r32;
	// 64 bit
	std::mt19937_64 r64(r32());
	if (m_parameters[5].value != 0.0)
		r64.seed(static_cast<size_t>(m_parameters[5].value));

	// For output
	std::cout << std::endl << " ===== Starting seed generation ===== " << std::endl << "Desired seed states: " << _desNumStates << std::endl;
	if (m_parameters[5].value != 0.0)
		std::cout << "WARNING: Fixed seed is used. Use with caution. Diffusion model will always generate same random forces/torques for each particle." << std::endl;
	if (m_checkSeeds)
		std::cout << "Checking for seed duplicates: ENABLED" << std::endl << "64 bit seeds used. Checking likely not necessary." << std::endl;
	else
		std::cout << "Checking for seed duplicates: DISABLED" << std::endl << "64 bit seeds used. Issues unlikely." << std::endl;
	//clock_t tLast = clock();
	time_t tLast = time(nullptr);
	int progressPerc = 0;
	const int progressIntervalPerc = 10;
	const int modFactor = _desNumStates / (100 / progressIntervalPerc);
	unsigned totalDuplicatesFound = 0;

	// Initialize thread pool
	const size_t threadsNumber = GetThreadsNumber();
	bool* pThreadFlags = new bool[threadsNumber];
	std::fill_n(pThreadFlags, threadsNumber, false);
	auto* pThreadRegions = new size_t[threadsNumber];

	// Create random seeds
	for (unsigned i = 0; i < _desNumStates; ++i)
	{
		uint64_t tempSeed = r64()*r64() + r64() + i*i*i + i*i + i;	// create initial seed
		if (m_checkSeeds)
		{
			bool check = true;										// set need to check to true
			while (check)											// as long as seed check is not false re-calculate seeds
			{
				unsigned n_ranSame = 0;
				if (i < 10000)
				{
					for (unsigned a = 0; a < i; a++)				// go over all previous seeds
					{
						if (_pSeeds64[a] == tempSeed)
						{
							n_ranSame++;							// check if any seed with value tempSeed exists
							break;
						}
					}
				}
				else
				{
					// Distribute array over thread pool
					const size_t sizePerThread = ((i - 1) - (i - 1) % threadsNumber) / threadsNumber;
					for (size_t a = 0; a < threadsNumber; a++)
					{
						pThreadRegions[2 * a] = a * sizePerThread;
						pThreadRegions[2 * a + 1] = (a + 1) * sizePerThread - 1;
					}
					pThreadRegions[2 * threadsNumber - 1] = i - 1;

					// Start thread pool and execute search
					ParallelFor(threadsNumber, [&](size_t t)
					{
						for (size_t j = pThreadRegions[2 * t]; j < pThreadRegions[2 * t + 1]; j++)
						{
							if (_pSeeds64[j] == tempSeed)
							{
								pThreadFlags[t] = true;				// check if any seed with value tempSeed exists
								break;
							}
						}
					});

					// Reduction
					for (size_t red = 0; red < threadsNumber; red++)
					{
						if (pThreadFlags[red] == true)
						{
							n_ranSame++;
							totalDuplicatesFound++;
						}
						pThreadFlags[red] = false;					// reset flag
					}

				}

				if (n_ranSame == 0)
					check = false;									// if there are no duplicates present -> exit while loop
				else
					tempSeed = r64()*r64() + r64() + i*i*i + i*i + i;	// else generate new seed
			}
			_pSeeds64[i] = tempSeed;
		}
		else
			_pSeeds64[i] = tempSeed;

		// Progress in output file
		if (_desNumStates > 1000)									// avoid singularity of mod_factor
		{
			if ((i + 1) % modFactor == 0)
			{
				progressPerc = progressPerc + progressIntervalPerc;
				//clock_t tCurrent = clock();
				const time_t tCurrent = time(nullptr);
				const auto tElapsed = static_cast<double>(tCurrent - tLast);
				std::cout << progressPerc << " % of desired seed states generated. Elapsed time since last print: " << tElapsed << " s" << std::endl;
				//tLast = clock();
				tLast = time(nullptr);
			}
		}
	}

	if (m_checkSeeds)
		std::cout << "Total number of duplicates detected during seed creation: " << totalDuplicatesFound << std::endl;
	std::cout << " ===== Finished seed generation ===== " << std::endl << std::endl << " ===== Starting time-stepping ===== " << std::endl;

	// Free previously allocated memory
	delete[] pThreadFlags;
	delete[] pThreadRegions;
}

void CModelEFLiquidDiffusion::CreateUniqueSeeds32(uint32_t * pSeeds32, unsigned _desNumStates)
{
	// 32 bit
	std::random_device r32;

	// For output
	std::cout << std::endl << " ===== Starting seed generation ===== " << std::endl << "Desired seed states: " << _desNumStates << std::endl;
	if (m_checkSeeds)
		std::cout << "Checking for seed duplicates: ENABLED" << std::endl << "32 bit seeds used. Checking likely necessary." << std::endl;
	else
		std::cout << "Checking for seed duplicates: DISABLED" << std::endl << "32 bit seeds used. Issues are likely. Consider changing to checking-mode." << std::endl;
	//clock_t tLast = clock();
	time_t tLast = time(nullptr);
	int progressPerc = 0;
	const int progressIntervalPerc = 10;
	const int modFactor = _desNumStates / (100 / progressIntervalPerc);
	unsigned totalDuplicatesFound = 0;

	// Initialize thread pool
	const size_t threadsNumber = GetThreadsNumber();
	bool* pThreadFlags = new bool[threadsNumber];
	std::fill_n(pThreadFlags, threadsNumber, false);
	auto* pThreadRegions = new size_t[2 * threadsNumber];

	// Create random seeds
	for (unsigned i = 0; i < _desNumStates; i++)
	{
		uint32_t tempSeed = r32()*r32() + r32() + i*i*i + i*i + i;		// create initial seed

		if (m_checkSeeds)
		{
			bool check = true;											// set need to check to true
			while (check)												// as long as seed check is not false re-calculate seeds
			{
				unsigned n_ranSame = 0;

				if (i < 10000)
				{
					for (unsigned a = 0; a < i; a++)					// go over all previous seeds
					{
						if (pSeeds32[a] == tempSeed)
						{
							n_ranSame++;								// check if any seed with value tempSeed exists
							break;
						}
					}
				}
				else
				{
					// Distribute array over thread pool
					const size_t sizePerThread = ((i - 1) - (i - 1) % threadsNumber) / threadsNumber;
					for (size_t a = 0; a < threadsNumber; a++)
					{
						pThreadRegions[2 * a] = a * sizePerThread;
						pThreadRegions[2 * a + 1] = (a + 1) * sizePerThread - 1;
					}
					pThreadRegions[2 * threadsNumber - 1] = i - 1;

					// Start thread pool and execute search
					ParallelFor(threadsNumber, [&](size_t t)
					{
						for (size_t j = pThreadRegions[2 * t]; j < pThreadRegions[2 * t + 1]; j++)
						{
							if (pSeeds32[j] == tempSeed)
							{
								pThreadFlags[t] = true;					// check if any seed with value tempSeed exists
								break;
							}
						}
					});

					// Reduction
					for (size_t red = 0; red < threadsNumber; red++)
					{
						if (pThreadFlags[red] == true)
						{
							n_ranSame++;
							totalDuplicatesFound++;
						}
						pThreadFlags[red] = false;						// reset flag
					}

				}

				if (n_ranSame == 0)
					check = false;										// if there are no duplicates present -> exit while loop
				else
					tempSeed = r32()*r32() + r32() + i*i*i + i*i + i;	// else generate new seed
			}
			pSeeds32[i] = tempSeed;
		}
		else
			pSeeds32[i] = tempSeed;

		// Progress in output file
		if (_desNumStates > 1000)										// avoid singularity of mod_factor
		{
			if ((i + 1) % modFactor == 0)
			{
				progressPerc = progressPerc + progressIntervalPerc;
				//clock_t tCurrent = clock();
				const time_t tCurrent = time(nullptr);
				const double tElapsed = static_cast<double>(tCurrent - tLast);
				std::cout << progressPerc << " % of desired seed states generated. Elapsed time since last print: " << tElapsed << " s" << std::endl;
				//tLast = clock();
				tLast = time(nullptr);
			}
		}
	}

	if (m_checkSeeds)
		std::cout << "Total number of duplicates detected during seed creation: " << totalDuplicatesFound << std::endl;
	std::cout << " ===== Finished seed generation ===== " << std::endl << std::endl << " ===== Starting time-stepping ===== " << std::endl;

	// Free previously allocated memory
	delete[] pThreadFlags;
	delete[] pThreadRegions;
}
