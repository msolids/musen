/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ScriptRunner.h"
#include "ConsoleResultsAnalyzer.h"
#include "ConsoleSimulator.h"
#include "FileConverter.h"
#include "ImportFromText.h"
#include "ExportAsText.h"
#include "BondsGenerator.h"
#include "PackageGenerator.h"

CScriptRunner::CScriptRunner() : m_out(std::cout.rdbuf()), m_err(std::cerr.rdbuf())
{
	m_out.copyfmt(std::cout);				// copy everything except rdstate and rdbuf
	m_out.clear(std::cout.rdstate());	// copy rdstate
	m_err.copyfmt(std::cerr);				// copy everything except rdstate and rdbuf
	m_err.clear(std::cerr.rdstate());	// copy rdstate
}

void CScriptRunner::RunScriptJob(const SJob& _job)
{
	m_job = _job;

	m_out << "Source file: " << m_job.sourceFileName << std::endl;
	m_out << "Result file: " << m_job.resultFileName << std::endl;

	switch (m_job.component)
	{
	case SJob::EComponent::SIMULATOR:          PerformSimulation();	break;
	case SJob::EComponent::PACKAGE_GENERATOR:  GeneratePackage();	break;
	case SJob::EComponent::BONDS_GENERATOR:    GenerateBonds();		break;
	case SJob::EComponent::RESULTS_ANALYZER:   AnalyzeResults();	break;
	case SJob::EComponent::SNAPSHOT_GENERATOR: GenerateSnapshot();	break;
	case SJob::EComponent::EXPORT_TO_TEXT:     ExportToText();		break;
	case SJob::EComponent::IMPORT_FROM_TEXT:   ImportFromText();	break;
	case SJob::EComponent::COMPARE_FILES:      CompareFiles();		break;
	}
}

void CScriptRunner::PerformSimulation()
{
	m_out << "Selected component: Simulator" << std::endl << std::endl;

	// file I/O
	if (!LoadAndResaveSystemStructure()) return;

	// set material parameters
	ApplyMaterialParameters();

	CConsoleSimulator simulator(m_systemStructure, m_out, m_err);
	simulator.Simulate(&m_job);
}

void CScriptRunner::GeneratePackage()
{
	m_out << "Selected component: Package generator" << std::endl << std::endl;

	// file I/O
	if (!LoadAndResaveSystemStructure()) return;

	// set material parameters
	ApplyMaterialParameters();

	// delete all existing particles
	m_systemStructure.DeleteAllParticles();

	// setup generators
	CPackageGenerator generator;
	generator.SetSystemStructure(&m_systemStructure);
	generator.LoadConfiguration();
	if (m_job.verletCoef != 0)
		generator.SetVerletCoefficient(m_job.verletCoef);
	if (m_job.simulatorType != ESimulatorType::BASE)
		generator.SetSimulatorType(m_job.simulatorType);
	for (const auto& g : m_job.packageGenerators)
	{
		const size_t index = g.first - 1;
		while (index >= generator.GeneratorsNumber())
			generator.AddGenerator({});
		if (!g.second.volume.empty())
		{
			if (const auto* volume = m_systemStructure.AnalysisVolumeByName(g.second.volume))
				generator.Generator(index)->volumeKey = volume->Key();
			else
				generator.Generator(index)->volumeKey = g.second.volume;
		}
		if (!g.second.mixture.empty())
		{
			if (const auto* mixture = m_systemStructure.m_MaterialDatabase.GetMixtureByName(g.second.mixture))
				generator.Generator(index)->mixtureKey = mixture->GetKey();
			else
				generator.Generator(index)->mixtureKey = g.second.mixture;
		}
		if (g.second.porosity != 0.0)		generator.Generator(index)->targetPorosity   = g.second.porosity;
		if (g.second.overlap != 0.0)		generator.Generator(index)->targetMaxOverlap = g.second.overlap;
		if (g.second.iterations != 0)		generator.Generator(index)->maxIterations    = (unsigned)g.second.iterations;
		if (!g.second.velocity.IsInf())		generator.Generator(index)->initVelocity     = g.second.velocity;
		if (g.second.inside.IsDefined())	generator.Generator(index)->insideGeometry   = g.second.inside.ToBool();
	}

	// check data correctness
	if (!generator.IsDataCorrect())
	{
		m_err << generator.ErrorMessage();
		return;
	}

	m_out << "Generation started" << std::endl;
	generator.StartGeneration();
	m_out << "Generation finished" << std::endl;
	m_out << "Saving started" << std::endl;
	generator.SaveConfiguration();
	m_systemStructure.SaveToFile(m_job.resultFileName);
	m_out << "Saving finished" << std::endl;
}

void CScriptRunner::GenerateBonds()
{
	m_out << "Selected component: Bonds generator" << std::endl << std::endl;

	// file I/O
	if (!LoadAndResaveSystemStructure()) return;

	// delete all existing bonds
	m_systemStructure.DeleteAllBonds();

	// setup generators
	CBondsGenerator generator;
	generator.SetSystemStructure(&m_systemStructure);
	generator.LoadConfiguration();
	for (const auto& g : m_job.bondGenerators)
	{
		const size_t index = g.first - 1;
		while (index >= generator.GeneratorsNumber())
			generator.AddGenerator({});
		if (!g.second.material.empty())
		{
			if (const auto* material = m_systemStructure.m_MaterialDatabase.GetCompoundByName(g.second.material))
				generator.Generator(index)->compoundKey = material->GetKey();
			else
				generator.Generator(index)->compoundKey = g.second.material;
		}
		if (std::isfinite(g.second.minDistance))	generator.Generator(index)->minDistance      = g.second.minDistance;
		if (std::isfinite(g.second.maxDistance))	generator.Generator(index)->maxDistance      = g.second.maxDistance;
		if (g.second.diameter != 0.0)				generator.Generator(index)->diameter         = g.second.diameter;
		if (g.second.overlay.IsDefined())			generator.Generator(index)->isOverlayAllowed = g.second.overlay.ToBool();
	}

	// check data correctness
	if (!generator.IsDataCorrect())
	{
		m_out << generator.ErrorMessage() << std::endl;
		return;
	}

	m_out << "Generation started" << std::endl;
	generator.StartGeneration();
	m_out << "Generation finished" << std::endl;
	m_out << "Saving started" << std::endl;
	generator.SaveConfiguration();
	m_systemStructure.SaveToFile(m_job.resultFileName);
	m_out << "Saving finished" << std::endl;
}

void CScriptRunner::AnalyzeResults()
{
	m_out << "Selected component: Results analyzer" << std::endl << std::endl;

	// load source file
	if (!LoadSourceFile()) return;

	m_out << "Analysis started" << std::endl;
	CConsoleResultsAnalyzer analyzer(m_out, m_err);
	analyzer.EvaluateResults(m_job, m_systemStructure);
	m_out << "Analysis finished" << std::endl;
}

void CScriptRunner::GenerateSnapshot()
{
	m_out << "Selected component: Snapshot generator" << std::endl << std::endl;
	m_out << "Snapshot time point: " << m_job.dSnapshotTP << std::endl;

	// check files names
	if (m_job.sourceFileName == m_job.resultFileName)
	{
		m_out << "Error: Cannot generate snapshot: source file and result file are the same." << std::endl;
		return;
	}

	// load source file
	if (!LoadSourceFile()) return;

	m_out << "Generation started" << std::endl;

	// create snapshot file
	CSystemStructure snapshot;
	snapshot.SaveToFile(m_job.resultFileName);
	snapshot.CreateFromSystemStructure(&m_systemStructure, m_job.dSnapshotTP);

	// create, load and save all managers
	CSimulatorManager simulatorManager;
	CPackageGenerator packageGenerator;
	CBondsGenerator bondsGenerator;
	CGenerationManager generationManager;
	CModelManager modelsManager;
	std::vector<CMusenComponent*> componets { &simulatorManager, &packageGenerator, &bondsGenerator, &generationManager, &modelsManager };
	for (CMusenComponent* c : componets)
	{
		c->SetSystemStructure(&m_systemStructure);
		c->LoadConfiguration();

		// switch off dynamic generators in snapshot
		if (auto* gm = dynamic_cast<CGenerationManager*>(c))
			for (size_t i = 0; i < gm->GetGeneratorsNumber(); ++i)
				gm->GetGenerator(i)->m_bActive = false;

		c->SetSystemStructure(&snapshot);
		c->SaveConfiguration();
	}

	// final save
	snapshot.SaveToFile(m_job.resultFileName);

	m_out << "Generation finished" << std::endl;
}

void CScriptRunner::ExportToText()
{
	m_out << "Selected component: Text exporter" << std::endl << std::endl;

	// load source file
	if (!LoadSourceFile()) return;

	m_out << "Export to text started" << std::endl;
	CPackageGenerator packageGenerator;
	packageGenerator.SetSystemStructure(&m_systemStructure);
	packageGenerator.LoadConfiguration();
	CBondsGenerator bondsGenerator;
	bondsGenerator.SetSystemStructure(&m_systemStructure);
	bondsGenerator.LoadConfiguration();
	CExportAsText exporter;
	CConstraints constraints;
	exporter.SetPointers(&m_systemStructure, &constraints, &packageGenerator, &bondsGenerator);
	exporter.SetFileName(m_job.resultFileName);
	exporter.SetTimePoints(m_systemStructure.GetAllTimePoints());
	exporter.SetFlags(m_job.txtExportObjects, m_job.txtExportScene, m_job.txtExportConst, m_job.txtExportTD, m_job.txtExportGeometries, m_job.txtExportMaterials, m_job.txtExportGenerators);
	exporter.SetPrecision(m_job.txtPrecision);
	exporter.Export();
	m_out << "Export to text finished" << std::endl;
}

void CScriptRunner::ImportFromText()
{
	m_out << "Selected component: Text importer" << std::endl << std::endl;

	m_out << "Import from text started" << std::endl;
	m_systemStructure.SaveToFile(m_job.resultFileName);
	CPackageGenerator packageGenerator;
	CBondsGenerator bondsGenerator;
	packageGenerator.SetSystemStructure(&m_systemStructure);
	bondsGenerator.SetSystemStructure(&m_systemStructure);
	CImportFromText importer(&m_systemStructure, &packageGenerator, &bondsGenerator);
	importer.Import(m_job.sourceFileName);
	packageGenerator.SaveConfiguration();
	bondsGenerator.SaveConfiguration();
	m_systemStructure.SaveToFile(m_job.resultFileName);
	m_out << "Import from text finished" << std::endl;
}


void CScriptRunner::CompareFiles()
{
	m_out << "Selected component: File comparison" << std::endl << std::endl;

	std::ofstream outStream;
	outStream.open(m_job.logFileName, std::ios::app);
	CSystemStructure scene1, scene2;
	const bool loaded1 = LoadMusenFile(m_job.sourceFileName, scene1);
	const bool loaded2 = LoadMusenFile(m_job.resultFileName, scene2);
	if (!loaded1 || !loaded2)
	{
		m_err << "Unable to load two files." << std::endl;
		return;
	}
	m_resultsComparer.CompareScenes(outStream, &scene1, &scene2);
	outStream.close();
}

bool CScriptRunner::LoadAndResaveSystemStructure()
{
	// try to load source file into m_systemStructure
	if (!LoadSourceFile()) return false;
	// check that result file name is defined
	if (m_job.resultFileName.empty()) return true;
	// save the loaded m_systemStructure into result file
	m_out << "Creating result file ... " << std::flush;
	m_systemStructure.SaveToFile(m_job.resultFileName);
	m_systemStructure.UpdateAllObjectsCompoundsProperties();
	m_out << "complete" << std::endl;
	return true;
}

bool CScriptRunner::LoadSourceFile()
{
	// check whether the file is already loaded
	if (m_job.sourceFileName.empty() && !m_systemStructure.GetFileName().empty() || !m_job.sourceFileName.empty() && m_systemStructure.GetFileName() == m_job.sourceFileName)
	{
		m_job.sourceFileName = m_systemStructure.GetFileName();
		m_out << "The source file is already loaded" << std::endl;
		return true;
	}

	// additional check that input file exists and readable
	std::ifstream fs(m_job.sourceFileName);
	const bool isGood = fs.good();
	fs.close();
	if (!isGood) // file can not be loaded
	{
		m_err << "Error: The source file cannot be opened. It may not exist or is not readable." << std::endl;
		return false;
	}

	// load file
	return LoadMusenFile(m_job.sourceFileName, m_systemStructure);
}

bool CScriptRunner::LoadMusenFile(const std::string& _sourceFileName, CSystemStructure& _systemStructure)
{
	// check file version
	if (CSystemStructure::IsOldFileVersion(_sourceFileName) || (CSystemStructure::FileVersion(_sourceFileName) < 2))
	{
		m_out << "The source file is in the old format. Converting it to a new format ... " << std::flush;
		CFileConverter fileConverter(_sourceFileName);
		fileConverter.ConvertFileToNewFormat();
		m_out << "complete" << std::endl;
	}

	// load system structure
	m_out << "Loading source file ... " << std::flush;
	const auto status = _systemStructure.LoadFromFile(_sourceFileName);
	m_out << "complete" << std::endl;

	// print information about loaded file if needed
	PrintFileLoadingInfo(status);

	return status != CSystemStructure::ELoadFileResult::IsNotDEMFile;
}

void CScriptRunner::ApplyMaterialParameters()
{
	for (const auto& prop : m_job.materialProperties)
		m_systemStructure.m_MaterialDatabase.SetPropertyValue(prop.compoundKey, prop.propertyKey, prop.value);
	for (const auto& inter : m_job.interactionProperties)
		m_systemStructure.m_MaterialDatabase.SetInteractionValue(inter.compoundKey1, inter.compoundKey2, inter.propertyKey, inter.value);
	for (const auto& mix : m_job.mixtureProperties)
	{
		if (!m_systemStructure.m_MaterialDatabase.GetMixture(mix.iMixture)) continue;
		m_systemStructure.m_MaterialDatabase.GetMixture(mix.iMixture)->SetFractionCompound(mix.iFraction, mix.compoundKey);
		m_systemStructure.m_MaterialDatabase.GetMixture(mix.iMixture)->SetFractionDiameter(mix.iFraction, mix.diameter);
		m_systemStructure.m_MaterialDatabase.GetMixture(mix.iMixture)->SetFractionContactDiameter(mix.iFraction, mix.diameter);
		m_systemStructure.m_MaterialDatabase.GetMixture(mix.iMixture)->SetFractionValue(mix.iFraction, mix.fraction);
	}
}

void CScriptRunner::PrintFileLoadingInfo(const CSystemStructure::ELoadFileResult& _status)
{
	switch (_status)
	{
	case CSystemStructure::ELoadFileResult::OK:
		break;
	case CSystemStructure::ELoadFileResult::IsNotDEMFile:
		m_err << "Error: Unable to open the file. It has the wrong format or is corrupted." << std::endl;
		break;
	case CSystemStructure::ELoadFileResult::PartiallyLoaded:
		m_out << "Warning: The file was partially loaded. An error may have occurred during the previous simulation." << std::endl;
		break;
	case CSystemStructure::ELoadFileResult::SelectivelySaved:
		m_out << "Warning: The file was saved in selective mode. Some properties of objects may not be available." << std::endl;
		break;
	}
}