/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ParticlesAnalyzerTab.h"

CParticlesAnalyzerTab::CParticlesAnalyzerTab(QWidget *parent)
	: CResultsAnalyzerTab(parent)
{
	m_pAnalyzer = new CParticlesAnalyzer();

	// working with ComboBoxProperty
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Coordinate,        "Coordinate [m]",				"Distribution of particles according to coordinate");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::CoordinationNumber,"Coordination number [-]",		"Distribution of particles according to coordination number");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Distance,          "Distance [m]",					"Distribution of particles according to the distance to the specified point");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::ForceTotal,        "Force total [N]",				"Distribution of particles according to the force");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::KineticEnergy,		"Kinetic energy [J]",			"Distribution of particles according to kinetic energy");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::MaxOverlap,		"Max overlap [m]",				"Distribution of particles according to maximum overlap with other particles");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Number,            "Number [-]",					"Total number of particles");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::PotentialEnergy,	"Potential energy [J]",			"Distribution of particles according to potential energy");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::ResidenceTime,     "Residence time [s]",			"Residence time of particles");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Temperature,		"Temperature [K]",				"Temperature of particles");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::TotalVolume,       "Total volume [m3]",			"Total volume of objects");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::VelocityTotal,     "Velocity total [m/s]",			"Distribution of particles according to velocity");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::VelocityRotational,"Velocity rotational [rad/s]",	"Distribution of particles according to rotation velocity");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Stress, "Normal stress [Pa]", "SigmaXX, SigmaYY, SigmaZZ");

	SetWindowTitle("Particles Analyzer");
	m_sHelpFileName = "Users Guide/Particles Analyzer.pdf";
}
void CParticlesAnalyzerTab::InitializeAnalyzerTab()
{
	SetMultiplePropertySelection(false);
	SetResultsTypeVisible(true);
	SetDistanceVisible(true);
	SetComponentVisible(true);
	SetCollisionsVisible(false);
	SetGeometryVisible(false);
	SetConstraintsVisible(true);
	SetConstraintMaterialsVisible(true);
	SetConstraintVolumesVisible(true);
	SetConstraintDiametersVisible(true);
}

void CParticlesAnalyzerTab::UpdateWholeView()
{
	UpdateSelectedProperty();
	UpdateSelectedDistance();
	UpdateSelectedComponent();
	UpdateConstraints();
}