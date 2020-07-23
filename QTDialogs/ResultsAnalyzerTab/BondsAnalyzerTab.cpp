/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "BondsAnalyzerTab.h"
#include "qtOperations.h"

CBondsAnalyzerTab::CBondsAnalyzerTab(QWidget *parent)
	: CResultsAnalyzerTab(parent)
{
	m_pAnalyzer = new CBondsAnalyzer();

	// working with ComboBoxProperty
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::BondForce,     "Bond force[N]",        "Bond force");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Deformation, "Deformation [m]", "Bonds deformation (positive-compression, negative-tension");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Diameter,      "Diameter [m]",         "Diameter of the bond");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::ForceTotal,    "Force total [N]",      "Total force");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Length,        "Length [m]",           "Current length of the bond");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Number,        "Number [-]",           "Number of solid bonds");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Strain, "Strain [%]", "Relative bonds strain (positive-compression, negative-tension)");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Stress, "Stress [Pa]", "Bond normal stress (positive-compression, negative-tension)");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::VelocityTotal, "Velocity total [m/s]", "Calculated as middle velocity of contact partners");

	SetWindowTitle("Bonds Analyzer");
	m_sHelpFileName = "Users Guide/Bonds Analyzer.pdf";
}


void CBondsAnalyzerTab::InitializeAnalyzerTab()
{
	SetMultiplePropertySelection(false);
	SetResultsTypeVisible(true);
	SetDistanceVisible(false);
	SetComponentVisible(true);
	SetCollisionsVisible(false);
	SetGeometryVisible(false);
	SetConstraintsVisible(true);
	SetConstraintMaterialsVisible(true);
	SetConstraintVolumesVisible(true);
	SetConstraintDiametersVisible(true);
}

void CBondsAnalyzerTab::UpdateWholeView()
{
	UpdateSelectedProperty();
	UpdateSelectedComponent();
	UpdateConstraints();
}