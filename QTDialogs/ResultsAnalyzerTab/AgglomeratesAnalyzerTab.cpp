/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "AgglomeratesAnalyzerTab.h"

CAgglomeratesAnalyzerTab::CAgglomeratesAnalyzerTab(QWidget *parent)
	: CResultsAnalyzerTab(parent)
{
	m_pAnalyzer = new CAgglomeratesAnalyzer();

	// working with ComboBoxProperty
	CResultsAnalyzerTab::AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Coordinate,	 "Coordinate [m]",		    "Distribution of agglomerates according coordinate");
	CResultsAnalyzerTab::AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Diameter,		 "Diameter [m]",		    "Distribution of agglomerates by diameters of equivalent volume");
	CResultsAnalyzerTab::AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Number,		 "Number [-]",			    "Total number of agglomerates");
	CResultsAnalyzerTab::AddAnalysisProperty(CResultsAnalyzer::EPropertyType::BondNumber,    "Number of bonds [-]",     "Distribution of agglomerates by number of bonds");
	CResultsAnalyzerTab::AddAnalysisProperty(CResultsAnalyzer::EPropertyType::PartNumber,    "Number of particles [-]", "Distribution of agglomerates by number of primary particles");
	CResultsAnalyzerTab::AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Orientation,	 "Orientation [-]",		    "Orientation of agglomerates in 3D space");
	CResultsAnalyzerTab::AddAnalysisProperty(CResultsAnalyzer::EPropertyType::VelocityTotal, "Velocity total [m/s]",    "Distribution of agglomerates according velocity");
	CResultsAnalyzerTab::AddAnalysisProperty(CResultsAnalyzer::EPropertyType::ExportToFile,	 "Export to file",	        "Export of coordinates to the file");

	SetWindowTitle("Agglomerates Analyzer");
	m_sHelpFileName = "Users Guide/Agglomerates Analyzer.pdf";
}

void CAgglomeratesAnalyzerTab::InitializeAnalyzerTab()
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
}

void CAgglomeratesAnalyzerTab::UpdateWholeView()
{
	UpdateSelectedProperty();
	UpdateSelectedDistance();
	UpdateSelectedComponent();
	UpdateConstraints();
}