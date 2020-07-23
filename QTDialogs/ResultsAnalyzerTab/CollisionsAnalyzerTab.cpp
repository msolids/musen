/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "CollisionsAnalyzerTab.h"

CCollisionsAnalyzerTab::CCollisionsAnalyzerTab(QWidget *parent)
	: CResultsAnalyzerTab(parent)
{
	m_pAnalyzer = new CCollisionsAnalyzer();

	// working with ComboBoxProperty
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Coordinate,        "Coordinate [m]",            "Coordinate of the contact");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Distance,          "Distance [m]",              "Distance to the contact");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Duration,          "Duration [s]",              "Duration of contacts");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Energy,            "Energy [J]",                "Collision energy");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::ForceNormal,       "Force normal [N]",          "Maximum normal force during the collision");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::ForceTangential,   "Force tangential [N]",      "Maximum tangential force during the collision");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::ForceTotal,		"Force total [N]",			 "Maximum total force during the collision");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Number,            "Number [-]",                "Total number of contacts");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::VelocityNormal,    "Velocity normal [m/s]",     "Normal relative velocity of particles at the moment of contact");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::VelocityTangential,"Velocity tangential [m/s]", "Tangential relative velocity of particles at the moment of contact");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::VelocityTotal,     "Velocity total [m/s]",      "Total relative velocity of particles at the moment of contact");

	SetWindowTitle("Collisions Analyzer");
	m_sHelpFileName = "Users Guide/Collisions Analyzer.pdf";
}

void CCollisionsAnalyzerTab::Initialize()
{
	SetMultiplePropertySelection(false);
	SetResultsTypeVisible(true);
	SetDistanceVisible(true);
	SetComponentVisible(true);
	SetCollisionsVisible(true);
	SetGeometryVisible(false);
	SetConstraintsVisible(true);
	SetConstraintMaterialsVisible(true);
	SetConstraintMaterials2Visible(true);
	SetConstraintVolumesVisible(true);
	SetConstraintDiametersVisible(true);
	SetConstraintDiameters2Visible(true);
}


void CCollisionsAnalyzerTab::UpdateWholeView()
{
	UpdateSelectedProperty();
	UpdateSelectedDistance();
	UpdateSelectedComponent();
	UpdateSelectedRelation();
	UpdateConstraints();
}