/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GeometriesAnalyzerTab.h"

CGeometriesAnalyzerTab::CGeometriesAnalyzerTab(QWidget *parent)
	: CResultsAnalyzerTab(parent)
{
	m_pAnalyzer = new CGeometriesAnalyzer();

	//working with ComboBoxProperty
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::ForceTotal,	"Force total [N]",	"Force total [N]");
	AddAnalysisProperty(CResultsAnalyzer::EPropertyType::Distance,		"Distance [m]",		"Returns distance to the initial position" );

	SetWindowTitle("Geometries analyzer");
	m_sHelpFileName = "Users Guide/Geometries Analyzer.pdf";
}

void CGeometriesAnalyzerTab::InitializeAnalyzerTab()
{
	SetMultiplePropertySelection(true);
	SetResultsTypeVisible(false);
	SetDistanceVisible(false);
	SetComponentVisible(false);
	SetCollisionsVisible(false);
	SetGeometryVisible(true);
	SetConstraintsVisible(false);
	SetResultsTypeVisible(false);
}

void CGeometriesAnalyzerTab::UpdateDistanceVisibility()
{

}

void CGeometriesAnalyzerTab::UpdateWholeView()
{
	SetupGeometryCombo();
	UpdateSelectedProperty();
	UpdateSelectedGeometry();
	adjustSize();
}
