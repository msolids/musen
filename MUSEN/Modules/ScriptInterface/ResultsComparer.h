/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ScriptJob.h"

class CResultsComparer
{
public:
	CResultsComparer();
	void CompareScenes(std::ofstream& _outStream, CSystemStructure* _pScene1, CSystemStructure* _pScene2);

private:
	double m_dRelTolerance; // max relative tolerance 
	CSystemStructure* m_pScene1; 
	CSystemStructure* m_pScene2;
	std::ofstream* m_pOutStream;

private:
	bool CompareTwoValues( std::string _sMessage, double _dValue1, double _dValue2, double _dRelTol=0);
	bool CompareTwoValues(std::string _sMessage, CVector3 _vVec1, CVector3 _vVec2, double _dRelTol = 0);
	bool CompareTwoValues(std::string _sMessage, CVector3 _vVec1, CVector3 _vVec2, double _dRelTol, double _dAbsTol);
	
	bool CompareTimeIndependentData();
	bool CompareWallForces(double _dTime);
	bool CompareKineticEnergies(double _dTime);
	bool CompareCoordinates(double _dTime);
	bool CompareInterparticleContacts(double _dTime);
};
