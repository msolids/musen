/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ResultsAnalyzer.h"
#include <unordered_map>
#include <unordered_set>
#include <iomanip>

class CAgglomeratesAnalyzer : public CResultsAnalyzer
{
private:
	std::vector<std::vector<size_t>> m_vAgglomParticles;	    // list of agglomerates; each row contains list of particles' IDs, which belong to one agglomerate
	std::vector<std::vector<size_t>> m_vAgglomBonds;		    // list of agglomerates; each row contains list of bonds' IDs, which belong to one agglomerate
	std::vector<std::vector<size_t>> m_vSuperAgglomerates;      // list of super agglomerates (agglomerates at a certain distance); each row contains index of super agglomerate, row agglomerate index (m_vAgglomerates first)
	std::vector<size_t> m_vSuperAgglomeratesNumParticles;	    // number of individual particles in each superAgglomerate
	std::vector<CVector3> m_vSuperAgglomCOM;		            //
	std::vector<double> m_vSuperAgglomRadiusGyration;		    //
	std::vector<std::vector<double>> m_vSuperAgglomBondLengths;	// bond lengths in each super agglomerate
	std::vector<CVector3> m_vAgglomeratesMassCenters;		    //
	std::vector<CVector3> m_vAgglomeratesMassCenters_Initial;   //

public:
	CAgglomeratesAnalyzer();

	size_t GetAgglomeratesNumber() const;
	void FindAgglomerates(double _dTime);
	void FindSuperAgglomerates(double _dTime);	// Identify all agglomerates at a certain distance
	void CalculateMSD(double _dTime);
	bool Export() override;

	const std::vector<std::vector<size_t>>& GetAgglomeratesParticles() const { return m_vAgglomParticles; }
	const std::vector<std::vector<size_t>>& GetAgglomeratesBonds() const { return m_vAgglomBonds; }

	std::vector<std::vector<size_t>>* GetPointerToSuperAgglomerates() { return &m_vSuperAgglomerates; }
	std::vector<CVector3>* GetPointerToSuperAgglomCOM() { return &m_vSuperAgglomCOM; }
	std::vector<double>* GetPointerToSuperAgglomRadiusGyration() { return &m_vSuperAgglomRadiusGyration; }
	size_t GetNumberOfSuperAgglomerates() const { return m_vSuperAgglomerates.size(); }
	std::vector<size_t>* GetPointerToSuperAgglomNumParticles() { return &m_vSuperAgglomeratesNumParticles; }

private:
	std::vector<size_t> FilterAgglomerates(double _dTime) const;
	std::set<size_t> ApplyMaterialFilter(double _dTime, const std::set<size_t>& _vIndexes) const;
	void CalculateAgglomsCenterOfMass( double _dTime);
	void CalculateSuperAgglomsCOM(double _dTime);
	void CalculateSuperAgglomsRadiusGyration(double _dTime);
	CVector3 CalculateAgglomVelocity(size_t _iAgglom, double _dTime);
	CVector3 CalculateAgglomOrientation(size_t _iAgglom, double _dTime);
	double CalculateAgglomDiameter(size_t _iAgglom, double _dTime);

	void ExportCoordinatesAgglomerates(size_t _iAgglom, double _dTime);
	std::vector<CVector3> DuplicateAgglomerate(const std::vector<CVector3>& _vCoords, const CVector3& _vShiftDirection, const SPBC& _PBC) const;
	void MakeStandardSphere(const CVector3& _point, double _dRadius, size_t _nLatitude, size_t _nLongitude);
};
