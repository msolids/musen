/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ui_SampleAnalyzerTab.h"
#include "SystemStructure.h"
#include "GeneralMUSENDialog.h"
#include "UnitConvertor.h"

class CSampleAnalyzerTab: public CMusenDialog
{
	Q_OBJECT
private:
	Ui::sampleAnalyzerTab ui;

public:
	CVector3 m_vCenter;
	double m_dRadius;
	double m_dPSDStart; // start diameter to export [m]
	double m_dPSDEnd; // end diameter to export [m]
	unsigned m_nPSDIntervalsNumber; // number of classes
	double m_dVelStart; // start velocity to export [m/s]
	double m_dVelEnd; // end velocity to export [m/s]
	unsigned m_nVelIntervalsNumber; // number of classes
	double m_dTimeStart;
	double m_dTimeEnd;
	unsigned m_nTimeIntervalsNumber;

private:
	// creates the list of all particles which situated within some region ( if bPartialyInVolume than also particle the centers of which in volume are also included)
	void GetAllParticlesInRegion(double _dTime, std::vector<const CSphere*> *_pAllParticles, double* _pTotalMass, double* _pTotalVolume, bool bPartialyInVolume = false, bool bConsiderBonds = false);
	double GetCoordinationNumber(double _time, const std::vector<const CSphere*> &_particles) const; // approximate coordination number of specified particles
	bool CheckDataCorrectness(); // check that all fields in the edit are correct (if correct return true )
	void InitializeConnections();
	// Returns all particles that are totally or partially inside the sphere.
	std::vector<const CSphere*> GetParticlesInsideSphere(double _time, const CVector3& _center, double _radius) const;
	// Returns all bonds that are connected to particles, which are totally or partially inside the sphere.
	std::vector<const CSolidBond*> GetBondsInsideSphere(double _time, const CVector3& _center, double _radius) const;

public slots:
	void UpdateExportView();
	void UpdateVolumeView(); // update information in the table view
	void UpdateMainProperties();
	void VolumeDataWasChanged();
	void ExportDataWasChanged();

	//void PorosityVolumeWasChanged(); // called when properties of porosity volume has been changed by user
	//void UpdatePorosityView();
	void UpdateWholeView();
	//void RecalculatePorosity();
	void ExportData();

protected:
	//void keyPressEvent(QKeyEvent *e);

public:
	CSampleAnalyzerTab( QWidget *parent = 0 );
};
