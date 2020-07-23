/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ui_PackageGeneratorTab.h"
#include "GeneralMUSENDialog.h"
#include "PackageGenerator.h"
#include <QTimer>

 // thread for packing generation
class CPackageGeneratorThread : public QObject
{
	Q_OBJECT
public:
	CPackageGenerator* m_pPackageGenerator;

public:
	CPackageGeneratorThread(CPackageGenerator* _pPackageGenerator, QObject *parent = 0);
	~CPackageGeneratorThread();

public slots:
	void StartGeneration();
	void StopGeneration();

signals:
	void UpdateStatisticsSignal();
	void finished();
};



class CPackageGeneratorTab: public CMusenDialog
{
	Q_OBJECT
private:
	enum ETableIndex
	{
		VOLUME_NAME = 0,
		PARTICLES = 1,
		MAX_OVERLAP = 2,
		AVER_OVERLAP = 3,
		COMPLETNESS = 4
	};

	Ui::packageGeneratorTab ui;

	CPackageGenerator* m_pPackageGenerator;
	CPackageGeneratorThread* m_pPackageGeneratorThread;
	QThread* m_pQTThread;

	bool m_bGenerationStarted; // flag indicate about current status of generation

	// timer which is used to update statistic
	QTimer m_UpdateTimer;

public:
	CPackageGeneratorTab(CPackageGenerator* _pPackageGenerator, QWidget *parent = 0);

	void InitializeConnections();

public slots:
	void SelectedVolumeChanged(); // if selected index has been changed
	void UpdateWholeView();
	void GeneratorItemChanged(QListWidgetItem* _pItem); // called when user changes the name of the volume in the list
	void StartGeneration();

	void GenerationFinished();
	void UpdateGenerationStatistics();

protected:
	void keyPressEvent(QKeyEvent *e);

private slots:
	void DeleteAllParticles();
	void DeleteGenerator(); // remove specified volume class
	void AddGenerator(); // add new volume class
	void UpGenerator();
	void DownGenerator();
	void GenerationDataChanged();
	void GeneratorTypeChanged() const;

private:
	void UpdateGeneratorsList();
	void UpdateSelectedGeneratorInfo();
	void UpdateGeneratorsTable();
	void UpdateVolumesCombo();
	void UpdateMixturesCombo();
	void UpdateUnitsInLabels();
	void UpdateSimulatorType() const;

	void ShowParticlesNumberInTable(unsigned _iGenerator);
	void EnableControls(bool _bEnable);

signals:
	void OpenGLViewShouldBeCentrated();
	void ObjectsChanged();
};