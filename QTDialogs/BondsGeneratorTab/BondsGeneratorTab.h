/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ui_BondsGeneratorTab.h"
#include "GeneralMUSENDialog.h"
#include "BondsGenerator.h"
#include <QThread>
#include <QTimer>

class CBondsGeneratorProcess : public QObject
{
	Q_OBJECT
public:
	CBondsGenerator* m_bondsGenerator;	/// Pointer to a generator.
public:
	CBondsGeneratorProcess(CBondsGenerator* _bondsGenerator, QObject* _parent = nullptr);
public slots:
	void StartGeneration();
	void StopGeneration() const;
signals:
	void finished();
};


class CBondsGeneratorTab : public CMusenDialog
{
	Q_OBJECT

	Ui::bondsGeneratorTab ui;

	enum EColumn
	{
		ACTIVITY = 0,
		NAME     = 1,
		MATERIAL = 2,
		MIN_DIST = 3,
		MAX_DIST = 4,
		DIAMETER = 5,
		OVERLAY  = 6,
		MAT_SPEC = 7,
		NUMBER   = 8,
		PROGRESS = 9,
	};

	CBondsGenerator* m_generator{};				/// Pointer to a generator itself.
	CBondsGeneratorProcess m_generatorProcess;	/// Process with the generator.
	QThread m_generatorThread;					/// Thread to execute generator process.
	QTimer m_statisticsTimer;					/// Timer to update statistics.

public:
	CBondsGeneratorTab(CBondsGenerator* _bondsGenerator, QWidget* _parent = nullptr);

public slots:
	void UpdateWholeView() override;

private:
	void InitializeConnections() const;

	void UpdateGeneratorsTableHeaders() const;	/// Updates headers in the main information table.
	void UpdateGeneratorsTable() const;			/// Updates main information about available bonds generators.
	void UpdateMaterialsLists() const;			/// Updates list of selected materials.
	void UpdateMaterialsSelection() const;		/// Updates activity of elements in the list of selected materials.

	void StartGeneration();	/// Starts generation of bonds.
	void StopGeneration();	/// Stops generation of bonds.

	void EnableControls(bool _enable) const;	/// Enables/disables all controls when generation is started/finished.
	std::string GenerateClassName() const;		/// Generates a unique name for new bonds class.

private slots:
	void AddClass() const;		/// Adds new bonds class.
	void RemoveClass() const;	/// Removes selected bond class.
	void UpClass() const;		/// Moves selected class one position upwards.
	void DownClass() const;		/// Moves selected class one position downwards.

	void BondClassChanged() const;															/// Is called when any bond class is changed.
	void BondClassSelected(int _currRow, int _currCol, int _prevRow, int _prevCol) const;	/// Is called when new bond class must be selected.
	void BondClassDeselected() const;														/// Is called when current bond class must be deselected.
	void SelectedMaterialsChanged() const;													/// Is called when user chooses specific materials.

	void DeleteBondsClicked();	/// Removes all existing bonds from the scene.
	void StartStopClicked();	/// Reacts on pressing of the start/stop button.

	void GenerationFinished();					/// Is called when generation is finished.
	void UpdateGenerationStatistics() const;	/// Updates generation statistics of all generators.

signals:
	void ObjectsChanged();
};