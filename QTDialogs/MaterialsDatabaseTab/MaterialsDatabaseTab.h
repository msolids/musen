/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ui_MaterialsDatabaseTab.h"
#include "GeneralMUSENDialog.h"
#include "MaterialsDatabase.h"
#include <QSignalMapper>

class CMaterialsDatabaseTab : public CMusenDialog
{
	Q_OBJECT
protected:
	enum EFractionsColumns : int { NAME = 0, COMPOUND = 1, DIAMETER = 2, CONTACT_DIAMETER = 3, NUMBER_FRACTION = 4 };
	Ui::CMaterialsEditorTab ui;
	CMaterialsDatabase* m_pMaterialsDB;	// Pointer to current database. Hides the same named variable from CMusenDialog.
	bool m_bGlobal;						// Determines whether this tab works with global or local database.

private:
	bool m_bAvoidUpdate;
	QSignalMapper *m_pSignalMapperFracs;
	std::vector<ETPPropertyTypes> m_vMusenActiveProperies;
	std::vector<EIntPropertyTypes> m_vMusenActiveInteractions;

public:
	CMaterialsDatabaseTab(CMaterialsDatabase* _pMaterialsDB, QWidget *parent = 0);

	// Overrides setting pointers to disable resetting pointer to materials database
	void SetPointers(CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor, CMaterialsDatabase* _pMaterialsDB, CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB) override;

public slots:
	virtual void UpdateWholeView();

protected:
	void InitializeConnections();
	virtual void UpdateWindowTitle();

	// Updates list of available compounds.
	void UpdateCompoundsList();
	// Updates list of available mixtures.
	void UpdateMixturesList();

	// Makes specified compound currently selected.
	void SelectCompound(const CCompound* _pCompound);
	// Makes specified mixture currently selected.
	void SelectMixture(const CMixture* _pMixture);

private:
	// Update activity of new/save/save as/load buttons
	void UpdateButtons();

	// Updates common information about selected compound.
	void UpdateCompoundInfo();
	// Updates properties of selected compound.
	void UpdateCompoundProperties();

	// Updates lists of compounds for interactions view.
	void UpdateInteractionsCompoundsLists();
	// Updates interaction properties between selected compounds.
	void UpdateInteractions();

	// Updates list of fractions of selected mixture.
	void UpdateFractionsList();
	// Updates info about sum of all fractions in selected mixture.
	void UpdateTotalFraction();
	// Creates combobox with fractions and selects the specified one.
	void AddCombobBoxOnFractionsTable(int _iRow, int _iCol, const std::string& _sSelected);

	// Returns list of names of mixtures.
	std::vector<std::string> GetMixturesNames() const;
	// Returns list of names of fractions.
	std::vector<std::string> GetFractionNames(const CMixture& _mixture) const;
	// Returns name which is unique for provided list.
	std::string PickUniqueName(const std::vector<std::string>& _vNames, const std::string& _sBaseName) const;
	// Returns key of the currently selected (_nRow == -1) or specified (_nRow != -1) element.
	std::string GetElementKey(const QListWidget *_pListWidget, int _nRow = -1) const;
	// Returns type of specified property.
	int GetPropertyType(QTableWidget *_pTable, int _nRow) const;
	// Returns path to the previous file to use in the GetFileName dialogs.
	QString FilePathToOpen() const;

private slots:
	// Creates file for new database.
	void NewDatabase();
	// Loads database from file.
	void LoadDatabase();
	// Saves database into current file.
	void SaveDatabase();
	// Saves database into new file.
	void SaveDatabaseAs();

	// Adds new compound into current database.
	void AddCompound();
	// Creates a copy of selected compound.
	void DuplicateCompound();
	// Removes selected compound.
	void RemoveCompound();
	// Moves selected compound upwards in the list of compounds.
	void UpCompound();
	// Moves selected compound downwards in the list of compounds.
	void DownCompound();
	// Changes current selection to a cell in the last row of the table.
	void CompoundPropertySelectionChanged(int _iRow, int _iCol, int _iPrevRow, int _iPrevCol);
	// Updates info about selected compound.
	void CompoundSelected();
	// Changes name of corresponding compound.
	void CompoundNameChanged(QListWidgetItem* _pItem);
	// Changes color of selected compound.
	void CompoundColorChanged();
	// Changes author of selected compound.
	void CompoundAuthorChanged();
	// Changes property value of selected compound.
	void PropertyValueChanged(int _nRow, int _nCol);

	// Changes current selection to a cell in the last row of the table.
	void InteractionPropertySelectionChanged(int _iRow, int _iCol, int _iPrevRow, int _iPrevCol);
	// Selected first compound for interaction.
	void Compound1Selected();
	// Selected second compound for interaction.
	void Compound2Selected();
	// Changes value of selected interaction property between selected compounds.
	void InteractionValueChanged(int _nRow, int _nCol);

	// Adds new mixture into current database.
	void AddMixture();
	// Creates a copy of selected mixture.
	void DuplicateMixture();
	// Removes selected mixture.
	void RemoveMixture();
	// Moves selected mixture upwards in the list of mixtures.
	void UpMixture();
	// Moves selected mixture downwards in the list of mixtures.
	void DownMixture();
	// Updates info about selected mixture.
	void MixtureSelected();
	// Changes name of corresponding mixture.
	void MixtureNameChanged(QListWidgetItem* _pItem);

	// Add fraction to selected mixture.
	void AddFraction();
	// Creates a copy of selected fraction.
	void DuplicateFraction();
	// Removes selected fraction.
	void RemoveFraction();
	// Moves selected fraction upwards in the list of fractions.
	void UpFraction();
	// Moves selected fraction downwards in the list of fractions.
	void DownFraction();
	// Normalizes fractions of selected mixture.
	void NormalizeFractions();
	// Changes some values of selected fraction.
	void FractionChanged(int _iRow, int _iCol);
	// Select another compound for fraction.
	void FractionCompoundChanged(int _iRow);

	// Handles keyboard events
	void keyPressEvent(QKeyEvent* event);

signals:
	// Called when material database has been changed.
	void MaterialDatabaseWasChanged();
	// Called when path to material database file has been changed.
	void MaterialDatabaseFileWasChanged();
};
