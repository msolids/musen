/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ui_ObjectsEditorTab.h"
#include "GeneralMUSENDialog.h"
#include "SystemStructure.h"
#include "ExportTDPTab.h"
#include "UnitConvertor.h"
#include <QInputDialog>
#include <QSettings>
#include <utility>

class CViewSettings;

class CObjectsEditorTab: public CMusenDialog
{
	Q_OBJECT

	const QString c_AUTO_UPDATE        = "OET_AUTO_UPDATE";
	const QString c_OBJECTS_TYPE       = "OET_OBJECTS_TYPE";
	const QString c_FIELDS_PARTICLES   = "OET_FIELDS_PARTICLES";
	const QString c_FIELDS_SOLID_BONDS = "OET_FIELDS_SOLID_BONDS";
	const QString c_FIELDS_WALLS       = "OET_FIELDS_WALLS";

	Ui::objectsEditorTab ui;

	enum EObjectTypes : int
	{
		PARTICLES   = 0,
		SOLID_BONDS = 1,
		WALLS       = 2,
	};

	/// Available data fields.
	enum class EFieldTypes : int
	{
		ID,
		PARTNERS_ID,
		MATERIAL,
		COORDINATE,
		VELOCITY,
		ROTATION_VELOCITY,
		DIAMETER,
		CONTACT_DIAMETER,
		COORDINATION_NUMBER,
		MAX_OVERLAP,
		FORCE,
		TEMPERATURE,
		ORIENTATION,
		LENGTH,
		INITIAL_LENGTH,
		NORMAL,
		GEOMETRY,
	};

	struct SDataField
	{
		EFieldTypes type;					/// Type of the field for its identification.
		std::string name;					/// Field name to show in checkboxes.
		std::vector<std::string> headers;	/// Field name to show in table headers.
		EUnitType units;					/// Measurement units.
		bool active;						/// Whether it is selected by user.
		SDataField(EFieldTypes _type, std::string _name, std::vector<std::string> _headers, EUnitType _units = EUnitType::NONE)
			: type{_type}, name{std::move(_name)}, headers{std::move(_headers)}, units{_units}, active(false)	{}
	};

	enum class EDirection { X, Y, Z };

	QSettings* m_settings;
	CExportTDPTab* m_exportTDPTab;
	CViewSettings* m_viewSettings;

	EObjectTypes m_currentObjectsType{ PARTICLES }; /// Currently selected type of objects.
	std::vector<SDataField> m_dataFieldsPart; /// Description of all available data fields for particles.
	std::vector<SDataField> m_dataFieldsBond; /// Description of all available data fields for solid bonds.
	std::vector<SDataField> m_dataFieldsWall; /// Description of all available data fields for triangular walls.

	bool m_autoUpdate{ true };	/// Whether to update the main table automatically when the settings are changed.

public:
	CObjectsEditorTab(CExportTDPTab* _pTab, CViewSettings* _viewSettings, QSettings* _settings, QWidget *parent = nullptr);

	void Initialize() override;
	void SetEditEnabled(bool _enable) const; /// Enables/disables edit possibilities.

	void SaveConfiguration() const;	/// Saves configuration of the dialog into ini-file.
	void LoadConfiguration();		/// Loads configuration of the dialog from ini-file.

public slots:
	void UpdateWholeView() override;	/// Updates all.

	void UpdateSelectedObjects() const; /// Updates objects that were selected elsewhere.

	void ObjectsSelectionChanged() const;				/// Is called when user selects rows in the main table.
	void ObjectDataChanged(QTableWidgetItem* _item);	/// Is called when user changes some entries in the table.
	void DataPasted() const;							/// Is called when user pastes data into the main table from clipboard.

	void ShowContextMenu(const QPoint& _pos);	/// Shows context menu for the selected objects.

protected:
	void keyPressEvent(QKeyEvent* _event) override;

private:
	void InitializeConnections() const;	/// Connects all signals.

	void SetupObjectTypesCombo() const;	/// Creates a combobox for selection of object types.

	void UpdateTimeView() const;		/// Updates time label.
	void UpdateVisibleFields();			/// Updates check boxes of active data fields.
	void UpdateAutoButtonBlock() const;	/// Updated the auto update block.
	void TryUpdateTable();				/// Updates the main table if necessary.
	void UpdateTable();					/// Updates the main table with the information about the selected objects.
	void UpdateTableParts() const;		/// Updates the main table with the information about particles.
	void UpdateTableBonds() const;		/// Updates the main table with the information about solid bonds.
	void UpdateTableWalls() const;		/// Updates the main table with the information about triangular walls.

	void SetObjectData(int _row) const;								/// Sets data from the specified table's _row to the corresponding object.
	void SetObjectDataPart(int _row, CSphere& _part) const;			/// Sets data from the specified table's _row to the corresponding particle.
	void SetObjectDataBond(int _row, CSolidBond& _bond) const;		/// Sets data from the specified table's _row to the corresponding solid bond.
	void SetObjectDataWall(int _row, CTriangularWall& _wall) const;	/// Sets data from the specified table's _row to the corresponding triangular wall.

	void DeleteSelectedObjects();	/// Removes the selected objects.

	EObjectTypes SelectedObjectsType() const;					/// Returns currently selected type of objects.
	std::vector<SDataField>& CurrentDataFields();				/// Returns a reference to a vector of data fields of the currently selected type.
	const std::vector<SDataField>& CurrentDataFields() const;	/// Returns a const reference to a vector of data fields of the currently selected type.
	std::vector<SDataField> ActiveDataFields() const;			/// Returns a vector of active data fields of the currently selected type.

	void ShowTDPExportTab() const;									/// Is called when in context menu Export is selected.
	void SetNewDiameter(double _diameter);							/// Is called when in context menu new diameter is set.
	void SetNewVelocity(double _velocity, EDirection _direction);	/// Is called when in context menu new velocity is set.
	void SetNewMaterial(size_t _iCompound);							/// Is called when in context menu new material is set.

	void FitBondsDiameters() const;		/// Checks that bonds diameters are not larger than the diameters of particles.
	void FitPartsContactRadii() const;	/// Checks that contact radii of particles correspond to their radii.

private slots:
	void ObjectTypeChanged();								/// Is called when user selects another objects type.
	void FieldActivityChanged(const QCheckBox* _checkbox);	/// Is called when user changes activity of any data field.
	void UpdateButtonPressed();								/// Update the main table according to the selected objects type and data fields.
	void AutoUpdateToggled();								/// Is called when user toggles the auto update check box.
	void ToggleAddPanelVisibility() const;					/// Shows/hides Add panel.
	void ObjectAdded();										/// Is called when new object is added.

signals:
	void ObjectsSelected() const;	/// Is emitted when some objects where selected.
	void MaterialsChanged();		/// Is emitted if materials of objects were changed.
};
