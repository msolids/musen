/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ui_GeometriesEditorTab.h"
#include "GeneralMUSENDialog.h"
#include <QButtonGroup>

class CGeometriesEditorTab : public CMusenDialog
{
	Q_OBJECT

	Ui::geometriesEditorTab ui{};

	// Type of geometry.
	enum class EType
	{
		NONE, GEOMETRY, VOLUME
	};

	// Keys to find properties in the tree widget.
	enum EProperty : uint32_t
	{
		MATERIAL,
		COLOR,
		MOTION,
		TRIANGLES,
		ROTATE_X,
		ROTATE_Y,
		ROTATE_Z,
		POSITION_X,
		POSITION_Y,
		POSITION_Z,
		SCALE,
		SIZE_X,
		SIZE_Y,
		SIZE_Z,
		WIDTH,
		DEPTH,
		HEIGHT,
		RADIUS,
		INNER_RADIUS,
	};

	// Keys to find properties in the motion table.
	enum ERowMotion : int
	{
		TIME_BEG = 0,
		TIME_END,
		FORCE,
		LIMIT_TYPE,
		VEL_X,
		VEL_Y,
		VEL_Z,
		ROT_VEL_X,
		ROT_VEL_Y,
		ROT_VEL_Z,
		ROT_CENTER_X,
		ROT_CENTER_Y,
		ROT_CENTER_Z,
	};

	enum class EAxis { X = 0, Y, Z };
	enum class EPosition { MIN, MAX };

	int m_motionWidth{};								// Width of the motion table needed to hide/show it with the same size.
	std::map<EType, QTreeWidgetItem*> m_list;			// Pointers to top-level items in the list of geometries.
	std::map<EProperty, QTreeWidgetItem*> m_properties;	// Pointers to tree items with properties of the selected geometry.

	CBaseGeometry* m_object{ nullptr };	// Currently selected geometry.

public:
	CGeometriesEditorTab(QWidget* parent = nullptr);

public slots:
	void setVisible(bool _visible) override;	// Is called when window visibility changes.

	void UpdateWholeView() override;	// Updates all controls.

private:
	void InitializeConnections() const;	// Connects all signals.
	void Initialize() override;			// Class is initialized with all required pointers.

	void SetupGeometriesList();			// Configures the main list with geometries.
	void SetupPropertiesList();			// Configures the list with properties of selected geometry.

	void UpdateMeasurementUnits() const;	// Updates all texts according to selected measurement units.
	void UpdateAddButtons();				// Updates content of add buttons.
	void UpdateMaterialsCombo() const;		// Updates combo box with materials.
	void UpdateMotionCombo() const;			// Updates combo box with motion type.
	void UpdateGeometriesList();			// Updates the list of existing geometries and volumes.
	void UpdatePropertiesInfoEmpty() const;	// Updates properties table view if no geometry or volume is selected.
	void UpdatePropertiesInfo() const;		// Updates information about properties of selected geometry or volume.
	void UpdateMotionInfo();				// Updates information about motion of selected geometry or volume.
	void UpdateMotionVisibility();			// Updates visibility of the motion table.

	void AddGeometryStd(EVolumeShape _shape, EType _type);		// Adds a standard geometry or volume of selected _type.
	void AddGeometryLib(const std::string& _key, EType _type);	// Adds an object with _key from geometries database as a geometry or volume.
	void DeleteGeometry();										// Removes selected geometry or volume.
	void UpGeometry();											// Moves selected geometry or volume upwards in the list.
	void DownGeometry();										// Moves selected geometry or volume downwards in the list.

	EType Type() const;									// Returns type of the currently selected object.
	void EmitChangeSignals(EType _type = EType::NONE);	// Emits signals depending on the provided geometry type or the type of the selected geometry.
	bool CheckAndConfirmTPRemoval();					// Asks user to confirm removal of all time points from system structure.

	void keyPressEvent(QKeyEvent* _event) override;

private slots:
	void NameChanged();					// Is called when user changes the name of a geometry or volume.
	void GeometrySelected();			// Is called when user selects new geometry or volume.
	void MaterialChanged();				// Is called when user changes material of a geometry.
	void ColorChanged();				// Is called when user changes color of a geometry or volume.
	void MotionTypeChanged();			// Is called when user changes motion type of a geometry or volume.
	void RotationChanged(EAxis _axis);	// Is called when user applies rotation to a geometry or volume.
	void PositionChanged();				// Is called when user changes position of a geometry or volume.
	void SizeChanged();					// Is called when user changes size-related parameters of a geometry or volume.
	void QualityChanged();				// Is called when user changes quality of geometry or volume.
	void ScalingChanged();				// Is called when user applies scaling to a geometry or volume.

	void AddMotion();			// Adds new motion step to a geometry or volume.
	void DeleteMotion();		// Removes a motion step from a geometry or volume.
	void MotionTableChanged();	// Is called when user changes table of motion parameters of a geometry or volume.
	void MotionChanged();		// Is called when user changes other motion parameters of a geometry or volume.

	void PlaceAside(EAxis _axis, EPosition _pos); // Places a geometry or volume on the selected side of all particles.

signals:
	void ObjectsChanged();
	void AnalysisGeometriesChanged();
	void GeometryAdded(const std::string& _key);	// Is emitted when a new real geometry is created in the system.
	void VolumeAdded(const std::string& _key);		// Is emitted when a new analysis volume is created in the system.
};
