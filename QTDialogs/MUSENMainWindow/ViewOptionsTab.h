/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ui_ViewOptions.h"
#include "GeneralMUSENDialog.h"
#include "UnitConvertor.h"
#include "ViewManager.h"
#include <QButtonGroup>

class CViewOptionsTab : public CMusenDialog
{
	Q_OBJECT

	enum ETab : int
	{
		CUTTING    = 0,
		VISIBILITY = 1,
		SELECTION  = 2,
		COLORING   = 3
	};
	enum EInfoRow : int
	{
		TIME,
		TYPE,
		ID,
		RADIUS,
		VOLUME,
		COORDX,
		COORDY,
		COORDZ,
		VELOX,
		VELOY,
		VELOZ,
		ANGVELOX,
		ANGVELOY,
		ANGVELOZ,
		MASS,
		TEMPERATURE,
		FORCE,
		MATERIAL
	};

public:
	Ui::viewOptionsTab ui;

private:
	CViewManager* m_viewManager;
	CViewSettings* m_viewSettings;

	QButtonGroup m_coloringGroup;				// Group of radio buttons with coloring types.
	QButtonGroup m_componentGroup;				// Group of radio buttons with component types for coloring.
	QButtonGroup m_slicingGroup;				// Group of radio buttons with planes coordinates for slicing.
	std::vector<EColoring> m_coloringVectors;	// List of coloring types that have vector (not double) values.

public:
	CViewOptionsTab(CViewManager* _viewManager, CViewSettings* _viewSettings, QWidget* parent = nullptr);

private:
	void InitializeConnections() const; // Initialize all connections on the form.

public slots:
	void UpdateWholeView() override;
	void NewSceneLoaded() override;			// Must be called when new scene was loaded.
	void UpdateMaterials() const;			// Must be called when materials set was changed.
	void UpdateGeometries() const;			// Must be called when geometries/volumes set was changed.
	void UpdateSelectedObjects() const;		// Must be called when new objects were selected.

private slots:
	void OnCuttingChanged() const;					// Called when cutting settings have been changed.
	void OnSlicingChanged() const;					// Called when slicing settings have been changed.
	void OnParticlesVisibilityChanged() const;		// Called when visibility options of particles have been changed.
	void OnBondsVisibilityChanged() const;			// Called when visibility options of bonds have been changed.
	void OnGeometriesVisibilityChanged() const;		// Called when visibility options of geometries have been changed.
	void OnVolumesVisibilityChanged() const;		// Called when visibility options of volumes have been changed.
	void OnDomainVisibilityChanged() const;			// Called when visibility options of simulation domain have been changed.
	void OnPBCVisibilityChanged() const;			// Called when visibility options of PBC have been changed.

	void OnColoringChanged() const;					// Called when coloring parameters have been changed.
	void OnAutoColorLimits() const;					// Called when auto coloring limits button has been pressed.

protected:
	void keyPressEvent(QKeyEvent *e) override;

private:
	void UpdateCuttingPlanes() const;			// Updates the list of cutting planes, and the current selection.
	void UpdateCuttingVolumes() const;			// Updates the list of cutting volumes, and the current selection.
	void UpdateSlicing() const;					// Updates information about slicing planes, and the current selection.

	void UpdateParticlesVisibility() const;		// Updates the list of materials, available for particles, and the current selection.
	void UpdateBondsVisibility() const;			// Updates the list of materials, available for bonds, and the current selection.
	void UpdateGeometriesVisibility() const;	// Updates the list of geometries, and the current selection.
	void UpdateVolumesVisibility() const;		// Updates the list of volumes, and the current selection.
	void UpdateVisibilityCheckboxes() const;	// Updates the rest visibility options.

	void SelectAllParticleMaterials() const;	// Selects all particles' materials to be shown.
	void SelectAllBondMaterials() const;		// Selects all bonds' materials to be shown.
	void SelectAllGeometries() const;			// Selects all geometries to be shown.
	void SelectAllVolumes() const;				// Selects all volumes to be shown.

	void UpdateSelectedObjectsInfo() const;	// Updates information about selected objects.
	void UpdateOneObjectInfo() const;		// Updates information about selected objects if a single object is selected.
	void UpdateGroupObjectsInfo() const;	// Updates information about selected objects if a group of objects is selected.

	void UpdateColoringTypes();				// Updates radio buttons of coloring types according to current settings of units converter.
	void UpdateColoringLimits() const;		// Updates coloring limits and their labels.
	void UpdateColoringComponent() const;	// Updates activity of group and selected coloring component.
	void UpdateColoringColors() const;		// Updates selected colors.

	void UpdateLabels() const;		// Updates text labels according to current settings of units converter.
};

