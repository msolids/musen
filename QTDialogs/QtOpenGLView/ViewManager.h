/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "SampleAnalyzerTab.h"
#include "ViewSettings.h"
#include "EventMonitor.h"
#include "OpenGLTypes.h"

/* Manages switching between different visualization options.
 * Shows the selected viewer as m_widget, placed in m_layout.
 */
class CViewManager : public QObject
{
	Q_OBJECT

	QWidget* m_widget;	// Pointer to a widget, where the current viewer is shown.
	QLayout* m_layout;	// Pointer to a layout, where the current widget is placed.
	CEventMonitor m_eventMonitor; // Monitors mouse events in viewer.

	CViewSettings* m_viewSettings;  // Contains all visualization settings.

	// Tabs to access current view settings.
	const CSampleAnalyzerTab* m_sampleAnalyzer{}; // Pointer to CSampleAnalyzerTab.

	CSystemStructure* m_systemStructure{};		  // Pointer to current system structure.
	double m_time;								  // Current time point.

public:
	CViewManager(QWidget* _showWidget, QLayout* _layout, CViewSettings* _viewSettings, QObject* _parent);

	void SetPointers(CSystemStructure* _systemStructure, const CSampleAnalyzerTab* _sampleAnalyzer);

	void UpdateRenderType();			// Render type needs to be updated.
	void UpdateViewQuality() const;		// Render quality needs to be updated.
	void UpdateParticleTexture() const; // Particle texture needs to be updated.

	void UpdateParticlesVisible() const;	// Updates visibility of particles.
	void UpdateBondsVisible() const;		// Updates visibility of bonds.
	void UpdateGeometriesVisible() const;	// Updates visibility of geometries.
	void UpdateVolumesVisible() const;		// Updates visibility of volumes.
	void UpdateSlicesVisible() const;		// Updates visibility of slices.
	void UpdateDomainVisible() const;		// Updates visibility of simulation domain.
	void UpdatePBCVisible() const;			// Updates visibility of PBC.
	void UpdateAxesVisible() const;			// Updates visibility of axes.
	void UpdateTimeVisible() const;			// Updates visibility of time label.
	void UpdateLegendVisible() const;		// Updates visibility of legend.


	void UpdateFontAxes() const;	// Update font of axes labels.
	void UpdateFontTime() const;	// Update font of time.
	void UpdateFontLegend() const;	// Update font of legend labels.

	void UpdateColors() const;	// Update coloring of all particles and bonds.

	void SetTime(double _time);

	QImage GetSnapshot(uint8_t _scaling = 1) const; // Returns a scaled snapshot of a current view.
	void SetCameraStandardView(const CVector3& _position = CVector3{ 0, -1, 0 }) const;
	SCameraSettings GetCameraSettings() const;						// Returns current settings of the camera.
	void SetCameraSettings(const SCameraSettings& _settings) const;	// Sets new camera settings.

	std::vector<double> GetColoringValues(const std::vector<CSphere*>& _parts, const std::vector<CSolidBond*>& _bonds) const; // Returns list of values, for which coloring is applied.

public slots:
	void UpdateAllObjects() const;
	void EnableView() const;
	void DisableView() const;
	void UpdateSelectedObjects() const;		// Updates objects that have been selected.

private slots:
	void SelectObject(const QPoint& _pos);			// Selects a single object according to pointed position.
	void SelectGroup(const QPoint& _pos);			// Selects a group of connected objects according to pointed position.

private:
	size_t GetPointedObject(const QPoint& _pos) const;	// Returns ID of the pointed object or -1 if no objects pointed.

	void Initialize();

	void SetRenderGlu();	// Switches render to GLU
	void SetRenderMixed();	// Switches render to Mixed
	void SetRenderShader();	// Switches render to full OpenGL shader

	ERenderType WidgetRenderType() const; // Determines render type, which is selected in current widget.

	void SetViewSettings() const;

	void ClearWidget();

	// Update different components from system structure.
	void UpdateParticles() const;        // Updates particles.
	void UpdateBonds() const;            // Updates bonds.
	void UpdateWalls() const;            // Updates physical walls.
	void UpdateVolumes() const;			 // Updates analysis volumes.
	void UpdateSlices() const;			 // Updates slices.
	void UpdateSimulationDomain() const; // Updates simulation domain.
	void UpdatePBC() const;              // Updates PBC.
	void UpdateAxes() const;			 // Updates coordinate axes.
	void UpdateTime() const;             // Updates time.
	void UpdateLegend() const;           // Updates legend.

	std::vector<CSphere*> GetVisibleParticles() const;	// Returns all the particles that should be shown.
	std::vector<CSolidBond*> GetVisibleBonds() const;	// Returns all the bonds that should be shown.
	std::vector<CSphere*> GetVisibleSlices() const;		// Returns all the particles that should be shown in slice view.

	bool IsCutByPlanes(const CVector3& _coord) const; // Checks whether the point is in not cut space.

	std::vector<QColor> GetObjectsColors(const std::vector<CSphere*>& _parts, const std::vector<CSolidBond*>& _bonds) const;	// Returns colors of the object according to selected coloring options.
	QColor InterpolateColor(double _value) const; // Returns the color that matches the passed _value, according to selected coloring options.

	static QVector3D C2Q(const CVector3& _v);		// Converts CVector3 to QVector3D.
	static QQuaternion C2Q(const CQuaternion& _q);	// Converts CQuaternion to QQuaternion.
	static QColor C2Q(const CColor& _c);			// Converts CColor to QColor.
	static CVector3 Q2C(const QVector3D& _v);		// Converts QVector3D to CVector3.

signals:
	void ObjectsSelected();
	void CameraChanged();	// Is emitted when camera settings change.
};

