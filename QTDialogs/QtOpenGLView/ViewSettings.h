/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "OpenGLTypes.h"
#include <QFont>
#include <QColor>
#include <QSettings>
#include <set>

class CViewSettings
{
public:
	struct SVisibility
	{
		bool particles{ true };
		bool bonds{ true };
		bool geometries{ true };
		bool volumes{ true };
		bool domain{ false };
		bool pbc{ false };
		bool orientations{ false };
		bool axes{ true };
		bool time{ true };
		bool legend{ true };
	};
	struct SFont
	{
		QFont font{};
		QColor color = Qt::black;
	};
	struct SColoring
	{
		EColoring type{ EColoring::NONE };
		EColorComponent component{ EColorComponent::TOTAL };
		double minValue{ 0 };
		double maxValue{ 0 };
		double minDisplayValue{ 0 };
		double maxDisplayValue{ 0 };
		QColor maxColor{ Qt::red };
		QColor midColor{ Qt::green };
		QColor minColor{ Qt::blue };
	};
	struct SCutting
	{
		bool cutByX{ false };
		double minX{ 0 };
		double maxX{ 1 };
		bool cutByY{ false };
		double minY{ 0 };
		double maxY{ 1 };
		bool cutByZ{ false };
		double minZ{ 0 };
		double maxZ{ 1 };
		bool cutByVolumes{ false };
		std::set<std::string> volumes; // List of volumes' keys to cut.
		bool CutByPlanes() const { return cutByX || cutByY || cutByZ; } // Whether is cut by any plane
	};
	struct SSlicing
	{
		bool active{ false };					// Whether slicing is active.
		ESlicePlane plane{ ESlicePlane::NONE };	// Slicing plane
		double coordinate{ 0.0 };				// Coordinate of the selected slicing plane.
	};
	struct SBrokenBonds
	{
		bool show{ false };
		double startTime{ 0.0 };
		double endTime{ 1.0 };
		QColor color{ Qt::gray };
		bool IsInInterval(double _time) const { return startTime <= _time && _time <= endTime; }
	};

	const QString c_defaultPartTexture = ":/QT_GUI/Pictures/SphereTexture0.png";

private:
	const QString c_RENDER_TYPE               = "RENDER_TYPE";
	const QString c_RENDER_QUALITY            = "RENDER_QUALITY";
	const QString c_RENDER_PART_TEXTURE       = "RENDER_PART_TEXTURE";
	const QString c_DISPLAY_SHOW_PARTICLES    = "DISPLAY_SHOW_PARTICLES";
	const QString c_DISPLAY_SHOW_BONDS        = "DISPLAY_SHOW_BONDS";
	const QString c_DISPLAY_SHOW_WALLS        = "DISPLAY_SHOW_WALLS";
	const QString c_DISPLAY_SHOW_VOLUMES      = "DISPLAY_SHOW_VOLUMES";
	const QString c_DISPLAY_SHOW_DOMAIN       = "DISPLAY_SHOW_DOMAIN";
	const QString c_DISPLAY_SHOW_ORIENTATION  = "DISPLAY_SHOW_ORIENTATION";
	const QString c_DISPLAY_SHOW_AXES         = "DISPLAY_SHOW_AXES";
	const QString c_DISPLAY_SHOW_TIME         = "DISPLAY_SHOW_TIME";
	const QString c_DISPLAY_SHOW_LEGEND       = "DISPLAY_SHOW_LEGEND";
	const QString c_DISPLAY_FONT_AXES         = "DISPLAY_FONT_AXES";
	const QString c_DISPLAY_FONT_COLOR_AXES   = "DISPLAY_FONT_COLOR_AXES";
	const QString c_DISPLAY_FONT_TIME         = "DISPLAY_FONT_TIME";
	const QString c_DISPLAY_FONT_COLOR_TIME   = "DISPLAY_FONT_COLOR_TIME";
	const QString c_DISPLAY_FONT_LEGEND       = "DISPLAY_FONT_LEGEND";
	const QString c_DISPLAY_FONT_COLOR_LEGEND = "DISPLAY_FONT_COLOR_LEGEND";

	QSettings* m_settings;	  // File to save application specific settings.

	ERenderType m_renderType{ ERenderType::SHADER };	// Selected rendering mechanism.
	uint8_t m_renderQuality{ 50 };	                    // Rendering quality [1:100].
	QString m_particleTexture{ c_defaultPartTexture };	// Path to the current texture for particles.
	SVisibility m_visibility;                           // Determines, which types of objects are visible.
	SFont m_fontAxes;									// Font for rendering of axes labels.
	SFont m_fontTime;									// Font for rendering of time label.
	SFont m_fontLegend;									// Font for rendering of legend labels.

	SColoring m_coloring;       // Coloring settings.
	SCutting m_cutting;         // Cutting settings.
	SSlicing m_slicing;			// Slicing settings.
	SBrokenBonds m_brokenBonds; // Settings for visualization of broken bonds.

	std::set<std::string> m_visiblePartMaterials;	// List of keys of particles' materials to show.
	std::set<std::string> m_visibleBondMaterials;	// List of keys of bonds' materials to show.
	std::set<std::string> m_visibleGeometries;		// List of keys of geometries to show.
	std::set<std::string> m_visibleVolumes;			// List of keys of volumes to show.

	float m_geometriesTransparency = 0.0; // Overall transparency for all physical geometries [0.0:1.0].

	std::vector<size_t> m_selectedObjects; // Vector with IDs of currently selected objects.


public:
	CViewSettings(QSettings* _settings);

	ERenderType RenderType() const { return m_renderType; }
	void RenderType(ERenderType _val);

	uint8_t RenderQuality() const { return m_renderQuality; }
	void RenderQuality(uint8_t _val);

	QString ParticleTexture() const { return m_particleTexture; }
	void ParticleTexture(const QString& _val);

	SVisibility Visibility() const { return m_visibility; }
	void Visibility(const SVisibility& _val);

	SFont FontAxes() const { return m_fontAxes; }
	void FontAxes(const SFont& _val);
	SFont FontTime() const { return m_fontTime; }
	void FontTime(const SFont& _val);
	SFont FontLegend() const { return m_fontLegend; }
	void FontLegend(const SFont& _val);

	SColoring Coloring() const { return m_coloring; }
	void Coloring(const SColoring& _val) { m_coloring = _val; }

	SCutting Cutting() const { return m_cutting; }
	void Cutting(const SCutting& _val) { m_cutting = _val; }

	SSlicing Slicing() const { return m_slicing; }
	void Slicing(const SSlicing& _val) { m_slicing = _val; }

	SBrokenBonds BrokenBonds() const { return m_brokenBonds; }
	void BrokenBonds(const SBrokenBonds& _val) { m_brokenBonds = _val; }

	std::set<std::string> VisiblePartMaterials() const { return m_visiblePartMaterials; }
	void VisiblePartMaterials(const std::set<std::string>& _val) { m_visiblePartMaterials = _val; }
	std::set<std::string> VisibleBondMaterials() const { return m_visibleBondMaterials; }
	void VisibleBondMaterials(const std::set<std::string>& _val) { m_visibleBondMaterials = _val; }
	std::set<std::string> VisibleGeometries() const { return m_visibleGeometries; }
	void VisibleGeometries(const std::set<std::string>& _val) { m_visibleGeometries = _val; }
	std::set<std::string> VisibleVolumes() const { return m_visibleVolumes; }
	void VisibleVolumes(const std::set<std::string>& _val) { m_visibleVolumes = _val; }

	float GeometriesTransparency() const { return m_geometriesTransparency; }
	void GeometriesTransparency(const float _val) { m_geometriesTransparency = _val; }

	std::vector<size_t> SelectedObjects() const { return m_selectedObjects; }
	void SelectedObjects(const std::vector<size_t>& _val) { m_selectedObjects = _val; }

private:
	void LoadConfiguration();
};
