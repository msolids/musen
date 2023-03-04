/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "SystemStructure.h"

class CConstraints
{
public:
	struct SInterval
	{
		double dMin;
		double dMax;
	};

private:
	CSystemStructure* m_pSystemStructure;
	CMaterialsDatabase* m_pMaterialsDB;

	bool m_bVolumesActive;
	std::set<unsigned> m_vVolumes;

	bool m_bMaterialsActive;
	std::set<std::string> m_MaterialsSet;
	std::map<std::string, std::set<std::string>> m_MaterialsMap;

	bool m_bGeometriesActive;
	std::set<unsigned> m_vGeometries;
	std::set<size_t> m_vGeometriesPlanes;

	bool m_bDiametersActive;
	SInterval m_Diameters;
	SInterval m_Diameters2;

public:
	CConstraints();
	~CConstraints();

	void UpdateSettings();
	void SetPointers(CSystemStructure* _pSystemStructure, CMaterialsDatabase* _pMaterialsDB);

	bool IsAllVolumeSelected() const;
	bool IsAllMaterialsSelected() const;
	bool IsAllMaterials2Selected() const;
	bool IsAllMaterials2Selected(const std::string& _sKey) const;
	bool IsAllGeometriesSelected() const;
	bool IsAllDiametersSelected() const;

	//////////////////////////////////////////////////////////////////////////
	// Getters

	bool IsVolumesActive() const;
	std::vector<unsigned> GetVolumes() const;

	bool IsMaterialsActive() const;
	std::vector<std::string> GetMaterials() const;
	std::vector<std::string> GetMaterials2(const std::string& _sKey) const;
	std::vector<std::string> GetCommonMaterials() const;	// list of materials, which connected with all materials

	bool IsGeometriesActive() const;
	std::vector<unsigned> GetGeometries() const;
	std::set<size_t> GetGeometriesPlanes() const;

	bool IsDiametersActive() const;
	SInterval GetDiameter() const;
	double GetDiameterMin() const;
	double GetDiameterMax() const;
	SInterval GetDiameter2() const;
	double GetDiameter2Min() const;
	double GetDiameter2Max() const;

	//////////////////////////////////////////////////////////////////////////
	// Setters

	void SetVolumesActive(bool _bActive);
	void AddVolume(unsigned _nIndex);
	void RemoveVolume(unsigned _nIndex);
	void SetVolumes(const std::vector<unsigned>& _vIndexes);
	void ClearVolumes();

	void SetMateralsActive(bool _bActive);
	void AddMaterial(const std::string& _sKey);
	void RemoveMaterial(const std::string& _sKey);
	void SetMaterials(const std::vector<std::string>& _vKeys);
	void ClearMaterials();
	void AddMaterial2(const std::string& _sKey);
	void AddMaterial2(const std::string& _sKey1, const std::string& _sKey2);
	void RemoveMaterial2(const std::string& _sKey);
	void RemoveMaterial2(const std::string& _sKey1, const std::string& _sKey2);
	void SetMaterials2(const std::string& _sKey1, const std::vector<std::string>& _vKeys2);
	void ClearMaterials2();
	void ClearMaterials2(const std::string& _sKey);

	void SetGeometriesActive(bool _bActive);
	void AddGeometry(unsigned _nIndex);
	void RemoveGeometry(unsigned _nIndex);
	void SetGeometries(const std::vector<unsigned>& _vIndexes);
	void ClearGeometries();

	void SetDiametersActive(bool _bActive);
	void SetDiameter(double _dMin, double _dMax);
	void SetDiameter2(double _dMin, double _dMax);
	void ClearDiameters();

	//////////////////////////////////////////////////////////////////////////
	/// Checkers

	// Returns 'true' if volume with specified index is selected. Doesn't analyze empty list: IsAllVolumeSelected() should be called in client.
	bool CheckVolume(unsigned _nIndex) const;
	// Returns 'true' if material with specified key is selected. Doesn't analyze empty list: IsAllMaterialsSelected() should be called in client.
	bool CheckMaterial(const std::string& _sKey) const;
	/* Returns 'true' if material of object with specified index is selected. Does not perform any checks on indexes or corresponding objects.
	 * Doesn't analyze empty list: IsAllMaterialsSelected() should be called in client. */
	bool CheckMaterial(size_t _nIndex) const;
	// Returns 'true' if materials with specified keys are selected and should be analyzed. Doesn't analyze empty list: IsAllMaterials2Selected() should be called in client.
	bool CheckMaterial(const std::string& _sKey1, const std::string& _sKey2) const;
	/* Returns 'true' if materials of objects with specified indexes are selected and should be analyzed. Does not perform any checks on indexes or corresponding objects.
	 * Doesn't analyze empty list: IsAllMaterials2Selected() should be called in client. */
	bool CheckMaterial(size_t _nIndex1, size_t _nIndex2) const;
	// Returns 'true' if object with specified index is plane and belongs to selected geometries. Doesn't analyze empty list: IsAllGeometriesSelected() should be called in client.
	bool CheckGeometry(size_t _nIndex) const;
	// Returns 'true' if specified diameter is selected. Doesn't analyze empty list: IsAllDiametersSelected() should be called in client.
	bool CheckDiameter(double _dDiam) const;
	/* Returns 'true' if sphere with specified diameters are selected and should be analyzed.
	 * Index must belong to spheres - there is no internal checks of indexes or corresponding objects.
	 * Doesn't analyze empty list: IsAllDiametersSelected() should be called in client. */
	bool CheckDiameter(size_t _nIndex) const;
	// Returns 'true' if specified diameters are selected and should be analyzed. Doesn't analyze empty list: IsAllDiametersSelected() should be called in client.
	bool CheckDiameter(double _dDiam1, double _dDiam2) const;
	/* Returns 'true' if spheres with specified diameters are selected and should be analyzed.
	 * Both indexes must belong to spheres - there is no internal checks of indexes or corresponding objects.
	 * Doesn't analyze empty list: IsAllDiametersSelected() should be called in client. */
	bool CheckDiameter(size_t _nIndex1, size_t _nIndex2) const;

	//////////////////////////////////////////////////////////////////////////
	/// Analyzers

	// Returns list of particles filtered by volume constraints. _vIndexes will be filtered if specified, or global list of objects from SystemStructure otherwise.
	std::set<size_t> ApplyVolumeFilter(double _dTime, unsigned _nObjType, std::set<size_t>* _vIndexes = nullptr) const;
	// Filter specified coordinates by all selected volumes and returns indexes of passed.
	std::set<size_t> ApplyVolumeFilter(double _dTime, const std::vector<CVector3>& _vCoords) const;
	// Returns list of objects with type _nObjType filtered by material constraints. _vIndexes will be filtered if specified, or global list of objects from SystemStructure otherwise.
	std::set<size_t> ApplyMaterialFilter(double _dTime, unsigned _nObjType, std::set<size_t>* _vIndexes = nullptr) const;
	// Returns list of objects with type _nObjType filtered by diameters constraints. _vIndexes will be filtered if specified, or global list of objects from SystemStructure otherwise.
	std::set<size_t> ApplyDiameterFilter(double _dTime, unsigned _nObjType, std::set<size_t>* _vIndexes = nullptr) const;

	// Returns a list of objects filtered by materials.
	[[nodiscard]] std::set<size_t> ApplyMaterialFilter(const std::set<size_t>& _ids) const;
	// Returns a list of objects filtered by diameters.
	[[nodiscard]] std::set<size_t> ApplyDiameterFilter(const std::set<size_t>& _ids) const;
	// Returns a list of objects filtered by position in volumes at the selected time point.
	[[nodiscard]] std::set<size_t> ApplyVolumeFilter(const std::set<size_t>& _ids, double _time) const;

	// Returns list of active particles filtered by all active constraints.
	std::vector<size_t> FilteredParticles(double _dTime) const;
	// Returns list of active bonds filtered by all active constraints.
	std::vector<size_t> FilteredBonds(double _dTime) const;
};

