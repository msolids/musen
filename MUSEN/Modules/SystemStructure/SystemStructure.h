/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include <memory>
#include "Sphere.h"
#include "TriangularWall.h"
#include "SolidBond.h"
#include "LiquidBond.h"
#include "InsideVolumeChecker.h"
#include "ContactCalculator.h"
#include "STLFileHandler.h"
#include "MaterialsDatabase.h"
#include "ProtoFunctions.h"
#include "DemStorage.h"
#include "RealGeometry.h"

class CDemStorage;

/** This class is used for describing the structure of the whole system. */
class CSystemStructure
{
public:
	enum class ELoadFileResult
	{
		OK,
		IsNotDEMFile,
		PartiallyLoaded,
		SelectivelySaved
	};

	CMaterialsDatabase m_MaterialDatabase;

private:
	std::vector<CPhysicalObject*> objects;				// all physical objects in the system
	std::vector<std::vector<size_t>> m_Multispheres;
	std::vector<std::unique_ptr<CRealGeometry>> m_geometries;			// List of defined real geometries.
	std::vector<std::unique_ptr<CAnalysisVolume>> m_analysisVolumes;	// List of defined analysis volumes.

	//////////////////////////////////////////////////////////////////////////
	/// Simulation and scene settings

	SVolumeType m_SimulationDomain;	// Defines the volume where all objects are situated.
	mutable SPBC m_PBC;				// Periodic boundary conditions.
	bool m_bAnisotropy;				// If enabled anisotropy of non-spherical objects will be considered during simulation.
	bool m_bContactRadius;			// If enabled contact radius of particles will be available.

	std::string m_sDEMFileName; // the name of the file from which the data has been loaded

	/* This vector contains the key of local compounds to obtain accordance between local compounds and global database */
	std::shared_ptr<CDemStorage> m_storage;

private:
	/** The objects are connected with a bonds. Each bond have an indexes of the object, which connected by this bond.
	The UpdateBondedIDs() insert into the objects indexes of the bonds, by which they are connected. */
	void UpdateBondedIDs();

	void Compress(); // delete all empty objects from the end of system structure

	void FlushToStorage();
	void Reset();

public:
	CSystemStructure();
	~CSystemStructure();

	void ClearAllData();

	ELoadFileResult LoadFromFile(const std::string& _sFileName);
	void SaveToFile(const std::string& _sFileName = "");
	void NewFile();
	// Reduces size of simulated file by removing free space in file. Should be called when simulation is finished.
	void FinalFileTruncate();

	// Import data from DEM Text format
	void ImportFromEDEMTextFormat(const std::string& _sFileName);
	// Export geometries as STL
	void ExportGeometriesToSTL(const std::string& _sFileName, double _dTime);

	// make a copy of source system structure, however copy data only from one time point, which will be defined as zero time point
	void CreateFromSystemStructure(CSystemStructure* _pSource, double _dTime);

	// deletes on object from the system
	void DeleteObjects(const std::vector<size_t>& _vIndexes); // delete all objects with specified indexes
	void DeleteAllObjects(); // deletes all objects from the system
	void DeleteAllParticles();
	size_t DeleteAllNonConnectedParticles(); // returns number of removed particles
	size_t DeleteAllParticlesWithNoContacts(); // removes particle with coordination number 0, returns number of removed particles
	void DeleteAllBonds(); // delete liquid and solid bonds

	void ClearAllStatesFrom(double _dTime); // clear all states from specified time point up to the last time point

	size_t GetTotalObjectsCount() const; // returns the number of all objects in the system
	size_t GetNumberOfSpecificObjects(unsigned _nObjectType); // return number of specific objects in the scene
	unsigned GetMultispheresNumber() const { return static_cast<unsigned>(m_Multispheres.size()); }

	/** This function returns the vector which contains the indexes of all objects which related to input parameters */
	void GetAllObjectsOfSpecifiedCompound(double _dTime, std::vector<CPhysicalObject*>* _vecIndexes, unsigned _nObjectType, const std::string& _sCompoundKey = "");

	std::vector<CPhysicalObject*> GetAllActiveObjects(double _dTime, unsigned _nObjectType = UNKNOWN_OBJECT);
	std::vector<CSphere*> GetAllSpheres(double _time, bool _onlyActive = true);
	std::vector<const CSphere*> GetAllSpheres(double _time, bool _onlyActive = true) const;
	std::vector<CSolidBond*> GetAllSolidBonds(double _time, bool _onlyActive = true) const;
	std::vector<CLiquidBond*> GetAllLiquidBonds(double _time, bool _onlyActive = true) const;
	std::vector<CBond*> GetAllBonds(double _time, bool _onlyActive = true) const;
	std::vector<CTriangularWall*> GetAllWalls(double _time, bool _onlyActive = true);
	std::vector<const CTriangularWall*> GetAllWalls(double _time, bool _onlyActive = true) const;
	std::vector<CTriangularWall*> GetAllWallsForGeometry(double _time, const std::string& _geometryKey, bool _onlyActive = true);

	bool IsParticlesExist() const;		// Returns whether particles are defined on scene.
	bool IsSolidBondsExist() const;		// Returns whether solid bonds are defined on scene.
	bool IsLiquidBondsExist() const;	// Returns whether liquid bonds are defined on scene.
	bool IsBondsExist() const;			// Returns whether bonds are defined on scene.
	bool IsWallsExist() const;			// Returns whether walls are defined on scene.

	// Goes through all objects and analyze Rayleigh time step
	double GetRecommendedTimeStep(double _dTime = 0) const; // for all spherical objects

	/** This function allocates memory for the new object with specified type.
	It returns the pointer to this object. if the object with specified ID already exists, than  the memory will not be allocated */
	CPhysicalObject* AddObject(unsigned objectType);
	CPhysicalObject* AddObject(unsigned _objectType, size_t _nObjectID);
	std::vector<CPhysicalObject*> AddSeveralObjects(unsigned _objectType, size_t _nObjectsCount);
	std::vector<size_t> GetFreeIDs(size_t _objectsCount); // Returns the list of IDs, available to add _objectsCount new objects.
	void AddMultisphere(const std::vector<size_t>& _vIndexes); // specify new multisphere which consists from several objects

	// returns the object reference due to the given ID
	CPhysicalObject* GetObjectByIndex(size_t _ObjectID);
	const CPhysicalObject* GetObjectByIndex(size_t _ObjectID) const;

	std::vector<size_t> GetMultisphere(unsigned _nIndex);
	long long int GetMultisphereIndex(unsigned _nParticleIndex);

	// returns the volume of the bond
	double GetBondVolume(double _dTime, size_t _nBondID);
	CVector3 GetBondVelocity(double _dTime, size_t _nBondID);
	CVector3 GetBondCoordinate(double _dTime, size_t _nBondID) const;
	CVector3 GetBond(double _dTime, size_t _nBondID) const;

	// return the maximal time for which the objects have been defined
	double GetMaxTime() const;
	double GetMinTime() const;
	std::vector<double> GetAllTimePoints() const;
	std::vector<double> GetAllTimePointsOldFormat();

	// return maximal/minimal values of coordinates for specified time point
	SVolumeType GetBoundingBox(double _time = 0) const;
	CVector3 GetMaxCoordinate(const double _dTime = 0);
	CVector3 GetMinCoordinate(const double _dTime = 0);

	// returns the number of particles in the classes for specified time point (agglomerates)
	std::vector<unsigned> GetFragmentsPSD(double _dTime, double _dMin, double _dMax, unsigned _nClasses);
	std::vector<unsigned> GetPrimaryPSD(double _dTime, double _dMin, double _dMax, unsigned _nClasses); // returns the PSD of primary particles

	// return the smallest diameter from all particle; // -1 returns if there are no particles
	double GetMinParticleDiameter() const;
	double GetMaxParticleDiameter() const;

	// functions returns the ID of a particles which are connected to the particle with a specified ID
	void GetGroup(double _dTime, size_t _nSourceParticleID, std::vector<size_t>* _pVecGroup);
	/// Returns IDs of all particles, which are in the same agglomerate as _particleID.
	std::vector<size_t> GetAgglomerate(double _time, size_t _particleID);

	// Returns file name
	std::string GetFileName() const { return m_sDEMFileName; }

	/** Calculates the center of a mass of a system for specified time point.
	Now it is considered just spheres. The center of a mass can be also given for a part of a system. */
	CVector3 GetCenterOfMass(double _dTime, size_t _nFirstObjectID = 0, size_t _nLastObjectID = 0);

	/** rotate the system above the center of a mass to the specified angle
	The rotation can be performed just for a small number of spheres (agglomerate)	*/
	void RotateSystem(double _dTime, const CVector3& _RotCenter, const CVector3& _RotAngleRad, size_t _nFirstObjectID = 0, size_t _nLastObjectID = 0);

	// Add to the coordinates of all objects, on all time points additional offset
	void MoveSystem(double _dTime, const CVector3& _vOffset);

	// set specific velocity to all spheres in the scene
	void SetSystemVelocity(double _dTime, const CVector3& _vNewVelocity);

	/** Return vector which contains coordination number of all particles (coupling by bond is considered)	*/
	std::vector<unsigned> GetCoordinationNumbers(double _dTime);

	// returns the vector of indexes of particles which are situated outside of specified box volume
	// (criteria  - at least some part of particle is outside this volume
	std::vector<size_t> GetParticlesOutsideVolume(double _dTime, const CVector3& vBottomLeft, const CVector3& vTopRight) const;

	// update material properties of selected objects
	void UpdateObjectsCompoundProperties(const std::vector<size_t>& _IDs);
	// update material properties of all objects
	void UpdateAllObjectsCompoundsProperties();

	// Returns all compounds used by particles
	std::set<std::string> GetAllParticlesCompounds() const;
	// Returns all compounds used by bonds
	std::set<std::string> GetAllBondsCompounds() const;

	// Returns maximal overlap between particles for specified time point
	double GetMaxOverlap(const double& _dTime = 0);

	std::vector<double> GetMaxOverlaps(const double& _dTime);
	// Returns list of max overlaps, considering only particles with specified IDs.
	std::vector<double> GetMaxOverlaps(const double& _dTime, const std::vector<size_t>& _vIDs);
	void GetOverlaps(const double& _dTime, std::vector<unsigned>& _vID1, std::vector<unsigned>& _vID2, std::vector<double>& _vOverlap);
	void GetOverlaps(double& _dMaxOverlap, unsigned& _nMaxOverlapParticleID1, unsigned& _nMaxOverlapParticleID2, double& _dTotalOverlap, const double& _dTime = 0);
	// returns vector of bonds which are connected to specified particle.
	std::vector<CSolidBond*> GetParticleBonds(unsigned _nID);
	// check that exists all materials of all objects in the database.
	std::string IsAllCompoundsDefined() const;

	ProtoModulesData* GetProtoModulesData() const;
	ProtoSimulationInfo* GetSimulationInfo() const;

	// Makes initialization to allow for parallel read from the selected time point.
	void PrepareTimePointForRead(double _time) const;
	// Makes initialization to allow for parallel write to the selected time point.
	void PrepareTimePointForWrite(double _time) const;

	//////////////////////////////////////////////////////////////////////////
	/// Functions to work with simulation and scene settings

	// Returns volume representing current simulation domain.
	SVolumeType GetSimulationDomain() const;
	// Sets new simulation domain.
	void SetSimulationDomain(const SVolumeType& _domain);

	// Returns current settings of periodic boundary conditions.
	SPBC GetPBC() const;
	// Sets new settings of periodic boundary conditions.
	void SetPBC(const SPBC& _PBC);

	// Returns the flag whether anisotropy is currently considered.
	bool IsAnisotropyEnabled() const;
	// Sets the flag to consider anisotropy.
	void EnableAnisotropy(bool _bEnable);

	// Returns the flag whether contact radius of particles is currently considered.
	bool IsContactRadiusEnabled() const;
	// Sets the flag to consider the contact radius of particles.
	void EnableContactRadius(bool _bEnable);

	// TODO: remove unused
	//////////////////////////////////////////////////////////////////////////
	/// Functions to work with real geometries

	// Returns number of defined real geometries.
	size_t GeometriesNumber() const;
	// Returns pointers to a all real geometry.
	std::vector<CRealGeometry*> AllGeometries();
	// Returns pointers to a all real geometry.
	std::vector<const CRealGeometry*> AllGeometries() const;
	// Returns constant pointer to a specified real geometry.
	const CRealGeometry* Geometry(size_t _index) const;
	// Returns pointer to a specified real geometry.
	CRealGeometry* Geometry(size_t _index);
	// Returns constant pointer to a specified real geometry.
	const CRealGeometry* Geometry(const std::string& _key) const;
	// Returns pointer to a specified real geometry.
	CRealGeometry* Geometry(const std::string& _key);
	// Returns constant pointer to a specified real geometry.
	const CRealGeometry* GeometryByName(const std::string& _name) const;
	// Returns pointer to a specified real geometry.
	CRealGeometry* GeometryByName(const std::string& _name);
	// Returns index of a specified real geometry or -1 if geometry with such key has not been defined.
	size_t GeometryIndex(const std::string& _key) const;

	// Adds new real geometry and returns pointer to it.
	CRealGeometry* AddGeometry();
	// Adds new real geometry with provided mesh and returns pointer to it.
	CRealGeometry* AddGeometry(const CTriangularMesh& _mesh);
	// Adds new real geometry of specified type with selected parameters and returns pointer to it.
	CRealGeometry* AddGeometry(const EVolumeShape& _type, const CGeometrySizes& _sizes, const CVector3& _center);
	// Removes specified geometry.
	void DeleteGeometry(size_t _index);
	// Removes specified geometry.
	void DeleteGeometry(const std::string& _key);
	// Removes all geometries.
	void DeleteAllGeometries();
	// Moves selected geometry upwards in the list of geometries.
	void UpGeometry(const std::string& _key);
	// Moves selected geometry downwards in the list of geometries.
	void DownGeometry(const std::string& _key);

	// Returns all unique keys of real geometries.
	std::vector<std::string> GeometriesKeys() const;

	//////////////////////////////////////////////////////////////////////////
	/// Functions to work with analysis volumes

	// Returns number of defined analysis volumes.
	size_t AnalysisVolumesNumber() const;
	// Returns pointers to all specified analysis volumes.
	std::vector<CAnalysisVolume*> AllAnalysisVolumes();
	// Returns pointers to all specified analysis volumes.
	std::vector<const CAnalysisVolume*> AllAnalysisVolumes() const;
	// Returns constant pointer to a specified analysis volume.
	const CAnalysisVolume* AnalysisVolume(size_t _index) const;
	// Returns pointer to a specified analysis volume.
	CAnalysisVolume* AnalysisVolume(size_t _index);
	// Returns constant pointer to a specified analysis volume.
	const CAnalysisVolume* AnalysisVolume(const std::string& _key) const;
	// Returns pointer to a specified analysis volume.
	CAnalysisVolume* AnalysisVolume(const std::string& _key);
	// Returns constant pointer to a first analysis volume. with the specified name.
	const CAnalysisVolume* AnalysisVolumeByName(const std::string& _name) const;
	// Returns pointer to a first analysis volume. with the specified name.
	CAnalysisVolume* AnalysisVolumeByName(const std::string& _name);
	// Returns index of a specified analysis volume or -1 if volume with such key has not been defined.
	size_t AnalysisVolumeIndex(const std::string& _key) const;

	// Adds a new analysis volume and returns a pointer to it.
	CAnalysisVolume* AddAnalysisVolume();
	// Adds a new analysis volume with provided mesh and returns a pointer to it.
	CAnalysisVolume* AddAnalysisVolume(const CTriangularMesh& _mesh);
	// Adds a new analysis volume of specified type with selected parameters and returns a pointer to it.
	CAnalysisVolume* AddAnalysisVolume(const EVolumeShape& _type, const CGeometrySizes& _sizes, const CVector3& _center);
	// Removes specified analysis volume.
	void DeleteAnalysisVolume(size_t _index);
	// Removes specified analysis volume.
	void DeleteAnalysisVolume(const std::string& _key);
	// Moves selected analysis volume upwards in the list of analysis volumes.
	void UpAnalysisVolume(const std::string& _key);
	// Moves selected analysis volume downwards in the list of analysis volumes.
	void DownAnalysisVolume(const std::string& _key);

	// Returns all unique keys of analysis volumes.
	std::vector<std::string> AnalysisVolumesKeys() const;

	// Clears all time-dependent data for zero time point.
	void ClearAllTDData();
	// Returns true if file has old format.
	static bool IsOldFileVersion(const std::string& _sFileName);
	// Returns current file version.
	static uint32_t FileVersion(const std::string& _sFileName);
	// Returns current file version.
	uint32_t FileVersion() const;
};
