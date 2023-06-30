#pragma once

#include "SystemStructure.h"
#include "SceneOptionalVariables.h"
#include "SceneTypes.h"

/* This class is compressed representation of the data from the system structure which will be used for the calculation. */
class CSimplifiedScene
{
private:
	struct SObjects
	{
		std::shared_ptr<SParticleStruct> vParticles;
		std::shared_ptr<SSolidBondStruct> vSolidBonds;
		std::shared_ptr<SLiquidBondStruct> vLiquidBonds;
		std::shared_ptr<SWallStruct> vWalls;
		std::shared_ptr<SMultiSphere> vMultiSpheres;
		size_t nVirtualParticles;
	};
	// precalculated properties of compounds' interactions. this is a 2D symmetric matrix stored as 1D array
	std::shared_ptr<std::vector<SInteractProps>> m_vInteractProps;
	std::shared_ptr<std::vector<std::vector<unsigned>>> m_vParticlesToSolidBonds; // array contain information about indexes of bonds which are connected to specific particle

	SObjects m_Objects;	// all objects for consideration in the scene, including virtual ones (in case of periodic boundary conditions)
public:
	std::vector<size_t> m_vNewIndexes; // corresponds to the new indexes

	//////////////////////////////////////////////////////////////////////////
	// Periodic Boundary Conditions
	SPBC m_PBC;							   // Information about periodic boundary conditions
	std::vector<uint8_t> m_vPBCVirtShift; // Information to shift real particle in order to find position of its virtual one. For BOX: {x, y, z}, for CYLINDER: {cos(a), sin (a), 0}. The vector length is equal to the number of virtual particles.
	//////////////////////////////////////////////////////////////////////////

	std::vector<std::vector<unsigned>> m_adjacentWalls; // Contains list of adjacent walls for each wall.

private:
	CSystemStructure* m_pSystemStructure;
	size_t m_vCompoundsNumber; // number of unique compounds in this simplified scene

	SOptionalVariables m_ActiveVariables;
public:
	CSimplifiedScene();
	~CSimplifiedScene();

	void SetSystemStructure(CSystemStructure* _pSystemStructure);
	void InitializeScene(double _dStartTime,const SOptionalVariables& _activeOptionalVariables);	// Initialize scene by system structure at specific time point
	void AddParticle(size_t _index, double _dTime);				// Add new particle from the system structure
	void AddSolidBond(size_t _index, double _dTime);			// Add new solid bond from the system structure
	void AddLiquidBond(size_t _index, double _dTime);			// Add new liquid bond from the system structure
	void AddWall(size_t _index, double _dTime);					// Add new wall from the system structure
	void AddMultisphere(const std::vector<size_t>& _vIndexes);	// Add new multisphere from the system structure

	void InitializeMaterials();			// Parameters and interactions of materials
	void ClearState() const;            // Sets current values of running variables (force, moment, heat flux) to 0.
	void ClearHeatFluxes() const;

	void AddVirtualParticles(double _dVerletDistance);
	void RemoveVirtualParticles();

	void SaveVerletCoords(); // save coordinates of all objects which were used for calculation of verlet
	double GetMaxPartVerletDistance(); // get maximal distance which was made by particle from last verlet updata

	void GetAllParticlesInVolume(const SVolumeType& _volume, std::vector<unsigned>* _pvIndexes) const;
	void GetAllWallsInVolume(const SVolumeType& _volume, std::vector<unsigned>* _pvIndexes) const;

	double GetMaxParticleVelocity() const;
	double GetMaxParticleTemperature() const;
	double GetMaxParticleRadius() const;
	double GetMinParticleRadius() const;
	double GetMaxParticleContactRadius() const;
	double GetMinParticleContactRadius() const;
	double GetMaxWallVelocity() const;

	void UpdateParticlesToBonds();

	SPBC GetPBC() const { return m_PBC; }

	// a set of slow function // should be called often
	CVector3 GetObjectCoord(size_t _index) const { return m_Objects.vParticles->Coord(_index); };
	CVector3 GetObjectVel(size_t _index) const { return m_Objects.vParticles->Vel(_index); };
	CVector3 GetObjectAnglVel(size_t _index) const { return  m_Objects.vParticles->AnglVel(_index); };
	double   GetParticleTemperature(size_t _index) const;

	inline size_t GetCompoundsNumber() const { return m_vCompoundsNumber; }
	inline std::shared_ptr<std::vector<SInteractProps>> GetPointerToInteractProperties() { return m_vInteractProps; }
	inline SInteractProps& GetInteractProp(size_t _nPropIndex) { return (*m_vInteractProps)[_nPropIndex]; }

	SParticleStruct& GetRefToParticles() { return *m_Objects.vParticles; }
	const SParticleStruct& GetRefToParticles() const { return *m_Objects.vParticles; }


	SSolidBondStruct& GetRefToSolidBonds(){ return *m_Objects.vSolidBonds; }

	SLiquidBondStruct& GetRefToLiquidBonds(){ return *m_Objects.vLiquidBonds; }

	SWallStruct& GetRefToWalls() {return *m_Objects.vWalls; }
	const SWallStruct& GetRefToWalls() const { return *m_Objects.vWalls; }

	SMultiSphere& GetRefToMultispheres(){ return *m_Objects.vMultiSpheres; }


	std::shared_ptr<SParticleStruct> GetPointerToParticles() { return m_Objects.vParticles; }
	std::shared_ptr<SSolidBondStruct> GetPointerToSolidBonds() { return m_Objects.vSolidBonds; }
	std::shared_ptr<SLiquidBondStruct> GetPointerToLiquidBonds() { return m_Objects.vLiquidBonds; }
	std::shared_ptr<SWallStruct> GetPointerToWalls() { return m_Objects.vWalls; }


	std::shared_ptr<std::vector<std::vector<unsigned>>> GetPointerToPartToSolidBonds() { return m_vParticlesToSolidBonds; }

	size_t GetTotalParticlesNumber() const { return m_Objects.vParticles->Size(); }
	size_t GetVirtualParticlesNumber()const { return m_Objects.nVirtualParticles;  }
	size_t GetRealParticlesNumber()const { return m_Objects.vParticles->Size() - m_Objects.nVirtualParticles; }
	size_t GetWallsNumber() const { return m_Objects.vWalls->Size(); }
	size_t GetBondsNumber() const { return m_Objects.vSolidBonds->Size(); }
	size_t GetLiquidBondsNumber() const { return m_Objects.vLiquidBonds->Size(); }
	size_t GetMultiSpheresNumber() const { return m_Objects.vMultiSpheres->Size(); }

	double GetDistance(unsigned _nIndex1, unsigned _nIndex2) const { return Length(m_Objects.vParticles->Coord(_nIndex1), m_Objects.vParticles->Coord(_nIndex2)); }
protected:
	void ClearAllData(); // Deletes all initialized memory
	void InitializeLiquidBondsCharacteristics(double _dTime = 0); // Sets the initial length of bonds and sets the new indexes of particles in the new array // !!!!!!!!!!!!! WTF?
	void InitializeGeometricalObjects(double _dTime);
	SInteractProps CalculateInteractionProperty(const std::string& _sCompound1, const std::string& _sCompound2) const;
	void AddVirtualParticleBox(size_t _nSourceID, const CVector3& _vShift);

	void FindAdjacentWalls(); // Constructs a list of adjacent walls for each wall.
};


