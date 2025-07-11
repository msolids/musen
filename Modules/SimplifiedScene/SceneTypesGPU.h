/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicGPUFunctions.cuh"
#include "Quaternion.h"
#include <vector>

#define ADD_FIELD(x) AddField((void**)&x, sizeof(*x));

// General structure to describe set of arrays for specific object on GPU.
struct SBasicGPUStruct
{
	enum class EMemType { DEVICE, HOST };

protected:
	EMemType m_memType; // where to store data: on device or on host
public:
	size_t nElements;
	size_t nAllocated; //memory size which was already allocated
	std::vector<void**> vPointers; // list of pointers on all data structures
	std::vector<size_t> vTypeSize; // in bytes

	SBasicGPUStruct(EMemType _memType = EMemType::DEVICE) : m_memType(_memType), nElements(0), nAllocated(0) { }
	virtual ~SBasicGPUStruct()
	{
		Clear();
	}

	void Resize(size_t _nElements) // allocate memory for specific number of elements
	{
		if (nAllocated >= _nElements)
		{
			nElements = _nElements;
			return;
		}
		Clear();
		switch (m_memType)
		{
		case EMemType::DEVICE:
			for (size_t i = 0; i < vPointers.size(); ++i)
				CUDA_MALLOC_D(vPointers[i], _nElements*vTypeSize[i]);
			break;
		case EMemType::HOST:
			for (size_t i = 0; i < vPointers.size(); ++i)
				CUDA_MALLOC_H(vPointers[i], _nElements*vTypeSize[i]);
			break;
		}
		nAllocated = _nElements;
		nElements = _nElements;
	}

	void CopyFrom(const SBasicGPUStruct& _source)
	{
		Resize(_source.nElements); // resize if needed
		if (!_source.nElements) return;
		if (_source.m_memType == EMemType::DEVICE && m_memType == EMemType::HOST)			// device to host
			for (size_t i = 0; i < vPointers.size(); ++i)
				CUDA_MEMCPY_D2H(*vPointers[i], *_source.vPointers[i], _source.vTypeSize[i] * _source.nElements);
		else if (_source.m_memType == EMemType::HOST && m_memType == EMemType::DEVICE)		// host to device
			for (size_t i = 0; i < vPointers.size(); ++i)
				CUDA_MEMCPY_H2D(*vPointers[i], *_source.vPointers[i], _source.vTypeSize[i] * _source.nElements);
		else if (_source.m_memType == EMemType::DEVICE && m_memType == EMemType::DEVICE)	// device to device
			for (size_t i = 0; i < vPointers.size(); ++i)
				CUDA_MEMCPY_D2D(*vPointers[i], *_source.vPointers[i], _source.vTypeSize[i] * _source.nElements);
		else if (_source.m_memType == EMemType::HOST && m_memType == EMemType::HOST)		// host to host
			for (size_t i = 0; i < vPointers.size(); ++i)
				CUDA_MEMCPY_H2H(*vPointers[i], *_source.vPointers[i], _source.vTypeSize[i] * _source.nElements);
	}

	void Clear()
	{
		if (!nAllocated) return;
		switch (m_memType)
		{
		case EMemType::DEVICE:
			for (auto& pointer : vPointers)
				CUDA_FREE_D(*pointer);
			break;
		case EMemType::HOST:
			for (auto& pointer : vPointers)
				CUDA_FREE_H(*pointer);
			break;
		}
		nAllocated = nElements = 0;
	}

protected:
	void AddField(void** _pointer, size_t _bytes)
	{
		vPointers.push_back(_pointer);
		vTypeSize.push_back(_bytes);
	}
};

// GPU definition of particle
struct SGPUParticles : SBasicGPUStruct
{
	CVector3* Coords;
	CVector3* CoordsVerlet;
	CVector3* Vels;
	CVector3* AnglVels;
	CVector3* Forces;
	CVector3* Moments;
	CQuaternion* Quaternions;
	double* Masses;
	double* Radii;
	double* ContactRadii;
	double* InertiaMoments;
	unsigned* InitIndexes;  // corresponds to the indexes in the initial array (in the systemStructure) for real particles. for virtual it corresponds to index of particle in simplified scene
	unsigned* CompoundIndices;
	unsigned* Activities;		// List of flags for particles activity.
	double* StartActivities;	// Time point when particles appear.
	double* EndActivities;		// End time of activity for particles.
	double* Temperatures;
	double* HeatFluxes;
	double* HeatCapacities;
	double* TempDouble1;		// Temporary data to use in GPU::GetMaxParticleVelocity().
	double* TempDouble2;		// Temporary data to use in GPU::GetMaxParticleVelocity().
	unsigned *TempUInt;			// Temporary data to use in GPU::CheckParticlesInDomain().

private:
	void Init()
	{
		ADD_FIELD(Coords);
		ADD_FIELD(CoordsVerlet);
		ADD_FIELD(Vels);
		ADD_FIELD(AnglVels);
		ADD_FIELD(Forces);
		ADD_FIELD(Moments);
		ADD_FIELD(Quaternions);
		ADD_FIELD(Masses);
		ADD_FIELD(Radii);
		ADD_FIELD(ContactRadii);
		ADD_FIELD(InitIndexes);
		ADD_FIELD(InertiaMoments);
		ADD_FIELD(CompoundIndices);
		ADD_FIELD(Activities);
		ADD_FIELD(StartActivities);
		ADD_FIELD(EndActivities);
		ADD_FIELD(Temperatures);
		ADD_FIELD(HeatFluxes);
		ADD_FIELD(HeatCapacities);
		ADD_FIELD(TempDouble1);
		ADD_FIELD(TempDouble2);
		ADD_FIELD(TempUInt);
	}
public:
	SGPUParticles(EMemType _memType = EMemType::DEVICE) : SBasicGPUStruct(_memType) { Init(); }
	SGPUParticles(size_t _size, EMemType _memType = EMemType::DEVICE) : SBasicGPUStruct(_memType) { Init(); Resize(_size); }
};

// GPU definition of wall
struct SGPUWalls : SBasicGPUStruct
{
	CVector3* Vertices1;
	CVector3* Vertices2;
	CVector3* Vertices3;
	CVector3* NormalVectors;
	CVector3* MinCoords;		// minimal coordinates of the bounding box
	CVector3* MaxCoords;		// maximal coordinates of the bounding box
	CVector3* Vels;
	CVector3* Forces;
	CVector3* RotVels;
	CVector3* RotCenters;
	unsigned* CompoundIndices;
	double* TempVels1;			// Temporary data to use in GPU::GetMaxWallVelocity()
	double* TempVels2;			// Temporary data to use in GPU::GetMaxWallVelocity()
private:
	void Init()
	{
		ADD_FIELD(Vertices1);
		ADD_FIELD(Vertices2);
		ADD_FIELD(Vertices3);
		ADD_FIELD(NormalVectors);
		ADD_FIELD(MinCoords);
		ADD_FIELD(MaxCoords);
		ADD_FIELD(Vels);
		ADD_FIELD(Forces);
		ADD_FIELD(RotVels);
		ADD_FIELD(RotCenters);
		ADD_FIELD(CompoundIndices);
		ADD_FIELD(TempVels1);
		ADD_FIELD(TempVels2);
	}
public:
	SGPUWalls(EMemType _memType = EMemType::DEVICE) : SBasicGPUStruct(_memType) { Init(); }
	SGPUWalls(size_t _size, EMemType _memType = EMemType::DEVICE) : SBasicGPUStruct(_memType) { Init(); Resize(_size); }
};

// GPU definition of solid bond
struct SGPUSolidBonds : SBasicGPUStruct
{
	unsigned* Activities;
	unsigned* LeftIDs;
	unsigned* RightIDs;
	// const parameters
	unsigned* InitialIndices;
	double* Diameters;					// Bond diameter
	double* CrossCuts;
	double* InitialLengths;
	double* NormalStrengths;
	double* TangentialStrengths;
	// material properties
	double* Viscosities;
	double* TimeThermExpCoeffs;
	double* NormalStiffnesses;
	double* TangentialStiffnesses;		// [Pa]
	double* YieldStrengths;				// [Pa]
	double* AxialMoments;
	// non-const parameters
	double* EndActivities;
	CVector3* TangentialOverlaps;
	CVector3* PrevBonds;
	// important for plastic model
	double* NormalPlasticStrains;		// In [-]
	CVector3* TangentialPlasticStrains;	// In [-,-,-]
	double* ThermalConductivities;
	// forces and moments
	CVector3* TotalForces;				// Normal + Tangential
	CVector3* NormalMoments;
	CVector3* TangentialMoments;
private:
	void Init()
	{
		ADD_FIELD(Activities);
		ADD_FIELD(LeftIDs);
		ADD_FIELD(RightIDs);
		ADD_FIELD(InitialIndices);
		ADD_FIELD(Diameters);
		ADD_FIELD(CrossCuts);
		ADD_FIELD(InitialLengths);
		ADD_FIELD(NormalStrengths);
		ADD_FIELD(TangentialStrengths);
		ADD_FIELD(Viscosities);
		ADD_FIELD(TimeThermExpCoeffs);
		ADD_FIELD(NormalStiffnesses);
		ADD_FIELD(TangentialStiffnesses);
		ADD_FIELD(YieldStrengths);
		ADD_FIELD(AxialMoments);
		ADD_FIELD(EndActivities);
		ADD_FIELD(TangentialOverlaps);
		ADD_FIELD(PrevBonds);
		ADD_FIELD(NormalPlasticStrains);
		ADD_FIELD(TangentialPlasticStrains);
		ADD_FIELD(ThermalConductivities);
		ADD_FIELD(TotalForces);
		ADD_FIELD(NormalMoments);
		ADD_FIELD(TangentialMoments);
	}
public:
	SGPUSolidBonds(EMemType _memType = EMemType::DEVICE) : SBasicGPUStruct(_memType) { Init(); }
	SGPUSolidBonds(size_t _size, EMemType _memType = EMemType::DEVICE) : SBasicGPUStruct(_memType) { Init(); Resize(_size); }
};

// GPU definition of collision
struct SGPUCollisions : SBasicGPUStruct
{
	unsigned* SrcIDs;			// Identifier of first contact partner or wall (nWallID).
	unsigned* DstIDs;			// Identifier of second contact partner.
	double* NormalOverlaps;
	double* InitNormalOverlaps;
	double* EquivMasses;		// Equivalent mass.
	double* EquivRadii;			// Equivalent radius.
	double* SumRadii;			// summarized radii
	uint16_t* InteractPropIDs;	// ID of interaction properties
	bool* ActivityFlags;		// List of active collisions in form of flags for particle-particle contacts.
	unsigned* ActivityIndices;	// List of active collisions in form of indices for particle-particle contacts.
	CVector3* TangOverlaps;		// Old tangential overlap.
	CVector3* TotalForces;
	uint8_t* VirtualShifts;		// Virtual shift if this is a virtual contact.
	CVector3* ContactVectors;	// For PP contact: dstCoord - srcCoord. For PW contact: contact point.

	// Not in the list of pointers
	unsigned* ActiveCollisionsNum;	// Number of currently active collisions.

private:
	void Init()
	{
		ADD_FIELD(SrcIDs);
		ADD_FIELD(DstIDs)
		ADD_FIELD(NormalOverlaps);
		ADD_FIELD(InitNormalOverlaps);
		ADD_FIELD(EquivMasses);
		ADD_FIELD(EquivRadii);
		ADD_FIELD(SumRadii);
		ADD_FIELD(InteractPropIDs);
		ADD_FIELD(ActivityFlags);
		ADD_FIELD(ActivityIndices);
		ADD_FIELD(TangOverlaps);
		ADD_FIELD(TotalForces);
		ADD_FIELD(VirtualShifts);
		ADD_FIELD(ContactVectors);
	}
public:
	SGPUCollisions(EMemType _memType = EMemType::DEVICE) : SBasicGPUStruct(_memType) { Init(); InitInternal(); }
	SGPUCollisions(size_t _size, EMemType _memType = EMemType::DEVICE) : SBasicGPUStruct(_memType) { Init(); Resize(_size); InitInternal(); }
	~SGPUCollisions()
	{
		switch (m_memType)
		{
		case EMemType::DEVICE:	CUDA_FREE_D(ActiveCollisionsNum);	break;
		case EMemType::HOST:	CUDA_FREE_H(ActiveCollisionsNum);	break;
		}
	}
	void CopyFrom(const SGPUCollisions& _source)
	{
		SBasicGPUStruct::CopyFrom(_source);
		if (_source.m_memType == EMemType::DEVICE && m_memType == EMemType::HOST)			// device to host
			CUDA_MEMCPY_D2H(ActiveCollisionsNum, _source.ActiveCollisionsNum, sizeof(*ActiveCollisionsNum));
		else if (_source.m_memType == EMemType::HOST && m_memType == EMemType::DEVICE)		// host to device
			CUDA_MEMCPY_H2D(ActiveCollisionsNum, _source.ActiveCollisionsNum, sizeof(*ActiveCollisionsNum));
		else if (_source.m_memType == EMemType::DEVICE && m_memType == EMemType::DEVICE)	// device to device
			CUDA_MEMCPY_D2D(ActiveCollisionsNum, _source.ActiveCollisionsNum, sizeof(*ActiveCollisionsNum));
		else if (_source.m_memType == EMemType::HOST && m_memType == EMemType::HOST)		// host to host
			CUDA_MEMCPY_H2H(ActiveCollisionsNum, _source.ActiveCollisionsNum, sizeof(*ActiveCollisionsNum));
	}
private:
	void InitInternal()
	{
		switch (m_memType)
		{
		case EMemType::DEVICE:	CUDA_MALLOC_D(&ActiveCollisionsNum, sizeof(*ActiveCollisionsNum));	break;
		case EMemType::HOST:	CUDA_MALLOC_H(&ActiveCollisionsNum, sizeof(*ActiveCollisionsNum));	break;
		}
	}
};

#undef ADD_FIELD