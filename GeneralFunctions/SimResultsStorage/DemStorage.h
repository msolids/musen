/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "MDEMFile.h"
#include "TimePointR.h"
#include "TimePointW.h"

#define TIME_INDEPENDENT_DATA	0
#define TIME_DEPENDENT_DATA		1

class CDemStorage
{
private:
	// General information and time-independent data.
	ProtoSimulationStorage m_simStorage;

	// Vector of pointers to proto time-dependent data blocks, where each block is contained a group of time points with time-dependent data for each time point.
	std::vector<std::unique_ptr<ProtoBlockOfTimePoints>> m_blocksOfTimePoints;

	std::shared_ptr<CMDEMFile> m_pMDEMFile; // Pointer to m_pMDEMfile.

	uint32_t	 m_iLeftLoadedBlock;	// Index of left loaded block.
	double	     m_timeEnd;				// Last successfully loaded time point.
	std::string  m_sLastError;			// String with last error description.

	// Time point cache structure.
	// It contains value of time point and pointer to proto time point (contains time-dependent data for all objects).
	struct STimePointCache
	{
		double time{ -1 };
		ProtoTimePoint* pTimePoint{ nullptr };
	};

	// Time points pair cache structure.
	// It contains value of requested time and pointer to pair of proto time points (are needed to get requested time by interpolation).
	struct STimePointsPairCache
	{
		double requestedTime{ -1 };
		ProtoTimePoint* pTimePointL{ nullptr }; // Left time point for interpolation.
		ProtoTimePoint* pTimePointR{ nullptr }; // Right time point for interpolation.
	};

	STimePointCache m_SWriteCache; // It is used to cache data when writing to file.
	STimePointsPairCache m_SCache; // It is used to cache data when reading from file.

	CTimePointR m_timePointR; // Time point for write access.
	CTimePointW m_timePointW; // Time point for write access.

public:
	CDemStorage();

	// Returns true if file is existed and has not error.
	bool IsValid() const;
	// Returns string with description of the last error.
	std::string GetLastError() const;
	// Opens file _sFileName.
	void OpenFile(const std::string& _sFileName);
	// Loads initial data from mdem file to RAM (mdem file headers, time-independent data and first block of time-dependent data are loaded). OpenFile() function has to be called before.
	bool LoadFromFile();
	// Creates new mdem file (mdem file headers with default values are written).
	bool CreateNewFile(const std::string& _sFileName);
	// Writes data from RAM into mdem file.
	void FlushToDisk(bool _isSaveLastBlock);
	// Truncates file to significant data. Has to be called at the end of simulation process.
	void FinalTruncate();
	// Returns file version from file header. OpenFile() function has to be called before.
	uint32_t GetFileVersion() const;
	// Sets file version to file header. OpenFile() function has to be called before.
	void SetFileVersion(uint32_t _nFileVersion) const;

	// Returns pointer to mutable simulation info in protofile.
	ProtoSimulationInfo* SimulationInfo();
	// Returns pointer to simulation info in protofile.
	const ProtoSimulationInfo* SimulationInfo() const;
	// Returns pointer to mutable modules data in protofile.
	ProtoModulesData* ModulesData();

	// Returns value of last saved time point.
	double GetRealTimeEnd() const;
	// Gets saved time points.
	std::vector<double> GetAllTimePoints() const;
	// Gets saved time points for old format file.
	std::vector<double> GetAllTimePointsOldFormat();

	// Remove all time-dependent data after _dTime.
	void RemoveAllAfterTime(double _dTime);
	// Remove all data.
	void RemoveAll();

	// Copies all data from old storage to current storage.
	bool CopyFrom(CDemStorage *_pOldStorage);

	// Returns pointer to ProtoParticleInfo of object with index = _nObjectIndex.
	ProtoParticleInfo* Object(uint32_t _nObjectIndex);
	// Removes object with index = _nObjectIndex from protofile.
	void RemoveObject(uint32_t _nObjectIndex);
	// Returns number of all objects in protofile.
	uint32_t ObjectsCount();

	//// Returns pointer to time point structure for reading time-dependent data.
	//const TimePoint* GetTimePoint(double _dTime, uint32_t _nObjectIndex);

	// Makes initialization to allow read from the selected time point. _objectsCount is an amount of objects that will be available for read.
	void PrepareTimePointForRead(double _time, int _objectsCount);
	// Returns pointer to a time point for reading time-dependent data. Can not be accessed in parallel.
	CTimePointR* GetTimePointR(double _time, int _objectID);
	// Returns pointer to a time point for reading time-dependent data. The return value can be accessed in parallel.
	CTimePointR* GetTimePointR();

	// Makes initialization to allow write to the selected time point. _objectsCount is an amount of objects that will be available for write.
	void PrepareTimePointForWrite(double _time, int _objectsCount);
	// Returns pointer to a mutable time point for writing time-dependent data. Can not be accessed in parallel.
	CTimePointW* GetTimePointW(double _time, int _objectID);
	// Returns pointer to a mutable time point for writing time-dependent data. The return value can be accessed in parallel.
	CTimePointW* GetTimePointW();

private:
	// Returns index of block with start time <= _dTime.
	uint32_t FindBlockByTime(double _dTime) const;
	// Returns pointers to left and right time points relatively _dTIime.
	static STimePointsPairCache FindPairInBlock(ProtoBlockOfTimePoints& _protoBlockOfTimePoints, double _dTime);
	// Save block with time-dependent data to file.
	void SaveLastBlock();
	// Loads time-dependent data block wtih _nBlockIndex and next data block into RAM.
	bool LoadBlock(uint32_t _iBlock);
	// Loads time-dependent data block with _nBlockIndex into RAM.
	bool LoadBlockInternal(uint32_t _iBlock);
	// Unload unused time-dependent data blocks from RAM.
	void UnloadUnusedBlocks(uint32_t _iBlockInUse);
	// Initializes empty file.
	bool InitEmptyFile();
	// Reads time-independent data and two time-dependent data blocks.
	bool ReadInitialData();

	// Serialize and write data from  protobuf _message to buffer  _pBuffer, returns size of serialized data.
	static int32_t WriteToBuf(char *&_pBuffer, const google::protobuf::Message& _message);
	// Read and desirialize data from _pBuffer to protobufe _message, returns true if success.
	static bool ReadFromBuf(const char *_pBuffer, google::protobuf::Message& _message, int _nSize, int _sizeFull);

	// Returns pair of pointers to proto time points which contain time-dependent data for all objects.
	STimePointsPairCache& TimeDependentProperties(double _dTime);
	// Returns pointer to proto time point for writing. New block is allocated in file if it is needed.
	ProtoTimePoint* MutableTimeDependentProperties(double _dTime);
};