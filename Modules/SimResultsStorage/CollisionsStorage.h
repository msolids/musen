/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

/*	File structure

	+0		-----------------------	0
			m_cSignatureStart
			m_cVersion
	+32b	-----------------------	32b	(INFO_BLOCK_OFFSET)
			SSavedBlockInfo
	+32b	-----------------------	64b (INFO_BLOCK_OFFSET)
			......
	+32b	-----------------------
			SSavedBlockInfo
	+32b	-----------------------
			m_cSignatureEnd
	+32b	-----------------------
			......
			-----------------------	1Mb (FIRST_DATA_OFFSET)
			ProtoBlockOfCollisionsInfo
			ProtoBlockOfCollisions
			-----------------------
			ProtoBlockOfCollisionsInfo
			ProtoBlockOfCollisions
			-----------------------

	m_cSignatureStart marks that file stores collisions.
	m_cSignatureEnd mark the end of descriptors.
	Each SSavedBlockInfo describes a pair: ProtoBlockOfCollisionsInfo and ProtoBlockOfCollisions.
	Data blocks are not aligned, their positions and sizes are specified in corresponding SSavedBlockInfo.
	Maximum possible number of storing pairs: 32767. Minimum number of collisions in each block is described with COLLISIONS_NUMBER_TO_SAVE.
	Thus, guaranteed number of collisions is 32767*COLLISIONS_NUMBER_TO_SAVE. Each block, except last one, contains >=COLLISIONS_NUMBER_TO_SAVE collisions.
	Last block is usually larger, as it contains additionally all collisions, which have not been finished at the end of simulation.
*/

#pragma once

#include <fstream>
#include <cstdint>
#include "MUSENStringFunctions.h"

#define COLLISIONS_NUMBER_TO_SAVE	1000000
#define INFO_BLOCK_OFFSET			32
#define FIRST_DATA_OFFSET			1024*1024
#define COLL_FILE_EXT				".coll"

class ProtoBlockOfCollisions;
class ProtoBlockOfCollisionsInfo;
namespace google::protobuf
{
	class Message;
}

class CCollisionsStorage
{
public:
	// For storing data in RAM. This data will always be present in memory if file is loaded successfully.
	struct SRuntimeBlockInfo
	{
		uint64_t nOffset;		// Offset in file, points to corresponding ProtoBlockOfCollisionsInfo block.
		uint32_t nDescrSize;	// Size of the ProtoBlockOfCollisionsInfo block. (nOffset + nDescrSize) points to corresponding ProtoBlockOfCollisions.
		uint32_t nDataSize;		// Size of ProtoBlockOfCollisions block.
		double dTimeStart;		// minimum start time of collision in current block
		double dTimeEnd;		// maximum end time of collision in current block
	};

private:
	static const uint64_t m_cSignatureStart = 0xBEDABEDA;	// Signature at the begin of the file.
	static const uint64_t m_cVersion = 100;					// Version of CollisionStorage file.
	static const uint64_t m_cSignatureEnd = 0x00000013;		// Signature at the end of block.

	// For storing data in file
	struct SSavedBlockInfo
	{
		uint64_t nOffset;		// Offset in file, points to corresponding ProtoBlockOfCollisionsInfo block.
		uint32_t nDescrSize;	// Size of the ProtoBlockOfCollisionsInfo block. (nOffset + nDescrSize) points to corresponding ProtoBlockOfCollisions.
		uint32_t nDataSize;		// Size of the ProtoBlockOfCollisions block. (nOffset + nDescrSize) points to corresponding ProtoBlockOfCollisions.
	};

	std::string m_sFileName;	// Current file to store collisions.
	std::fstream m_file;		// Current file.
	std::vector<SRuntimeBlockInfo*> m_vDescriptors;	// List of descriptors for fast file navigation.
	std::string m_sLastError;
	unsigned m_nCurrVersion;	// Version of loaded CollisionStorage file.

public:
	CCollisionsStorage();
	~CCollisionsStorage();

	// Returns true if file is properly loaded and ready to save/load data.
	bool IsValid() const;
	// Sets file name with collisions.
	void SetFileName(const std::string& _sFileName);
	// Returns name of the current file with collisions.
	std::string GetFileName() const;
	// Loads collisions data from specified file. Prepares file for loading but not really reads data to RAM.
	void LoadFromFile(const std::string& _sFile = "");
	// Create new file to store collisions information.
	void CreateNewFile(const std::string& _sFile = "");
	// Saves data with descriptor in current file.
	void SaveBlock(const ProtoBlockOfCollisionsInfo& _descriptor, const ProtoBlockOfCollisions& _collisions);
	// Returns total number of saved blocks.
	unsigned GetTotalBlocksNumber() const;
	// Returns list of descriptors.
	std::vector<SRuntimeBlockInfo*>& GetDesriptors();
	// Returns pointer to a descriptor.
	SRuntimeBlockInfo* GetDesriptor(unsigned _nIndex);
	// Returns data info block with current index.
	ProtoBlockOfCollisionsInfo* GetDataInfo(size_t _nIndex);
	// Returns data with current index.
	ProtoBlockOfCollisions* GetData(unsigned _nIndex);
	// Removes all data from opened file and resets descriptors to allow write of new data.
	void Reset();
	// Returns string with description of last occurred error.
	std::string GetLastError() const;
	// Flushes all cached data in file and closes file descriptor.
	void FlushAndCloseFile();
	// Returns true if it's possible to read from file: file is specified and descriptors are loaded.
	bool IsReadyToRead() const;
	// Returns version of opened file. LoadFromFile() should be run previously to get actual info.
	unsigned GetFileVersion() const;

private:
	// Returns true if it's possible to write in file.
	bool IsReadyToWrite() const;
	// Clears all runtime descriptors.
	void ClearDescriptors();
	// Writes message to buffer using protobuf. Returns size of actual data after saving.
	int32_t WriteToBuf(char *&_pBuffer, const google::protobuf::Message& _message);
	// Reads message from buffer using protobuf. Returns 'true' if success.
	bool ReadFromBuf(const char *_pBuffer, google::protobuf::Message& _message, int _nSize);
	// Saves data to file at the specified offset. Returns 'true' if success.
	bool WriteToFile(const char *_pBuffer, int32_t _nSize, int64_t _nOffset);
	// Reads data from file at the specified offset. Returns 'true' if success.
	bool ReadFromFile(char *_pBuffer, int32_t _nSize, int64_t _nOffset);
};

