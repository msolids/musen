/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

/*	MDEM File structure:

	+0		---------------------------------------- 0b
			SFileHeader
	+64b	---------------------------------------- 64b
			SFileThreadHeader ("simulation info")
	+1024b	---------------------------------------- 1088b
			SFileThreadHeader ("time-dependent")
	+1024b	---------------------------------------- 2111b
			block0
			block1
			block2
			......
			block(n-1)
			----------------------------------------

	There are two file threads (for time-independent data and time-dependent data).
	Each block in file relates to one thread. Blocks of different threads can be mixed in file.
	There is not strong recording order.
	Start positions of all blocks is listed in FileThreadHeader.
	Each block has a fixed size which depends on its number in file. (see CMDEMFile::GetBlockSizeByIndex). */

#pragma once

#include "FileHandler.h"
#include "MUSENDefinitions.h"

struct SFileHeader											 // File header structure (64 bytes).
{
	static const uint32_t kSignature = 0x12345678;			 // Default start signature.
	static const uint32_t kFileVersion = MDEM_FILE_VERSION;	 // Default file version.
	static const uint32_t kMaxNumberOfThreads = 10000;		 // Default maximum number of threads in file.
	static const uint32_t kClusterSize = 4 * 1024;			 // Default cluster size.
	static const uint32_t kCopyBlockSize = 10 * 1024 * 1024; // Default copy block size. It is used for file copy operation.

	uint32_t nSignature;									 // Start signature. It is used to define start of file.
	uint32_t nVersion;										 // File version. This field is not used now. It can be used further.
	uint32_t nNumberOfThreads;								 // Number of threads in file.
	uint32_t nSizeOfCluster;								 // Cluster size. It is used for calculating of block size.

	char vReserved[64 - 4 * 4];								 // Reserved values to align size of structure to 64 bytes.
};

struct SFileThreadHeader									 // Thread header structure (1024 bytes).
{
	char vName[64];											 // Thread name (can be only "simulation info" or "time-dependent").
	uint64_t nSize;											 // Size of filethread (calculated as SRuntimeThreadInfo.nCurrentOffset + size of saved data).
	uint64_t nReserved;										 // Reserved value to align size of structure to 1024 bytes.

	static const int kNumberOfBlocks = (1024 - 64 - 8 - 8) / sizeof(uint64_t); // 118 blocks is possible for each thread.
	uint64_t vOffsetsInFile[kNumberOfBlocks];								   // This array contains offsets (start positions) of each block in .mdem file.
};

class CMDEMFile
{
private:
	struct SRuntimeThreadInfo					    // Runtime thread information.
	{
		SFileThreadHeader SFThreadHeader;			// Thread header.
		int nThreadIndex;							// Current index of filethread.
		uint64_t nCurrentOffset;					// Current offset in .mdem file.
	};

	struct SRuntimeFileInfo							// Runtime file information.
	{
		SFileHeader SFHeader;						// File header.
		std::vector<SRuntimeThreadInfo> vThreads;	// Runtime threads info.
	} m_SRunTimeFileInfo;

	std::shared_ptr<CFileHandler> m_pFileInterface; // Pointer to file interface.
	std::string m_sErrorMessage;					// Error description.
	uint32_t m_fileVersion{};						// Version of the file.

public:
	CMDEMFile();
	~CMDEMFile();

	// Checks error message field.
	bool IsValid() const;
	// Gets error message ("File not found", etc.).
	std::string GetLastError() const;
	// Sets file interface to current *.mdem file.
	void SetFileInterface(const  std::shared_ptr<CFileHandler> &_pFileAccess);
	// Returns file version from file header.
	uint32_t GetFileVersion() const;
	// Sets file version to file header.
	void SetFileVersion(uint32_t _nFileVersion);
	// Initializes when file is not empty (reading from the .mdem file file header and threads header).
	void Init();
	// Initializes when file is empty (writing to the .mdem file file header and threads header).
	void InitEmpty(const std::vector<std::string> &_vThreadNames);
	// Writes .mdem file header, threads headers and calls fflush.
	void WriteHeadersToDisk();
	// Copies from olf *.mdem file to the current file.
	bool CopyFrom(CMDEMFile* _pOldMDEMFile);
	// Reduces generated file by removing free space in last blocks of two different threads.
	void FinalTruncate();

	/* File threads functions.
	File threads are used for logical separation of the time-independent data and time-dependent data when handling .mdem file.
	Onlly two different file threads are used ("simulator info" with nThreadIndex = 0  and "time-dependent data" with nThreadIndex = 1).
	Both filethreads are referred to the same .mdem file. */

	// Returns name of file thread by index.
	std::string FileThreadGetName(uint32_t _nThreadIndex);
	// Returns size of file thread by index.
	uint64_t FileThreadGetSize(uint32_t _nThreadIndex);
	// Sets size of file thread with index _nThreadIndex.
	bool FileThreadSetSize(uint32_t _nThreadIndex, uint64_t _nNewThreadSize);
	// Sets current offset in file to _nOffset for file thread with _nThreadIndex.
	void FileThreadSetPointerTo(uint32_t _nThreadIndex, uint64_t _nOffset);
	// Writes data to file using_nThreadIndex file thread.
	uint32_t FileThreadWrite(uint32_t _nThreadIndex, void* _pData, uint32_t _nSize);
	// Reads data from file using _nThreadIndex file thread.
	uint32_t FileThreadRead(uint32_t _nThreadIndex, void* _pData, uint32_t _nSize);

	 // Gets info about file headers. This function can be used for debugging and will be removed further.
	//void GetHeadersValues(uint64_t& _nCurrOffs_1, uint64_t& _nSize_1, std::vector<uint64_t>& _vOffsetsInFile_1, std::vector<int>& _vInd_1, uint64_t& _nCurrOffs_2, uint64_t& _nSize_2, std::vector<uint64_t>& _vOffsetsInFile_2, std::vector<int>& _vInd_2)
	//{
	//	// thread header 0
	//	_nCurrOffs_1 = m_SRunTimeFileInfo.vThreads[0].nCurrentOffset;
	//	_nSize_1 = m_SRunTimeFileInfo.vThreads[0].SFThreadHeader.nSize;
	//	std::vector<uint64_t> vTempVector;
	//	std::vector<int> vTempVectorInd;
	//	for (int i = 0; i < 117; i++)
	//	{
	//		if (m_SRunTimeFileInfo.vThreads[0].SFThreadHeader.vOffsetsInFile[i] != 0)
	//		{
	//			vTempVectorInd.push_back(i);
	//			vTempVector.push_back(m_SRunTimeFileInfo.vThreads[0].SFThreadHeader.vOffsetsInFile[i]);
	//		}
	//	}
	//	_vOffsetsInFile_1 = vTempVector;
	//	_vInd_1 = vTempVectorInd;

	//	// thread header 1
	//	_nCurrOffs_2 = m_SRunTimeFileInfo.vThreads[1].nCurrentOffset;
	//	_nSize_2 = m_SRunTimeFileInfo.vThreads[1].SFThreadHeader.nSize;
	//	vTempVector.clear();
	//	vTempVectorInd.clear();
	//	for (int i = 0; i < 117; i++)
	//	{
	//		if (m_SRunTimeFileInfo.vThreads[1].SFThreadHeader.vOffsetsInFile[i] != 0)
	//		{
	//			vTempVectorInd.push_back(i);
	//			vTempVector.push_back(m_SRunTimeFileInfo.vThreads[1].SFThreadHeader.vOffsetsInFile[i]);
	//		}
	//	}
	//	_vOffsetsInFile_2 = vTempVector;
	//	_vInd_2 = vTempVectorInd;
	//}

private:
	typedef uint32_t(CFileHandler:: *IoOperationFunction)(void* _pBuffer, uint32_t _nBufferSize);
	// Performs reading or writing operation.
	uint32_t FileIoOperation(uint32_t _nThreadIndex, void* _pData, uint32_t _nSize, IoOperationFunction _operation);
	// Return address to store new block in file.
	uint64_t FindAddressToAppendNewBlock();
	// Returns size of block which strictly depends on its index.
	uint64_t GetBlockSizeByIndex(uint32_t _nBlockIndex);
	// Gets index of block and offset in this block by address in file.
	bool GetBlockByAddress(uint64_t _nAddress, uint32_t &_nBlockIndex, uint64_t &_nOffsetInBlock);
};