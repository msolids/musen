/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "MDEMFile.h"

CMDEMFile::CMDEMFile()
{
}

CMDEMFile::~CMDEMFile()
{
	// write file header and threads headers to file
	if (m_SRunTimeFileInfo.SFHeader.nSignature == SFileHeader::kSignature) // if headers have been filled
		WriteHeadersToDisk();
}

void CMDEMFile::SetFileInterface(const  std::shared_ptr<CFileHandler> &_pFileInterface)
{
	m_pFileInterface = _pFileInterface;

	// get the file version
	m_fileVersion = 0;
	m_pFileInterface->SetPointerInFileTo(4, SEEK_SET); // 4 offset to fileVersion
	m_pFileInterface->ReadFromFile(&m_fileVersion, sizeof(m_fileVersion));
}

uint32_t CMDEMFile::GetFileVersion() const
{
	return m_fileVersion;
}

void CMDEMFile::SetFileVersion(uint32_t _nFileVersion)
{
	m_pFileInterface->SetPointerInFileTo(4, SEEK_SET); // 4 offset to nFileVersion
	m_pFileInterface->WriteToFile(&_nFileVersion, sizeof(_nFileVersion));
	m_SRunTimeFileInfo.SFHeader.nVersion = _nFileVersion;
}

bool CMDEMFile::IsValid() const
{
	return m_sErrorMessage.empty();
}

std::string CMDEMFile::GetLastError() const
{
	return m_sErrorMessage;
}

void CMDEMFile::Init()
{
	assert(m_pFileInterface);

	// read file header
	m_pFileInterface->SetPointerInFileTo(0, SEEK_SET);
	m_pFileInterface->ReadFromFile(&m_SRunTimeFileInfo.SFHeader, sizeof(m_SRunTimeFileInfo.SFHeader));

	// check reading error
	m_sErrorMessage = m_pFileInterface->GetLastError();
	if (!m_sErrorMessage.empty()) return;

	// check file
	if (m_SRunTimeFileInfo.SFHeader.nSignature != SFileHeader::kSignature || m_SRunTimeFileInfo.SFHeader.nNumberOfThreads > SFileHeader::kMaxNumberOfThreads)
	{
		m_sErrorMessage = "Is not a *.mdem file";
		return;
	}

	// read file threads headers
	m_SRunTimeFileInfo.vThreads.resize(m_SRunTimeFileInfo.SFHeader.nNumberOfThreads);
	for (uint32_t i = 0; i < m_SRunTimeFileInfo.vThreads.size(); i++)
	{
		auto &item = m_SRunTimeFileInfo.vThreads[i];
		m_pFileInterface->ReadFromFile(&item.SFThreadHeader, sizeof(item.SFThreadHeader));
		item.nThreadIndex = i;
		item.nCurrentOffset = 0;
	}

	// check reading error
	m_sErrorMessage = m_pFileInterface->GetLastError();
}

void CMDEMFile::InitEmpty(const std::vector<std::string> &_vThreadNames)
{
	assert(m_pFileInterface);

	// set default values for file header
	memset(&m_SRunTimeFileInfo.SFHeader, 0, sizeof(m_SRunTimeFileInfo.SFHeader));
	m_SRunTimeFileInfo.SFHeader.nSignature = SFileHeader::kSignature;
	m_SRunTimeFileInfo.SFHeader.nNumberOfThreads = (uint32_t)_vThreadNames.size();
	m_SRunTimeFileInfo.SFHeader.nSizeOfCluster = SFileHeader::kClusterSize;
	m_SRunTimeFileInfo.SFHeader.nVersion = SFileHeader::kFileVersion;

	// set defult value for file threads headers
	m_SRunTimeFileInfo.vThreads.resize(m_SRunTimeFileInfo.SFHeader.nNumberOfThreads);
	for (uint32_t i = 0; i < m_SRunTimeFileInfo.vThreads.size(); i++)
	{
		auto &item = m_SRunTimeFileInfo.vThreads[i];
		memset(&item.SFThreadHeader, 0, sizeof(item.SFThreadHeader));
		// set name of current thread (can be only "simulation info" or "time-dependent")
#ifdef PATH_CONFIGURED
		strncpy(item.SFThreadHeader.vName, _vThreadNames[i].c_str(), sizeof(item.SFThreadHeader.vName) - 1);
#else
		strncpy_s(item.SFThreadHeader.vName, 64, _vThreadNames[i].c_str(), sizeof(item.SFThreadHeader.vName) - 1);
#endif
		item.SFThreadHeader.nSize = 0;
		item.nThreadIndex = i;
		item.nCurrentOffset = 0;
	}

	// write file header and file threads headers to file
	WriteHeadersToDisk();
}

void CMDEMFile::WriteHeadersToDisk()
{
	// check errors
	if (!IsValid())
		return;

	// write file header to file
	m_pFileInterface->SetPointerInFileTo(0, SEEK_SET);
	m_pFileInterface->WriteToFile(&m_SRunTimeFileInfo.SFHeader, sizeof(SFileHeader));

	// write file threads headers to file
	for (unsigned i = 0; i < m_SRunTimeFileInfo.vThreads.size(); i++)
	{
		m_pFileInterface->WriteToFile(&m_SRunTimeFileInfo.vThreads.at(i).SFThreadHeader, sizeof(SFileThreadHeader));
	}

	// flush data from system buffer to file
	m_pFileInterface->FlushToFile();

	// check writing error
	m_sErrorMessage = m_pFileInterface->GetLastError();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string CMDEMFile::FileThreadGetName(uint32_t _nThreadIndex)
{
	assert(_nThreadIndex < m_SRunTimeFileInfo.vThreads.size());
	// return name of current thread
	return std::string(m_SRunTimeFileInfo.vThreads[_nThreadIndex].SFThreadHeader.vName);
}

uint64_t CMDEMFile::FileThreadGetSize(uint32_t _nThreadIndex)
{
	assert(_nThreadIndex < m_SRunTimeFileInfo.vThreads.size());
	// return size of current thread
	return m_SRunTimeFileInfo.vThreads[_nThreadIndex].SFThreadHeader.nSize;
}

void CMDEMFile::FileThreadSetPointerTo(uint32_t _nThreadIndex, uint64_t _nOffset)
{
	assert(_nThreadIndex < m_SRunTimeFileInfo.vThreads.size());
	// set offset in file for current thread
	SRuntimeThreadInfo &item = m_SRunTimeFileInfo.vThreads[_nThreadIndex];
	item.nCurrentOffset = _nOffset;
}

bool CMDEMFile::FileThreadSetSize(uint32_t _nThreadIndex, uint64_t _nNewThreadSize)
{
	assert(_nThreadIndex < m_SRunTimeFileInfo.vThreads.size());

	// set new size value
	SRuntimeThreadInfo &item = m_SRunTimeFileInfo.vThreads[_nThreadIndex];
	item.SFThreadHeader.nSize = _nNewThreadSize;

	// get index of block and offset in this block by new size
	uint32_t nBlockIndex;
	uint64_t nOffsetInBlock;
	bool bRes = GetBlockByAddress(item.SFThreadHeader.nSize, nBlockIndex, nOffsetInBlock);
	if (!bRes) return false;

	// if found block is partially filled go to next block
	if (nOffsetInBlock != 0) nBlockIndex++;

	// clear offsets in file for all blocks with higher index
	while (nBlockIndex < SFileThreadHeader::kNumberOfBlocks && item.SFThreadHeader.vOffsetsInFile[nBlockIndex] != 0)
	{
		item.SFThreadHeader.vOffsetsInFile[nBlockIndex] = 0;
		nBlockIndex++;
	}

	// get new address to append next block in file
	uint64_t nFurtherPosInFile = FindAddressToAppendNewBlock();

	// truncate file to calculated address
	m_pFileInterface->SetFileSize(nFurtherPosInFile);

	return true;
}

uint32_t CMDEMFile::FileThreadWrite(uint32_t _nThreadIndex, void *_pData, uint32_t _nSize)
{
	assert(_nThreadIndex < m_SRunTimeFileInfo.vThreads.size());
	SRuntimeThreadInfo &item = m_SRunTimeFileInfo.vThreads[_nThreadIndex];

	// recalculate current thread size
	if (item.nCurrentOffset + _nSize > item.SFThreadHeader.nSize) item.SFThreadHeader.nSize = item.nCurrentOffset + _nSize;

	// perform writing operation
	return FileIoOperation(_nThreadIndex, _pData, _nSize, &CFileHandler::WriteToFile);
}

uint32_t CMDEMFile::FileThreadRead(uint32_t _nThreadIndex, void *_pData, uint32_t _nSize)
{
	assert(_nThreadIndex < m_SRunTimeFileInfo.vThreads.size());
	SRuntimeThreadInfo &item = m_SRunTimeFileInfo.vThreads[_nThreadIndex];

	// recalculate read data size
	if (item.nCurrentOffset + _nSize > item.SFThreadHeader.nSize)
	{
		assert(item.nCurrentOffset <= item.SFThreadHeader.nSize);
		_nSize = static_cast<uint32_t>(item.SFThreadHeader.nSize - item.nCurrentOffset);
	}

	// perform reading operation
	return FileIoOperation(_nThreadIndex, _pData, _nSize, &CFileHandler::ReadFromFile);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t CMDEMFile::FileIoOperation(uint32_t _nThreadIndex, void *_pData, uint32_t _nSize, CMDEMFile::IoOperationFunction _operation)
{
	assert(_nThreadIndex < m_SRunTimeFileInfo.vThreads.size());
	SRuntimeThreadInfo &item = m_SRunTimeFileInfo.vThreads[_nThreadIndex];

	uint32_t nActuallyProcessed = 0;
	char* pData = (char *)_pData;

	// get index of block and offset in this block by offset of current thread in file
	uint32_t nBlockIndex;
	uint64_t nOffsetInBlock;
	bool bRes = GetBlockByAddress(item.nCurrentOffset, nBlockIndex, nOffsetInBlock);
	if (!bRes) return false;

	// while size of currently processed data < size of data which has to be processed
	while (nActuallyProcessed < _nSize && m_pFileInterface->IsFileValid())
	{
		// only when writing operation
		for (unsigned i = 0; i <= nBlockIndex && item.SFThreadHeader.vOffsetsInFile[nBlockIndex] == 0; i++)
		{
			if (item.SFThreadHeader.vOffsetsInFile[i] != 0) continue;
			item.SFThreadHeader.vOffsetsInFile[i] = FindAddressToAppendNewBlock();
		}

		// calculate size of data to process
		uint64_t nBlockSize = GetBlockSizeByIndex(nBlockIndex);
		uint64_t nSizeToEndOfBlock = nBlockSize - nOffsetInBlock;
		uint32_t nSizeToProcess = (uint32_t)std::min(nSizeToEndOfBlock, (uint64_t)(_nSize - nActuallyProcessed));

		// set pointer to free space in current block
		m_pFileInterface->SetPointerInFileTo(item.SFThreadHeader.vOffsetsInFile[nBlockIndex] + nOffsetInBlock, SEEK_SET);

		// read or write operation is performed
		(&*m_pFileInterface->*_operation)(pData + nActuallyProcessed, nSizeToProcess);

		// recalculate acctually processed size and current thread offset
		nActuallyProcessed += nSizeToProcess;
		item.nCurrentOffset += nSizeToProcess;

		// recalculate index of block and clear offset in block
		nBlockIndex++;
		nOffsetInBlock = 0;
	}

	// check error
	m_sErrorMessage = m_pFileInterface->GetLastError();
	return nActuallyProcessed;
}

uint64_t CMDEMFile::FindAddressToAppendNewBlock()
{
	// calculcate start position of blocks (64 + 1024 * 2 = 2112 bytes)
	uint64_t nFirstDataOffset = sizeof(SFileHeader) + sizeof(SFileThreadHeader) * m_SRunTimeFileInfo.vThreads.size();

	// loop for all file threads
	for (unsigned i = 0; i < m_SRunTimeFileInfo.vThreads.size(); i++)
	{
		auto& item = m_SRunTimeFileInfo.vThreads.at(i);
		uint32_t nIndexOfFreeBlock = 0;

		// loop for all blocks
		for (uint32_t i = 0; i < SFileThreadHeader::kNumberOfBlocks; i++)
		{
			// look for free block
			if (item.SFThreadHeader.vOffsetsInFile[i] == 0)
			{
				nIndexOfFreeBlock = i;
				break;
			}
		}

		// if free block has not found go to next file thread
		if (nIndexOfFreeBlock == 0) continue;

		// recalculate offset to append new block
		uint64_t nCandidate = item.SFThreadHeader.vOffsetsInFile[nIndexOfFreeBlock - 1] + GetBlockSizeByIndex(nIndexOfFreeBlock - 1);
		if (nFirstDataOffset < nCandidate) nFirstDataOffset = nCandidate;
	}
	return nFirstDataOffset;
}

uint64_t CMDEMFile::GetBlockSizeByIndex(uint32_t _nBlockIndex)
{
 /* Index	Size
	  0		4096 bytes, 4 KB
	  1		4096 bytes, 4 KB
	  2		8192 bytes, 8 KB
	  3		12288 bytes, 12 KB
	  4		16384 bytes, 16 KB
	  5		24576 bytes, 24 KB
	  6		32768 bytes, 32 KB
	  7		49152 bytes, 48 KB
	  8		65536 bytes, 64 KB
	  9		98304 bytes, 96 KB
	  10	131072 bytes, 128 KB */

	if (_nBlockIndex == 0) return m_SRunTimeFileInfo.SFHeader.nSizeOfCluster;
	if (_nBlockIndex < 5) return _nBlockIndex * m_SRunTimeFileInfo.SFHeader.nSizeOfCluster;

	uint32_t nCurrentIndex = 4;
	uint64_t nVal = 4;

	int nStepNumber = 0;
	while (nCurrentIndex < _nBlockIndex)
	{
		if (nStepNumber % 2 == 0) nVal = nVal / 2 * 3;
		else nVal = nVal / 3 * 4;

		nStepNumber++;
		nCurrentIndex++;
	}

	return nVal * m_SRunTimeFileInfo.SFHeader.nSizeOfCluster;
}

bool CMDEMFile::GetBlockByAddress(uint64_t _nAddress, uint32_t &_nBlockIndex, uint64_t &_nOffsetInBlock)
{
	// _nAddress - input argument which means offset in file
	_nBlockIndex = 0;
	uint64_t nCurrertStartAddress = 0;

	// loop for all blocks (118 pieces)
	while (_nBlockIndex < SFileThreadHeader::kNumberOfBlocks)
	{
		// get size of current block
		uint64_t nCurrentSize = GetBlockSizeByIndex(_nBlockIndex);

		// if this is not suitable block go to next block
		if (nCurrertStartAddress + nCurrentSize <= _nAddress)
		{
			_nBlockIndex++;
			nCurrertStartAddress += nCurrentSize;
			continue;
		}

		// calculate offset in found block
		_nOffsetInBlock = _nAddress - nCurrertStartAddress;
		return true;
	}
	return false;
}

bool CMDEMFile::CopyFrom(CMDEMFile* _pOldMDEMFile)
{
	// check new and old files
	if (!IsValid() || !_pOldMDEMFile->IsValid()) return false;

	// calculate inital copy position (64 + 1024 * 2 = 2112 bytes)
	uint64_t nCurrentPos = sizeof(SFileHeader) + sizeof(SFileThreadHeader) * m_SRunTimeFileInfo.vThreads.size();

	m_SRunTimeFileInfo = _pOldMDEMFile->m_SRunTimeFileInfo;

	// loop for all data blocks (available 118 blocks)
	for (uint32_t i = 0; i < SFileThreadHeader::kNumberOfBlocks; i++)
	{
		// loop for all threads (2 threads)
		for (unsigned j = 0; j < m_SRunTimeFileInfo.vThreads.size(); j++)
		{
			SRuntimeThreadInfo& SCurrentThread = m_SRunTimeFileInfo.vThreads.at(j);

			// if current block is empty, skip this block
			if (SCurrentThread.SFThreadHeader.vOffsetsInFile[i] == 0)
				continue;

			// get size of current block to copy
			uint64_t nSizeToCopy = GetBlockSizeByIndex(i);

			// set pointers in new and old ile in positions for writing and reading accordingly
			m_pFileInterface->SetPointerInFileTo(nCurrentPos, SEEK_SET);
			_pOldMDEMFile->m_pFileInterface->SetPointerInFileTo(SCurrentThread.SFThreadHeader.vOffsetsInFile[i], SEEK_SET);

			// set offset of current block in file and recalculate next copy position
			SCurrentThread.SFThreadHeader.vOffsetsInFile[i] = nCurrentPos;
			nCurrentPos += nSizeToCopy;

			// copy data from old file to new file
			std::vector<char> vBuffer(SFileHeader::kCopyBlockSize);
			while (nSizeToCopy)
			{
				uint32_t nCopySize = (uint32_t)std::min(nSizeToCopy, (uint64_t)vBuffer.size());

				_pOldMDEMFile->m_pFileInterface->ReadFromFile(&vBuffer[0], nCopySize);
				m_pFileInterface->WriteToFile(&vBuffer[0], nCopySize);

				nSizeToCopy -= nCopySize;
			}
		}
	}
	// write file header and file threads headers to file
	WriteHeadersToDisk();
	// check file
	return IsValid();
}

void CMDEMFile::FinalTruncate(ProtoSimulationStorage& _protoSimStorage)
{
	// get index, offset in file, offset inside block for last time-independent data block by TIDthread size
	uint32_t nIndexLastBlockTID;
	uint64_t nOffsetInsideLastBlockTID;
	SRuntimeThreadInfo &itemTID = m_SRunTimeFileInfo.vThreads[0];
	bool bRes = GetBlockByAddress(itemTID.SFThreadHeader.nSize, nIndexLastBlockTID, nOffsetInsideLastBlockTID);
	if (!bRes) return;
	uint64_t nOffsetLastBlockTID = itemTID.SFThreadHeader.vOffsetsInFile[nIndexLastBlockTID];

	// get index, offset in file, offset inside block for last time-dependent data block by TDDthread size
	uint32_t nIndexLastBlockTDD;
	uint64_t nOffsetInsideLastBlockTDD;
	SRuntimeThreadInfo &itemTDD = m_SRunTimeFileInfo.vThreads[1];
	bRes = GetBlockByAddress(itemTDD.SFThreadHeader.nSize, nIndexLastBlockTDD, nOffsetInsideLastBlockTDD);
	if (!bRes) return;
	uint64_t nOffsetLastBlockTDD = itemTDD.SFThreadHeader.vOffsetsInFile[nIndexLastBlockTDD];

	// calculate real end for both threads
	uint64_t nEndOfTIDThread = nOffsetLastBlockTID + nOffsetInsideLastBlockTID;
	uint64_t nEndOfTDDThread = nOffsetLastBlockTDD + nOffsetInsideLastBlockTDD;

	uint64_t nFreeSpaceInLastBlockTID = GetBlockSizeByIndex(nIndexLastBlockTID) - nOffsetInsideLastBlockTID;

	// if last saved block relates to time-dependent data thread
	if (nOffsetLastBlockTID < nOffsetLastBlockTDD)
	{
		// Problem with removing data before resimulation has to be resolved
		// then the next transformation can be applied for file size reducing:

		//// find index of first TDD block for shifting
		//uint32_t nBlockIndTDDFirst = 0;
		//for (auto i = 0; i < itemTDD.SFThreadHeader.kNumberOfBlocks; i++)
		//{
		//	if (itemTDD.SFThreadHeader.vOffsetsInFile[i] == (nEndOfTIDThread + nFreeSpaceInLastBlockTID))
		//	{
		//		nBlockIndTDDFirst = i;
		//		break;
		//	}
		//}

		//// shift all TDD block to last TID block
		//for (auto i = nBlockIndTDDFirst; i <= nIndexLastBlockTDD; i++)
		//{
		//	uint64_t nCurrSizeBlockTDD = GetBlockSizeByIndex(i);
		//	char* pBuffer = new char[nCurrSizeBlockTDD];
		//	m_pFileInterface->SetPointerInFileTo(itemTDD.SFThreadHeader.vOffsetsInFile[i], SEEK_SET);
		//	m_pFileInterface->ReadFromFile(pBuffer, nCurrSizeBlockTDD);
		//	m_pFileInterface->SetPointerInFileTo(itemTDD.SFThreadHeader.vOffsetsInFile[i] - nFreeSpaceInLastBlockTID, SEEK_SET);
		//	m_pFileInterface->WriteToFile(pBuffer, nCurrSizeBlockTDD);
		//	delete[] pBuffer;
		//	itemTDD.SFThreadHeader.vOffsetsInFile[i] = itemTDD.SFThreadHeader.vOffsetsInFile[i] - nFreeSpaceInLastBlockTID;
		//}

		//// set new values for offset_in_file in each proto block descriptor
		//for (auto i = 0; i < _protoSimStorage.data_blocks().size(); i++)
		//{
		//	ProtoBlockDescriptor* pCurrBlock = _protoSimStorage.mutable_data_blocks(i);
		//	if (pCurrBlock->offset_in_file() >= itemTID.SFThreadHeader.nSize)
		//	{
		//		uint64_t nNewOffset = _protoSimStorage.data_blocks(i).offset_in_file() - nFreeSpaceInLastBlockTID;
		//		pCurrBlock->set_offset_in_file(nNewOffset);
		//	}
		//}

		// truncate file to real end of time-dependent data thread
		m_pFileInterface->SetFileSize(nEndOfTDDThread);// -nFreeSpaceInLastBlockTID);
	}
	else
	{
		// shift significant data from last time-independent block to real end of last time-dependent block
	/*	char* pBufferTemp = new char[nOffsetInsideLastBlockTID];
		m_pFileInterface->SetPointerInFileTo(nOffsetLastBlockTID, SEEK_SET);
		m_pFileInterface->ReadFromFile(pBufferTemp, nOffsetInsideLastBlockTID);
		m_pFileInterface->SetPointerInFileTo(nEndOfTDDThread, SEEK_SET);
		m_pFileInterface->WriteToFile(pBufferTemp, nOffsetInsideLastBlockTID);
		delete[] pBufferTemp;
		itemTID.SFThreadHeader.vOffsetsInFile[nIndexLastBlockTID] = nEndOfTDDThread;*/

		// truncate file to real end of time-independent data thread
		m_pFileInterface->SetFileSize(nEndOfTIDThread);//(nEndOfTDDThread + nOffsetInsideLastBlockTID);
	}

	WriteHeadersToDisk();
}