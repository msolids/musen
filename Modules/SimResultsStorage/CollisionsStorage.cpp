/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "CollisionsStorage.h"
#include "DisableWarningHelper.h"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include "GeneratedFiles/SimulationDescription.pb.h"
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
PRAGMA_WARNING_POP

CCollisionsStorage::CCollisionsStorage()
{
	m_sFileName = "";
	m_nCurrVersion = m_cVersion;
}

CCollisionsStorage::~CCollisionsStorage()
{
	ClearDescriptors();
	m_file.close();
}

bool CCollisionsStorage::IsValid() const
{
	return m_file.good();
}

void CCollisionsStorage::SetFileName(const std::string& _sFileName)
{
	m_sFileName = _sFileName;
}

std::string CCollisionsStorage::GetFileName() const
{
	return m_sFileName;
}

void CCollisionsStorage::LoadFromFile(const std::string& _sFile /*= ""*/)
{
	// clear runtime descriptors
	ClearDescriptors();

	// close previous file
	m_file.close();

	// open new file
	if (_sFile != "")
		m_sFileName = _sFile;
	m_file.open(UnicodePath(m_sFileName), std::ios::in | std::ios::binary);

	// check opened file
	if (!m_file.good())
	{
		m_sLastError = "CCollisionsStorage::LoadFromFile(). Cannot load file.";
		return;
	}

	// check signature
	uint64_t nBeginSignature = 0;
	ReadFromFile((char*)&nBeginSignature, sizeof(m_cSignatureStart), 0);
	if (nBeginSignature != m_cSignatureStart)
	{
		m_sLastError = "CCollisionsStorage::LoadFromFile(). Wrong start signature.";
		m_file.close();
		return;
	}

	// get file version
	uint64_t nTempVersion = 0;
	ReadFromFile((char*)&nTempVersion, sizeof(m_cVersion), sizeof(m_cSignatureStart));
	m_nCurrVersion = static_cast<unsigned>(nTempVersion);

	// read data descriptors
	for (;;)
	{
		// read position descriptor
		SSavedBlockInfo savedInfo;
		ReadFromFile((char*)&savedInfo, sizeof(SSavedBlockInfo), INFO_BLOCK_OFFSET*(m_vDescriptors.size() + 1));
		if (savedInfo.nOffset == m_cSignatureEnd)	// last block
			break;

		// create runtime descriptor
		SRuntimeBlockInfo *pInfo = new SRuntimeBlockInfo;
		m_vDescriptors.push_back(pInfo);

		// save positions data to runtime descriptor
		pInfo->nOffset = savedInfo.nOffset;
		pInfo->nDescrSize = savedInfo.nDescrSize;
		pInfo->nDataSize = savedInfo.nDataSize;

		//// read descriptor
		ProtoBlockOfCollisionsInfo *pDescr = GetDataInfo(m_vDescriptors.size() - 1);

		// save info data to runtime descriptor
		pInfo->dTimeStart = pDescr->time_min();
		pInfo->dTimeEnd = pDescr->time_max();

		delete pDescr;
	}
}

void CCollisionsStorage::CreateNewFile(const std::string& _sFile /*= ""*/)
{
	// clear runtime descriptors
	ClearDescriptors();

	// create new file (or truncate existing)
	m_file.close();
	if (_sFile != "")
		m_sFileName = _sFile;
	m_file.open(UnicodePath(m_sFileName), std::ios::in | std::ios::out | std::ios::trunc | std::ios::binary);

	// check created file
	if (!m_file.good())
	{
		m_sLastError = "CCollisionsStorage::CreateNewFile(). Cannot create new file.";
		return;
	}

	// write initial signature and version
	uint64_t *tmp1 = new uint64_t(m_cSignatureStart);
	m_file.write((char*)tmp1, sizeof(m_cSignatureStart));
	uint64_t *tmp2 = new uint64_t(m_cVersion);
	m_file.write((char*)tmp2, sizeof(m_cVersion));
	delete tmp1;
	delete tmp2;

	m_nCurrVersion = static_cast<unsigned>(m_cVersion);
}

void CCollisionsStorage::SaveBlock(const ProtoBlockOfCollisionsInfo& _descriptor, const ProtoBlockOfCollisions& _collisions)
{
	if (!IsReadyToWrite())
	{
		m_sLastError = "CCollisionsStorage::SaveBlock(). Maximum number of data blocks is reached";
		return;
	}

	// current offset of data
	uint64_t nCurrOffset = m_vDescriptors.empty() ? FIRST_DATA_OFFSET : (m_vDescriptors.back()->nOffset + m_vDescriptors.back()->nDescrSize + m_vDescriptors.back()->nDataSize);

	using namespace google::protobuf::io;

	// write descriptor to buffer with protobuf
	char *pDescrBuffer;
	int32_t nDescrSize = WriteToBuf(pDescrBuffer, _descriptor);

	// write data to buffer with protobuf
	char *pDataBuffer;
	int32_t nDataSize = WriteToBuf(pDataBuffer, _collisions);

	// fill up runtime descriptor
	SRuntimeBlockInfo *pInfo = new SRuntimeBlockInfo;
	pInfo->nOffset = nCurrOffset;
	pInfo->nDescrSize = nDescrSize;
	pInfo->nDataSize = nDataSize;
	pInfo->dTimeStart = _descriptor.time_min();
	pInfo->dTimeEnd = _descriptor.time_max();
	m_vDescriptors.push_back(pInfo);

	// fill up save descriptor
	SSavedBlockInfo savedInfo = { nCurrOffset, static_cast<uint32_t>(nDescrSize), static_cast<uint32_t>(nDataSize) };

	// save buffers to file
	uint64_t *tmp = new uint64_t(m_cSignatureEnd);
	WriteToFile(static_cast<char*>(static_cast<void*>(&savedInfo)), sizeof(SSavedBlockInfo), INFO_BLOCK_OFFSET*m_vDescriptors.size());	// position descriptor
	WriteToFile((char*)tmp, sizeof(m_cSignatureEnd), INFO_BLOCK_OFFSET*(m_vDescriptors.size() + 1));	// last descriptor signature
	WriteToFile(pDescrBuffer, nDescrSize, nCurrOffset);	// data descriptor
	WriteToFile(pDataBuffer, nDataSize, nCurrOffset + nDescrSize);	// data
	delete tmp;

	delete[] pDataBuffer;
	delete[] pDescrBuffer;
}

unsigned CCollisionsStorage::GetTotalBlocksNumber() const
{
	return static_cast<unsigned>(m_vDescriptors.size());
}

ProtoBlockOfCollisionsInfo* CCollisionsStorage::GetDataInfo(size_t _nIndex)
{
	if (_nIndex >= m_vDescriptors.size())
		return (new ProtoBlockOfCollisionsInfo());

	// read descriptor
	char *pBuff = new char[m_vDescriptors[_nIndex]->nDescrSize];
	ReadFromFile(pBuff, m_vDescriptors[_nIndex]->nDescrSize, m_vDescriptors[_nIndex]->nOffset);

	// parse descriptor
	ProtoBlockOfCollisionsInfo *pDescr = new ProtoBlockOfCollisionsInfo;
	ReadFromBuf(pBuff, *pDescr, m_vDescriptors[_nIndex]->nDescrSize);

	// remove all temporary data
	delete[] pBuff;

	return pDescr;
}

ProtoBlockOfCollisions* CCollisionsStorage::GetData(unsigned _nIndex)
{
	if (_nIndex >= m_vDescriptors.size())
		return (new ProtoBlockOfCollisions());

	// read descriptor
	char *pBuff = new char[m_vDescriptors[_nIndex]->nDataSize];
	ReadFromFile(pBuff, m_vDescriptors[_nIndex]->nDataSize, m_vDescriptors[_nIndex]->nOffset + m_vDescriptors[_nIndex]->nDescrSize);

	// parse descriptor
	ProtoBlockOfCollisions *pData = new ProtoBlockOfCollisions;
	ReadFromBuf(pBuff, *pData, m_vDescriptors[_nIndex]->nDataSize);

	// remove all temporary data
	delete[] pBuff;

	return pData;
}

std::vector<CCollisionsStorage::SRuntimeBlockInfo*>& CCollisionsStorage::GetDesriptors()
{
	return m_vDescriptors;
}

CCollisionsStorage::SRuntimeBlockInfo* CCollisionsStorage::GetDesriptor(unsigned _nIndex)
{
	if (_nIndex < m_vDescriptors.size())
		return m_vDescriptors[_nIndex];
	return nullptr;
}

void CCollisionsStorage::Reset()
{
	CreateNewFile();
}

std::string CCollisionsStorage::GetLastError() const
{
	return m_sLastError;
}

bool CCollisionsStorage::IsReadyToRead() const
{
	return (IsValid() && !m_vDescriptors.empty());
}

unsigned CCollisionsStorage::GetFileVersion() const
{
	return m_nCurrVersion;
}

bool CCollisionsStorage::IsReadyToWrite() const
{
	return (IsValid() && (m_vDescriptors.size() < FIRST_DATA_OFFSET / INFO_BLOCK_OFFSET - 1));
}

void CCollisionsStorage::ClearDescriptors()
{
	for (unsigned i = 0; i < m_vDescriptors.size(); ++i)
		delete m_vDescriptors[i];
	m_vDescriptors.clear();
}

int32_t CCollisionsStorage::WriteToBuf(char *&_pBuffer, const google::protobuf::Message& _message)
{
	using namespace google::protobuf::io;
	int nInitSize = (int)_message.ByteSizeLong() + 10;		// initial size of data. 10 for black magic
	GzipOutputStream::Options options;
	options.format = GzipOutputStream::ZLIB;
	options.compression_level = 9;
	_pBuffer = new char[nInitSize];
	ArrayOutputStream arrayStream(_pBuffer, nInitSize);
	GzipOutputStream gzipStream(&arrayStream, options);
	_message.SerializeToZeroCopyStream(&gzipStream);
	gzipStream.Close();
	return static_cast<int32_t>(arrayStream.ByteCount());	// size of data after saving
}

bool CCollisionsStorage::ReadFromBuf(const char *_pBuffer, google::protobuf::Message& _message, int _nSize)
{
	using namespace google::protobuf::io;
	ArrayInputStream arrayStream(_pBuffer, _nSize);
	GzipInputStream gzipStream(&arrayStream, GzipInputStream::ZLIB);
	return _message.ParseFromZeroCopyStream(&gzipStream);
}

bool CCollisionsStorage::WriteToFile(const char *_pBuffer, int32_t _nSize, int64_t _nOffset)
{
	// check file
	if (!IsValid())
	{
		m_sLastError = "CCollisionsStorage::WriteToFile(). Cannot write to file: file is not valid";
		return false;
	}

	// write data
	m_file.seekp(static_cast<std::streamoff>(_nOffset));
	m_file.write((char*)_pBuffer, _nSize);

	// verify success
	if (m_file.fail())
	{
		m_sLastError = "CCollisionsStorage::WriteToFile(). Cannot write to file: file is not writable";
		return false;
	}

	return true;
}

bool CCollisionsStorage::ReadFromFile(char *_pBuffer, int32_t _nSize, int64_t _nOffset)
{
	// check file
	if (!IsValid())
	{
		m_sLastError = "CCollisionsStorage::ReadFromFile(). Cannot read file: file is not valid";
		return false;
	}

	m_file.seekg(static_cast<std::streamoff>(_nOffset));
	m_file.read((char*)_pBuffer, _nSize);

	// verify success
	if (m_file.fail())
	{
		m_sLastError = "CCollisionsStorage::ReadFromFile(). Cannot read to file: file is not readable";
		return false;
	}

	return true;
}

void CCollisionsStorage::FlushAndCloseFile()
{
	if (IsValid())
	{
		m_file.flush();
		m_file.close();
	}
}
