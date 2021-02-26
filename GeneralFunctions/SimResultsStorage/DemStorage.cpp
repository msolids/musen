/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "DemStorage.h"
#include "DisableWarningHelper.h"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
PRAGMA_WARNING_POP

CDemStorage::CDemStorage()
{
	m_timeEnd = 0;
}

bool CDemStorage::IsValid() const
{
	return m_pMDEMFile && GetLastError().empty();
}

std::string CDemStorage::GetLastError() const
{
	return m_sLastError.empty() && m_pMDEMFile ? m_pMDEMFile->GetLastError() : m_sLastError;
}

double CDemStorage::GetRealTimeEnd() const
{
	return m_timeEnd;
}

void CDemStorage::FinalTruncate()
{
	m_pMDEMFile->FinalTruncate(/*m_simStorage*/);
}

bool CDemStorage::CopyFrom(CDemStorage *_pOldStorage)
{
	// check new and old file
	if (!_pOldStorage || !IsValid() || !_pOldStorage->IsValid()) return false;

	// flush data from memory to disk for old file
	_pOldStorage->FlushToDisk(true);
	_pOldStorage->FinalTruncate();

	// copy data from old file to new file
	m_pMDEMFile->CopyFrom(&*_pOldStorage->m_pMDEMFile);

	// read time-independent data and two proto block of time points with time-dependent data
	return ReadInitialData();
}

void CDemStorage::RemoveAll()
{
	InitEmptyFile();
}

uint32_t CDemStorage::GetFileVersion() const
{
	return m_pMDEMFile->GetFileVersion();
}

void CDemStorage::SetFileVersion(uint32_t _nFileVersion) const
{
	m_pMDEMFile->SetFileVersion(_nFileVersion);
}

void CDemStorage::OpenFile(const std::string& _sFileName)
{
	// open file
	std::shared_ptr<CFileHandler> pFileInterface = std::make_shared<CFileHandler>();
	pFileInterface->UseFile(_sFileName);

	// set file interface to file
	m_pMDEMFile = std::make_shared<CMDEMFile>();
	m_pMDEMFile->SetFileInterface(pFileInterface);
}

bool CDemStorage::LoadFromFile()
{
	m_sLastError.clear();

	// read file header and file threads headers from file
	m_pMDEMFile->Init();

	// read time-independent data and two proto block of time points with time-dependent data
	return ReadInitialData();
}

bool CDemStorage::ReadInitialData()
{
	// check file
	if (!m_pMDEMFile || m_pMDEMFile->IsValid() == false) return false;

	// get size of time-independent data thread
	const auto nSize = static_cast<uint32_t>(m_pMDEMFile->FileThreadGetSize(TIME_INDEPENDENT_DATA));

	// allocate data buffer for reading
	const std::unique_ptr<void, void(*)(void *)> pTmpBuffer(malloc(nSize), std::free);
	if (!pTmpBuffer)
	{
		m_sLastError = "memory allocation error";
		m_pMDEMFile.reset();
		return false;
	}

	// read time-independent data from file to data buffer
	m_pMDEMFile->FileThreadSetPointerTo(TIME_INDEPENDENT_DATA, 0);
	m_pMDEMFile->FileThreadRead(TIME_INDEPENDENT_DATA, pTmpBuffer.get(), nSize);

	// read from data buffer to proto message (ProtoSimulationStorage)
	const bool bRes = ReadFromBuf(static_cast<const char*>(pTmpBuffer.get()), m_simStorage, nSize, -1);
	if (!bRes)
	{
		m_sLastError = "invalid protobuf in proto simulation storage";
		m_pMDEMFile.reset();
		return false;
	}

	// initialize array contained proto blocks of time points
	m_blocksOfTimePoints.clear();
	m_blocksOfTimePoints.reserve(m_simStorage.data_blocks_size());
	while (m_blocksOfTimePoints.size() < static_cast<uint32_t>(m_simStorage.data_blocks_size()))
		m_blocksOfTimePoints.push_back(std::make_unique<ProtoBlockOfTimePoints>());

	// check array with proto block of time points
	if (m_blocksOfTimePoints.empty())
	{
		m_sLastError = "proto blocks of time points cannot be found";
		m_pMDEMFile.reset();
		return false;
	}

	// load the last proto block of time points to memory
	LoadBlockInternal(static_cast<uint32_t>(m_blocksOfTimePoints.size()) - 1);
	m_iLeftLoadedBlock = static_cast<uint32_t>(m_blocksOfTimePoints.size()) - 1;

	// load zero time-dependent block (and first if size_of_blocks > 2), m_iLeftLoadedBlock = 0
	LoadBlock(0);
	if (!IsValid()) return false;

	// save value of last saved time point
	if (m_blocksOfTimePoints.back()->time_points_size() != 0)
		m_timeEnd = m_blocksOfTimePoints.back()->time_points(m_blocksOfTimePoints.back()->time_points_size() - 1).time();
	else
		m_timeEnd = 0;

	// set default value for caches
	m_SWriteCache.time = 1e+300;
	m_SWriteCache.pTimePoint = nullptr;
	m_SCache.requestedTime = 1e+300;
	m_SCache.pTimePointL = nullptr;
	m_SCache.pTimePointR = nullptr;

	return true;
}

bool CDemStorage::CreateNewFile(const std::string& _sFileName)
{
	m_sLastError.clear();

	// create pointer to file interface (in this case new file is created and opened for reading and writing in binary mode)
	std::shared_ptr<CFileHandler> pFileInterface = std::make_shared<CFileHandler>();
	pFileInterface->UseFile(_sFileName);

	// create pointer to MDEMFile and set file interface
	m_pMDEMFile = std::make_shared<CMDEMFile>();
	m_pMDEMFile->SetFileInterface(pFileInterface);

	// init empty file (set default headers values and write file header and file threads headers to file)
	m_pMDEMFile->InitEmpty(std::vector<std::string>{"simulation info", "time-dependent"});

	// check created file
	if (!m_pMDEMFile || !m_pMDEMFile->IsValid())
		return false;

	// set default values for proto structures in RAM
	const bool res = InitEmptyFile();

	FlushToDisk(true);

	return res;
}

bool CDemStorage::InitEmptyFile()
{
	// create new block descriptor for block 0 with time-dependent data for 0 time point
	m_simStorage = ProtoSimulationStorage();
	ProtoBlockDescriptor* pBlock = m_simStorage.add_data_blocks();
	pBlock->set_offset_in_file(0);
	pBlock->set_start_time(0);

	// create new time-dependent data block with 0 time point
	m_blocksOfTimePoints.clear();
	m_blocksOfTimePoints.push_back(std::make_unique<ProtoBlockOfTimePoints>());
	ProtoTimePoint* pTimePoint = m_blocksOfTimePoints[0]->add_time_points();
	pTimePoint->set_time(0);

	m_iLeftLoadedBlock = 0;

	// set default values in caches
	m_SCache.requestedTime = 1e+300;
	m_SCache.pTimePointL = nullptr;
	m_SCache.pTimePointR = nullptr;
	m_SWriteCache.time = 1e+300;
	m_SWriteCache.pTimePoint = nullptr;

	return true;
}

void CDemStorage::FlushToDisk(bool _isSaveLastBlock)
{
	// check file
	if (!IsValid()) return;

	// save the last block
	if (_isSaveLastBlock)
		SaveLastBlock();

	// clear cache for writing
	m_SWriteCache.time = 1e+300;
	m_SWriteCache.pTimePoint = nullptr;

	// serialize time-independent data from proto file to tmpBuffer, binSize - size of tmpBuffer after serialization
	char* tmpBuffer;
	const int nBinSize = WriteToBuf(tmpBuffer, m_simStorage);

	// write time-independent data to file
	m_pMDEMFile->FileThreadSetPointerTo(TIME_INDEPENDENT_DATA, 0);
	m_pMDEMFile->FileThreadWrite(TIME_INDEPENDENT_DATA, tmpBuffer, nBinSize);

	delete[] tmpBuffer;

	// pointer to last saved block
	const ProtoBlockDescriptor* pLastBlock = &m_simStorage.data_blocks(m_simStorage.data_blocks_size() - 1);
	// offset for saving of the next block
	const uint64_t nEndAddress = pLastBlock->offset_in_file() + pLastBlock->size();

	// set size of file
	m_pMDEMFile->FileThreadSetSize(TIME_INDEPENDENT_DATA, nBinSize);
	m_pMDEMFile->FileThreadSetSize(TIME_DEPENDENT_DATA, nEndAddress);

	// writing file header and threads headers
	m_pMDEMFile->WriteHeadersToDisk();
}

ProtoSimulationInfo* CDemStorage::SimulationInfo()
{
	return m_simStorage.mutable_info();
}

const ProtoSimulationInfo* CDemStorage::SimulationInfo() const
{
	return &m_simStorage.info();
}

ProtoModulesData* CDemStorage::ModulesData()
{
	return m_simStorage.mutable_modules_data();
}

void CDemStorage::RemoveAllAfterTime(double _dTime)
{
	assert(_dTime >= 0);

	// check file
	if (!IsValid()) return;

	// if _dTime has a bigger value then the last cached
	if (m_SWriteCache.time <= _dTime) return;

	// if _dTime has a bigger value then the value of the first time point in the last block which was saved in proto file
	if (!m_blocksOfTimePoints.empty() && !m_blocksOfTimePoints.back()->time_points().empty() && m_blocksOfTimePoints.back()->time_points().rbegin()->time() <= _dTime) return;

	// removing
	while (m_simStorage.data_blocks().rbegin()->start_time() > _dTime)
	{
		m_simStorage.mutable_data_blocks()->RemoveLast();
		m_blocksOfTimePoints.pop_back();
	}

	assert(!m_blocksOfTimePoints.empty());

	// load the last TD data block from file to proto file
	LoadBlockInternal(static_cast<uint32_t>(m_blocksOfTimePoints.size()) - 1);

	std::unique_ptr<ProtoBlockOfTimePoints> &pBlock = *m_blocksOfTimePoints.rbegin();
	assert(pBlock->time_points_size() > 0);
	assert(pBlock->time_points(0).time() <= _dTime);

	while (pBlock->time_points().rbegin()->time() > _dTime)
		pBlock->mutable_time_points()->RemoveLast();

	if (m_SCache.requestedTime > pBlock->time_points().rbegin()->time())
		m_SCache.requestedTime = 1e+300;

	m_simStorage.mutable_info()->set_end_time(pBlock->time_points().rbegin()->time());
	while (m_simStorage.mutable_info()->savedtimepoints_size() > 0)
		if (m_simStorage.mutable_info()->savedtimepoints(m_simStorage.mutable_info()->savedtimepoints_size() - 1) > pBlock->time_points().rbegin()->time())
			SimulationInfo()->mutable_savedtimepoints()->RemoveLast();
		else
			break;

	// HACK: nasty hack to remove all time points in case of time = 0.
	// sometimes time points are not removed correctly (maybe due to program crash or in the debug mode) and new smaller time points appear after old larger ones.
	// TODO: find and fix it!
	if (_dTime == 0 && m_simStorage.mutable_info()->savedtimepoints_size() > 1)
		while (m_simStorage.mutable_info()->savedtimepoints_size() > 1)
			SimulationInfo()->mutable_savedtimepoints()->RemoveLast();

	m_SWriteCache.time = 1e+300;
}

CDemStorage::STimePointsPairCache& CDemStorage::TimeDependentProperties(double _dTime)
{
	// if result for this time has been already cached
	if (m_SCache.requestedTime == _dTime) return m_SCache;

	// find related block by time
	const uint32_t nBlockIndex = FindBlockByTime(_dTime);

	// if this is not current left loaded block -> load found block to RAM
	if (m_iLeftLoadedBlock != nBlockIndex)
		LoadBlock(nBlockIndex);

	// get pair of time points in found block
	m_SCache = FindPairInBlock(*m_blocksOfTimePoints[nBlockIndex], _dTime);

	// if
	if (m_SCache.pTimePointR->time() < _dTime && nBlockIndex < m_blocksOfTimePoints.size() - 1)
		m_SCache.pTimePointR = m_blocksOfTimePoints[nBlockIndex + 1]->mutable_time_points(0);

	m_SCache.requestedTime = _dTime;
	return m_SCache;
}

ProtoTimePoint* CDemStorage::MutableTimeDependentProperties(double _dTime)
{
	if (m_SWriteCache.time == _dTime) return m_SWriteCache.pTimePoint;

	RemoveAllAfterTime(_dTime);

	ProtoTimePoint* pLastTimePoint = &*m_blocksOfTimePoints.back()->mutable_time_points()->rbegin();
	if (pLastTimePoint->time() == _dTime)
	{
		m_SWriteCache.time = _dTime;
		m_SWriteCache.pTimePoint = pLastTimePoint;
		return m_SWriteCache.pTimePoint;
	}

	SaveLastBlock();
	FlushToDisk(false); // flush to disk without SaveLastBlock() because such a function has been already called
	m_blocksOfTimePoints.push_back(std::make_unique<ProtoBlockOfTimePoints>());
	ProtoBlockDescriptor* pNewBlock = m_simStorage.mutable_data_blocks()->Add();
	ProtoBlockDescriptor* pPrewBlock = m_simStorage.mutable_data_blocks(m_simStorage.data_blocks_size() - 2);
	pNewBlock->set_start_time(_dTime);
	pNewBlock->set_offset_in_file(pPrewBlock->offset_in_file() + pPrewBlock->size());

	// allocate new time point
	ProtoTimePoint* pTimePoint = m_blocksOfTimePoints.back()->mutable_time_points()->Add();
	pTimePoint->set_time(_dTime);

	m_SWriteCache.time = _dTime;
	m_SWriteCache.pTimePoint = pTimePoint;
	m_SCache.requestedTime = 1e+300;

	UnloadUnusedBlocks(m_iLeftLoadedBlock);

	return pTimePoint;
}

uint32_t CDemStorage::FindBlockByTime(double _dTime) const
{
	// binary searching of suitable block by time
	int nLeft = 0;
	int nRight = m_simStorage.data_blocks_size();
	while (nLeft + 1 < nRight)
	{
		const int nCentre = (nLeft + nRight) / 2;
		if (m_simStorage.data_blocks(nCentre).start_time() > _dTime)
			nRight = nCentre;
		else
			nLeft = nCentre;
	}
	return nLeft;
}

CDemStorage::STimePointsPairCache CDemStorage::FindPairInBlock(ProtoBlockOfTimePoints &_protoBlockOfTimePoints, double _dTime)
{
	assert(_protoBlockOfTimePoints.time_points_size() != 0);

	// binary searching of suitable time point by time
	int nLeft = 0;
	int nRight = _protoBlockOfTimePoints.time_points_size();
	while (nLeft + 1 < nRight)
	{
		const int nCentre = (nLeft + nRight) / 2;
		if (_protoBlockOfTimePoints.time_points(nCentre).time() > _dTime)
			nRight = nCentre;
		else
			nLeft = nCentre;
	}

	// cache left and right time point
	STimePointsPairCache TimePointsPairCache;
	TimePointsPairCache.pTimePointL = _protoBlockOfTimePoints.mutable_time_points(nLeft);

	// if left time point less than required time point and the next time point is specified
	if (_protoBlockOfTimePoints.time_points(nLeft).time() < _dTime && nLeft + 1 < _protoBlockOfTimePoints.time_points_size())
		TimePointsPairCache.pTimePointR = _protoBlockOfTimePoints.mutable_time_points(nLeft + 1); // cache right time point
	else // else only one the same time point has to be cached
		TimePointsPairCache.pTimePointR = TimePointsPairCache.pTimePointL;

	// return pair of cached proto time points
	return TimePointsPairCache;
}

void CDemStorage::SaveLastBlock()
{
	// get pointers to last proto block of time points and related proto block descriptor
	ProtoBlockOfTimePoints* pProtoBlockOfTimePoints = &*m_blocksOfTimePoints.back();
	ProtoBlockDescriptor* pBlockDescriptor = &*m_simStorage.mutable_data_blocks()->rbegin();

	// uncompressed size of block with time-dependena data and value of time points
	uint32_t nBinSize = (uint32_t)pProtoBlockOfTimePoints->ByteSizeLong();

	// set type saved time-dependent data block and uncompressed size
	pBlockDescriptor->set_format(ProtoBlockDescriptor::kZippedProtoBuff);
	pBlockDescriptor->set_uncompressed_size(nBinSize);

	// serialize time-dependent data block and save compressed size
	char* pTmpBuffer;
	nBinSize = WriteToBuf(pTmpBuffer, *pProtoBlockOfTimePoints);
	pBlockDescriptor->set_size(nBinSize);

	// set pointer to current offset in file and write time-dependent data block
	m_pMDEMFile->FileThreadSetPointerTo(TIME_DEPENDENT_DATA, pBlockDescriptor->offset_in_file());
	m_pMDEMFile->FileThreadWrite(TIME_DEPENDENT_DATA, pTmpBuffer, nBinSize);

	// clear buffer and set required number of time points for saving to default value
	delete[] pTmpBuffer;
}

void CDemStorage::UnloadUnusedBlocks(uint32_t _iBlockInUse)
{
	if (m_iLeftLoadedBlock != _iBlockInUse && m_iLeftLoadedBlock != _iBlockInUse + 1 && m_iLeftLoadedBlock < m_blocksOfTimePoints.size() - 1)
		m_blocksOfTimePoints[m_iLeftLoadedBlock].reset(new ProtoBlockOfTimePoints);

	if (m_iLeftLoadedBlock + 1 != _iBlockInUse && m_iLeftLoadedBlock + 1 != _iBlockInUse + 1 && m_iLeftLoadedBlock + 1 < m_blocksOfTimePoints.size() - 1)
		m_blocksOfTimePoints[m_iLeftLoadedBlock + 1].reset(new ProtoBlockOfTimePoints);

	// remove penultimate block in case we finish writing it
	if (m_blocksOfTimePoints.size() >= 2 && m_blocksOfTimePoints.size() - 2 != m_iLeftLoadedBlock && m_blocksOfTimePoints.size() - 2 != m_iLeftLoadedBlock + 1)
		m_blocksOfTimePoints[m_blocksOfTimePoints.size() - 2].reset(new ProtoBlockOfTimePoints);
}

bool CDemStorage::LoadBlock(uint32_t _iBlock)
{
	// always current and next time block should be loaded into memory. This is needed for data interpolation and for saving.
	UnloadUnusedBlocks(_iBlock);

	bool bRes = true;
	if (_iBlock < m_blocksOfTimePoints.size() - 1) // check that end of blocks is not reached
		if (m_iLeftLoadedBlock != _iBlock && m_iLeftLoadedBlock + 1 != _iBlock)
			bRes = LoadBlockInternal(_iBlock);

	if (_iBlock + 1 < m_blocksOfTimePoints.size() - 1)
		if (bRes && m_iLeftLoadedBlock != _iBlock + 1 && m_iLeftLoadedBlock + 1 != _iBlock + 1)
			bRes = LoadBlockInternal(_iBlock + 1);

	m_iLeftLoadedBlock = _iBlock;
	return bRes;
}

bool CDemStorage::LoadBlockInternal(uint32_t _iBlock)
{
	assert(_iBlock < static_cast<uint32_t>(m_simStorage.data_blocks().size()));

	// get offset and size of current proto block of time points
	const uint64_t nBlockOffset = m_simStorage.data_blocks(_iBlock).offset_in_file();
	const uint32_t nBlockSize = m_simStorage.data_blocks(_iBlock).size();

	// set pointer to offset of current block
	m_pMDEMFile->FileThreadSetPointerTo(TIME_DEPENDENT_DATA, nBlockOffset);

	// memory allocation
	const std::unique_ptr<void, void(*)(void *)> pTmpBuffer(malloc(nBlockSize), std::free);
	if (!pTmpBuffer)
	{
		m_sLastError = "memory allocation error";
		m_pMDEMFile.reset();
		return false;
	}

	// read current proto block of time points from file to memory
	m_pMDEMFile->FileThreadRead(TIME_DEPENDENT_DATA, pTmpBuffer.get(), nBlockSize);
	const bool bRes = ReadFromBuf(static_cast<const char*>(pTmpBuffer.get()), *m_blocksOfTimePoints[_iBlock], nBlockSize, m_simStorage.data_blocks(_iBlock).uncompressed_size());
	if (!bRes) m_sLastError = "protobuf can not be parsed";

	return bRes;
}

std::vector<double> CDemStorage::GetAllTimePoints() const
{
	std::vector<double> vResult;
	vResult.reserve(SimulationInfo()->savedtimepoints_size());
	for (int i = 0; i < SimulationInfo()->savedtimepoints_size(); i++)
		vResult.push_back(SimulationInfo()->savedtimepoints(i));

	return vResult;
}

std::vector<double> CDemStorage::GetAllTimePointsOldFormat()
{
	std::vector<double> vResult;
	if (m_blocksOfTimePoints.empty()) return vResult;

	for (uint32_t i = 0; i < m_blocksOfTimePoints.size(); i++)
	{
		LoadBlock(i);
		for (int j = 0; j < m_blocksOfTimePoints[i]->time_points_size(); j++)
			vResult.push_back(m_blocksOfTimePoints[i]->time_points(j).time());
	}

	LoadBlock(0);
	m_SCache.requestedTime = 1e+300;
	m_SCache.pTimePointL = nullptr;
	m_SCache.pTimePointR = nullptr;
	m_SWriteCache.time = 1e+300;
	m_SWriteCache.pTimePoint = nullptr;

	return vResult;
}

bool CDemStorage::ReadFromBuf(const char *_pBuffer, google::protobuf::Message& _message, int _nSize, int _sizeFull)
{
	using namespace google::protobuf::io;
	ArrayInputStream arrayStream(_pBuffer, _nSize);
	GzipInputStream gzipStream(&arrayStream, GzipInputStream::ZLIB, _sizeFull);
	if(_sizeFull != -1)
		return _message.ParseFromBoundedZeroCopyStream(&gzipStream, _sizeFull);
	else
		return _message.ParseFromZeroCopyStream(&gzipStream);
}

int32_t CDemStorage::WriteToBuf(char *&_pBuffer, const google::protobuf::Message& _message)
{
	using namespace google::protobuf::io;
	const int nInitSize = (int)_message.ByteSizeLong() + 10;			// initial size of data. 10 for black magic
	GzipOutputStream::Options options;
	options.format = GzipOutputStream::ZLIB;
	options.compression_level = 1;
	_pBuffer = new char[nInitSize];
	ArrayOutputStream arrayStream(_pBuffer, nInitSize);
	GzipOutputStream gzipStream(&arrayStream, options);
	_message.SerializeToZeroCopyStream(&gzipStream);
	gzipStream.Close();
	return static_cast<int32_t>(arrayStream.ByteCount());	// size of data after saving
}

/////////////////////////////////////////////////////////////////////////////////////////////////

ProtoParticleInfo* CDemStorage::Object(uint32_t _nObjectId)
{
	ProtoSimulationInfo* pProtoSimulationInfo = SimulationInfo();
	while (pProtoSimulationInfo->particles_size() <= static_cast<int>(_nObjectId))
		pProtoSimulationInfo->add_particles();

	pProtoSimulationInfo->mutable_particles(_nObjectId)->set_id(_nObjectId);
	return pProtoSimulationInfo->mutable_particles(_nObjectId);
}

void CDemStorage::RemoveObject(uint32_t _nObjectId)
{
	ProtoSimulationInfo* pProtoSimulationInfo = SimulationInfo();

	// if object which should be removed is existed
	if (static_cast<uint32_t>(pProtoSimulationInfo->particles_size()) > _nObjectId)
	{
		pProtoSimulationInfo->mutable_particles(_nObjectId)->clear_id();
		pProtoSimulationInfo->mutable_particles(_nObjectId)->clear_type(); // for proto3
	}

	while (pProtoSimulationInfo->particles_size() > 0 && pProtoSimulationInfo->particles().rbegin()->type() == 0) //proto3 (in proto2 was: has_id == false)
		pProtoSimulationInfo->mutable_particles()->RemoveLast(); // remove last ProtoParticleInfo from proto file
}

uint32_t CDemStorage::ObjectsCount()
{
	ProtoSimulationInfo* pProtoSimulationInfo = SimulationInfo();
	return pProtoSimulationInfo->particles_size();
}

void CDemStorage::PrepareTimePointForRead(double _time, int _objectsCount)
{
	// get pair of proto time points
	const auto pair = TimeDependentProperties(_time);
	m_timePointR.protoTimePointL = pair.pTimePointL;
	m_timePointR.protoTimePointR = pair.pTimePointR;

	// add all required objects
	while (m_timePointR.protoTimePointL->particles_size() < _objectsCount)
		m_timePointR.protoTimePointL->add_particles();
	while (m_timePointR.protoTimePointR->particles_size() < _objectsCount)
		m_timePointR.protoTimePointR->add_particles();

	// store additional info
	m_timePointR.timeRequest = _time;
	m_timePointR.timeL = m_timePointR.protoTimePointL->time();
	m_timePointR.timeR = m_timePointR.protoTimePointR->time();
	m_timePointR.count = _objectsCount;
}

CTimePointR* CDemStorage::GetTimePointR(double _time, int _objectID)
{
	// if needed time point was already cached
	if (m_timePointR.timeRequest == _time && m_timePointR.count > _objectID)
		return &m_timePointR;

	// if not - load and prepare it to read
	PrepareTimePointForRead(_time, _objectID + 1);

	// return prepared requested value
	return &m_timePointR;
}

CTimePointR* CDemStorage::GetTimePointR()
{
	return &m_timePointR;
}

void CDemStorage::PrepareTimePointForWrite(double _time, int _objectsCount)
{
	// add this time point to the array of saved time points, if it is not there
	const int timePointsNum = m_simStorage.mutable_info()->savedtimepoints_size();
	if (timePointsNum == 0 || m_simStorage.mutable_info()->savedtimepoints(timePointsNum - 1) != _time)
		m_simStorage.mutable_info()->add_savedtimepoints(_time);

	// set end time and begin time if needed
	if (_time > m_simStorage.mutable_info()->end_time())
		m_simStorage.mutable_info()->set_end_time(_time);
	if (_time < m_simStorage.mutable_info()->begin_time())
		m_simStorage.mutable_info()->set_begin_time(_time);

	// add all required objects
	m_timePointW.protoTimePoint = MutableTimeDependentProperties(_time);
	while (m_timePointW.protoTimePoint->particles_size() < _objectsCount)
		m_timePointW.protoTimePoint->add_particles();

	// store additional info
	m_timePointW.time = _time;
	m_timePointW.count = _objectsCount;
}

CTimePointW* CDemStorage::GetTimePointW(double _time, int _objectID)
{
	// if needed time point was already cached
	if (m_timePointW.time == _time && m_timePointW.count > _objectID)
		return &m_timePointW;

	// if not - load and prepare it to write
	PrepareTimePointForWrite(_time, _objectID + 1);

	// return prepared requested value
	return &m_timePointW;
}

CTimePointW* CDemStorage::GetTimePointW()
{
	return &m_timePointW;
}
