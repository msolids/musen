/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "FileHandler.h"

CFileHandler::CFileHandler()
{
	m_pFile = nullptr;
}

CFileHandler::~CFileHandler()
{
	if (m_pFile) fclose(m_pFile);
}

void CFileHandler::UseFile(const std::string &_sName)
{
	if (m_pFile)		    // if file is already exist
	{
		fclose(m_pFile);	// close current file
		m_pFile = nullptr;  // clear pointer to file
	}
	m_sError.clear();		// clear error string

#ifdef PATH_CONFIGURED      // linux

	auto f = fopen(_sName.c_str(), "rb+");
	if (f == nullptr)
	{
		f = fopen(_sName.c_str(), "ab+");
		if (f) fclose(f);
		f = fopen(_sName.c_str(), "rb+");
	}
#else					    // windows
	FILE* f;
	_wfopen_s(&f, UnicodePath(_sName).c_str(), L"rb+");		    // open file for binary reading + writing
	if (f == nullptr)										    // if file is not exsit
	{
		_wfopen_s(&f, UnicodePath(_sName).c_str(), L"ab+");		// create file with name = _sName
		if (f) fclose(f);										// if file is exist -> close file
		_wfopen_s(&f, UnicodePath(_sName).c_str(), L"rb+");		// open file for binary reading + writing
	}
#endif

	m_pFile = f;
	if (!m_pFile) m_sError = std::string("Can not open file") + _sName;
}

bool CFileHandler::IsFileValid()
{
	return m_pFile != nullptr;
}

std::string CFileHandler::GetLastError()
{
	return m_sError;
}

void CFileHandler::SetPointerInFileTo(uint64_t _nOffset, int _nOrigin)
{
	if (!m_pFile) return;

#ifdef PATH_CONFIGURED
	bool res = 0 == fseeko64(m_pFile, _nOffset, _nOrigin);
#else
	bool res = 0 == _fseeki64(m_pFile, _nOffset, _nOrigin);
#endif

	if (!res)
	{
		m_sError = "_fseeki64 error ";
		//m_error += strerror(errno);
		fclose(m_pFile);
		m_pFile = nullptr;
	}
}

uint64_t CFileHandler::GetPointerInFile()
{
	if (!m_pFile) return 0;

#ifdef PATH_CONFIGURED
	return ftello64(m_pFile);
#else
	return _ftelli64(m_pFile);
#endif
}

void CFileHandler::SetFileSize(uint64_t _nSize)
{
	if (!m_pFile) return;

#ifdef PATH_CONFIGURED
	int filedes = fileno(m_pFile);
	int res = ftruncate(filedes, _nSize);
#else
	int filedes = _fileno(m_pFile);
	int res = _chsize_s(filedes, _nSize); // ftruncate
#endif
}

uint32_t CFileHandler::ReadFromFile(void *_pBuffer, uint32_t _nBufferSize)
{
	if (!m_pFile) return 0;
	return (uint32_t)fread(_pBuffer, 1, _nBufferSize, m_pFile);
}

uint32_t CFileHandler::WriteToFile(void *_pBuffer, uint32_t _nBufferSize)
{
	if (!m_pFile) return 0;
	uint32_t res = (uint32_t)fwrite(_pBuffer, 1, _nBufferSize, m_pFile);

	if (res != _nBufferSize)
	{
		m_sError = "Write error";
		fclose(m_pFile);
		m_pFile = nullptr;
		return 0;
	}
	return res;
}

void CFileHandler::FlushToFile()
{
	if (m_pFile) fflush(m_pFile);
}