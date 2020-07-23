/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#ifdef __linux__
#include <unistd.h>
#include <sys/types.h>
#else
#include <io.h>
#endif
#include <cstdio>
#include <memory>
#include "MUSENStringFunctions.h"

// CFileHandler is the class which realizes the main functions for handling of a binary file.
// Low-level C functions are used.
class CFileHandler
{
public:
	CFileHandler();
	~CFileHandler();

	// Prepares file for using (create, open).
	void UseFile(const std::string &_sName);
	// Checks file for the existence.
	bool IsFileValid();
	// Returns string with the last error description.
	std::string GetLastError();
	// Sets pointer to offset inside of the file, _nOrigin can be equal to SEEK_SET = 0 or SEEK_CUR = 1 or SEEK_END = 2.
	void SetPointerInFileTo(uint64_t _nOffset, int _nOrigin);
	// Returns current offest inside of the file.
	uint64_t GetPointerInFile();
	// Truncates file to size.
	void SetFileSize(uint64_t _nSize);
	// Reads from file to buffer.
	uint32_t ReadFromFile(void *_pBuffer, uint32_t _nBufferSize);
	// Writes to file from buffer.
	uint32_t WriteToFile(void *_pBuffer, uint32_t _nBufferSize);
	// If the given stream was open for writing any unwritten data in its output buffer is written to the file.
	void FlushToFile();

private:
	std::string m_sError; // String with error description.
	FILE* m_pFile;		  // Pointer to file.
};

