/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <string>
#include <sys/stat.h>
#ifdef PATH_CONFIGURED
#include <dirent.h>
#include <cstring>
#else
#ifdef NOMINMAX
#include <Windows.h>
#else
#define NOMINMAX
#include <Windows.h>
#undef NOMINMAX
#endif
#endif

namespace MUSENFileFunctions
{
	inline std::string getFileExt(const std::string& _filePath)
	{
		const size_t slashPos = _filePath.find_last_of("/\\");
		const std::string lastWord = slashPos != std::string::npos ? _filePath.substr(slashPos + 1) : _filePath;
		const size_t dotPos = lastWord.find_last_of('.');
		return dotPos != std::string::npos ? lastWord.substr(dotPos + 1) : "";
	}

	inline size_t getFileSize(const std::string& _filePath)
	{
		struct stat info{};
		const int err = stat(_filePath.c_str(), &info);
		return err == 0 ? info.st_size : 0;
	}

	inline std::string removeFileExt(const std::string& _filePath)
	{
		const size_t extLength = getFileExt(_filePath).length();
		if (extLength == 0) return _filePath;
		return _filePath.substr(0, _filePath.length() - extLength - 1);
	}

	inline std::vector<std::string> filesList(const std::string& _sPath, const std::string& _sFilter)
#ifdef PATH_CONFIGURED
	{
		std::vector<std::string> vFiles;
		DIR* dir = opendir(_sPath.c_str());
		const char* sExt = _sFilter.substr(1).c_str();
		if (dir)
		{
			struct dirent* hDir;
			while ((hDir = readdir(dir)) != NULL)
			{
				if (!strcmp(hDir->d_name, ".")) continue;
				if (!strcmp(hDir->d_name, "..")) continue;
				if (strstr(hDir->d_name, sExt))
					vFiles.push_back(hDir->d_name);
			}
			closedir(dir);
		}
		return vFiles;
	}
#else
	{
		std::vector<std::string> vFiles;
		std::string sSearchPath = _sPath + "/" + _sFilter;
		WIN32_FIND_DATAA fileData;
		HANDLE hFile = ::FindFirstFileA(sSearchPath.c_str(), &fileData);
		if (hFile != INVALID_HANDLE_VALUE)
		{
			do
			{
				if (!(fileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
					vFiles.push_back(_sPath + "/" + fileData.cFileName);
			} while (::FindNextFileA(hFile, &fileData));
			::FindClose(hFile);
		}
		return vFiles;
	}
#endif

	inline bool removeFile(const std::string& _sFileName)
	{
		return(std::remove(_sFileName.c_str()) == 0);
	}

	inline bool renameFile(const std::string& _oldName, const std::string& _newName)
	{
		return std::rename(_oldName.c_str(), _newName.c_str()) == 0;
	}
}