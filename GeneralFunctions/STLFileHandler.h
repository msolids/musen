/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "BasicTypes.h"
#include "MUSENStringFunctions.h"
#include "MUSENFileFunctions.h"
#include <fstream>
#include <cstdint>
#include "TriangularMesh.h"

class CSTLFileHandler
{
private:
#pragma pack (push, 1)
	struct SSTLTriangle	// relies on the sizeof(float) == 4 and doesn't take the endianness into account
	{
		float norm[3];
		float v1[3];
		float v2[3];
		float v3[3];
		uint16_t attr;
		SSTLTriangle()
		{
			norm[0] = norm[1] = norm[2] = 0;
			v1[0] = v1[1] = v1[2] = 0;
			v2[0] = v2[1] = v2[2] = 0;
			v3[0] = v3[1] = v3[2] = 0;
			attr = 0;
		}
		SSTLTriangle(const CVector3& _v1, const CVector3& _v2, const CVector3& _v3)
		{
			norm[0] = norm[1] = norm[2] = 0;
			v1[0] = static_cast<float>(_v1.x); v1[1] = static_cast<float>(_v1.y); v1[2] = static_cast<float>(_v1.z);
			v2[0] = static_cast<float>(_v2.x); v2[1] = static_cast<float>(_v2.y); v2[2] = static_cast<float>(_v2.z);
			v3[0] = static_cast<float>(_v3.x); v3[1] = static_cast<float>(_v3.y); v3[2] = static_cast<float>(_v3.z);
			attr = 0;
		}
	};
#pragma pack (pop)
	enum class FileType
	{
		UNKNOWN_TYPE, STLA, STLB
	};
	enum class FileAccess
	{
		READ, WRITE
	};

public:
	CSTLFileHandler();
	~CSTLFileHandler();

	CTriangularMesh ReadFromFile(const std::string& _filePath) const;
	void WriteToFile(const CTriangularMesh& _geometry, const std::string& _filePath) const;
	void WriteToFile(const CTriangularMesh& _geometry, std::ofstream& _file) const;

private:
	FileType GetFileType(const std::string& _filePath, const FileAccess& _access) const;

	bool IsSTLA(const std::string& _filePath) const;
	CTriangularMesh Read_STLA(const std::string& _filePath) const;
	void Write_STLA(const CTriangularMesh& _geometry, const std::string& _filePath) const;

	bool IsSTLB(const std::string& _filePath) const;
	CTriangularMesh Read_STLB(const std::string& _filePath) const;
	void Write_STLB(const CTriangularMesh& _geometry, const std::string& _filePath) const;
	void Write_STLB(const CTriangularMesh& _geometry, std::ofstream& _file) const;
};

