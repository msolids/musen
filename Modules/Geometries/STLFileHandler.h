/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <cstdint>
#include "TriangularMesh.h"

class CSTLFileHandler
{
	static const size_t STL_HEADER_SIZE   = 80 * sizeof(uint8_t);	// Size of header in the binary STL file.
	static const size_t STL_NUMBER_SIZE   = sizeof(uint32_t);		// Size of number field in the binary STL file.
	static const size_t STL_TRIANGLE_SIZE = 50;						// Size of each triangle in the binary STL file.

#pragma pack (push, 1)
	struct SSTLTriangle	// relies on the sizeof(float) == 4 and doesn't take the byte order into account
	{
		float norm[3]{ 0 };
		float v1[3]{ 0 };
		float v2[3]{ 0 };
		float v3[3]{ 0 };
		uint16_t attr{ 0 };
		SSTLTriangle() = default;
		SSTLTriangle(const CVector3& _v1, const CVector3& _v2, const CVector3& _v3)
		{
			v1[0] = static_cast<float>(_v1.x); v1[1] = static_cast<float>(_v1.y); v1[2] = static_cast<float>(_v1.z);
			v2[0] = static_cast<float>(_v2.x); v2[1] = static_cast<float>(_v2.y); v2[2] = static_cast<float>(_v2.z);
			v3[0] = static_cast<float>(_v3.x); v3[1] = static_cast<float>(_v3.y); v3[2] = static_cast<float>(_v3.z);
		}
		CTriangle ToTriangle()
		{
			return CTriangle{ CVector3{ v1[0], v1[1], v1[2] }, CVector3{ v2[0], v2[1], v2[2] }, CVector3{ v3[0], v3[1], v3[2] } };
		}
	};
#pragma pack (pop)
	enum class EFileType { UNKNOWN, STL_ASCII, STL_BINARY };


public:
	/// Extracts an STL figure from the provided file.
	static CTriangularMesh ReadFromFile(const std::string& _filePath);
	/// Writes the provided STL figure into the file.
	static void WriteToFile(const CTriangularMesh& _geometry, const std::string& _filePath);

private:
	/// Determines the type of the given STL file for read.
	static EFileType GetFileTypeR(const std::string& _filePath);
	/// Determines the type of the given STL file for write.
	static EFileType GetFileTypeW(const std::string& _filePath);

	/// Determines whether the _filePath file is a file in ASCII STL format.
	static bool IsSTLAscii(const std::string& _filePath);
	/// Reads an ASCII STL file into a triangular mesh.
	static CTriangularMesh ReadSTLAscii(const std::string& _filePath);
	/// Writes a triangular mesh into an ASCII STL file.
	static void WriteSTLAscii(const CTriangularMesh& _geometry, const std::string& _filePath);

	/// Determines whether the _filePath file is a file in binary STL format.
	static bool IsSTLBinary(const std::string& _filePath);
	/// Reads a binary STL file into a triangular mesh.
	static CTriangularMesh ReadSTLBinary(const std::string& _filePath);
	/// Writes a triangular mesh into a binary STL file.
	static void WriteSTLBinary(const CTriangularMesh& _geometry, const std::string& _filePath);
};

