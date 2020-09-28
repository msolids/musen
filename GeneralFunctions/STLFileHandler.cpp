/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "STLFileHandler.h"
#include "MUSENFileFunctions.h"
#include "MUSENStringFunctions.h"
#include <fstream>

CTriangularMesh CSTLFileHandler::ReadFromFile(const std::string& _filePath)
{
	switch (GetFileTypeR(_filePath))
	{
	case EFileType::STL_ASCII:		return ReadSTLAscii(_filePath);
	case EFileType::STL_BINARY:		return ReadSTLBinary(_filePath);
	default:						return {};
	}
}

void CSTLFileHandler::WriteToFile(const CTriangularMesh& _geometry, const std::string& _filePath)
{
	switch (GetFileTypeW(_filePath))
	{
	case EFileType::STL_ASCII:		return WriteSTLAscii(_geometry, _filePath);
	case EFileType::STL_BINARY:
	default:						return WriteSTLBinary(_geometry, _filePath);
	}
}

CSTLFileHandler::EFileType CSTLFileHandler::GetFileTypeR(const std::string& _filePath)
{
	if (IsSTLBinary(_filePath))	return EFileType::STL_BINARY;
	if (IsSTLAscii(_filePath))	return EFileType::STL_ASCII;
	return EFileType::UNKNOWN;
}

CSTLFileHandler::EFileType CSTLFileHandler::GetFileTypeW(const std::string& _filePath)
{
	const std::string ext = ToLowerCase(MUSENFileFunctions::getFileExt(_filePath));
	if (ext == "stla" || MUSENFileFunctions::isFileExist(_filePath) && IsSTLAscii(_filePath))					return EFileType::STL_ASCII;
	if (ext == "stl" || ext == "stlb" || MUSENFileFunctions::isFileExist(_filePath) && IsSTLBinary(_filePath))	return EFileType::STL_BINARY;
	return EFileType::UNKNOWN;
}

bool CSTLFileHandler::IsSTLAscii(const std::string& _filePath)
{
	std::ifstream file(UnicodePath(_filePath), std::ifstream::in);
	if (!file) return false;

	std::string line;
	safeGetLine(file, line);
	file.close();
	std::stringstream ss{ line };
	return ToLowerCase(GetValueFromStream<std::string>(&ss)) == "solid";
}

CTriangularMesh CSTLFileHandler::ReadSTLAscii(const std::string& _filePath)
{
	std::ifstream file(UnicodePath(_filePath), std::ifstream::in);
	if (!file) return {};

	CTriangularMesh geometry;
	std::string line;
	std::vector<CVector3> face;
	while (safeGetLine(file, line).good())
	{
		std::stringstream ss{ line };
		const std::string key = GetValueFromStream<std::string>(&ss);
		if (ToLowerCase(key) == "solid")
			geometry.SetName(GetRestOfLine(&ss));
		else if (ToLowerCase(key) == "vertex")
			face.push_back(GetValueFromStream<CVector3>(&ss));
		else if (ToLowerCase(key) == "endloop")
		{
			if (face.size() != 3) return {}; // only handle triangles
			geometry.AddTriangle({ face[0], face[1], face[2] });
			face.clear();
		}
	}

	file.close();

	return geometry;
}

void CSTLFileHandler::WriteSTLAscii(const CTriangularMesh& _geometry, const std::string& _filePath)
{
	std::ofstream file(UnicodePath(_filePath), std::ofstream::out | std::ofstream::trunc);
	if (!file) return;				// cannot open file for writing
	file.imbue(std::locale("C"));	// use standard locale for output
	file.precision(6);				// set floating point precision
	file << std::scientific;		// sign-mantissa-"e"-sign-exponent format

	const CVector3 normal{ 0 }; // we do not store normals -> set them all to 0

	file << "solid " << _geometry.Name() << std::endl;
	for (const auto& t : _geometry.Triangles())
	{
		file << "  facet normal " << normal << std::endl;
		file << "    outer loop"            << std::endl;
		file << "      vertex  "  << t.p1   << std::endl;
		file << "      vertex  "  << t.p2   << std::endl;
		file << "      vertex  "  << t.p3   << std::endl;
		file << "    endloop"               << std::endl;
		file << "  endfacet"                << std::endl;
	}
	file << "endsolid " << _geometry.Name() << std::endl;

	file.close();
}

bool CSTLFileHandler::IsSTLBinary(const std::string& _filePath)
{
	const size_t fileSize = MUSENFileFunctions::getFileSize(_filePath);
	if (fileSize < STL_HEADER_SIZE + STL_NUMBER_SIZE) return false; // header + number of triangles

	std::ifstream file(UnicodePath(_filePath), std::ifstream::in | std::ifstream::binary);
	if (!file) return false;

	// get the number of triangles
	uint32_t trianglesNumber;
	file.seekg(STL_HEADER_SIZE);	// skip the header
	file.read(reinterpret_cast<char*>(&trianglesNumber), STL_NUMBER_SIZE);
	file.close();

	return fileSize == trianglesNumber * STL_TRIANGLE_SIZE + STL_HEADER_SIZE + STL_NUMBER_SIZE;
}

CTriangularMesh CSTLFileHandler::ReadSTLBinary(const std::string& _filePath)
{
	std::ifstream file(UnicodePath(_filePath), std::ifstream::in | std::ifstream::binary);
	if (!file) return {};

	// get the number of triangles
	uint32_t trianglesNumber;
	file.seekg(STL_HEADER_SIZE);	// skip the header
	file.read(reinterpret_cast<char*>(&trianglesNumber), STL_NUMBER_SIZE);

	// get triangles data
	std::vector<SSTLTriangle> stlTriangles(trianglesNumber);
	file.read(reinterpret_cast<char*>(stlTriangles.data()), trianglesNumber * sizeof(SSTLTriangle));
	file.close();

	// transform STLTriangle to CTriangle
	std::vector<CTriangle> triangles(trianglesNumber);
	for (size_t i = 0; i < trianglesNumber; ++i)
		triangles[i] = stlTriangles[i].ToTriangle();

	return CTriangularMesh{ MUSENFileFunctions::getFileName(_filePath), triangles };
}

void CSTLFileHandler::WriteSTLBinary(const CTriangularMesh& _geometry, const std::string& _filePath)
{
	std::ofstream file(UnicodePath(_filePath), std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
	if (!file) return;	// cannot open file for writing

	// write header
	char header[STL_HEADER_SIZE];
	file.write(reinterpret_cast<char*>(&header), STL_HEADER_SIZE);

	// write the number of triangles
	auto trianglesNumber = static_cast<uint32_t>(_geometry.TrianglesNumber());
	file.write(reinterpret_cast<char*>(&trianglesNumber), sizeof(uint32_t));

	// write triangles data
	for (const auto& t : _geometry.Triangles())
	{
		SSTLTriangle triangle(t.p1, t.p2, t.p3);
		file.write(reinterpret_cast<char*>(&triangle), sizeof(SSTLTriangle));
	}
}
