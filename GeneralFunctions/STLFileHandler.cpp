/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "STLFileHandler.h"

CSTLFileHandler::CSTLFileHandler()
{
}

CSTLFileHandler::~CSTLFileHandler()
{
}

CTriangularMesh CSTLFileHandler::ReadFromFile(const std::string& _filePath) const
{
	switch (GetFileType(_filePath, FileAccess::READ))
	{
	case FileType::STLA:
		return Read_STLA(_filePath);
	case FileType::STLB:
		return Read_STLB(_filePath);
	default:
		return CTriangularMesh();
	}
}

void CSTLFileHandler::WriteToFile(const CTriangularMesh& _geometry, const std::string& _filePath) const
{
	switch (GetFileType(_filePath, FileAccess::WRITE))
	{
	case FileType::STLA:
		return Write_STLA(_geometry, _filePath);
	case FileType::STLB:
		return Write_STLB(_geometry, _filePath);
	default:
		return Write_STLB(_geometry, _filePath);
	}
}

void CSTLFileHandler::WriteToFile(const CTriangularMesh& _geometry, std::ofstream& _file) const
{
	if (!_file) return;
	Write_STLB(_geometry, _file);
}

CSTLFileHandler::FileType CSTLFileHandler::GetFileType(const std::string& _filePath, const FileAccess& _access) const
{
	std::string ext = ToLowerCase(MUSENFileFunctions::getFileExt(_filePath));
	if ((ext == "stl" || ext == "stlb") && (_access == FileAccess::WRITE || IsSTLB(_filePath)))
		return FileType::STLB;
	if ((ext == "stl" || ext == "stla") && (_access == FileAccess::WRITE || IsSTLA(_filePath)))
		return FileType::STLA;
	return FileType::UNKNOWN_TYPE;
}

bool CSTLFileHandler::IsSTLA(const std::string& _filePath) const
{
	if (IsSTLB(_filePath))
		return false;

	std::ifstream file(UnicodePath(_filePath), std::ifstream::in);
	if (!file)
		return false;

	std::string line;
	safeGetLine(file, line);
	std::stringstream ss(line);
	ss >> line;
	file.close();
	return ToLowerCase(line) == "solid";
}

CTriangularMesh CSTLFileHandler::Read_STLA(const std::string& _filePath) const
{
	std::ifstream file(UnicodePath(_filePath), std::ifstream::in);
	if (!file)
		return CTriangularMesh();

	CTriangularMesh geometry;
	std::string key, line;
	std::vector<CVector3> face;
	while (safeGetLine(file, line).good())
	{
		std::stringstream ss(line);
		ss >> key;
		if (ToLowerCase(key) == "solid")
		{
			geometry.sName = GetRestOfLine(&ss);
		}
		else if (ToLowerCase(key) == "vertex")
		{
			CVector3 vertex;
			ss >> vertex.x >> vertex.y >> vertex.z;
			face.push_back(vertex);
		}
		else if (ToLowerCase(key) == "endloop")
		{
			if (face.size() != 3) // only handle triangles
				return CTriangularMesh();
			geometry.vTriangles.push_back(STriangleType(face[0], face[1], face[2]));
			face.clear();
		}
	}

	file.close();

	return geometry;
}

void CSTLFileHandler::Write_STLA(const CTriangularMesh& _geometry, const std::string& _filePath) const
{
	std::ofstream file(UnicodePath(_filePath), std::ofstream::out | std::ofstream::trunc);
	if (!file) return;	// cannot open file for writing
	file.imbue(std::locale("C"));	// use standard locale for output
	file.precision(6);				// set floating point precision
	file << std::scientific;		// sign-mantissa-"e"-sign-exponent format

	CVector3 normal(0); // we do not store normals -> set them all to 0

	file << "solid " << _geometry.sName << std::endl;
	for (size_t i = 0; i < _geometry.vTriangles.size(); ++i)
	{
		file << "  facet normal " << normal << std::endl;
		file << "    outer loop" << std::endl;
		file << "      vertex  " << _geometry.vTriangles[i].p1 << std::endl;
		file << "      vertex  " << _geometry.vTriangles[i].p2 << std::endl;
		file << "      vertex  " << _geometry.vTriangles[i].p3 << std::endl;
		file << "    endloop" << std::endl;
		file << "  endfacet" << std::endl;
	}
	file << "endsolid " << _geometry.sName << std::endl;

	file.close();
}

bool CSTLFileHandler::IsSTLB(const std::string& _filePath) const
{
	const unsigned HEADER_FIELD_SIZE = 80;
	const unsigned NUMBER_FIELD_SIZE = 4;
	const unsigned TRIANGLE_FIELD_SIZE = 50;

	size_t fileSize = MUSENFileFunctions::getFileSize(_filePath);
	if (fileSize < HEADER_FIELD_SIZE + NUMBER_FIELD_SIZE) // header + number of triangles
		return false;

	std::ifstream file(UnicodePath(_filePath), std::ifstream::in | std::ifstream::binary);
	if (!file)
		return false;

	file.seekg(HEADER_FIELD_SIZE);	// skip header
	uint32_t trianglesNumber;
	file.read((char*)(&trianglesNumber), sizeof(uint32_t));

	file.close();

	return fileSize == (trianglesNumber * TRIANGLE_FIELD_SIZE) + HEADER_FIELD_SIZE + NUMBER_FIELD_SIZE;
}

CTriangularMesh CSTLFileHandler::Read_STLB(const std::string& _filePath) const
{
	const unsigned HEADER_FIELD_SIZE = 80;

	std::ifstream file(UnicodePath(_filePath), std::ifstream::in | std::ifstream::binary);
	if (!file)
		return CTriangularMesh();

	CTriangularMesh geometry;
	geometry.sName = "STL_geometry";

	char header[HEADER_FIELD_SIZE];
	file.read(header, HEADER_FIELD_SIZE);
	uint32_t trianglesNumber;
	file.read((char*)(&trianglesNumber), sizeof(uint32_t));
	for (size_t i = 0; i < trianglesNumber; ++i)
	{
		SSTLTriangle triangle;
		file.read((char*)(&triangle), sizeof(SSTLTriangle));
		geometry.vTriangles.push_back(STriangleType(CVector3(triangle.v1[0], triangle.v1[1], triangle.v1[2]),
													CVector3(triangle.v2[0], triangle.v2[1], triangle.v2[2]),
													CVector3(triangle.v3[0], triangle.v3[1], triangle.v3[2])));
	}
	file.close();

	return geometry;
}

void CSTLFileHandler::Write_STLB(const CTriangularMesh& _geometry, const std::string& _filePath) const
{
	std::ofstream file(UnicodePath(_filePath), std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
	if (!file) return;	// cannot open file for writing
	Write_STLB(_geometry, file);
}

void CSTLFileHandler::Write_STLB(const CTriangularMesh& _geometry, std::ofstream& _file) const
{
	const unsigned HEADER_FIELD_SIZE = 80;

	char header[HEADER_FIELD_SIZE];
	_file.write((char*)(&header), HEADER_FIELD_SIZE);
	uint32_t trianglesNumber = static_cast<uint32_t>(_geometry.vTriangles.size());
	_file.write((char*)(&trianglesNumber), sizeof(uint32_t));
	for (size_t i = 0; i < trianglesNumber; ++i)
	{
		SSTLTriangle triangle(_geometry.vTriangles[i].p1, _geometry.vTriangles[i].p2, _geometry.vTriangles[i].p3);
		_file.write((char*)(&triangle), sizeof(SSTLTriangle));
	}
}
