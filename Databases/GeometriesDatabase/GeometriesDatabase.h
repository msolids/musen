/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "MixedFunctions.h"
#include "MUSENStringFunctions.h"
#include "STLFileHandler.h"
#pragma warning(push)
#pragma warning(disable: 6011 6387 26495)
#include "GeneratedFiles/GeometriesDatabase.pb.h"
#pragma warning(pop)

class CGeometriesDatabase
{
public:
	~CGeometriesDatabase();
	void AddGeometry(const std::string& _sSTLFileName); // add geometry from an STL file
	void ExportGeometry(size_t _index, const std::string& _sSTLFileName) const; // export selected geometry into an STL file

	size_t GetGeometriesNumber() const;
	CTriangularMesh* GetGeometry(size_t _nIndex);
	const CTriangularMesh* GetGeometry(size_t _nIndex) const;

	std::string GetFileName() const;

	void SaveToFile(const std::string& _sFileName); // save database into the file
	void LoadFromFile(const std::string& _sFileName); // load  database from the file
	void NewDatabase();

	void DeleteGeometry(size_t _nIndex);
	void UpGeometry(size_t _nIndex);
	void DownGeometry(size_t _nIndex);

private:
	std::string m_sDatabaseFileName;
	std::vector<CTriangularMesh*> m_vGeometries;

	void DeleteGeometries();
};