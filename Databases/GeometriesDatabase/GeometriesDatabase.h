/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "GeneratedFiles/GeometriesDatabase.pb.h"
#include "TriangularMesh.h"

class CGeometriesDatabase
{
public:
	struct SGeometry
	{
		CTriangularMesh mesh;	// Triangular mesh.
		std::string key;		// Unique key within this database.
		double scaleFactor;		// Currently applied scale factor.
	};

private:
	std::string m_fileName;					// Where the database is currently stored.
	std::vector<SGeometry> m_geometries;	// List of geometries.

public:
	void NewDatabase();					// Creates a new database by removing information about all geometries and the file name.
	std::string GetFileName() const;	// Returns the name of the current database file.

	void AddGeometry(const std::string& _filePath);							// Adds geometry from the STL file.
	void AddGeometry(const CTriangularMesh& _mesh, const std::string& _key = "", double _scale = 1.0);	// Adds geometry from the mesh.
	void DeleteGeometry(size_t _index);										// Removes geometry with the specified index.
	void UpGeometry(size_t _index);											// Moves the selected geometry upwards in the list.
	void DownGeometry(size_t _index);										// Moves the selected geometry downwards in the list.

	size_t GeometriesNumber() const;							// Returns the number of defined geometries.
	SGeometry* Geometry(size_t _index);							// Returns a pointer to a geometry by its index.
	const SGeometry* Geometry(size_t _index) const;				// Returns a constant pointer to a geometry by its index.
	SGeometry* Geometry(const std::string& _key);				// Returns a pointer to a geometry by its key.
	const SGeometry* Geometry(const std::string& _key) const;	// Returns a constant pointer to a geometry by its key.
	std::vector<SGeometry*> Geometries();						// Returns pointers to all defined geometries.
	std::vector<const SGeometry*> Geometries() const;			// Returns constant pointers to all defined geometries.

	void ScaleGeometry(size_t _index, double _factor);						// Scales a geometry and moves in it to the scaled position.
	void ExportGeometry(size_t _index, const std::string& _filePath) const; // Export the selected geometry into an STL file.

	void SaveToFile(const std::string& _filePath);		// Save the database into the file.
	void LoadFromFile(const std::string& _filePath);	// Load the database from the file.

private:
	std::string GenerateKey(const std::string& _key = "") const;	// Generates a key, unique within this database.
};
