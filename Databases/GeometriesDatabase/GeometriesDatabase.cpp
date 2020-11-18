/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GeometriesDatabase.h"
#include "STLFileHandler.h"
#include "MUSENStringFunctions.h"
#include <fstream>

void CGeometriesDatabase::NewDatabase()
{
	m_geometries.clear();
	m_fileName.clear();
}

std::string CGeometriesDatabase::GetFileName() const
{
	return m_fileName;
}

void CGeometriesDatabase::AddGeometry(const std::string& _filePath)
{
	AddGeometry(CSTLFileHandler::ReadFromFile(_filePath));
}

void CGeometriesDatabase::AddGeometry(const CTriangularMesh& _mesh, const std::string& _key/* = ""*/)
{
	if (_mesh.IsEmpty()) return;
	m_geometries.push_back({ _mesh, GenerateKey(_key) });
}

void CGeometriesDatabase::DeleteGeometry(size_t _index)
{
	if (_index < m_geometries.size())
		m_geometries.erase(m_geometries.begin() + _index);
}

void CGeometriesDatabase::UpGeometry(size_t _index)
{
	if (_index < m_geometries.size() && _index != 0)
		std::iter_swap(m_geometries.begin() + _index, m_geometries.begin() + _index - 1);
}

void CGeometriesDatabase::DownGeometry(size_t _index)
{
	if (_index < m_geometries.size() && _index != m_geometries.size() - 1)
		std::iter_swap(m_geometries.begin() + _index, m_geometries.begin() + _index + 1);
}

size_t CGeometriesDatabase::GeometriesNumber() const
{
	return m_geometries.size();
}

CGeometriesDatabase::SGeometry* CGeometriesDatabase::Geometry(size_t _index)
{
	return const_cast<SGeometry*>(static_cast<const CGeometriesDatabase&>(*this).Geometry(_index));
}

const CGeometriesDatabase::SGeometry* CGeometriesDatabase::Geometry(size_t _index) const
{
	if (_index >= m_geometries.size()) return nullptr;
	return &m_geometries[_index];
}

CGeometriesDatabase::SGeometry* CGeometriesDatabase::Geometry(const std::string& _key)
{
	return const_cast<SGeometry*>(static_cast<const CGeometriesDatabase&>(*this).Geometry(_key));
}

const CGeometriesDatabase::SGeometry* CGeometriesDatabase::Geometry(const std::string& _key) const
{
	for (const auto& geometry : m_geometries)
		if (geometry.key == _key)
			return &geometry;
	return nullptr;
}

std::vector<CGeometriesDatabase::SGeometry*> CGeometriesDatabase::Geometries()
{
	std::vector<SGeometry*> res;
	res.reserve(m_geometries.size());
	for (auto& g : m_geometries)
		res.push_back(&g);
	return res;
}

std::vector<const CGeometriesDatabase::SGeometry*> CGeometriesDatabase::Geometries() const
{
	std::vector<const SGeometry*> res;
	res.reserve(m_geometries.size());
	for (const auto& g : m_geometries)
		res.push_back(&g);
	return res;
}

void CGeometriesDatabase::ScaleGeometry(size_t _index, double _factor)
{
	if (_index >= m_geometries.size()) return;
	m_geometries[_index].mesh.Scale(_factor);											// scale
	m_geometries[_index].mesh.SetCenter(m_geometries[_index].mesh.Center() * _factor);	// move to scaled coordinates
}

void CGeometriesDatabase::ExportGeometry(size_t _index, const std::string& _filePath) const
{
	if (_index >= m_geometries.size()) return;
	CSTLFileHandler::WriteToFile(m_geometries[_index].mesh, _filePath);
}

void CGeometriesDatabase::SaveToFile(const std::string& _filePath)
{
	ProtoGeometriesDB protoGeometriesDB;

	for (const auto& g : m_geometries)
	{
		ProtoGeometry* protoGeom = protoGeometriesDB.add_geometry();
		protoGeom->set_name(g.mesh.Name());
		protoGeom->set_key(g.key);
		for (const auto& t : g.mesh.Triangles())
		{
			ProtoGeomVect* vert1 = protoGeom->add_vertices();
			ProtoGeomVect* vert2 = protoGeom->add_vertices();
			ProtoGeomVect* vert3 = protoGeom->add_vertices();
			vert1->set_x(t.p1.x);	vert1->set_y(t.p1.y);	vert1->set_z(t.p1.z);
			vert2->set_x(t.p2.x);	vert2->set_y(t.p2.y);	vert2->set_z(t.p2.z);
			vert3->set_x(t.p3.x);	vert3->set_y(t.p3.y);	vert3->set_z(t.p3.z);
		}
	}

	std::fstream outFile(UnicodePath(_filePath), std::ios::out | std::ios::trunc | std::ios::binary);
	std::string data;
	// TODO: consider to use SerializeToZeroCopyStream() for performance
	protoGeometriesDB.SerializeToString(&data);
	outFile << data;
	outFile.close();

	m_fileName = _filePath;
}

void CGeometriesDatabase::LoadFromFile(const std::string& _filePath)
{
	std::fstream inputFile(UnicodePath(_filePath), std::ios::in | std::ios::binary);
	if (!inputFile)	return;

	ProtoGeometriesDB protoGeometriesDB;
	// TODO: consider to use ParseFromZeroCopyStream() for performance
	if (!protoGeometriesDB.ParseFromString(std::string(std::istreambuf_iterator<char>(inputFile), std::istreambuf_iterator<char>()))) return;

	m_geometries.clear();

	for (int i = 0; i < protoGeometriesDB.geometry_size(); ++i)
	{
		ProtoGeometry* protoGeom = protoGeometriesDB.mutable_geometry(i);
		std::vector<CTriangle> triangles(protoGeom->vertices_size() / 3);
		for (int j = 0; j < protoGeom->vertices_size() / 3; ++j)
			triangles[j] = CTriangle {
				CVector3(protoGeom->mutable_vertices(j * 3 + 0)->x(), protoGeom->mutable_vertices(j * 3 + 0)->y(), protoGeom->mutable_vertices(j * 3 + 0)->z()),
				CVector3(protoGeom->mutable_vertices(j * 3 + 1)->x(), protoGeom->mutable_vertices(j * 3 + 1)->y(), protoGeom->mutable_vertices(j * 3 + 1)->z()),
				CVector3(protoGeom->mutable_vertices(j * 3 + 2)->x(), protoGeom->mutable_vertices(j * 3 + 2)->y(), protoGeom->mutable_vertices(j * 3 + 2)->z())};
		AddGeometry(CTriangularMesh{ protoGeom->name(), triangles }, protoGeom->key());
	}

	m_fileName = _filePath;
}

std::string CGeometriesDatabase::GenerateKey(const std::string& _key /*= ""*/) const
{
	std::vector<std::string> keys;
	for (const auto& geometry : m_geometries)
		keys.push_back(geometry.key);
	return GenerateUniqueKey(_key, keys);
}