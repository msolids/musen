/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "GeometriesDatabase.h"

CGeometriesDatabase::~CGeometriesDatabase()
{
	DeleteGeometries();
}

void CGeometriesDatabase::DeleteGeometries()
{
	for (size_t i = 0; i < m_vGeometries.size(); ++i)
		delete m_vGeometries[i];
	m_vGeometries.clear();
}

void CGeometriesDatabase::AddGeometry(const std::string& _sSTLFileName)
{
	CTriangularMesh* pGeometry = new CTriangularMesh;
	CSTLFileHandler loader;
	*pGeometry = loader.ReadFromFile(_sSTLFileName);
	if (!pGeometry->vTriangles.empty())
		m_vGeometries.push_back(pGeometry);
}

void CGeometriesDatabase::ExportGeometry(size_t _index, const std::string& _sSTLFileName) const
{
	if (_index >= m_vGeometries.size())
		return;
	CSTLFileHandler saver;
	saver.WriteToFile(*m_vGeometries[_index], _sSTLFileName);
}

CTriangularMesh* CGeometriesDatabase::GetGeometry(size_t _nIndex)
{
	if (_nIndex < m_vGeometries.size())
		return m_vGeometries[_nIndex];
	else
		return nullptr;
}

const CTriangularMesh* CGeometriesDatabase::GetGeometry(size_t _nIndex) const
{
	if (_nIndex < m_vGeometries.size())
		return m_vGeometries[_nIndex];
	else
		return nullptr;
}

void CGeometriesDatabase::SaveToFile(const std::string& _sFileName)
{
	ProtoGeometriesDB protoGeometriesDB;

	for (size_t i = 0; i < m_vGeometries.size(); ++i)
	{
		ProtoGeometry* protoGeom = protoGeometriesDB.add_geometry();
		protoGeom->set_name(m_vGeometries[i]->sName);
		for (size_t j = 0; j < m_vGeometries[i]->vTriangles.size(); ++j)
		{
			ProtoGeomVect* vert1 = protoGeom->add_vertices();
			vert1->set_x(m_vGeometries[i]->vTriangles[j].p1.x);
			vert1->set_y(m_vGeometries[i]->vTriangles[j].p1.y);
			vert1->set_z(m_vGeometries[i]->vTriangles[j].p1.z);
			ProtoGeomVect* vert2 = protoGeom->add_vertices();
			vert2->set_x(m_vGeometries[i]->vTriangles[j].p2.x);
			vert2->set_y(m_vGeometries[i]->vTriangles[j].p2.y);
			vert2->set_z(m_vGeometries[i]->vTriangles[j].p2.z);
			ProtoGeomVect* vert3 = protoGeom->add_vertices();
			vert3->set_x(m_vGeometries[i]->vTriangles[j].p3.x);
			vert3->set_y(m_vGeometries[i]->vTriangles[j].p3.y);
			vert3->set_z(m_vGeometries[i]->vTriangles[j].p3.z);
		}
	}

	std::fstream outFile(UnicodePath(_sFileName), std::ios::out | std::ios::trunc | std::ios::binary);
	std::string data;
	// TODO: consider to use SerializeToZeroCopyStream() for performance
	protoGeometriesDB.SerializeToString(&data);
	outFile << data;
	outFile.close();
	m_sDatabaseFileName = _sFileName;
}

void CGeometriesDatabase::LoadFromFile(const std::string& _sFileName)
{
	std::fstream inputFile(UnicodePath(_sFileName), std::ios::in | std::ios::binary);
	if (!inputFile)
		return;

	ProtoGeometriesDB protoGeometriesDB;
	// TODO: consider to use ParseFromZeroCopyStream() for performance
	if (!protoGeometriesDB.ParseFromString(std::string(std::istreambuf_iterator<char>(inputFile), std::istreambuf_iterator<char>())))
		return;

	DeleteGeometries();
	for (int i = 0; i < protoGeometriesDB.geometry_size(); ++i)
	{
		CTriangularMesh* pNewGeometry = new CTriangularMesh;
		ProtoGeometry* protoGeom = protoGeometriesDB.mutable_geometry(i);
		pNewGeometry->sName = protoGeom->name();
		for (int j = 0; j < protoGeom->vertices_size() / 3; ++j)
			pNewGeometry->vTriangles.push_back(STriangleType(
				CVector3(protoGeom->mutable_vertices(j * 3 + 0)->x(), protoGeom->mutable_vertices(j * 3 + 0)->y(), protoGeom->mutable_vertices(j * 3 + 0)->z()),
				CVector3(protoGeom->mutable_vertices(j * 3 + 1)->x(), protoGeom->mutable_vertices(j * 3 + 1)->y(), protoGeom->mutable_vertices(j * 3 + 1)->z()),
				CVector3(protoGeom->mutable_vertices(j * 3 + 2)->x(), protoGeom->mutable_vertices(j * 3 + 2)->y(), protoGeom->mutable_vertices(j * 3 + 2)->z())));
		m_vGeometries.push_back(pNewGeometry);
	}
	m_sDatabaseFileName = _sFileName;
}

void CGeometriesDatabase::NewDatabase()
{
	DeleteGeometries();
	m_sDatabaseFileName.clear();
}

void CGeometriesDatabase::DeleteGeometry(size_t _nIndex)
{
	if (_nIndex < m_vGeometries.size())
		m_vGeometries.erase(m_vGeometries.begin() + _nIndex);
}

void CGeometriesDatabase::UpGeometry(size_t _nIndex)
{
	if (_nIndex < m_vGeometries.size() && _nIndex != 0)
		std::iter_swap(m_vGeometries.begin() + _nIndex, m_vGeometries.begin() + _nIndex - 1);
}

void CGeometriesDatabase::DownGeometry(size_t _nIndex)
{
	if ((_nIndex < m_vGeometries.size()) && (_nIndex != (m_vGeometries.size() - 1)))
		std::iter_swap(m_vGeometries.begin() + _nIndex, m_vGeometries.begin() + _nIndex + 1);
}

std::string CGeometriesDatabase::GetFileName() const
{
	return m_sDatabaseFileName;
}

size_t CGeometriesDatabase::GetGeometriesNumber() const
{
	return m_vGeometries.size();
}
