/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "ObjectsGenerator.h"
#include "MUSENComponent.h"

class CGenerationManager : public CMusenComponent
{
private:
	std::vector< CObjectsGenerator* > m_vGenerators;
	CAgglomeratesDatabase* m_pAgglomDB;

private:
	void DeleteAllGenerators();

public:
	CGenerationManager();
	~CGenerationManager();

	void SetAgglomeratesDatabase( CAgglomeratesDatabase* _pDatabase );
	size_t GetGeneratorsNumber() const;
	size_t GetActiveGeneratorsNumber() const;
	CObjectsGenerator* GetGenerator(size_t _nIndex);
	const CObjectsGenerator* GetGenerator(size_t _nIndex) const;
	// Returns const pointers to all defined generators
	std::vector<const CObjectsGenerator*> GetGenerators() const;

	void CreateNewGenerator();
	void DeleteGenerator( size_t _nIndex );

	void Initialize();

	// uses the sane file as system structure to store configuration
	void LoadConfiguration();
	void SaveConfiguration();

	// check that everything defined correctly
	std::string IsDataCorrect() const;

	// Adds objects to simplified scene and fills _newObjects for later adding them to system structure. Returns number of generated objects.
	size_t GenerateObjects(double _dTime, CSimplifiedScene& _pScene, std::vector<SGeneratedObject>& _newObjects);
	// Returns true if some particles must be generated at the specified time point in any generator.
	bool IsNeedToBeGenerated(double _dTime) const;
};