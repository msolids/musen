/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "MaterialsDatabaseTab.h"
#include "PackageGenerator.h"
#include "BondsGenerator.h"

class CMaterialsDatabaseLocalTab : public CMaterialsDatabaseTab
{
private:
	const CMaterialsDatabase* m_pMaterialsDBGlobal;
	const CPackageGenerator* m_pPackageGenerator;
	const CBondsGenerator* m_pBondsGenerator;
	const CGenerationManager* m_pDynamicGenerator;

	QSignalMapper* m_pCompoundsMapper;
	QSignalMapper* m_pMixturesMapper;

public:
	CMaterialsDatabaseLocalTab(const CMaterialsDatabase* _pMaterialsDBGlobal, CMaterialsDatabase* _pMaterialsDBLocal,
		const CPackageGenerator* _pPackageGenerator, const CBondsGenerator* _pBondsGenerator, const CGenerationManager* _pDynamicGenerator,
		QWidget *parent);

	// Overrides setting pointers to disable resetting pointer to materials database
	void SetPointers(CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor, CMaterialsDatabase* _pMaterialsDB, CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB) override;

public slots:
	void UpdateWholeView() override;

private slots:
	// Adds specified compound from global database into current one.
	void AddCompound(int _iCompound);
	// Adds specified mixture from global database into current one.
	void AddMixture(int _iMixture);
	// Adds all compounds used in system structure and generators, if they are available in global database.
	void AddAllCompounds();
	// Adds all mixtures used in system structure and generators, if they are available in global database.
	void AddAllMixtures();
	// Removes all compounds not currently used in system structure and generators.
	void RemoveUnusedCompounds();
	// Removes all mixtures not currently used in system structure and generators.
	void RemoveUnusedMixtures();

private:
	void InitializeConnections();

	void UpdateWindowTitle() override;
	// Updates compounds available for adding according to global database.
	void UpdateGlobalCompounds();
	// Updates mixtures available for adding according to global database.
	void UpdateGlobalMixtures();

	// Returns true if all compounds in mixture are added into current database.
	bool IsAllCompoundsAvailable(const CMixture& _mixture) const;

	// Adds specified compound and related interactions from global database into the current database if it has not been added yet and returns pointer to an added compound.
	CCompound* AddCompoundFromGlobalIfNotYetExist(const std::string& _sCompoundKey);
	// Adds specified compound and related interactions from global database into the current database and returns pointer to an added compound.
	CCompound* AddCompoundFromGlobal(int _nGlobalCompoundIndex);
	// Adds specified compound and related interactions from global database into the current database and returns pointer to an added compound.
	CCompound* AddCompoundFromGlobal(const std::string& _sGlobalCompoundKey);

	// Adds specified mixture from global database into the current database if it has not been added yet and returns pointer to an added mixture.
	CMixture* AddMixtureFromGlobalIfNotYetExist(const std::string& _sMixtureKey);
	// Adds specified mixture from global database into the current database and returns pointer to an added mixture.
	CMixture* AddMixtureFromGlobal(int _nGlobalMixtureIndex);
	// Adds specified mixture from global database into the current database and returns pointer to an added mixture.
	CMixture* AddMixtureFromGlobal(const std::string& _sGlobalMixtureKey);
};

