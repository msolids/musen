/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "MaterialsDatabaseLocalTab.h"
#include "qtOperations.h"
#include <QMenu>
#include <QMessageBox>

CMaterialsDatabaseLocalTab::CMaterialsDatabaseLocalTab(const CMaterialsDatabase* _pMaterialsDBGlobal, CMaterialsDatabase* _pMaterialsDBLocal,
	const CPackageGenerator* _pPackageGenerator, const CBondsGenerator* _pBondsGenerator, const CGenerationManager* _pDynamicGenerator, QWidget *parent)
	: CMaterialsDatabaseTab(_pMaterialsDBLocal, parent)
{
	m_pMaterialsDBGlobal = _pMaterialsDBGlobal;
	m_pPackageGenerator = _pPackageGenerator;
	m_pBondsGenerator = _pBondsGenerator;
	m_pDynamicGenerator = _pDynamicGenerator;

	// set visibility of controls
	ui.frameFileButtons->setEnabled(false);
	ui.frameFileButtons->setVisible(false);
	ui.buttonAddAllCompounds->setVisible(true);
	ui.buttonAddAllMixtures->setVisible(true);
	ui.buttonRemoveUnusedCompounds->setVisible(true);
	ui.buttonRemoveUnusedMixtures->setVisible(true);

	// disable duplicate buttons
	ui.buttonDuplicateCompound->setEnabled(false);
	ui.buttonDuplicateCompound->setVisible(false);
	ui.buttonDuplicateMixture->setEnabled(false);
	ui.buttonDuplicateMixture->setVisible(false);

	// add menu to "add compound" button
	m_pCompoundsMapper = new QSignalMapper(this);
	ui.buttonAddCompound->setMenu(new QMenu(this));
	UpdateGlobalCompounds();

	// add menu to "add mixture" button
	m_pMixturesMapper = new QSignalMapper(this);
	ui.buttonAddMixture->setMenu(new QMenu(this));
	UpdateGlobalMixtures();

	m_bGlobal = false;

	m_sHelpFileName = "Users Guide/Materials Database.pdf";

	InitializeConnections();
}

void CMaterialsDatabaseLocalTab::SetPointers(CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor, CMaterialsDatabase* _pMaterialsDB, CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB)
{
	m_pSystemStructure = _pSystemStructure;
	m_pUnitConverter = _pUnitConvertor;
	m_pGeometriesDB = _pGeometriesDB;
	m_pAgglomDB = _pAgglomDB;
}

void CMaterialsDatabaseLocalTab::InitializeConnections()
{
	connect(m_pCompoundsMapper, &QSignalMapper::mappedInt, this, &CMaterialsDatabaseLocalTab::AddCompound);
	connect(m_pMixturesMapper,	&QSignalMapper::mappedInt, this, &CMaterialsDatabaseLocalTab::AddMixture);

	connect(ui.buttonAddAllCompounds,	&QPushButton::clicked, this, &CMaterialsDatabaseLocalTab::AddAllCompounds);
	connect(ui.buttonAddAllMixtures,	&QPushButton::clicked, this, &CMaterialsDatabaseLocalTab::AddAllMixtures);

	connect(ui.buttonRemoveUnusedCompounds, &QPushButton::clicked, this, &CMaterialsDatabaseLocalTab::RemoveUnusedCompounds);
	connect(ui.buttonRemoveUnusedMixtures,	&QPushButton::clicked, this, &CMaterialsDatabaseLocalTab::RemoveUnusedMixtures);
}

void CMaterialsDatabaseLocalTab::UpdateWholeView()
{
	CMaterialsDatabaseTab::UpdateWholeView();

	UpdateGlobalCompounds();
	UpdateGlobalMixtures();
}

void CMaterialsDatabaseLocalTab::AddCompound(int _iCompound)
{
	CCompound* pCompound = nullptr;

	std::string sBaseKey = m_pMaterialsDBGlobal->GetCompoundKey(_iCompound);

	if (!m_pMaterialsDB->GetCompound(sBaseKey)) // required compound has not been added yet
		pCompound = AddCompoundFromGlobal(sBaseKey);
	else										// compound with such key has already been added
	{
		QMessageBox msgBox(QMessageBox::Question, tr("Add compound"),
			tr("A compound '%1' with such a key is already added. Do you want to add a new compound as a copy or to overwrite an existing compound with properties from the database?").arg(ss2qs(m_pMaterialsDB->GetCompoundName(sBaseKey))),
			QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel, this);
		msgBox.setButtonText(QMessageBox::Yes, tr("Copy"));
		msgBox.setButtonText(QMessageBox::No, tr("Overwrite"));
		int reply = msgBox.exec();

		if (reply == QMessageBox::Yes)		// add a copy
			pCompound = AddCompoundFromGlobal(_iCompound);
		else if (reply == QMessageBox::No)	// overwrite existing compound
		{
			std::string sOldName = m_pMaterialsDB->GetCompoundName(sBaseKey);
			int nOldIndex = m_pMaterialsDB->GetCompoundIndex(sBaseKey);
			m_pMaterialsDB->RemoveCompound(sBaseKey);
			pCompound = AddCompoundFromGlobal(sBaseKey);
			// move to the old position
			for (size_t i = 0; i < m_pMaterialsDB->CompoundsNumber() - nOldIndex - 1; ++i)
				m_pMaterialsDB->UpCompound(m_pMaterialsDB->CompoundsNumber() - 1 - i);
			pCompound->SetName(sOldName);
		}
	}

	UpdateCompoundsList();
	SelectCompound(pCompound);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseLocalTab::AddMixture(int _iMixture)
{
	CMixture* pMixture = nullptr;

	std::string sBaseKey = m_pMaterialsDBGlobal->GetMixtureKey(_iMixture);

	if (!m_pMaterialsDB->GetMixture(sBaseKey))	// required mixture has not been added yet
		pMixture = AddMixtureFromGlobal(sBaseKey);
	else										// mixture with such key has already been added
	{
		QMessageBox msgBox(QMessageBox::Question, tr("Add Mixture"),
			tr("A mixture '%1' with such a key is already added. Do you want to add a new mixture as a copy or to overwrite an existing mixture with properties from the database?").arg(ss2qs(m_pMaterialsDB->GetMixtureName(sBaseKey))),
			QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel, this);
		msgBox.setButtonText(QMessageBox::Yes, tr("Copy"));
		msgBox.setButtonText(QMessageBox::No, tr("Overwrite"));
		int reply = msgBox.exec();

		if (reply == QMessageBox::Yes)		// add a copy
			pMixture = AddMixtureFromGlobal(_iMixture);
		else if (reply == QMessageBox::No)	// overwrite existing mixture
		{
			std::string sOldName = m_pMaterialsDB->GetMixtureName(sBaseKey);
			int nOldIndex = m_pMaterialsDB->GetMixtureIndex(sBaseKey);
			m_pMaterialsDB->RemoveMixture(sBaseKey);
			pMixture = AddMixtureFromGlobal(sBaseKey);
			// move to the old position
			for (size_t i = 0; i < m_pMaterialsDB->MixturesNumber() - nOldIndex - 1; ++i)
				m_pMaterialsDB->UpMixture(m_pMaterialsDB->MixturesNumber() - 1 - i);
			pMixture->SetName(sOldName);
		}
	}

	UpdateCompoundsList();
	UpdateMixturesList();
	SelectMixture(pMixture);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseLocalTab::AddAllCompounds()
{
	const CCompound* pLastAddedCompound = nullptr;

	// from objects
	for (size_t i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); ++i)
	{
		CPhysicalObject* pObject = m_pSystemStructure->GetObjectByIndex(i);
		if(!pObject) continue;
		pLastAddedCompound = AddCompoundFromGlobalIfNotYetExist(pObject->GetCompoundKey());
	}
	// from bonds generator
	for (size_t i = 0; i < m_pBondsGenerator->GeneratorsNumber(); ++i)
		pLastAddedCompound = AddCompoundFromGlobalIfNotYetExist(m_pBondsGenerator->Generator(i)->compoundKey);
	// from dynamic generator
	for (size_t i = 0; i < m_pDynamicGenerator->GetGeneratorsNumber(); ++i)
	{
		for(const auto& it : m_pDynamicGenerator->GetGenerator(i)->m_partMaterials)
			pLastAddedCompound = AddCompoundFromGlobalIfNotYetExist(it.second);
		for (const auto& it : m_pDynamicGenerator->GetGenerator(i)->m_bondMaterials)
			pLastAddedCompound = AddCompoundFromGlobalIfNotYetExist(it.second);
	}

	UpdateCompoundsList();
	SelectCompound(pLastAddedCompound);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseLocalTab::AddAllMixtures()
{
	// from package generators
	for (size_t i = 0; i < m_pPackageGenerator->GeneratorsNumber(); ++i)
		AddMixtureFromGlobalIfNotYetExist(m_pPackageGenerator->Generator(i)->mixtureKey);
	// from dynamic generator
	for (size_t i = 0; i < m_pDynamicGenerator->GetGeneratorsNumber(); ++i)
		AddMixtureFromGlobalIfNotYetExist(m_pDynamicGenerator->GetGenerator(i)->m_sMixtureKey);
}

void CMaterialsDatabaseLocalTab::RemoveUnusedCompounds()
{
	if (QMessageBox::question(this, tr("Remove compounds"), tr("All compounds that are not currently used in the scene and generators will be removed. Do you really want to continue?"),
		QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes) return;

	std::set<std::string> usedCompounds;
	// from objects
	for (size_t i = 0; i < m_pSystemStructure->GetTotalObjectsCount(); ++i)
	{
		CPhysicalObject* pObject = m_pSystemStructure->GetObjectByIndex(i);
		if (!pObject) continue;
		usedCompounds.insert(pObject->GetCompoundKey());
	}
	// from bonds generator
	for (size_t i = 0; i < m_pBondsGenerator->GeneratorsNumber(); ++i)
		usedCompounds.insert(m_pBondsGenerator->Generator(i)->compoundKey);
	// from dynamic generator
	for (size_t i = 0; i < m_pDynamicGenerator->GetGeneratorsNumber(); ++i)
	{
		for (const auto& it : m_pDynamicGenerator->GetGenerator(i)->m_partMaterials)
			usedCompounds.insert(it.second);
		for (const auto& it : m_pDynamicGenerator->GetGenerator(i)->m_bondMaterials)
			usedCompounds.insert(it.second);
	}

	for (size_t i = 0; i < m_pMaterialsDB->CompoundsNumber();)
		if (usedCompounds.find(m_pMaterialsDB->GetCompoundKey(i)) == usedCompounds.end())
			m_pMaterialsDB->RemoveCompound(i);
		else
			i++;

	UpdateCompoundsList();
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseLocalTab::RemoveUnusedMixtures()
{
	if (QMessageBox::question(this, tr("Remove mixtures"), tr("All mixtures that are not currently used in the scene and generators will be removed. Do you really want to continue?"),
		QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes) return;

	std::set<std::string> usedMixtures;
	// from package generators
	for (size_t i = 0; i < m_pPackageGenerator->GeneratorsNumber(); ++i)
		usedMixtures.insert(m_pPackageGenerator->Generator(i)->mixtureKey);
	// from dynamic generator
	for (size_t i = 0; i < m_pDynamicGenerator->GetGeneratorsNumber(); ++i)
		usedMixtures.insert(m_pDynamicGenerator->GetGenerator(i)->m_sMixtureKey);

	for (size_t i = 0; i < m_pMaterialsDB->MixturesNumber();)
		if (usedMixtures.find(m_pMaterialsDB->GetMixtureKey(i)) == usedMixtures.end())
			m_pMaterialsDB->RemoveMixture(i);
		else
			i++;

	UpdateMixturesList();
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseLocalTab::UpdateWindowTitle()
{
	setWindowTitle("Current Materials Editor");
}

void CMaterialsDatabaseLocalTab::UpdateGlobalCompounds()
{
	QMenu* pMenu = ui.buttonAddCompound->menu();

	// remove previous actions
	pMenu->clear();

	// create menu
	for (int i = 0; i < static_cast<int>(m_pMaterialsDBGlobal->CompoundsNumber()); ++i)
	{
		QAction* pAction = new QAction(ss2qs(m_pMaterialsDBGlobal->GetCompoundName(i)), this);
		pMenu->addAction(pAction);
		connect(pAction, &QAction::triggered, m_pCompoundsMapper, static_cast<void (QSignalMapper::*)()>(&QSignalMapper::map));
		m_pCompoundsMapper->setMapping(pAction, i);
	}
}

void CMaterialsDatabaseLocalTab::UpdateGlobalMixtures()
{
	QMenu* pMenu = ui.buttonAddMixture->menu();

	// remove previous actions
	pMenu->clear();

	// create menu
	for (int i = 0; i < static_cast<int>(m_pMaterialsDBGlobal->MixturesNumber()); ++i)
	{
		QAction* pAction = new QAction(ss2qs(m_pMaterialsDBGlobal->GetMixtureName(i)), this);
		pMenu->addAction(pAction);
		connect(pAction, &QAction::triggered, m_pMixturesMapper, static_cast<void (QSignalMapper::*)()>(&QSignalMapper::map));
		m_pMixturesMapper->setMapping(pAction, i);
	}
}

bool CMaterialsDatabaseLocalTab::IsAllCompoundsAvailable(const CMixture& _mixture) const
{
	for (size_t i = 0; i < _mixture.FractionsNumber(); ++i)
		if (!m_pMaterialsDB->GetCompound(_mixture.GetFractionCompound(i)))
			return false;
	return true;
}

CCompound* CMaterialsDatabaseLocalTab::AddCompoundFromGlobalIfNotYetExist(const std::string& _sCompoundKey)
{
	if (!m_pMaterialsDB->GetCompound(_sCompoundKey))	// compound is not yet in database
		return AddCompoundFromGlobal(_sCompoundKey);
	else												// compound is already in database
		return nullptr;
}

CCompound* CMaterialsDatabaseLocalTab::AddCompoundFromGlobal(int _nGlobalCompoundIndex)
{
	if (_nGlobalCompoundIndex < 0) return nullptr;
	return AddCompoundFromGlobal(m_pMaterialsDBGlobal->GetCompoundKey(_nGlobalCompoundIndex));
}

CCompound* CMaterialsDatabaseLocalTab::AddCompoundFromGlobal(const std::string& _sGlobalCompoundKey)
{
	if (_sGlobalCompoundKey.empty()) return nullptr;

	const CCompound* pBaseCompound = m_pMaterialsDBGlobal->GetCompound(_sGlobalCompoundKey);
	if (!pBaseCompound) return nullptr;	// nothing to add

	CCompound* pCompound = m_pMaterialsDB->AddCompound(*pBaseCompound);

	// copy interactions. unique key of pCompound may not be the same as of pBaseCompound, if base compound added more then once. resolve it by treating sKey and sBaseKey separately
	std::string sKey1 = pCompound->GetKey();
	std::string sBaseKey1 = pBaseCompound->GetKey();
	for (size_t i = 0; i < m_pMaterialsDB->CompoundsNumber(); ++i)
	{
		std::string sKey2 = m_pMaterialsDB->GetCompoundKey(i);
		std::string sBaseKey2 = (sKey2 == sKey1 ? sBaseKey1 : sKey2);	// if interaction with itself, use both keys from base class.
		CInteraction* pInteraction = m_pMaterialsDB->GetInteraction(sKey1, sKey2);
		const CInteraction* pBaseInteraction = m_pMaterialsDBGlobal->GetInteraction(sBaseKey1, sBaseKey2);
		if (pInteraction && pBaseInteraction)
		{
			*pInteraction = *pBaseInteraction;
			pInteraction->SetKeys(sKey1, sKey2);
		}
	}

	return pCompound;
}

CMixture* CMaterialsDatabaseLocalTab::AddMixtureFromGlobalIfNotYetExist(const std::string& _sMixtureKey)
{
	if (!m_pMaterialsDB->GetMixture(_sMixtureKey))	// mixture is not yet in database
		return AddMixtureFromGlobal(_sMixtureKey);
	else											// mixture is already in database
		return nullptr;
}

CMixture* CMaterialsDatabaseLocalTab::AddMixtureFromGlobal(int _nGlobalMixtureIndex)
{
	if (_nGlobalMixtureIndex < 0) return nullptr;
	return AddMixtureFromGlobal(m_pMaterialsDBGlobal->GetMixtureKey(_nGlobalMixtureIndex));
}

CMixture* CMaterialsDatabaseLocalTab::AddMixtureFromGlobal(const std::string& _sGlobalMixtureKey)
{
	if (_sGlobalMixtureKey.empty()) return nullptr;

	const CMixture* pBaseMixture = m_pMaterialsDBGlobal->GetMixture(_sGlobalMixtureKey);
	if (!pBaseMixture) return nullptr;

	CMixture* pMixture = m_pMaterialsDB->AddMixture(*pBaseMixture);

	// add all currently unavailable compounds from mixture into database
	if (!IsAllCompoundsAvailable(*pMixture))
	{
		if (QMessageBox::question(this, tr("Add compounds"), tr("Some compounds from the added mixture '%1' are not available in the scene. Add them from current database?").arg(ss2qs(pMixture->GetName())),
			QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) == QMessageBox::Yes)
			for (size_t i = 0; i < pMixture->FractionsNumber(); ++i)
				AddCompoundFromGlobalIfNotYetExist(pMixture->GetFractionCompound(i));
	}

	return pMixture;
}
