/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "MaterialsDatabaseTab.h"
#include "qtOperations.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QLockFile>
#include <QComboBox>

CMaterialsDatabaseTab::CMaterialsDatabaseTab(CMaterialsDatabase* _pMaterialsDB, QWidget *parent) : CMusenDialog(parent)
{
	ui.setupUi(this);
	ui.buttonAddAllCompounds->setVisible(false);
	ui.buttonAddAllMixtures->setVisible(false);
	ui.buttonRemoveUnusedCompounds->setVisible(false);
	ui.buttonRemoveUnusedMixtures->setVisible(false);

	m_pMaterialsDB = _pMaterialsDB;

	m_pSignalMapperFracs = new QSignalMapper(this);

	ui.tableProperties->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
	ui.tableProperties->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
	ui.tableProperties->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);

	ui.tableInterProperties->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
	ui.tableInterProperties->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
	ui.tableInterProperties->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);

	ui.tableFractions->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

	m_bAvoidSignal = false;
	m_bAvoidUpdate = false;
	m_bGlobal = true;

	m_vMusenActiveProperies = {	PROPERTY_DENSITY, PROPERTY_DYNAMIC_VISCOSITY, PROPERTY_YOUNG_MODULUS, PROPERTY_NORMAL_STRENGTH, PROPERTY_TANGENTIAL_STRENGTH, PROPERTY_POISSON_RATIO,
		PROPERTY_SURFACE_ENERGY, PROPERTY_ATOMIC_VOLUME, PROPERTY_SURFACE_TENSION, PROPERTY_TIME_THERM_EXP_COEFF, PROPERTY_YIELD_STRENGTH };
	m_vMusenActiveInteractions = { PROPERTY_RESTITUTION_COEFFICIENT, PROPERTY_STATIC_FRICTION, PROPERTY_ROLLING_FRICTION };

	m_sHelpFileName = "Users Guide/Materials Database.pdf";

	UpdateWholeView();
	InitializeConnections();
}

void CMaterialsDatabaseTab::InitializeConnections()
{
	//////////////////////////////////////////////////////////////////////////
	/// signals from files

	// file actions
	connect(ui.actionNewDatabase,		&QAction::triggered, this, &CMaterialsDatabaseTab::NewDatabase);
	connect(ui.actionLoadDatabase,		&QAction::triggered, this, &CMaterialsDatabaseTab::LoadDatabase);
	connect(ui.actionSaveDatabase,		&QAction::triggered, this, &CMaterialsDatabaseTab::SaveDatabase);
	connect(ui.actionSaveDatabaseAs,	&QAction::triggered, this, &CMaterialsDatabaseTab::SaveDatabaseAs);

	//////////////////////////////////////////////////////////////////////////
	/// signals from compounds

	// signals from buttons
	connect(ui.buttonAddCompound,		&QPushButton::clicked, this, &CMaterialsDatabaseTab::AddCompound);
	connect(ui.buttonDuplicateCompound, &QPushButton::clicked, this, &CMaterialsDatabaseTab::DuplicateCompound);
	connect(ui.buttonRemoveCompound,	&QPushButton::clicked, this, &CMaterialsDatabaseTab::RemoveCompound);
	connect(ui.upCompound,				&QPushButton::clicked, this, &CMaterialsDatabaseTab::UpCompound);
	connect(ui.downCompound,			&QPushButton::clicked, this, &CMaterialsDatabaseTab::DownCompound);

	// signals from compounds
	connect(ui.listCompounds,		&QListWidget::itemSelectionChanged,	this, &CMaterialsDatabaseTab::CompoundSelected);
	connect(ui.listCompounds,		&QListWidget::itemChanged,			this, &CMaterialsDatabaseTab::CompoundNameChanged);
	connect(ui.widgetColorView,		&CColorView::ColorChanged,			this, &CMaterialsDatabaseTab::CompoundColorChanged);
	connect(ui.lineCompoundAuthor,	&QLineEdit::editingFinished,		this, &CMaterialsDatabaseTab::CompoundAuthorChanged);
	connect(ui.tableProperties,		&CQtTable::currentCellChanged,		this, &CMaterialsDatabaseTab::CompoundPropertySelectionChanged);
	connect(ui.tableProperties,		&CQtTable::cellChanged,				this, &CMaterialsDatabaseTab::PropertyValueChanged);

	//////////////////////////////////////////////////////////////////////////
	/// signals from interactions

	connect(ui.listCompounds1,			&QListWidget::itemSelectionChanged, this, &CMaterialsDatabaseTab::Compound1Selected);
	connect(ui.listCompounds2,			&QListWidget::itemSelectionChanged, this, &CMaterialsDatabaseTab::Compound2Selected);
	connect(ui.tableInterProperties,	&CQtTable::currentCellChanged,		this, &CMaterialsDatabaseTab::InteractionPropertySelectionChanged);
	connect(ui.tableInterProperties,	&CQtTable::cellChanged,				this, &CMaterialsDatabaseTab::InteractionValueChanged);

	//////////////////////////////////////////////////////////////////////////
	/// signals from mixtures

	// signals from mixture buttons
	connect(ui.buttonAddMixture,		&QPushButton::clicked, this, &CMaterialsDatabaseTab::AddMixture);
	connect(ui.buttonDuplicateMixture,	&QPushButton::clicked, this, &CMaterialsDatabaseTab::DuplicateMixture);
	connect(ui.buttonRemoveMixture,		&QPushButton::clicked, this, &CMaterialsDatabaseTab::RemoveMixture);
	connect(ui.buttonUpMixture,			&QPushButton::clicked, this, &CMaterialsDatabaseTab::UpMixture);
	connect(ui.buttonDownMixture,		&QPushButton::clicked, this, &CMaterialsDatabaseTab::DownMixture);

	// signals from mixtures
	connect(ui.listMixtures, &QListWidget::itemSelectionChanged,	this, &CMaterialsDatabaseTab::MixtureSelected);
	connect(ui.listMixtures, &QListWidget::itemChanged,				this, &CMaterialsDatabaseTab::MixtureNameChanged);

	// signals from fractions buttons
	connect(ui.buttonAddFraction,		&QPushButton::clicked, this, &CMaterialsDatabaseTab::AddFraction);
	connect(ui.buttonDuplicateFraction, &QPushButton::clicked, this, &CMaterialsDatabaseTab::DuplicateFraction);
	connect(ui.buttonRemoveFraction,	&QPushButton::clicked, this, &CMaterialsDatabaseTab::RemoveFraction);
	connect(ui.buttonUpFraction,		&QPushButton::clicked, this, &CMaterialsDatabaseTab::UpFraction);
	connect(ui.buttonDownFraction,		&QPushButton::clicked, this, &CMaterialsDatabaseTab::DownFraction);
	connect(ui.buttonNormalize,			&QPushButton::clicked, this, &CMaterialsDatabaseTab::NormalizeFractions);

	// signals from fractions
	connect(ui.tableFractions,		&CQtTable::cellChanged,												this, &CMaterialsDatabaseTab::FractionChanged);
	connect(m_pSignalMapperFracs,	static_cast<void (QSignalMapper::*)(int)>(&QSignalMapper::mapped),	this, &CMaterialsDatabaseTab::FractionCompoundChanged);
}

void CMaterialsDatabaseTab::SetPointers(CSystemStructure* _pSystemStructure, CUnitConvertor* _pUnitConvertor, CMaterialsDatabase* _pMaterialsDB, CGeometriesDatabase* _pGeometriesDB, CAgglomeratesDatabase* _pAgglomDB)
{
	m_pSystemStructure = _pSystemStructure;
	m_pUnitConverter = _pUnitConvertor;
	m_pGeometriesDB = _pGeometriesDB;
	m_pAgglomDB = _pAgglomDB;
}

void CMaterialsDatabaseTab::UpdateWholeView()
{
	if (!m_pUnitConverter) return; // do not update untill all pointers will be set via SetPointers()
	if (m_bAvoidUpdate) return;
	UpdateWindowTitle();
	UpdateCompoundsList();
	UpdateCompoundInfo();
	UpdateCompoundProperties();
	UpdateInteractions();
	UpdateMixturesList();
	UpdateButtons();
}

void CMaterialsDatabaseTab::UpdateWindowTitle()
{
	setWindowTitle("Global Materials Database: " + ss2qs(m_pMaterialsDB->GetFileName()) + "[*]");
}

void CMaterialsDatabaseTab::NewDatabase()
{
	m_pMaterialsDB->CreateNewDatabase();

	UpdateWindowTitle();
	UpdateWholeView();

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
	emit MaterialDatabaseFileWasChanged();
}

void CMaterialsDatabaseTab::LoadDatabase()
{
	QString sFileName = QFileDialog::getOpenFileName(this, tr("Load database"), FilePathToOpen(), tr("Material database (*.mdb);;All files (*.*);;"));
	if (sFileName.simplified().isEmpty()) return;
	if (!QFile::exists(sFileName)) return;
	m_pMaterialsDB->LoadFromFile(qs2ss(sFileName));

	UpdateWholeView();

	setWindowModified(false);
	emit MaterialDatabaseWasChanged();
	emit MaterialDatabaseFileWasChanged();
}

void CMaterialsDatabaseTab::SaveDatabase()
{
	if (!ss2qs(m_pMaterialsDB->GetFileName()).simplified().isEmpty())
	{
		m_pMaterialsDB->SaveToFile(m_pMaterialsDB->GetFileName());
		setWindowModified(false);
	}
	else
		SaveDatabaseAs();
}

void CMaterialsDatabaseTab::SaveDatabaseAs()
{
	QString sFileName = QFileDialog::getSaveFileName(this, tr("Save database"), FilePathToOpen(), tr("Material database (*.mdb);;All files (*.*);;"));
	if (sFileName.simplified().isEmpty()) return;
	m_pMaterialsDB->SaveToFile(qs2ss(sFileName));

	UpdateWindowTitle();
	UpdateButtons();

	setWindowModified(false);
	emit MaterialDatabaseFileWasChanged();
}

void CMaterialsDatabaseTab::UpdateButtons()
{
	QString sLockerFileName = ss2qs(m_pMaterialsDB->GetFileName()) + ".lock";
	QLockFile *newFileLocker = new QLockFile(sLockerFileName);
	newFileLocker->setStaleLockTime(0);
	bool bSuccessfullyLocked = newFileLocker->tryLock(10);
	newFileLocker->unlock();
	ui.buttonSave->setEnabled(bSuccessfullyLocked);
}

void CMaterialsDatabaseTab::AddCompound()
{
	// pick name for new compound
	std::string sName = PickUniqueName(m_pMaterialsDB->GetCompoundsNames(), "Compound");
	// add new compound
	CCompound* pCompound = m_pMaterialsDB->AddCompound();
	// set new name
	pCompound->SetName(sName);

	// update everything
	UpdateCompoundsList();
	SelectCompound(pCompound);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::DuplicateCompound()
{
	const CCompound* pBaseCompound = m_pMaterialsDB->GetCompound(GetElementKey(ui.listCompounds));
	if (!pBaseCompound) return;
	CCompound* pCompound = m_pMaterialsDB->AddCompound(*pBaseCompound);

	// copy interactions.
	std::string sKey1 = pCompound->GetKey();
	std::string sBaseKey1 = pBaseCompound->GetKey();
	for (size_t i = 0; i < m_pMaterialsDB->CompoundsNumber(); ++i)
	{
		std::string sKey2 = m_pMaterialsDB->GetCompoundKey(i);
		std::string sBaseKey2 = sKey2;
		if ((sKey2 == sKey1) || (sKey2 == sBaseKey1))	// interaction with itself
			sBaseKey2 = sBaseKey1;
		CInteraction* pInteraction = m_pMaterialsDB->GetInteraction(sKey1, sKey2);
		const CInteraction* pBaseInteraction = m_pMaterialsDB->GetInteraction(sBaseKey1, sBaseKey2);
		if (pInteraction && pBaseInteraction)
		{
			*pInteraction = *pBaseInteraction;
			pInteraction->SetKeys(sKey1, sKey2);
		}
	}

	pCompound->SetName(pCompound->GetName() + "_copy");

	UpdateCompoundsList();
	SelectCompound(pCompound);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::RemoveCompound()
{
	std::string sKey = GetElementKey(ui.listCompounds);
	if (sKey.empty()) return;
	if (QMessageBox::question(this, tr("Remove compound"), tr("Do you really want to remove %1?").arg(ss2qs(m_pMaterialsDB->GetCompoundName(sKey))),
		QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) == QMessageBox::Yes)
	{
		m_pMaterialsDB->RemoveCompound(sKey);

		UpdateCompoundsList();

		setWindowModified(true);
		emit MaterialDatabaseWasChanged();
	}
}

void CMaterialsDatabaseTab::UpCompound()
{
	int iRow = ui.listCompounds->currentRow();
	if (iRow < 0) return;
	m_pMaterialsDB->UpCompound(iRow);
	UpdateCompoundsList();
	ui.listCompounds->setCurrentRow(iRow == 0 ? 0 : --iRow);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::DownCompound()
{
	int iRow = ui.listCompounds->currentRow();
	if (iRow < 0) return;
	m_pMaterialsDB->DownCompound(iRow);
	int lastRow = ui.listCompounds->count() - 1;
	UpdateCompoundsList();
	ui.listCompounds->setCurrentRow(iRow == lastRow ? lastRow : ++iRow);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::CompoundPropertySelectionChanged(int _iRow, int _iCol, int _iPrevRow, int _iPrevCol)
{
	if ((_iRow < 0) || (_iRow >= ui.tableProperties->rowCount())) return;
	ui.tableProperties->setCurrentCell(_iRow, 2);
}

void CMaterialsDatabaseTab::CompoundSelected()
{
	if (m_bAvoidSignal) return;
	UpdateCompoundInfo();
	UpdateCompoundProperties();
}

void CMaterialsDatabaseTab::CompoundNameChanged(QListWidgetItem* _pItem)
{
	if (m_bAvoidSignal) return;
	CCompound* pCompound = m_pMaterialsDB->GetCompound(GetElementKey(ui.listCompounds));
	if (!pCompound) return;
	QString sNewName = _pItem->text().simplified();
	if (sNewName.isEmpty()) return;
	pCompound->SetName(qs2ss(sNewName));

	UpdateCompoundsList();

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::CompoundColorChanged()
{
	if (m_bAvoidSignal) return;
	CCompound* pCompound = m_pMaterialsDB->GetCompound(GetElementKey(ui.listCompounds));
	if (!pCompound) return;
	qreal r, g, b, f;
	ui.widgetColorView->getColor().getRgbF(&r, &g, &b, &f);
	pCompound->SetColor(r, g, b, f);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::CompoundAuthorChanged()
{
	if (m_bAvoidSignal) return;
	CCompound* pCompound = m_pMaterialsDB->GetCompound(GetElementKey(ui.listCompounds));
	if (!pCompound) return;
	pCompound->SetAuthorName(qs2ss(ui.lineCompoundAuthor->text().simplified()));

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::PropertyValueChanged(int _nRow, int _nCol)
{
	if (m_bAvoidSignal) return;
	if (_nCol != 2) return;
	if (_nRow < 0) return;

	CCompound* pCompound = m_pMaterialsDB->GetCompound(GetElementKey(ui.listCompounds));
	if (!pCompound) return;

	int nType = GetPropertyType(ui.tableProperties, _nRow);
	if (nType == -1) return;
	double dValue = ui.tableProperties->item(_nRow, _nCol)->text().toDouble();
	pCompound->SetPropertyValue(static_cast<unsigned>(nType), dValue);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::UpdateCompoundsList()
{
	m_bAvoidSignal = true;

	int iOldRow = ui.listCompounds->currentRow();
	ui.listCompounds->clear();
	for (int i = 0; i < static_cast<int>(m_pMaterialsDB->CompoundsNumber()); ++i)
	{
		ui.listCompounds->insertItem(i, ss2qs(m_pMaterialsDB->GetCompound(i)->GetName()));
		ui.listCompounds->item(i)->setData(Qt::UserRole, ss2qs(m_pMaterialsDB->GetCompound(i)->GetKey()));
		ui.listCompounds->item(i)->setFlags(ui.listCompounds->item(i)->flags() | Qt::ItemIsEditable);
	}

	if ((iOldRow >= 0) && (iOldRow < static_cast<int>(m_pMaterialsDB->CompoundsNumber())))
		ui.listCompounds->setCurrentRow(iOldRow);
	else if (iOldRow != -1)
		ui.listCompounds->setCurrentRow(ui.listCompounds->count() - 1);
	else
		ui.listCompounds->setCurrentRow(-1); // empty list

	m_bAvoidSignal = false;

	CompoundSelected();
	UpdateInteractionsCompoundsLists();
	UpdateFractionsList();
}

void CMaterialsDatabaseTab::UpdateCompoundInfo()
{
	const CCompound* pCompound = m_pMaterialsDB->GetCompound(GetElementKey(ui.listCompounds));
	if (!pCompound)
	{
		ui.groupBoxInfo->setEnabled(false);
		ui.lineCompoundKey->clear();
		ui.lineCompoundAuthor->clear();
		ui.labelCompoundCreationDate->clear();
		ui.lineCompoundAuthor->setEnabled(false);
		ui.widgetColorView->setEnabled(false);
	}
	else
	{
		ui.groupBoxInfo->setEnabled(true);
		ui.lineCompoundAuthor->setEnabled(true);
		ui.widgetColorView->setEnabled(true);

		ui.lineCompoundKey->setText(ss2qs(pCompound->GetKey()));
		ui.lineCompoundAuthor->setText(ss2qs(pCompound->GetAuthorName()));
		QString sDate;
		sDate.sprintf("%02d.%02d.%04d", pCompound->GetCreationDate().nDay, pCompound->GetCreationDate().nMonth, pCompound->GetCreationDate().nYear);
		ui.labelCompoundCreationDate->setText(sDate);
		ui.widgetColorView->setColor(pCompound->GetColor());
	}
}

void CMaterialsDatabaseTab::UpdateCompoundProperties()
{
	if (m_bAvoidSignal) return;
	const CCompound* pCompound = m_pMaterialsDB->GetCompound(GetElementKey(ui.listCompounds));
	if (!pCompound)
	{
		ui.groupBoxProperties->setEnabled(false);
		ui.tableProperties->clearContents();
		return;
	}
	ui.groupBoxProperties->setEnabled(true);

	m_bAvoidSignal = true;

	ui.tableProperties->setRowCount(static_cast<int>(m_vMusenActiveProperies.size()));
	for (int i = 0; i < static_cast<int>(m_vMusenActiveProperies.size()); ++i)
	{
		const CTPProperty* pProp = pCompound->GetProperty(m_vMusenActiveProperies[i]);
		ui.tableProperties->SetItemNotEditable(i, 0, ss2qs(pProp->GetName()));
		ui.tableProperties->SetItemNotEditable(i, 1, ss2qs(pProp->GetUnits()));
		ui.tableProperties->SetItemEditable(i, 2, QString::number(pProp->GetValue()), pProp->GetType());
	}

	m_bAvoidSignal = false;
}

void CMaterialsDatabaseTab::InteractionPropertySelectionChanged(int _iRow, int _iCol, int _iPrevRow, int _iPrevCol)
{
	if ((_iRow < 0) || (_iRow >= ui.tableInterProperties->rowCount())) return;
	ui.tableInterProperties->setCurrentCell(_iRow, 2);
}

void CMaterialsDatabaseTab::Compound1Selected()
{
	const CCompound* pCompound = m_pMaterialsDB->GetCompound(GetElementKey(ui.listCompounds1));
	if (!pCompound)
		ui.labelCompound1->clear();
	else
		ui.labelCompound1->setText(ss2qs(pCompound->GetName()));
	UpdateInteractions();
}

void CMaterialsDatabaseTab::Compound2Selected()
{
	const CCompound* pCompound = m_pMaterialsDB->GetCompound(GetElementKey(ui.listCompounds2));
	if (!pCompound)
		ui.labelCompound2->clear();
	else
		ui.labelCompound2->setText(ss2qs(pCompound->GetName()));
	UpdateInteractions();
}

void CMaterialsDatabaseTab::InteractionValueChanged(int _nRow, int _nCol)
{
	if (m_bAvoidSignal) return;
	if (_nCol != 2) return;
	if (_nRow < 0) return;

	const CCompound* pCompound1 = m_pMaterialsDB->GetCompound(GetElementKey(ui.listCompounds1));
	const CCompound* pCompound2 = m_pMaterialsDB->GetCompound(GetElementKey(ui.listCompounds2));
	if (!pCompound1 || !pCompound2) return;
	CInteraction *pInteraction = m_pMaterialsDB->GetInteraction(pCompound1->GetKey(), pCompound2->GetKey());
	if (!pInteraction) return;

	int nType = GetPropertyType(ui.tableInterProperties, _nRow);
	if (nType == -1) return;
	double dValue = ui.tableInterProperties->item(_nRow, _nCol)->text().toDouble();
	pInteraction->SetPropertyValue(static_cast<unsigned>(nType), dValue);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::UpdateInteractionsCompoundsLists()
{
	m_bAvoidSignal = true;
	int iOldRow1 = ui.listCompounds1->currentRow();
	int iOldRow2 = ui.listCompounds2->currentRow();
	ui.listCompounds1->clear();
	ui.listCompounds2->clear();
	for (unsigned i = 0; i < m_pMaterialsDB->CompoundsNumber(); ++i)
	{
		const CCompound* pComp = m_pMaterialsDB->GetCompound(i);
		QListWidgetItem *pItem1 = new QListWidgetItem(ss2qs(pComp->GetName()));
		pItem1->setData(Qt::UserRole, ss2qs(pComp->GetKey()));
		ui.listCompounds1->insertItem(i, pItem1);

		QListWidgetItem *pItem2 = new QListWidgetItem(ss2qs(pComp->GetName()));
		pItem2->setData(Qt::UserRole, ss2qs(pComp->GetKey()));
		ui.listCompounds2->insertItem(i, pItem2);
	}
	m_bAvoidSignal = false;
	if ((iOldRow1 >= 0) && (iOldRow1 < static_cast<int>(m_pMaterialsDB->CompoundsNumber())))
		ui.listCompounds1->setCurrentRow(iOldRow1);
	if ((iOldRow2 >= 0) && (iOldRow2 < static_cast<int>(m_pMaterialsDB->CompoundsNumber())))
		ui.listCompounds2->setCurrentRow(iOldRow2);
}

void CMaterialsDatabaseTab::UpdateInteractions()
{
	if (m_bAvoidSignal) return;

	const CCompound* pCompound1 = m_pMaterialsDB->GetCompound(GetElementKey(ui.listCompounds1));
	const CCompound* pCompound2 = m_pMaterialsDB->GetCompound(GetElementKey(ui.listCompounds2));
	if (!pCompound1 || !pCompound2)
	{
		ui.tableInterProperties->setEnabled(false);
		ui.tableInterProperties->clearContents();
		return;
	}
	ui.tableInterProperties->setEnabled(true);
	const CInteraction *pInteraction = m_pMaterialsDB->GetInteraction(pCompound1->GetKey(), pCompound2->GetKey());
	if (!pInteraction) return;

	m_bAvoidSignal = true;

	ui.tableInterProperties->setRowCount(static_cast<int>(m_vMusenActiveInteractions.size()));

	for (int i = 0; i < static_cast<int>(m_vMusenActiveInteractions.size()); ++i)
	{
		const CTPProperty *pProperty = pInteraction->GetProperty(m_vMusenActiveInteractions[i]);
		ui.tableInterProperties->SetItemNotEditable(i, 0, ss2qs(pProperty->GetName()));
		ui.tableInterProperties->SetItemNotEditable(i, 1, ss2qs(pProperty->GetUnits()));
		ui.tableInterProperties->SetItemEditable(i, 2, QString::number(pProperty->GetValue()), pProperty->GetType());
	}

	m_bAvoidSignal = false;
}

void CMaterialsDatabaseTab::AddMixture()
{
	// pick name for new mixture
	std::string sName = PickUniqueName(GetMixturesNames(), "Mixture");
	// add new mixture
	CMixture *pMixture = m_pMaterialsDB->AddMixture();
	// set new name
	pMixture->SetName(sName);

	// update everything
	UpdateMixturesList();
	SelectMixture(pMixture);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::DuplicateMixture()
{
	const CMixture* pBaseMixture = m_pMaterialsDB->GetMixture(GetElementKey(ui.listMixtures));
	if (!pBaseMixture) return;
	CMixture* pMixture = m_pMaterialsDB->AddMixture(*pBaseMixture);
	pMixture->SetName(pMixture->GetName() + "_copy");

	UpdateMixturesList();
	SelectMixture(pMixture);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::RemoveMixture()
{
	std::string sKey = GetElementKey(ui.listMixtures);
	if (sKey.empty()) return;
	if (QMessageBox::question(this, tr("Remove mixture"), tr("Do you really want to remove %1?").arg(ss2qs(m_pMaterialsDB->GetMixture(sKey)->GetName())),
		QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) == QMessageBox::Yes)
	{
		m_pMaterialsDB->RemoveMixture(sKey);

		UpdateMixturesList();

		setWindowModified(true);
		emit MaterialDatabaseWasChanged();
	}
}

void CMaterialsDatabaseTab::UpMixture()
{
	int iRow = ui.listMixtures->currentRow();
	if (iRow < 0) return;
	m_pMaterialsDB->UpMixture(iRow);
	UpdateMixturesList();
	ui.listMixtures->setCurrentRow(iRow == 0 ? 0 : --iRow);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::DownMixture()
{
	int iRow = ui.listMixtures->currentRow();
	if (iRow < 0) return;
	m_pMaterialsDB->DownMixture(iRow);
	int lastRow = ui.listMixtures->count() - 1;
	UpdateMixturesList();
	ui.listMixtures->setCurrentRow(iRow == lastRow ? lastRow : ++iRow);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::MixtureSelected()
{
	if (m_bAvoidSignal) return;

	bool bEnable = (ui.listMixtures->currentRow() != -1);
	ui.buttonDuplicateMixture->setEnabled(bEnable);
	ui.buttonRemoveMixture->setEnabled(bEnable);
	ui.buttonUpMixture->setEnabled(bEnable);
	ui.buttonDownMixture->setEnabled(bEnable);

	UpdateFractionsList();
	UpdateTotalFraction();
}

void CMaterialsDatabaseTab::MixtureNameChanged(QListWidgetItem* _pItem)
{
	if (m_bAvoidSignal) return;
	CMixture* pMixture = m_pMaterialsDB->GetMixture(GetElementKey(ui.listMixtures));
	if (!pMixture) return;
	QString sNewName = _pItem->text().simplified();
	if (sNewName.isEmpty()) return;
	pMixture->SetName(qs2ss(sNewName));

	UpdateMixturesList();

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::UpdateMixturesList()
{
	m_bAvoidSignal = true;

	int nOldRow = ui.listMixtures->currentRow();
	ui.listMixtures->clear();
	for (int i = 0; i < static_cast<int>(m_pMaterialsDB->MixturesNumber()); ++i)
	{
		QListWidgetItem *pItem = new QListWidgetItem(ss2qs(m_pMaterialsDB->GetMixture(i)->GetName()));
		pItem->setData(Qt::UserRole, ss2qs(m_pMaterialsDB->GetMixtureKey(i)));
		pItem->setFlags(pItem->flags() | Qt::ItemIsEditable);
		ui.listMixtures->insertItem(static_cast<int>(i), pItem);
	}

	if ((nOldRow >= 0) && (nOldRow < static_cast<int>(m_pMaterialsDB->MixturesNumber())))
		ui.listMixtures->setCurrentRow(nOldRow);
	else if (nOldRow != -1)
		ui.listMixtures->setCurrentRow(ui.listMixtures->count() - 1);
	else
		ui.listMixtures->setCurrentRow(-1);	// empty list

	m_bAvoidSignal = false;

	MixtureSelected();
}

void CMaterialsDatabaseTab::AddFraction()
{
	// get and check mixture
	CMixture *pMixture = m_pMaterialsDB->GetMixture(GetElementKey(ui.listMixtures));
	if (!pMixture) return;
	// pick name for new compound
	std::string sName = PickUniqueName(GetFractionNames(*pMixture), "Fraction");
	// add new fraction
	size_t iFraction = pMixture->AddFraction();
	// set new name
	pMixture->SetFractionName(iFraction, sName);
	UpdateFractionsList();
	// select added fraction
	ui.tableFractions->setCurrentCell(ui.tableFractions->rowCount() - 1, EFractionsColumns::NUMBER_FRACTION);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::DuplicateFraction()
{
	// get and check mixture
	CMixture *pMixture = m_pMaterialsDB->GetMixture(GetElementKey(ui.listMixtures));
	if (!pMixture) return;
	QModelIndexList indexes = ui.tableFractions->selectionModel()->selection().indexes();
	if (indexes.empty()) return;
	int iRow = indexes.front().row();
	if (iRow < 0) return;
	// add new fraction
	pMixture->AddFraction(pMixture->GetFractionName(iRow) + "_copy", pMixture->GetFractionCompound(iRow), pMixture->GetFractionDiameter(iRow), pMixture->GetFractionContactDiameter(iRow), pMixture->GetFractionValue(iRow));
	UpdateFractionsList();
	// select added fraction
	ui.tableFractions->setCurrentCell(ui.tableFractions->rowCount() - 1, EFractionsColumns::NUMBER_FRACTION);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::RemoveFraction()
{
	// get and check mixture
	CMixture *pMixture = m_pMaterialsDB->GetMixture(GetElementKey(ui.listMixtures));
	if (!pMixture) return;
	QModelIndexList indexes = ui.tableFractions->selectionModel()->selection().indexes();
	if (indexes.empty()) return;
	int iRow = indexes.front().row();
	if (iRow < 0) return;
	// remove selected fraction
	pMixture->RemoveFraction(iRow);
	UpdateFractionsList();
	// select previous fraction
	if (iRow < ui.tableFractions->rowCount())
		ui.tableFractions->setCurrentCell(iRow, EFractionsColumns::NUMBER_FRACTION);
	else if (ui.tableFractions->rowCount() > 0)
		ui.tableFractions->setCurrentCell(ui.tableFractions->rowCount() - 1, EFractionsColumns::NUMBER_FRACTION);
	else
		ui.tableFractions->setCurrentCell(-1, -1);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::UpFraction()
{
	// get and check mixture
	CMixture *pMixture = m_pMaterialsDB->GetMixture(GetElementKey(ui.listMixtures));
	if (!pMixture) return;
	QModelIndexList indexes = ui.tableFractions->selectionModel()->selection().indexes();
	if (indexes.empty()) return;
	int iRow = indexes.front().row();
	if (iRow < 0) return;
	// move fraction
	pMixture->UpFraction(iRow);
	UpdateFractionsList();
	// select this fraction
	ui.tableFractions->setCurrentCell(iRow == 0 ? 0 : --iRow, EFractionsColumns::NUMBER_FRACTION);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::DownFraction()
{
	// get and check mixture
	CMixture *pMixture = m_pMaterialsDB->GetMixture(GetElementKey(ui.listMixtures));
	if (!pMixture) return;
	QModelIndexList indexes = ui.tableFractions->selectionModel()->selection().indexes();
	if (indexes.empty()) return;
	int iRow = indexes.front().row();
	if (iRow < 0) return;
	// move fraction
	pMixture->DownFraction(iRow);
	UpdateFractionsList();
	// select this fraction
	int lastRow = ui.tableFractions->rowCount() - 1;
	ui.tableFractions->setCurrentCell(iRow == lastRow ? lastRow : ++iRow, EFractionsColumns::NUMBER_FRACTION);

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::NormalizeFractions()
{
	CMixture *pMixture = m_pMaterialsDB->GetMixture(GetElementKey(ui.listMixtures));
	if (!pMixture) return;
	pMixture->NormalizeFractions();

	UpdateFractionsList();

	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
}

void CMaterialsDatabaseTab::FractionChanged(int _iRow, int _iCol)
{
	if (m_bAvoidSignal) return;
	if (_iRow < 0) return;
	CMixture* pMixture = m_pMaterialsDB->GetMixture(GetElementKey(ui.listMixtures));
	if (!pMixture) return;

	switch (_iCol)
	{
	case EFractionsColumns::NAME:
		pMixture->SetFractionName(_iRow, ui.tableFractions->item(_iRow, _iCol)->text().toStdString());
		break;
	case EFractionsColumns::DIAMETER:
		pMixture->SetFractionDiameter(_iRow, GetConvValue(ui.tableFractions->item(_iRow, _iCol), EUnitType::PARTICLE_DIAMETER));
		if(m_bGlobal || !m_pSystemStructure || !m_pSystemStructure->IsContactRadiusEnabled())
			pMixture->SetFractionContactDiameter(_iRow, GetConvValue(ui.tableFractions->item(_iRow, _iCol), EUnitType::PARTICLE_DIAMETER));
		break;
	case EFractionsColumns::CONTACT_DIAMETER:
		pMixture->SetFractionContactDiameter(_iRow, GetConvValue(ui.tableFractions->item(_iRow, _iCol), EUnitType::PARTICLE_DIAMETER));
		break;
	case EFractionsColumns::NUMBER_FRACTION:
		pMixture->SetFractionValue(_iRow, ui.tableFractions->item(_iRow, _iCol)->text().toDouble());
		UpdateTotalFraction();
		break;
	default:
		break;
	}

	m_bAvoidUpdate = true;
	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
	m_bAvoidUpdate = false;
}

void CMaterialsDatabaseTab::FractionCompoundChanged(int _iRow)
{
	if (m_bAvoidSignal) return;
	if (_iRow < 0) return;
	CMixture* pMixture = m_pMaterialsDB->GetMixture(GetElementKey(ui.listMixtures));
	if (!pMixture) return;
	int index = static_cast<QComboBox*>(ui.tableFractions->cellWidget(_iRow, EFractionsColumns::COMPOUND))->currentIndex();
	if (index == -1) return;
	QString sCompoundKey = static_cast<QComboBox*>(ui.tableFractions->cellWidget(_iRow, EFractionsColumns::COMPOUND))->itemData(index, Qt::UserRole).toString();
	pMixture->SetFractionCompound(_iRow, qs2ss(sCompoundKey));

	m_bAvoidUpdate = true;
	setWindowModified(true);
	emit MaterialDatabaseWasChanged();
	m_bAvoidUpdate = false;
}

void CMaterialsDatabaseTab::UpdateFractionsList()
{
	ui.tableFractions->setColumnHidden(EFractionsColumns::CONTACT_DIAMETER, m_bGlobal || !m_pSystemStructure || !m_pSystemStructure->IsContactRadiusEnabled());

	ShowConvLabel(ui.tableFractions->horizontalHeaderItem(EFractionsColumns::DIAMETER), "Diameter", EUnitType::PARTICLE_DIAMETER);
	ShowConvLabel(ui.tableFractions->horizontalHeaderItem(EFractionsColumns::CONTACT_DIAMETER), "Cont. diam.", EUnitType::PARTICLE_DIAMETER);
	const CMixture* pMixture = m_pMaterialsDB->GetMixture(GetElementKey(ui.listMixtures));

	QModelIndexList indexes = ui.tableFractions->selectionModel()->selection().indexes();
	int iRow = indexes.empty() ? -1 : indexes.front().row();

	bool bEnableFields = (pMixture != nullptr);
	ui.tableFractions->setEnabled(bEnableFields);
	ui.buttonAddFraction->setEnabled(bEnableFields);
	ui.buttonDuplicateFraction->setEnabled(bEnableFields);
	ui.buttonRemoveFraction->setEnabled(bEnableFields);
	ui.buttonUpFraction->setEnabled(bEnableFields);
	ui.buttonDownFraction->setEnabled(bEnableFields);
	ui.tableFractions->setRowCount(0);

	if (!pMixture) return;

	m_bAvoidSignal = true;
	for (int i = 0; i < static_cast<int>(pMixture->FractionsNumber()); ++i)
	{
		ui.tableFractions->insertRow(i);
		ui.tableFractions->SetItemEditable(i, EFractionsColumns::NAME, ss2qs(pMixture->GetFractionName(i)));
		AddCombobBoxOnFractionsTable(i, EFractionsColumns::COMPOUND, pMixture->GetFractionCompound(i));
		ui.tableFractions->SetItemEditable(i, EFractionsColumns::DIAMETER, QString::number(m_pUnitConverter->GetValue(EUnitType::PARTICLE_DIAMETER, pMixture->GetFractionDiameter(i))));
		ui.tableFractions->SetItemEditable(i, EFractionsColumns::CONTACT_DIAMETER, QString::number(m_pUnitConverter->GetValue(EUnitType::PARTICLE_DIAMETER, pMixture->GetFractionContactDiameter(i))));
		ui.tableFractions->SetItemEditable(i, EFractionsColumns::NUMBER_FRACTION, QString::number(pMixture->GetFractionValue(i)));
	}

	if ((iRow >= 0) && (iRow < ui.tableFractions->rowCount()))
		ui.tableFractions->setCurrentCell(iRow, EFractionsColumns::NUMBER_FRACTION);
	else if (iRow != -1)
		ui.tableFractions->setCurrentCell(ui.tableFractions->rowCount() - 1, EFractionsColumns::NUMBER_FRACTION);
	else
		ui.tableFractions->setCurrentCell(-1, -1); // select nothing

	UpdateTotalFraction();
	m_bAvoidSignal = false;
}

void CMaterialsDatabaseTab::UpdateTotalFraction()
{
	const CMixture* pMixture = m_pMaterialsDB->GetMixture(GetElementKey(ui.listMixtures));
	if (!pMixture)
	{
		ui.buttonNormalize->setEnabled(false);
		ui.labelFraction->setText("0");
		ui.labelFraction->setStyleSheet("QLabel { color : black; }");
		return;
	}
	double dTotalFraction = 0;
	for (size_t i = 0; i < pMixture->FractionsNumber(); ++i)
		dTotalFraction += pMixture->GetFractionValue(i);
	ui.labelFraction->setText(QString::number(dTotalFraction));
	if (fabs(dTotalFraction - 1) > 16 * DBL_EPSILON)
	{
		ui.buttonNormalize->setEnabled(true);
		ui.labelFraction->setStyleSheet("QLabel { color : red; }");
	}
	else
	{
		ui.buttonNormalize->setEnabled(false);
		ui.labelFraction->setStyleSheet("QLabel { color : black; }");
	}
}

void CMaterialsDatabaseTab::AddCombobBoxOnFractionsTable(int _iRow, int _iCol, const std::string& _sSelected)
{
	QComboBox *pComboBox = new QComboBox();
	int nCurrIndex = -1;
	for (int i = 0; i < static_cast<int>(m_pMaterialsDB->CompoundsNumber()); ++i)
	{
		pComboBox->insertItem(i, ss2qs(m_pMaterialsDB->GetCompoundName(i)));
		pComboBox->setItemData(i, ss2qs(m_pMaterialsDB->GetCompoundKey(i)), Qt::UserRole);
		if (m_pMaterialsDB->GetCompoundKey(i) == _sSelected)
			nCurrIndex = i;
	}
	pComboBox->setCurrentIndex(nCurrIndex);

	ui.tableFractions->setCellWidget(_iRow, _iCol, pComboBox);

	connect(pComboBox, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), m_pSignalMapperFracs, static_cast<void (QSignalMapper::*)()>(&QSignalMapper::map));
	m_pSignalMapperFracs->setMapping(pComboBox, _iRow);
}

int CMaterialsDatabaseTab::GetPropertyType(QTableWidget *_pTable, int _nRow) const
{
	QTableWidgetItem *pItem = _pTable->item(_nRow, 2);
	if (!pItem) return -1;
	return pItem->data(Qt::UserRole).toInt();
}

std::string CMaterialsDatabaseTab::GetElementKey(const QListWidget *_pListWidget, int _nRow /*= -1*/) const
{
	if (!_pListWidget) return "";

	if (_nRow != -1)
		return qs2ss(_pListWidget->item(_nRow)->data(Qt::UserRole).toString());
	else
		if (_pListWidget->currentItem())
			return qs2ss(_pListWidget->currentItem()->data(Qt::UserRole).toString());
		else
			return "";
}

void CMaterialsDatabaseTab::SelectCompound(const CCompound* _pCompound)
{
	if (!_pCompound) return;
	for (int i = 0; i < static_cast<int>(m_pMaterialsDB->CompoundsNumber()); ++i)
		if (GetElementKey(ui.listCompounds, i) == _pCompound->GetKey())
		{
			ui.listCompounds->setCurrentRow(i);
			break;
		}
}

void CMaterialsDatabaseTab::SelectMixture(const CMixture* _pMixture)
{
	if (!_pMixture) return;
	for (int i = 0; i < static_cast<int>(m_pMaterialsDB->MixturesNumber()); ++i)
		if (GetElementKey(ui.listMixtures, i) == _pMixture->GetKey())
		{
			ui.listMixtures->setCurrentRow(i);
			break;
		}
}

std::vector<std::string> CMaterialsDatabaseTab::GetMixturesNames() const
{
	std::vector<std::string> vRes;
	for (size_t i = 0; i < m_pMaterialsDB->MixturesNumber(); ++i)
		vRes.push_back(m_pMaterialsDB->GetMixtureName(i));
	return vRes;
}

std::vector<std::string> CMaterialsDatabaseTab::GetFractionNames(const CMixture& _mixture) const
{
	std::vector<std::string> vRes;
	for (size_t i = 0; i < _mixture.FractionsNumber(); ++i)
		vRes.push_back(_mixture.GetFractionName(i));
	return vRes;
}

std::string CMaterialsDatabaseTab::PickUniqueName(const std::vector<std::string>& _vNames, const std::string& _sBaseName) const
{
	QString sNewName;
	unsigned index = 0;
	bool bAlreadyExist;
	do
	{
		bAlreadyExist = false;
		sNewName = ss2qs(_sBaseName) + QString::number(index);
		for (size_t i = 0; i < _vNames.size(); ++i)
			if (ss2qs(_vNames[i]) == sNewName)
			{
				bAlreadyExist = true;
				break;
			}
		index++;
	} while (bAlreadyExist);
	return qs2ss(sNewName);
}

QString CMaterialsDatabaseTab::FilePathToOpen() const
{
	QString sPrevPath = ss2qs(m_pMaterialsDB->GetFileName());
	if (sPrevPath.isEmpty())
		sPrevPath = ss2qs(m_pSystemStructure->GetFileName());
	return sPrevPath;
}

void CMaterialsDatabaseTab::keyPressEvent(QKeyEvent* event)
{
	switch (event->key())
	{
	case Qt::Key_Delete:
		if (ui.listCompounds->hasFocus())	RemoveCompound();
		else if (ui.listMixtures->hasFocus())	RemoveMixture();
		else if (ui.tableFractions->hasFocus())	RemoveFraction();
		break;
	default: CMusenDialog::keyPressEvent(event);
	}
}
