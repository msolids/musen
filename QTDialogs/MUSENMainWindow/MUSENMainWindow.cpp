/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "MUSENMainWindow.h"
#include "AboutWindow.h"
#include "ImportFromText.h"

#undef GetCurrentTime

const QString MUSENMainWindow::m_sRecentFilesParamName	= "recentFiles";

MUSENMainWindow::MUSENMainWindow(const QString& _buildVersion, QWidget* parent /*= nullptr*/, Qt::WindowFlags flags /*= {}*/) :
	QMainWindow(parent, flags),
	m_pFileLocker(nullptr),
	m_buildVersion(_buildVersion)
{
	ui.setupUi(this);
	//ui.viewOptions->setWindowFlags(Qt::Widget);
	ui.timeSlider->setWindowFlags(Qt::Widget);

	m_pSettings = new QSettings(SettingsPath(), QSettings::IniFormat, this);

	m_GenerationManager.SetAgglomeratesDatabase(&m_AgglomeratesDB);

	m_SimulatorManager.GetSimulatorPtr()->SetGenerationManager(&m_GenerationManager);
	m_SimulatorManager.GetSimulatorPtr()->SetModelManager(&m_ModelsManager);

	m_vpGeneralComponents = { &m_ModelsManager, &m_SimulatorManager, &m_GenerationManager, &m_BondsGenerator, &m_PackageGenerator };
	for (auto& component : m_vpGeneralComponents)
		component->SetSystemStructure(&m_SystemStructure);

	m_pViewSettings            = new CViewSettings(m_pSettings);
	m_pViewManager             = new CViewManager(ui.sceneViewer, ui.verticalLayout, m_pViewSettings, this);
	m_pCameraSettings		   = new CCameraSettings(m_pViewManager, this);
	m_pConfigurationTab        = new CConfigurationTab(m_pViewManager, m_pViewSettings, m_pSettings, this); // should be loaded first!
	m_pModelManagerTab         = new CModelManagerTab(&m_ModelsManager, m_pSettings, this);
	m_pUnitConverterTab        = new CUnitConvertorTab(m_pSettings, this);
	m_pBondsGeneratorTab       = new CBondsGeneratorTab(&m_BondsGenerator, this);
	m_pSceneInfoTab            = new CSceneInfoTab(this);
	m_pPackageGeneratorTab     = new CPackageGeneratorTab(&m_PackageGenerator, this);
	m_pExportTDPTab            = new CExportTDPTab(this);
	m_pObjectsEditorTab        = new CObjectsEditorTab(m_pExportTDPTab, m_pViewSettings, m_pSettings, this);
	m_pSampleAnalyzerTab       = new CSampleAnalyzerTab(this);
	m_pImageGeneratorTab       = new CImageGeneratorTab(m_pViewManager, m_pSettings, this);
	m_pGeometriesDatabaseTab   = new CGeometriesDatabaseTab(this);
	m_pAgglomeratesDatabaseTab = new CAgglomeratesDatabaseTab(this);
	m_pObjectsGeneratorTab     = new CObjectsGeneratorTab(&m_GenerationManager, this);
	m_pGeometriesEditorTab     = new CGeometriesEditorTab(this);
	m_pSceneEditorTab          = new CSceneEditorTab(this);
	m_pMaterialDatabaseTab     = new CMaterialsDatabaseTab(&m_MaterialsDB, this);
	m_pMaterialEditorTab       = new CMaterialsDatabaseLocalTab(&m_MaterialsDB, &m_SystemStructure.m_MaterialDatabase, &m_PackageGenerator, &m_BondsGenerator, &m_GenerationManager, this);
	m_pSimulatorTab            = new CSimulatorTab(&m_SimulatorManager, m_pSettings, this);
	m_pParticlesAnalyzerTab    = new CParticlesAnalyzerTab(this);
	m_pBondsAnalyzerTab        = new CBondsAnalyzerTab(this);
	m_pAgglomeratesAnalyzerTab = new CAgglomeratesAnalyzerTab(this);
	m_pGeometriesAnalyzerTab   = new CGeometriesAnalyzerTab(this);
	//m_pCollisionsAnalyzerTab   = new CCollisionsAnalyzerTab(this);
	m_pExportAsTextTab         = new CExportAsTextTab(&m_PackageGenerator, &m_BondsGenerator, this);
	m_pFileMergerTab           = new CFileMergerTab(this);
	m_pFileConverterTab        = new CFileConverterTab(this);
	m_pSimulatorSettingsTab    = new CSimulatorSettingsTab(&m_SimulatorManager, this);
	m_pClearSpecificTPTab      = new CClearSpecificTPTab(this);
	m_pViewOptionsTab          = new CViewOptionsTab(m_pViewManager, m_pViewSettings, this);

	m_vpDialogTabs = { m_pViewOptionsTab, ui.timeSlider, m_pConfigurationTab, m_pModelManagerTab, m_pUnitConverterTab, m_pBondsGeneratorTab, m_pSceneInfoTab, m_pPackageGeneratorTab, m_pExportTDPTab,
		m_pObjectsEditorTab, m_pSampleAnalyzerTab, m_pImageGeneratorTab, m_pGeometriesDatabaseTab, m_pAgglomeratesDatabaseTab, m_pObjectsGeneratorTab, m_pGeometriesEditorTab, m_pSceneEditorTab,
		m_pMaterialDatabaseTab, m_pMaterialEditorTab, m_pSimulatorTab, m_pParticlesAnalyzerTab, m_pBondsAnalyzerTab, m_pAgglomeratesAnalyzerTab, m_pGeometriesAnalyzerTab, /*m_pCollisionsAnalyzerTab,*/
		m_pExportAsTextTab, m_pFileMergerTab, m_pFileConverterTab, m_pSimulatorSettingsTab, m_pClearSpecificTPTab };

	for (auto& dialogTab : m_vpDialogTabs)
	{
		dialogTab->SetPointers(&m_SystemStructure, &m_UnitConverter, &m_SystemStructure.m_MaterialDatabase, &m_GeometriesDB, &m_AgglomeratesDB);
		dialogTab->Initialize();
	}
	m_vpAnalyzerTabs = { m_pParticlesAnalyzerTab, m_pBondsAnalyzerTab, m_pAgglomeratesAnalyzerTab, m_pGeometriesAnalyzerTab/*, m_pCollisionsAnalyzerTab*/ };

	ui.centralWidget->layout()->replaceWidget(ui.viewOptions, m_pViewOptionsTab);

	m_pViewManager->SetPointers(&m_SystemStructure, m_pSampleAnalyzerTab);

	QWidget* pSpacer = new QWidget();
	pSpacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
	ui.mainToolBar->insertWidget(ui.actionAutoCentrateViewX, pSpacer);

	CreateRecentFilesMenu();
	UpdateRecentFilesMenu();
	CreateHelpMenu();

	LoadConfiguration();
	InitializeConnections();
	EnableControlsNoFile(false);
}

void MUSENMainWindow::InitializeConnections()
{
	// signals from menu entries
	connect(ui.actionMaterialsDatabaseGlobal,     &QAction::triggered, m_pMaterialDatabaseTab,       &CMaterialsDatabaseTab::ShowDialog);
	connect(ui.actionMaterialsDatabaseLocal,      &QAction::triggered, m_pMaterialEditorTab,         &CMaterialsDatabaseLocalTab::ShowDialog);
	connect(ui.actionPackageInfo,                 &QAction::triggered, m_pSceneInfoTab,              &CSceneInfoTab::ShowDialog);
	connect(ui.actionBondsGeneration,             &QAction::triggered, m_pBondsGeneratorTab,         &CBondsGeneratorTab::ShowDialog);
	connect(ui.actionPackageGenerator,            &QAction::triggered, m_pPackageGeneratorTab,       &CPackageGeneratorTab::ShowDialog);
	connect(ui.actionResultAnalyzer,              &QAction::triggered, m_pParticlesAnalyzerTab,      &CParticlesAnalyzerTab::ShowDialog);
	connect(ui.actionBondsAnalyzer,               &QAction::triggered, m_pBondsAnalyzerTab,          &CBondsAnalyzerTab::ShowDialog);
	connect(ui.actionAgglomeratesAnalyzer,        &QAction::triggered, m_pAgglomeratesAnalyzerTab,   &CAgglomeratesAnalyzerTab::ShowDialog);
	connect(ui.actionGeometriesAnalyzer,          &QAction::triggered, m_pGeometriesAnalyzerTab,     &CGeometriesAnalyzerTab::ShowDialog);
	//connect(ui.actionCollisionsAnalyzer,          &QAction::triggered, m_pCollisionsAnalyzerTab,     &CCollisionsAnalyzerTab::ShowDialog);
	connect(ui.actionSceneEditor,                 &QAction::triggered, m_pSceneEditorTab,            &CSceneEditorTab::ShowDialog);
	connect(ui.actionObjectsEditor,               &QAction::triggered, m_pObjectsEditorTab,          &CObjectsEditorTab::ShowDialog);
	connect(ui.actionSampleAnalyzer,              &QAction::triggered, m_pSampleAnalyzerTab,         &CSampleAnalyzerTab::ShowDialog);
	connect(ui.actionSaveAsImagesSet,             &QAction::triggered, m_pImageGeneratorTab,         &CImageGeneratorTab::ShowDialog);
	connect(ui.actionConfiguration,               &QAction::triggered, m_pConfigurationTab,          &CConfigurationTab::ShowDialog);
	connect(ui.actionModelManager,                &QAction::triggered, m_pModelManagerTab,           &CModelManagerTab::ShowDialog);
	connect(ui.actionDefaultUnits,                &QAction::triggered, m_pUnitConverterTab,          &CUnitConvertorTab::ShowDialog);
	connect(ui.actionGeometriesDatabase,          &QAction::triggered, m_pGeometriesDatabaseTab,     &CGeometriesDatabaseTab::ShowDialog);
	connect(ui.actionGeometriesEditor,            &QAction::triggered, m_pGeometriesEditorTab,       &CGeometriesEditorTab::ShowDialog);
	connect(ui.actionAgglomeratesDatabase,        &QAction::triggered, m_pAgglomeratesDatabaseTab,   &CAgglomeratesDatabaseTab::ShowDialog);
	connect(ui.actionObjectsGenerator,            &QAction::triggered, m_pObjectsGeneratorTab,       &CObjectsGeneratorTab::ShowDialog);
	connect(ui.actionStartSimulation,             &QAction::triggered, m_pSimulatorTab,              &CSimulatorTab::ShowDialog);
	connect(ui.actionSimulatorSettings,           &QAction::triggered, m_pSimulatorSettingsTab,      &CSimulatorSettingsTab::ShowDialog);
	connect(ui.actionExportToText,                &QAction::triggered, m_pExportAsTextTab,           &CExportAsTextTab::ShowDialog);
	connect(ui.actionMergeFiles,                  &QAction::triggered, m_pFileMergerTab,             &CFileMergerTab::ShowDialog);
	connect(ui.actionClearSpecificTimePoints,     &QAction::triggered, m_pClearSpecificTPTab,        &CClearSpecificTPTab::ShowDialog);
	connect(ui.actionCameraSettings,              &QAction::triggered, m_pCameraSettings,            &QDialog::show);
	connect(ui.actionAbout,                       &QAction::triggered, this,                         &MUSENMainWindow::ShowAboutWindow);
	connect(ui.actionNew,                         &QAction::triggered, this,                         &MUSENMainWindow::NewSystemStructure);
	connect(ui.actionLoad,                        &QAction::triggered, this,                         &MUSENMainWindow::LoadSystemStructure);
	connect(ui.actionSave,                        &QAction::triggered, this,                         &MUSENMainWindow::SaveSystemStructure);
	connect(ui.actionSaveAs,                      &QAction::triggered, this,                         &MUSENMainWindow::SaveSystemStructureAs);
	connect(ui.actionImportFromText,              &QAction::triggered, this,                         &MUSENMainWindow::LoadSystemStructureFromText);
	connect(ui.actionImportFromEDEM,              &QAction::triggered, this,                         &MUSENMainWindow::LoadSystemStructureFromEDEM);
	connect(ui.actionExportGeometries,            &QAction::triggered, this,                         &MUSENMainWindow::ExportGeometriesAsSTL);
	connect(ui.actionSaveAsImage,                 &QAction::triggered, this,                         &MUSENMainWindow::SaveAsImage);
	connect(ui.actionExit,                        &QAction::triggered, this,                         &MUSENMainWindow::close);
	connect(ui.actionWatchOnYoutube,              &QAction::triggered, this,                         &MUSENMainWindow::WatchOnYouTube);
	connect(ui.actionClearTimePoints,             &QAction::triggered, this,                         &MUSENMainWindow::ClearAllTimePoints);
	connect(ui.actionDeleteAllParticles,          &QAction::triggered, this,                         &MUSENMainWindow::DeleteAllParticles);
	connect(ui.actionDeleteNonConnectedParticles, &QAction::triggered, this,                         &MUSENMainWindow::DeleteAllNonConnectedParticles);
	connect(ui.actionDeleteSeparateParticles,     &QAction::triggered, this,                         &MUSENMainWindow::DeleteAllSeparateParticles);
	connect(ui.actionDeleteAllBonds,              &QAction::triggered, this,                         &MUSENMainWindow::DeleteAllBonds);
	connect(ui.actionSaveSnapshot,                &QAction::triggered, this,                         &MUSENMainWindow::SaveSnapshot);
	connect(ui.actionAutoCentrateView,            &QAction::triggered, this,                         [=]() { CenterView(ECenterView::AUTO); });
	connect(ui.actionAutoCentrateViewX,           &QAction::triggered, this,                         [=]() { CenterView(ECenterView::X); });
	connect(ui.actionAutoCentrateViewY,           &QAction::triggered, this,                         [=]() { CenterView(ECenterView::Y); });
	connect(ui.actionAutoCentrateViewZ,           &QAction::triggered, this,                         [=]() { CenterView(ECenterView::Z); });

	for (auto& tab : m_vpAnalyzerTabs)
	{
		connect(this,                   &MUSENMainWindow::NewSystemStructureGenerated,           tab, &CResultsAnalyzerTab::UpdateSettings);
		connect(m_pGeometriesEditorTab, &CGeometriesEditorTab::AnalysisGeometriesChanged, tab, &CResultsAnalyzerTab::UpdateWholeView);
	}

	for (auto& tab : m_vpDialogTabs)
	{
		connect(m_pUnitConverterTab,  &CUnitConvertorTab::NewUnitsSelected,                    tab,               &CMusenDialog::UpdateWholeView);
		connect(m_pMaterialEditorTab, &CMaterialsDatabaseLocalTab::MaterialDatabaseWasChanged, tab,               &CMusenDialog::UpdateWholeView);
		connect(m_pSceneEditorTab,    &CSceneEditorTab::ContactRadiusEnabled,                  tab,               &CMusenDialog::UpdateWholeView);
		connect(this,                 &MUSENMainWindow::NewSystemStructureGenerated,                  tab,               &CMusenDialog::UpdateWholeView);
		connect(this,                 &MUSENMainWindow::NewSystemStructureGenerated,                  tab,               &CMusenDialog::NewSceneLoaded);
		connect(tab,                  &CMusenDialog::UpdateOpenGLView,						   m_pViewManager,    &CViewManager::UpdateAllObjects);
		connect(tab,                  &CMusenDialog::EnableOpenGLView,                         m_pViewManager,    &CViewManager::EnableView);
		connect(tab,                  &CMusenDialog::DisableOpenGLView,                        m_pViewManager,    &CViewManager::DisableView);
		connect(tab, &CMusenDialog::UpdateViewParticles , m_pViewManager, &CViewManager::UpdateParticles );
		connect(tab, &CMusenDialog::UpdateViewBonds     , m_pViewManager, &CViewManager::UpdateBonds     );
		connect(tab, &CMusenDialog::UpdateViewGeometries, m_pViewManager, &CViewManager::UpdateGeometries);
		connect(tab, &CMusenDialog::UpdateViewVolumes   , m_pViewManager, &CViewManager::UpdateVolumes   );
		connect(tab, &CMusenDialog::UpdateViewSlices    , m_pViewManager, &CViewManager::UpdateSlices    );
		connect(tab, &CMusenDialog::UpdateViewDomain    , m_pViewManager, &CViewManager::UpdateDomain    );
		connect(tab, &CMusenDialog::UpdateViewPBC       , m_pViewManager, &CViewManager::UpdatePBC       );
		connect(tab, &CMusenDialog::UpdateViewAxes      , m_pViewManager, &CViewManager::UpdateAxes      );
		connect(tab, &CMusenDialog::UpdateViewTime      , m_pViewManager, &CViewManager::UpdateTime      );
		connect(tab, &CMusenDialog::UpdateViewLegend    , m_pViewManager, &CViewManager::UpdateLegend    );
	}

	connect(m_pGeometriesEditorTab, &CGeometriesEditorTab::GeometryAdded,	this,                   &MUSENMainWindow::MakeGeometryVisible);
	connect(m_pGeometriesEditorTab, &CGeometriesEditorTab::VolumeAdded,	    this,                   &MUSENMainWindow::MakeVolumeVisible);
	connect(m_pGeometriesEditorTab, &CGeometriesEditorTab::ObjectsChanged,	m_pPackageGeneratorTab, &CPackageGeneratorTab::UpdateWholeView);
	connect(m_pObjectsEditorTab,	&CObjectsEditorTab::MaterialsChanged,	m_pViewOptionsTab,      &CViewOptionsTab::UpdateMaterials);

	connect(m_pSimulatorTab,    &CSimulatorTab::SimulatorStatusChanged,  this, &MUSENMainWindow::EnableControlsSimulation);
	connect(m_pExportAsTextTab, &CExportAsTextTab::RunningStatusChanged, this, &MUSENMainWindow::EnableControlsSimulation);

	// materials database
	connect(m_pMaterialDatabaseTab, &CMaterialsDatabaseTab::MaterialDatabaseFileWasChanged,  this,                 &MUSENMainWindow::SaveConfiguration);
	connect(m_pMaterialDatabaseTab, &CMaterialsDatabaseTab::MaterialDatabaseWasChanged,      m_pMaterialEditorTab, &CMaterialsDatabaseLocalTab::UpdateWholeView);
	connect(m_pMaterialEditorTab,   &CMaterialsDatabaseLocalTab::MaterialDatabaseWasChanged, this,                 &MUSENMainWindow::UpdateMaterialsInSystemStructure);

	connect(m_pGeometriesDatabaseTab,   &CGeometriesDatabaseTab::GeometryAdded,      m_pGeometriesEditorTab, &CGeometriesEditorTab::UpdateWholeView);
	connect(m_pAgglomeratesDatabaseTab, &CAgglomeratesDatabaseTab::AgglomerateAdded, m_pObjectsEditorTab,    &CObjectsEditorTab::UpdateWholeView);

	connect(m_pPackageGeneratorTab,     &CPackageGeneratorTab::ObjectsChanged,         m_pObjectsEditorTab,    &CObjectsEditorTab::UpdateWholeView);
	connect(m_pPackageGeneratorTab,     &CPackageGeneratorTab::ObjectsChanged,         m_pViewOptionsTab,      &CViewOptionsTab::UpdateMaterials);
	connect(m_pPackageGeneratorTab,     &CPackageGeneratorTab::ObjectsChanged,         m_pViewManager,         &CViewManager::UpdateAllObjects);
	connect(m_pBondsGeneratorTab,       &CBondsGeneratorTab::ObjectsChanged,           m_pObjectsEditorTab,    &CObjectsEditorTab::UpdateWholeView);
	connect(m_pBondsGeneratorTab,       &CBondsGeneratorTab::ObjectsChanged,           m_pViewOptionsTab,      &CViewOptionsTab::UpdateMaterials);
	connect(m_pBondsGeneratorTab,       &CBondsGeneratorTab::ObjectsChanged,           m_pViewManager,         &CViewManager::UpdateAllObjects);
	connect(m_pGeometriesEditorTab,     &CGeometriesEditorTab::ObjectsChanged,         m_pViewOptionsTab,      &CViewOptionsTab::UpdateGeometries);
	connect(m_pGeometriesEditorTab,     &CGeometriesEditorTab::ObjectsChanged,         m_pViewManager,         &CViewManager::UpdateAllObjects);
	connect(m_pSampleAnalyzerTab,       &CSampleAnalyzerTab::finished,                 m_pViewManager,         &CViewManager::UpdateAllObjects);

	// time management
	connect(ui.timeSlider,   &CTimeSliderTab::NewTimeSelected,          this,          &MUSENMainWindow::ChangeCurrentTime);
	connect(this,			 &MUSENMainWindow::NumberOfTimePointsChanged,      ui.timeSlider, &CTimeSliderTab::SetTimeSliderEnabled);
	connect(m_pSimulatorTab, &CSimulatorTab::NumberOfTimePointsChanged, ui.timeSlider, &CTimeSliderTab::SetTimeSliderEnabled);

	connect(m_pFileMergerTab, &CFileMergerTab::LoadMergedSystemStrcuture,               this, &MUSENMainWindow::LoadFromFile);

	// rendering
	connect(m_pMaterialEditorTab, &CMaterialsDatabaseLocalTab::MaterialDatabaseWasChanged, m_pViewManager,      &CViewManager::UpdateAllObjects);
	connect(m_pObjectsEditorTab,  &CObjectsEditorTab::ObjectsSelected,					   m_pViewManager,      &CViewManager::UpdateSelectedObjects);
	connect(this,                 &MUSENMainWindow::NewSystemStructureGenerated,                  m_pViewManager,      &CViewManager::UpdateAllObjects);
	connect(m_pViewManager,       &CViewManager::ObjectsSelected,                          m_pViewOptionsTab,   &CViewOptionsTab::UpdateSelectedObjects);
	connect(m_pViewManager,       &CViewManager::ObjectsSelected,                          m_pObjectsEditorTab, &CObjectsEditorTab::UpdateSelectedObjects);
}

QString MUSENMainWindow::SettingsPath()
{
	const QString iniFileName = "/MUSEN.ini";
	const QString iniPath = QStandardPaths::standardLocations(QStandardPaths::AppDataLocation).front();
	(void)QDir{}.mkpath(iniPath); // create directory if does not exist yet
	return iniPath + iniFileName;
}

void MUSENMainWindow::ShowAboutWindow()
{
	CAboutWindow about(m_buildVersion, this);
	about.exec();
}

void MUSENMainWindow::CreateRecentFilesMenu()
{
	ui.menuFile->insertSeparator(ui.actionExit);
	for (int i = 0; i < MAX_RECENT_FILES; ++i)
	{
		auto* action = new QAction(this);
		action->setVisible(false);
		connect(action, &QAction::triggered, this, &MUSENMainWindow::LoadRecentFile);
		ui.menuFile->insertAction(ui.actionExit, action);
		m_vpRecentFilesActions.push_back(action);
	}
	ui.menuFile->insertSeparator(ui.actionExit);
}

void MUSENMainWindow::UpdateRecentFilesMenu()
{
	QStringList filesList = m_pSettings->value(m_sRecentFilesParamName).toStringList();
	for (int i = 0; i < filesList.size() && i < MAX_RECENT_FILES; ++i)
	{
		const QString sText = tr("&%1 %2").arg(i + 1).arg(QFileInfo(filesList[i]).fileName());
		m_vpRecentFilesActions[i]->setText(sText);
		m_vpRecentFilesActions[i]->setData(filesList[i]);
		m_vpRecentFilesActions[i]->setVisible(true);
		m_vpRecentFilesActions[i]->setToolTip(filesList[i]);
		m_vpRecentFilesActions[i]->setStatusTip(filesList[i]);
		m_vpRecentFilesActions[i]->setWhatsThis(filesList[i]);
	}
	for (int i = filesList.size(); i < MAX_RECENT_FILES; ++i)
		m_vpRecentFilesActions[i]->setVisible(false);
}

void MUSENMainWindow::CreateHelpMenu()
{
	std::vector<QString> vGuideNames1 = { "Definitions", "Graphical User Interface" };
	std::vector<QString> vGuideNames2 = { "Agglomerates Analyzer", "Agglomerates Database", "Bonds Analyzer", "Bonds Generator", /*"Collisions Analyzer",*/
		"Dynamic Generator", "Geometries Analyzer", "Geometries Database", "Geometries Editor", "Materials Database", "Package Generator",
		"Particles Analyzer", "Simulator", "Export as text", "Merge files"};
	std::vector<QString> vContactModelsNames = { "Hertz", "HertzMindlin", "JKR", "PopovJKR", "Sintering" };
	std::vector<QString> vSBNames = { "Elastic", "Kelvin", "Maxwell" };
	std::vector<QString> vExtForceNames = { "ViscousField" };
	std::vector<QString> vOtherNames = { "CMusen", "MUSEN Files", "Text File Format" };

	QMenu *pMenuGuide = ui.menuDocumentation->addMenu("Users Guide");
	for (const auto& name : vGuideNames1)
		CreateHelpAction("/Users Guide/", name, pMenuGuide);
	pMenuGuide->addSeparator();
	for (const auto& name : vGuideNames2)
		CreateHelpAction("/Users Guide/", name, pMenuGuide);

	QMenu *pMenuModels = ui.menuDocumentation->addMenu("Models");

	QMenu *pMenuPP = pMenuModels->addMenu("Contact Models");
	for (const auto& name : vContactModelsNames)
		CreateHelpAction("/Models/Contact Models/", name, pMenuPP);

	QMenu *pMenuSB = pMenuModels->addMenu("Solid Bonds");
	for (const auto& name : vSBNames)
		CreateHelpAction("/Models/Solid Bond/", name, pMenuSB);

	QMenu *pMenuExtForce = pMenuModels->addMenu("External Forces");
	for (const auto& name : vExtForceNames)
		CreateHelpAction("/Models/External Force/", name, pMenuExtForce);

	for (const auto& name : vOtherNames)
		CreateHelpAction("/", name, ui.menuDocumentation);
}

void MUSENMainWindow::CreateHelpAction(const QString& _path, const QString& _name, QMenu* _menu)
{
	const QString fullPath = QCoreApplication::applicationDirPath() + "/Documentation" + _path + _name + ".pdf";
	auto* action = new QAction(_name, this);
	action->setToolTip(fullPath);
	action->setStatusTip(fullPath);
	action->setWhatsThis(fullPath);
	connect(action, &QAction::triggered, this, [=] { QDesktopServices::openUrl(QUrl::fromLocalFile("file:///" + fullPath)); });
	_menu->addAction(action);
}

void MUSENMainWindow::NewSystemStructure()
{
	if (!m_sFileName.isEmpty())
	{
		const QMessageBox::StandardButton reply = QMessageBox::question(this, "Confirmation", "Save current simulation to the current file?", QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
		if (reply == QMessageBox::Cancel) return;
		if (reply == QMessageBox::Yes)
			SaveSystemStructureUtil(m_sFileName);
	}
	const QString newFileName = QFileDialog::getSaveFileName(this, "Create new structure", m_sFileName, "MUSEN files (*.mdem);;All files (*.*);;");
	if (newFileName.isEmpty()) return;
	m_SystemStructure.NewFile();

	if (!LockFile(newFileName)) return;

	SaveSystemStructureUtil(newFileName);
	for (auto& component : m_vpGeneralComponents)
		component->LoadConfiguration();
	m_ModelsManager.SetConnectedPPContact(true); // force to enable this option in new files
	emit NewSystemStructureGenerated();
}

void MUSENMainWindow::SaveSystemStructure()
{
	if (m_sFileName.isEmpty())
		SaveSystemStructureAs();
	else
		SaveSystemStructureUtil(m_sFileName);
}

void MUSENMainWindow::SaveSystemStructureAs()
{
	const QString fileName = QFileDialog::getSaveFileName(this, "Save structure", m_sFileName, "MUSEN files (*.mdem);;All files (*.*);;");
	if (fileName.isEmpty()) return;
	if (!LockFile(fileName)) return;
	SaveSystemStructureUtil(fileName);
}

void MUSENMainWindow::SaveSystemStructureUtil(const QString& _fileName)
{
	if (_fileName.isEmpty()) return;

	QApplication::setOverrideCursor(Qt::WaitCursor);
	m_pViewManager->DisableView();

	// if something is already loaded in system structure, m_storage will be used to save data; otherwise it is nothing to save yet
	for (auto& component : m_vpGeneralComponents)
		component->SaveConfiguration();
	m_SystemStructure.SaveToFile(qs2ss(_fileName));

	SetCurrentFile(_fileName);
	UpdateWindowTitle();
	EnableControlsNoFile(true);
	m_pViewManager->EnableView();
	QApplication::restoreOverrideCursor();
}

void MUSENMainWindow::LoadFromFile(const QString& _fileName)
{
	if (_fileName.isEmpty()) return;
	LoadSystemStructureUtil(_fileName);
}

void MUSENMainWindow::LoadRecentFile()
{
	const QAction* action = qobject_cast<QAction*>(sender());
	if (action)
		LoadSystemStructureUtil(action->data().toString());
}

void MUSENMainWindow::LoadSystemStructure()
{
	const QString fileName = QFileDialog::getOpenFileName(this, tr("Open structure"), m_sFileName, tr("MUSEN files (*.mdem);;All files (*.*);;"));
	if (fileName.isEmpty()) return;
	LoadSystemStructureUtil(fileName);
}

void MUSENMainWindow::LoadSystemStructureUtil(const QString& _fileName)
{
	// check that file exists
	if (_fileName.isEmpty() || !QFileInfo(_fileName).exists())
	{
		QMessageBox::critical(this, "MUSEN", "Unable to open '" + _fileName + "'. The selected file does not exist.");
		return;
	}

	if (!LockFile(_fileName)) return;

	// check file version
	if (CSystemStructure::IsOldFileVersion(qs2ss(_fileName)))
	{
		// converting
		QMessageBox::information(this, "MUSEN", "The selected file is in old format and will be converted now.");
		m_pFileConverterTab->StartConversion(qs2ss(_fileName));
	}
	if (CSystemStructure::FileVersion(qs2ss(_fileName)) < 2)
	{
		if (QMessageBox::question(this, "Outdated file", "The current file has an outdated format. Upgrade now?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes)
			m_pFileConverterTab->StartConversion(qs2ss(_fileName));
	}

	QApplication::setOverrideCursor(Qt::WaitCursor);

	// loading
	const CSystemStructure::ELoadFileResult status = m_SystemStructure.LoadFromFile(qs2ss(_fileName));
	switch (status)
	{
	case CSystemStructure::ELoadFileResult::OK:
		break;
	case CSystemStructure::ELoadFileResult::IsNotDEMFile:
		ShowMessage(QMessageBox::Icon::Critical, "The selected file cannot be opened. The file may be in the wrong format or damaged.");
		UnlockFile();
		QApplication::restoreOverrideCursor();
		return;
	case CSystemStructure::ELoadFileResult::PartiallyLoaded:
		ShowMessage(QMessageBox::Icon::Warning, "The selected file was only partially loaded. Probably, the simulation ended abnormally.");
		break;
	case CSystemStructure::ELoadFileResult::SelectivelySaved:
		ShowMessage(QMessageBox::Icon::Warning, "The selected file was selectively saved. Some objects properties may not be available. The scene may not display correctly at certain time points.");
		break;
	}

	SetCurrentFile(_fileName);
	// use m_storage in system structure to load all data; m_storage must be previously initialized in CSystemStructure::LoadFromFile()
	for (auto& component : m_vpGeneralComponents)
		component->LoadConfiguration();

	UpdateWindowTitle();
	emit NewSystemStructureGenerated();
	emit NumberOfTimePointsChanged();
	CenterView();
	EnableControlsNoFile(true);
	QApplication::restoreOverrideCursor();
}

void MUSENMainWindow::LoadSystemStructureFromText()
{
	const QString fileName = QFileDialog::getOpenFileName(this, tr("Import data"), m_sFileName, "MUSEN files (*.txt);;All files (*.*);;");
	if (fileName.isEmpty()) return;
	if (QMessageBox::question(this, "Confirmation", "The current file will be overwritten. Continue?", QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes) return;
	QApplication::setOverrideCursor(Qt::WaitCursor);
	m_pViewManager->DisableView();

	CImportFromText importer(&m_SystemStructure, &m_PackageGenerator, &m_BondsGenerator);
	const CImportFromText::SImportFileInfo importInfo = importer.Import(qs2ss(fileName));
	switch (importInfo.importResult)
	{
	case CImportFromText::EImportFileResult::ErrorOpening:
		QMessageBox::critical(this, "MUSEN", "Unable to open the selected file. The file may be incorrect or damaged.");
		break;
	case CImportFromText::EImportFileResult::ErrorNoID:
		QMessageBox::critical(this, "MUSEN", "Unable to import the selected file. Object ID is not specified in line: " + QString::number(importInfo.nErrorLineNumber) + ".");
		break;
	case CImportFromText::EImportFileResult::ErrorNoType:
		QMessageBox::critical(this, "MUSEN", "Unable to import the selected file. Object type is not specified in line: " + QString::number(importInfo.nErrorLineNumber) + ".");
		break;
	case CImportFromText::EImportFileResult::ErrorNoGeometry:
		QMessageBox::critical(this, "MUSEN", "Unable to import the selected file. Object geometry is not specified in line: " + QString::number(importInfo.nErrorLineNumber) + ".");
		break;
	case CImportFromText::EImportFileResult::OK:
		if (!importInfo.bMaterial)
			QMessageBox::warning(this, "MUSEN", "Materials of objects are not specified in file. Materials for all objects will be set as undefined.");
		if (!importInfo.bActivityInterval)
			QMessageBox::warning(this, "MUSEN", "Activity intervals of objects are not specified in file. All objects will be set active during the entire simulation time.");
		if (!importInfo.bParticleCoordinates)
			QMessageBox::warning(this, "MUSEN", "Coordinates of particles are not specified in file. The 3D scene view may not work properly.");
		break;
	}
	emit NewSystemStructureGenerated();
	emit NumberOfTimePointsChanged();
	CenterView();
	QApplication::restoreOverrideCursor();
}

void MUSENMainWindow::LoadSystemStructureFromEDEM()
{
	const QString fileName = QFileDialog::getOpenFileName(this, tr("Import data"), m_sFileName, tr("CSV files (*.csv);;Text files (*.txt);;All files (*.*);;"));
	if (fileName.isEmpty()) return;
	if (QMessageBox::question(this, "Confirmation", "The current file will be overwritten. Continue?", QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes) return;
	QApplication::setOverrideCursor(Qt::WaitCursor);
	m_pViewManager->DisableView();
	m_SystemStructure.ImportFromEDEMTextFormat(qs2ss(fileName));
	m_pViewManager->EnableView();
	CenterView();
	QApplication::restoreOverrideCursor();
}

bool MUSENMainWindow::LockFile(const QString& _fileName)
{
	if (_fileName == m_sFileName) return true;

	auto* fileLocker = new QLockFile(_fileName + ".lock");
	fileLocker->setStaleLockTime(0);
	const bool locked = fileLocker->tryLock(100);
	if (locked)
	{
		UnlockFile();
		m_pFileLocker = fileLocker;
	}
	else
		QMessageBox::warning(this, "Access denied", tr("File: %1\nCan not access the specified file: it is already in use by another instance of MUSEN or there is no write access to the specified location").arg(_fileName));
	return locked;
}

void MUSENMainWindow::UnlockFile()
{
	if (m_pFileLocker)
	{
		m_pFileLocker->unlock();
		m_pFileLocker->removeStaleLockFile();
		delete m_pFileLocker;
		m_pFileLocker = nullptr;
	}
}

void MUSENMainWindow::ShowMessage(QMessageBox::Icon _type, const QString& _message)
{
	auto* box = new QMessageBox{ this };
	box->setAttribute(Qt::WA_DeleteOnClose);
	box->setIcon(_type);
	box->setModal(false);
	box->setStandardButtons(QMessageBox::Ok);
	box->setText(_message);
	box->setWindowTitle("MUSEN");
	box->show();
}

void MUSENMainWindow::ChangeCurrentTime()
{
	for (auto& dialog : m_vpDialogTabs)
		dialog->SetCurrentTime(ui.timeSlider->GetCurrentTime());
	m_pViewManager->SetTime(ui.timeSlider->GetCurrentTime());
}

void MUSENMainWindow::UpdateMaterialsInSystemStructure()
{
	m_SystemStructure.UpdateAllObjectsCompoundsProperties();
}

void MUSENMainWindow::MakeGeometryVisible(const std::string& _key) const
{
	// force new geometry to be visible
	auto visibleGeometries = m_pViewSettings->VisibleGeometries();
	visibleGeometries.insert(_key);
	m_pViewSettings->VisibleGeometries(visibleGeometries);
}

void MUSENMainWindow::MakeVolumeVisible(const std::string& _key) const
{
	// force new volume to be visible
	auto visibleVolumes = m_pViewSettings->VisibleVolumes();
	visibleVolumes.insert(_key);
	m_pViewSettings->VisibleVolumes(visibleVolumes);
}

void MUSENMainWindow::ClearSystemStructure()
{
	if (QMessageBox::question(this, "Confirmation", "Delete all objects?", QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes) return;
	m_SystemStructure.DeleteAllObjects();
	emit NewSystemStructureGenerated();
}

void MUSENMainWindow::ClearAllTimePoints()
{
	if (QMessageBox::question(this, "Confirmation", "Clear all time points?", QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes) return;
	m_SystemStructure.ClearAllStatesFrom(0);
	ui.timeSlider->UpdateWholeView();
	emit NewSystemStructureGenerated();
	emit NumberOfTimePointsChanged();
}

void MUSENMainWindow::DeleteAllBonds()
{
	if (QMessageBox::question(this, "Confirmation", "Delete all bonds?", QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes) return;
	m_SystemStructure.DeleteAllBonds();
	emit NewSystemStructureGenerated();
}

void MUSENMainWindow::DeleteAllParticles()
{
	if (QMessageBox::question(this, "Confirmation", "Delete all particles?", QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes) return;
	m_SystemStructure.DeleteAllParticles();
	emit NewSystemStructureGenerated();
}


void MUSENMainWindow::DeleteAllSeparateParticles()
{
	if (QMessageBox::question(this, "Confirmation", "Delete all particles with zero coordination number?", QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes) return;
	const size_t removedParticles = m_SystemStructure.DeleteAllParticlesWithNoContacts();
	emit NewSystemStructureGenerated();
	QMessageBox::information(this, "Removed particles", QString::number(removedParticles) + " particles have been removed");
}

void MUSENMainWindow::DeleteAllNonConnectedParticles()
{
	if (QMessageBox::question(this, "Confirmation", "Delete all non-connected particles?", QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel) != QMessageBox::Yes) return;
	const size_t removedParticles = m_SystemStructure.DeleteAllNonConnectedParticles();
	emit NewSystemStructureGenerated();
	QMessageBox::information(this, "Removed particles", QString::number(removedParticles) + " particles have been removed");
}

void MUSENMainWindow::SaveSnapshot()
{
	if(!VectorContains(m_SystemStructure.GetAllTimePoints(), ui.timeSlider->GetCurrentTime()))
		if (QMessageBox::warning(this, "Warning", "The selected time point is obtained by interpolation. Using this data to restart simulation can lead to errors. Continue?", QMessageBox::Cancel, QMessageBox::Ok) == QMessageBox::Cancel) return;

	// get file name
	const QString fileName = QFileDialog::getSaveFileName(this, "Save data snapshot", m_sFileName, "MUSEN files (*.mdem);;All files (*.*);;");
	if (fileName.isEmpty()) return;

	// check file names
	if (m_SystemStructure.GetFileName() == fileName.toStdString())
	{
		QMessageBox::critical(this, "Error", "Can not generate snapshot: source file and result file are the same.");
		return;
	}

	// create new file
	CSystemStructure snapshot;
	snapshot.SaveToFile(fileName.toStdString());
	snapshot.CreateFromSystemStructure(&m_SystemStructure, ui.timeSlider->GetCurrentTime());

	// remember current activity of dynamic generators and switch them off for snapshot
	std::vector<bool> activ;
	for (size_t i = 0; i < m_GenerationManager.GetGeneratorsNumber(); ++i)
	{
		activ.push_back(m_GenerationManager.GetGenerator(i)->m_bActive);
		m_GenerationManager.GetGenerator(i)->m_bActive = false;
	}

	// save all modules
	for (auto& component : m_vpGeneralComponents)
	{
		component->SetSystemStructure(&snapshot);			// set new SS
		component->SaveConfiguration();						// save configuration
		component->SetSystemStructure(&m_SystemStructure);	// restore SS
	}

	// restore activity of dynamic generators
	for (size_t i = 0; i < m_GenerationManager.GetGeneratorsNumber(); ++i)
		m_GenerationManager.GetGenerator(i)->m_bActive = activ[i];

	snapshot.SaveToFile(fileName.toStdString());
}

void MUSENMainWindow::SaveAsImage()
{
	const QString fileName = QFileDialog::getSaveFileName(this, "Save as image", QFileInfo(m_sFileName).absolutePath(), "Image files(*.png);;Image files(*.jpg);;Image files(*.bmp);;All files (*.*);;");
	if (fileName.isEmpty()) return;
	const bool success = m_pViewManager->GetSnapshot().save(fileName);
}

void MUSENMainWindow::ExportGeometriesAsSTL()
{
	const QString fileName = QFileDialog::getSaveFileName(this, "Save geometries", QFileInfo(m_sFileName).absolutePath(), "STL files (*.stl);;All files (*.*);;");
	if (fileName.isEmpty()) return;
	if (!CMusenDialog::IsFileWritable(fileName))
	{
		QMessageBox::warning(this, "Access error", "Unable to save. The selected file is not writable");
		return;
	}
	m_SystemStructure.ExportGeometriesToSTL(qs2ss(fileName), ui.timeSlider->GetCurrentTime());
}

void MUSENMainWindow::WatchOnYouTube()
{
	QDesktopServices::openUrl(QUrl("http://www.youtube.com/channel/UCndqDrPpwcRqLeruavGmbbQ"));
}

void MUSENMainWindow::SaveConfiguration()
{
	m_pSettings->setValue("MATERIALS_DATABASE_PATH", ss2qs(m_MaterialsDB.GetFileName()));
	m_pSettings->setValue("GEOMETRIES_DATABASE_PATH", ss2qs(m_GeometriesDB.GetFileName()));
	m_pSettings->setValue("AGGLOMERATES_DATABASE_PATH", ss2qs(m_AgglomeratesDB.GetFileName()));
}

void MUSENMainWindow::LoadConfiguration()
{
	m_MaterialsDB.LoadFromFile(qs2ss(m_pSettings->value("MATERIALS_DATABASE_PATH").toString()));
	m_GeometriesDB.LoadFromFile(qs2ss(m_pSettings->value("GEOMETRIES_DATABASE_PATH").toString()));
	m_AgglomeratesDB.LoadFromFile(qs2ss(m_pSettings->value("AGGLOMERATES_DATABASE_PATH").toString()));
}

void MUSENMainWindow::SetCurrentFile(const QString& _fileName)
{
	QString filePath = QDir::fromNativeSeparators(_fileName);	// to bring the path into a qt-way view (slashes, backslashes, etc.)
	if (filePath.isEmpty() || m_sFileName == filePath) return;
	QStringList filesList = m_pSettings->value(m_sRecentFilesParamName).toStringList();
	filesList.removeAll(m_sFileName);
	filesList.removeAll(filePath);
	while (filesList.size() >= MAX_RECENT_FILES)
		filesList.pop_back();
	if (!m_sFileName.isEmpty())
		filesList.push_front(m_sFileName);
	filesList.push_front(filePath);
	m_pSettings->setValue(m_sRecentFilesParamName, filesList);
	UpdateRecentFilesMenu();
	m_sFileName = filePath;
}

void MUSENMainWindow::UpdateWindowTitle()
{
	setWindowTitle("MUSEN: " + m_sFileName);
	if (!m_sFileName.isEmpty())
		m_pSimulatorTab->setWindowTitle("Simulator:   " + QFileInfo(m_sFileName).fileName());
}

void MUSENMainWindow::EnableControlsNoFile(bool _bEnable) const
{
	ui.actionClearTimePoints->setEnabled(_bEnable);
	ui.actionDeleteAllParticles->setEnabled(_bEnable);
	ui.actionExportToText->setEnabled(_bEnable);
	ui.actionExportGeometries->setEnabled(_bEnable);
	ui.actionImportFromText->setEnabled(_bEnable);
	ui.actionImportFromEDEM->setEnabled(_bEnable);
	ui.actionSaveAsImage->setEnabled(_bEnable);
	ui.actionSaveAsImagesSet->setEnabled(_bEnable);
	ui.actionSaveSnapshot->setEnabled(_bEnable);
	ui.actionGeometriesEditor->setEnabled(_bEnable);
	ui.actionStartSimulation->setEnabled(_bEnable);
	ui.menuScene->setEnabled(_bEnable);
	ui.menuSimulation->setEnabled(_bEnable);
	ui.menuAnalysis->setEnabled(_bEnable);
	m_pAgglomeratesDatabaseTab->EnableInsertion(_bEnable);
}

void MUSENMainWindow::EnableControlsSimulation(ERunningStatus _simStatus) const
{
	const bool bEnabledOnlyInIdle = _simStatus == ERunningStatus::IDLE;
	const bool bEnabledAlsoInPause = (_simStatus == ERunningStatus::IDLE || _simStatus == ERunningStatus::PAUSED);

	ui.menuFile->setEnabled(bEnabledOnlyInIdle);
	ui.menuTools->setEnabled(bEnabledOnlyInIdle);
	ui.actionPackageGenerator->setEnabled(bEnabledOnlyInIdle);
	ui.actionBondsGeneration->setEnabled(bEnabledOnlyInIdle);
	ui.actionGeometriesEditor->setEnabled(bEnabledOnlyInIdle);
	ui.actionSceneEditor->setEnabled(bEnabledOnlyInIdle);
	ui.actionAgglomeratesDatabase->setEnabled(bEnabledOnlyInIdle);
	ui.actionGeometriesDatabase->setEnabled(bEnabledOnlyInIdle);
	ui.actionNew->setEnabled(bEnabledOnlyInIdle);
	ui.actionSave->setEnabled(bEnabledOnlyInIdle);
	ui.actionSaveAs->setEnabled(bEnabledOnlyInIdle);
	ui.actionLoad->setEnabled(bEnabledOnlyInIdle);
	ui.actionClearTimePoints->setEnabled(bEnabledOnlyInIdle);
	ui.actionClearSpecificTimePoints->setEnabled(bEnabledOnlyInIdle);
	ui.actionDeleteAllBonds->setEnabled(bEnabledOnlyInIdle);
	ui.actionDeleteAllParticles->setEnabled(bEnabledOnlyInIdle);
	ui.actionDeleteNonConnectedParticles->setEnabled(bEnabledOnlyInIdle);
	m_pModelManagerTab->setEnabled(bEnabledOnlyInIdle);
	m_pObjectsEditorTab->SetEditEnabled(bEnabledOnlyInIdle);

	ui.menuScene->setEnabled(bEnabledAlsoInPause);
	ui.menuAnalysis->setEnabled(bEnabledAlsoInPause);
	ui.menuDatabases->setEnabled(bEnabledAlsoInPause);
	ui.menuOptions->setEnabled(bEnabledAlsoInPause);
	ui.actionMaterialsDatabaseLocal->setEnabled(bEnabledAlsoInPause);
	ui.actionExportGeometries->setEnabled(bEnabledAlsoInPause);
	m_pViewOptionsTab->setEnabled(bEnabledAlsoInPause);
	ui.timeSlider->setEnabled(bEnabledAlsoInPause);
	ui.actionCameraSettings->setEnabled(bEnabledAlsoInPause);
}

void MUSENMainWindow::CenterView(ECenterView _type /*= ECenterView::AUTO*/) const
{
	static CVector3 viewX(1, 0, 0);	// camera position. X axis pointed to observer
	static CVector3 viewY(0, 1, 0);	// camera position. Y axis pointed to observer
	static CVector3 viewZ(0, 0, 1);	// camera position. Z axis pointed to observer
	switch (_type)
	{
	case ECenterView::AUTO: m_pViewManager->SetCameraStandardView();                          break;
	case ECenterView::X:    m_pViewManager->SetCameraStandardView(viewX); viewX.x = -viewX.x; break;
	case ECenterView::Y:    m_pViewManager->SetCameraStandardView(viewY); viewY.y = -viewY.y; break;
	case ECenterView::Z:    m_pViewManager->SetCameraStandardView(viewZ); viewZ.z = -viewZ.z; break;
	}
}

void MUSENMainWindow::closeEvent(QCloseEvent* _event)
{
	if (m_SimulatorManager.GetSimulatorPtr()->GetCurrentStatus() != ERunningStatus::IDLE)
		if (QMessageBox::warning(this, "Close MUSEN", "Simulation is still in progress. If you close the application now, all data will be lost.\nExit the application?",
			QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel, QMessageBox::No) != QMessageBox::Yes)
		{
			_event->ignore();
			return;
		}
	SaveConfiguration();
	m_pModelManagerTab->SaveConfiguration();
	m_pUnitConverterTab->SaveConfiguration();
	m_pObjectsEditorTab->SaveConfiguration();
	UnlockFile();
	_event->accept();
}

void MUSENMainWindow::keyPressEvent(QKeyEvent* _event)
{
	if (_event->key() == Qt::Key_F1)
		QDesktopServices::openUrl(QUrl::fromLocalFile("file:///" + QCoreApplication::applicationDirPath() + "/Documentation/Users Guide/Graphical User Interface.pdf"));
}

void MUSENMainWindow::showEvent(QShowEvent* _event)
{
	// HACK: Shaders view won't get initialized until it is really shown, but objects can be set only after initialization.
	// So set all objects to view here once more.
	static bool firstShow = true;
	if (firstShow)
	{
		m_pViewManager->UpdateAllObjects();
		firstShow = false;
	}
	QWidget::showEvent(_event);
}
