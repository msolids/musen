/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ui_MUSENMainWindow.h"
#include "BondsGeneratorTab.h"
#include "SceneInfoTab.h"
#include "PackageGeneratorTab.h"
#include "ObjectsEditorTab.h"
#include "ConfigurationTab.h"
#include "ModelManagerTab.h"
#include "UnitConverterTab.h"
#include "SampleAnalyzerTab.h"
#include "ImageGeneratorTab.h"
#include "SceneEditorTab.h"
#include "GeometriesDatabaseTab.h"
#include "AgglomeratesDatabaseTab.h"
#include "GeometriesEditorTab.h"
#include "ObjectsGeneratorTab.h"
#include "ParticlesAnalyzerTab.h"
#include "BondsAnalyzerTab.h"
#include "AgglomeratesAnalyzerTab.h"
#include "GeometriesAnalyzerTab.h"
//#include "CollisionsAnalyzerTab.h"
#include "SimulatorTab.h"
#include "MaterialsDatabaseLocalTab.h"
#include "ExportAsTextTab.h"
#include "FileMergerTab.h"
#include "FileConverterTab.h"
#include "SimulatorSettingsTab.h"
#include "ClearSpecificTPTab.h"
#include "ViewManager.h"
#include "ViewOptionsTab.h"
#include "CameraSettings.h"

#define MAX_RECENT_FILES 10
#define RENDER_SPHERE_TEXTURE "RNDR_SPHERE_TEXTURE"

class MUSENMainWindow : public QMainWindow
{
	Q_OBJECT

	enum class ECenterView { AUTO, X, Y, Z };

	static const QString m_sRecentFilesParamName;

	Ui::MUSENMainWindowClass ui;

	CSystemStructure m_SystemStructure;
	CMaterialsDatabase m_MaterialsDB;
	CGeometriesDatabase m_GeometriesDB;
	CAgglomeratesDatabase m_AgglomeratesDB;
	CGenerationManager m_GenerationManager;
	CModelManager m_ModelsManager;
	CSimulatorManager m_SimulatorManager;
	CBondsGenerator m_BondsGenerator;
	CPackageGenerator m_PackageGenerator;
	CUnitConvertor m_UnitConverter;

	CMaterialsDatabaseTab* m_pMaterialDatabaseTab;
	CGeometriesDatabaseTab* m_pGeometriesDatabaseTab;
	CAgglomeratesDatabaseTab* m_pAgglomeratesDatabaseTab;

	CMaterialsDatabaseLocalTab* m_pMaterialEditorTab;
	CBondsGeneratorTab* m_pBondsGeneratorTab;
	CSceneInfoTab* m_pSceneInfoTab;
	CPackageGeneratorTab* m_pPackageGeneratorTab;
	CObjectsEditorTab* m_pObjectsEditorTab;
	CSampleAnalyzerTab* m_pSampleAnalyzerTab;
	CImageGeneratorTab* m_pImageGeneratorTab;
	CGeometriesEditorTab* m_pGeometriesEditorTab;
	CSceneEditorTab* m_pSceneEditorTab;
	CConfigurationTab* m_pConfigurationTab;
	CModelManagerTab* m_pModelManagerTab;
	CUnitConvertorTab* m_pUnitConverterTab;
	CObjectsGeneratorTab* m_pObjectsGeneratorTab;
	CSimulatorTab* m_pSimulatorTab;
	CParticlesAnalyzerTab* m_pParticlesAnalyzerTab;
	CBondsAnalyzerTab* m_pBondsAnalyzerTab;
	CAgglomeratesAnalyzerTab* m_pAgglomeratesAnalyzerTab;
	CGeometriesAnalyzerTab* m_pGeometriesAnalyzerTab;
	//CCollisionsAnalyzerTab* m_pCollisionsAnalyzerTab;
	CExportTDPTab* m_pExportTDPTab;
	CExportAsTextTab* m_pExportAsTextTab;
	CFileMergerTab* m_pFileMergerTab;
	CFileConverterTab* m_pFileConverterTab;
	CSimulatorSettingsTab* m_pSimulatorSettingsTab;
	CClearSpecificTPTab* m_pClearSpecificTPTab;
	CViewManager* m_pViewManager;
	CViewSettings* m_pViewSettings;
	CViewOptionsTab* m_pViewOptionsTab;
	CCameraSettings* m_pCameraSettings;

	std::vector<CMusenDialog*> m_vpDialogTabs;
	std::vector<CResultsAnalyzerTab*> m_vpAnalyzerTabs;
	std::vector<CMusenComponent*> m_vpGeneralComponents; // Pointers to all general musen modules (GenerationManager, BondsGenerator, etc.).
	QList<QAction*> m_vpRecentFilesActions;
	QSettings* m_pSettings;
	QLockFile* m_pFileLocker;
	QString m_sFileName;	 // Name of the current file with system structure.
	QString m_buildVersion;

public:
	MUSENMainWindow(const QString& _buildVersion, QWidget* parent = nullptr, Qt::WindowFlags flags = {});

	void InitializeConnections(); // Initialize all connections on the form.

public slots:
	void LoadFromFile(const QString& _fileName);

private:
	static QString SettingsPath();
	void ShowAboutWindow();
	void CreateRecentFilesMenu();
	void UpdateRecentFilesMenu();
	void CreateHelpMenu();
	void CreateHelpAction(const QString& _path, const QString& _name, QMenu* _menu);

	void EnableControlsNoFile(bool _bEnable) const; // Changes activity of elements depending on whether the system structure is loaded.
	void SetCurrentFile(const QString& _fileName);
	void UpdateWindowTitle();                       // Update window title.

	void SaveSystemStructureUtil(const QString& _fileName); // Saves system structure to specified file.
	void LoadSystemStructureUtil(const QString& _fileName); // Loads system structure from specified file.
	bool LockFile(const QString& _fileName);                // Tries to lock the specified file to protect it from access by other processes. Returns true if lock was successful.
	void UnlockFile();                                      // Unlocks the current file for other processes.
	void ShowMessage(QMessageBox::Icon _type, const QString& _message);				// Shows a non-modal non-blocking window with default name, OK button and the specified _type and _message.

	void closeEvent(QCloseEvent* _event) override;
	void keyPressEvent(QKeyEvent* _event) override;
	void showEvent(QShowEvent* _event) override;

private slots:
	void EnableControlsSimulation(ERunningStatus _simStatus) const; // Changes activity of elements depending on the state of the simulator.
	void CenterView(ECenterView _type = ECenterView::AUTO) const;

	void NewSystemStructure();    // Create new system structure in new file.
	void SaveSystemStructure();   // Save system structure into current file.
	void SaveSystemStructureAs(); // Save system structure into new file.
	void LoadSystemStructure();   // Load system structure from the user specified file.
	void LoadSystemStructureFromText();
	void LoadSystemStructureFromEDEM();
	void LoadRecentFile();

	void ChangeCurrentTime(); // Change selected time point in all controls.
	void UpdateMaterialsInSystemStructure();

	void ClearSystemStructure();
	void ClearAllTimePoints(); // Delete all time points in the system structure.
	void DeleteAllBonds();
	void DeleteAllParticles();
	void DeleteAllNonConnectedParticles();
	void DeleteAllSeparateParticles();
	void SaveSnapshot();
	void SaveAsImage();
	void ExportGeometriesAsSTL();
	static void WatchOnYouTube();

	void SaveConfiguration();  // Saves current configuration to ini file.
	void LoadConfiguration();  // Loads configuration from ini file.

signals:
	void NewSystemStructureGenerated();
	void NumberOfTimePointsChanged();
};