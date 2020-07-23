/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GeneralMUSENDialog.h"
#include "ui_FileMergerTab.h"
#include <QDialog>
#include <QTimer>
#include "SystemStructure.h"
#include "FileMerger.h"

// Thread
class CFileMergerThread : public QObject
{
	Q_OBJECT

public:
	CFileMerger* m_pFileMerger;

public:
	CFileMergerThread(CFileMerger* _pFileMerger, QObject *parent = 0);
	~CFileMergerThread();

public slots:
	void StartMerging();
	void StopMerging();

signals:
	void finished();
};

// Tab
class CFileMergerTab : public CMusenDialog
{
	Q_OBJECT

private:
	QString m_sResultFile;					 // path to the output file
	QString m_sLastSelectedFile;			 // path to the last file which was chosen
	QString m_SLastMergedFile;				 // path to the last merged file
	bool    m_bMergingStarted;				 // flag of merge starting
	bool    m_bLoadMergedFile;				 // flag of loading file after merging
	std::vector<std::string> m_vListOfFiles; // list of paths to files for merging


	CFileMerger*           m_pFileMerger;
	QThread*               m_pQTThread;
	CFileMergerThread*	   m_pFileMergerThread;
	QTimer				   m_UpdateTimer;

public:
	CFileMergerTab(QWidget *parent = Q_NULLPTR);
	~CFileMergerTab();

private:
	Ui::CFileMergerTab ui;

	// Connects qt objects to slots
	void InitializeConnections();
	// Updates vector with files names for merging
	void UpdateFileList();
	// Updates GUI of file merger
	void UpdateGUI(bool _bIsMergingNotStarted);
	// Updates progress bar and status description
	void UpdateProgressInfo();

private slots:
	void SetOutputFileName();
	void EditOutputFileName();
	void SetFlagOfLoadMergedFile();
	void AddFileToList();
	void RemoveFileFromList();
	void RemoveAllFilesFromList();
	void MoveFileDown();
	void MoveFileUp();
	void MergeFilesPressed();
	void MergingFinished();

signals:
	void LoadMergedSystemStrcuture(const QString& _filePath);
};