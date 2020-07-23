/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include "GeneralMUSENDialog.h"
#include "ui_FileConverterTab.h"
#include <QDialog>
#include <QTimer>
#include "SystemStructure.h"
#include "FileConverter.h"

// Thread
class CFileConverterThread : public QObject
{
	Q_OBJECT

public:
	CFileConverter* m_pFileConverter;

public:
	CFileConverterThread(CFileConverter* _pFileMerger, QObject *parent = 0);
	~CFileConverterThread();

public slots:
	void StartConvertation();

signals:
	void finished();
};

// Tab
class CFileConverterTab : public CMusenDialog
{
	Q_OBJECT

private:
	Ui::CFileConverterTab  ui;
	CFileConverter*        m_pFileConverter;
	QThread*               m_pQTThread;
	CFileConverterThread*  m_pFileConverterThread;
	QTimer				   m_UpdateTimer;

public:
	CFileConverterTab(QWidget *parent = Q_NULLPTR);
	~CFileConverterTab();

private:
	void InitializeConnections();
	void UpdateProgressInfo();

public slots:
	void StartConversion(const std::string _sFileName);

private slots:
	void ConvertingFinished();
};