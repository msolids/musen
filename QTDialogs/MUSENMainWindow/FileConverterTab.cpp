/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "FileConverterTab.h"
#include "qtOperations.h"
#include <QThread>

// Thread
CFileConverterThread::CFileConverterThread(CFileConverter *_pFileConverter, QObject *parent /*= 0*/) : QObject(parent)
{
	m_pFileConverter = _pFileConverter;
}

CFileConverterThread::~CFileConverterThread()
{
}

void CFileConverterThread::StartConvertation()
{
	m_pFileConverter->ConvertFileToNewFormat();
	emit finished();
}

// Tab
CFileConverterTab::CFileConverterTab(QWidget *parent) : CMusenDialog(parent)
{
	ui.setupUi(this);

	m_pQTThread = nullptr;
	m_pFileConverter = nullptr;
	m_pFileConverterThread = nullptr;

	ui.progressBar->setValue(0);
	ui.labelStatus->setText("");

	InitializeConnections();
}

CFileConverterTab::~CFileConverterTab()
{
}

void CFileConverterTab::InitializeConnections()
{
	// timer
	connect(&m_UpdateTimer, &QTimer::timeout, this, &CFileConverterTab::UpdateProgressInfo);
}


void CFileConverterTab::UpdateProgressInfo()
{
	ui.progressBar->setValue(m_pFileConverter->GetProgressPercent());
	ui.labelStatus->setText(ss2qs(m_pFileConverter->GetProgressMessage()));
}

void CFileConverterTab::StartConversion(const std::string _sFileName)
{
	// start converting
	m_pFileConverter = new CFileConverter(_sFileName);
	m_pFileConverterThread = new CFileConverterThread(m_pFileConverter);
	m_pQTThread = new QThread();
	connect(m_pQTThread, SIGNAL(started()), m_pFileConverterThread, SLOT(StartConvertation()));
	connect(m_pFileConverterThread, SIGNAL(finished()), this, SLOT(ConvertingFinished()));
	m_pFileConverterThread->moveToThread(m_pQTThread);
	m_pQTThread->start();
	m_UpdateTimer.start(100);

	this->setWindowFlags(Qt::Window | Qt::WindowTitleHint | Qt::CustomizeWindowHint);
	this->exec();
}

void CFileConverterTab::ConvertingFinished()
{
	ui.labelStatus->setText("Transformation finished");
	ui.progressBar->setValue(100);
	m_UpdateTimer.stop();
	if (m_pQTThread != NULL)
	{
		m_pQTThread->exit();
		m_pQTThread = nullptr;
		delete m_pFileConverterThread;
		m_pFileConverterThread = nullptr;
	}
	this->accept();
}