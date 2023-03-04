/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "FileMergerTab.h"
#include "qtOperations.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QThread>

// Thread
CFileMergerThread::CFileMergerThread(CFileMerger *_pFileMerger, QObject *parent /*= 0*/) : QObject(parent)
{
	m_pFileMerger = _pFileMerger;
}

CFileMergerThread::~CFileMergerThread()
{
}

void CFileMergerThread::StartMerging()
{
	m_pFileMerger->Merge();
	emit finished();
}

void CFileMergerThread::StopMerging()
{
	if (m_pFileMerger->GetCurrentStatus() != ERunningStatus::IDLE)
		m_pFileMerger->SetCurrentStatus(ERunningStatus::TO_BE_STOPPED);
}

// Tab
CFileMergerTab::CFileMergerTab(QWidget *parent) : CMusenDialog(parent)
{
	ui.setupUi(this);

	m_pQTThread = nullptr;
	m_pFileMerger = nullptr;
	m_pFileMergerThread = nullptr;

	m_sResultFile.clear();
	m_sLastSelectedFile.clear();
	m_SLastMergedFile.clear();
	m_vListOfFiles.clear();

	m_bLoadMergedFile = false;
	m_bMergingStarted = false;
	ui.progressBar->setValue(0);
	ui.labelStatus->setText("");

	InitializeConnections();
	m_sHelpFileName = "Users Guide/Merge files.pdf";
}

CFileMergerTab::~CFileMergerTab()
{
}

void CFileMergerTab::InitializeConnections()
{
	// buttons
	connect(ui.pushButtonMerge, &QPushButton::clicked, this, &CFileMergerTab::MergeFilesPressed);
	connect(ui.pushButtonCancel, &QPushButton::clicked, this, &CFileMergerTab::reject);

	// buttons for handling list of files for merging:
	connect(ui.pushButtonAdd, &QPushButton::clicked, this, &CFileMergerTab::AddFileToList);
	connect(ui.pushButtonRemove, &QPushButton::clicked, this, &CFileMergerTab::RemoveFileFromList);
	connect(ui.pushButtonRemoveAll, &QPushButton::clicked, this, &CFileMergerTab::RemoveAllFilesFromList);

	connect(ui.pushButtonMoveUp, &QPushButton::clicked, this, &CFileMergerTab::MoveFileUp);
	connect(ui.pushButtonMoveDown, &QPushButton::clicked, this, &CFileMergerTab::MoveFileDown);

	// tool button for choose of output file
	connect(ui.toolButtonSetOuputtFile, &QPushButton::clicked, this, &CFileMergerTab::SetOutputFileName);

	// line edit
	connect(ui.lineEditOutputFilePath, &QLineEdit::editingFinished, this, &CFileMergerTab::EditOutputFileName);

	// checkbox
	connect(ui.checkBoxLoadFile, &QCheckBox::stateChanged, this, &CFileMergerTab::SetFlagOfLoadMergedFile);

	// timer
	connect(&m_UpdateTimer, &QTimer::timeout, this, &CFileMergerTab::UpdateProgressInfo);
}

void CFileMergerTab::UpdateFileList()
{
	m_vListOfFiles.clear();
	for (int i = 0; i < ui.listWidget->count(); ++i)
	{
		QListWidgetItem* item = ui.listWidget->item(i);
		m_vListOfFiles.push_back(qs2ss(item->text()));
	}
}

void CFileMergerTab::AddFileToList()
{
	QStringList listOfFile = QFileDialog::getOpenFileNames(this, tr("Path to the file for merging..."), m_sLastSelectedFile, tr("MUSEN files (*.mdem);;All files (*.*);;"));
	if (!listOfFile.isEmpty())
	{
		m_sLastSelectedFile = listOfFile.last();
		listOfFile.sort();
		ui.listWidget->addItems(listOfFile);
		UpdateFileList();
	}
}

void CFileMergerTab::RemoveFileFromList()
{
	if (!ui.listWidget->selectedItems().isEmpty())
	{
		if (ui.listWidget->count() > 0)
		{
			qDeleteAll(ui.listWidget->selectedItems());
			UpdateFileList();
		}
		else
		{
			QMessageBox::warning(this, "Remove file from list", "List of files for merging is already empty.");
		}
	}
	else
	{
		QMessageBox::warning(this, "Remove file from list", "File for removing is not selected.");
	}
}

void CFileMergerTab::RemoveAllFilesFromList()
{
	if (ui.listWidget->count() > 0)
	{
		ui.listWidget->clear();
		UpdateFileList();
	}
	else
	{
		QMessageBox::warning(this, "Remove all files from list", "List of files for merging is already empty.");
	}
}
void CFileMergerTab::MoveFileDown()
{
	QListWidgetItem *current = ui.listWidget->currentItem();
	int currIndex = ui.listWidget->row(current);

	QListWidgetItem *next = ui.listWidget->item(ui.listWidget->row(current) + 1);
	int nextIndex = ui.listWidget->row(next);

	QListWidgetItem *temp = ui.listWidget->takeItem(nextIndex);
	ui.listWidget->insertItem(currIndex, temp);
	ui.listWidget->insertItem(nextIndex, current);

	UpdateFileList();
}

void CFileMergerTab::MoveFileUp()
{
	QListWidgetItem *current = ui.listWidget->currentItem();
	int currIndex = ui.listWidget->row(current);

	QListWidgetItem *prev = ui.listWidget->item(ui.listWidget->row(current) - 1);
	int prevIndex = ui.listWidget->row(prev);

	QListWidgetItem *temp = ui.listWidget->takeItem(prevIndex);
	ui.listWidget->insertItem(prevIndex, current);
	ui.listWidget->insertItem(currIndex, temp);

	UpdateFileList();
}

void CFileMergerTab::SetFlagOfLoadMergedFile()
{
	m_bLoadMergedFile = ui.checkBoxLoadFile->isChecked();
}

void CFileMergerTab::SetOutputFileName()
{
	m_sResultFile = QFileDialog::getSaveFileName(this, tr("Path to the result merged file..."), m_sLastSelectedFile, tr("MUSEN files (*.mdem);;All files (*.*);;"));
	if (!m_sResultFile.isEmpty())
	{
		ui.lineEditOutputFilePath->setText(m_sResultFile);
	}
	m_sLastSelectedFile = m_sResultFile;
}

void CFileMergerTab::EditOutputFileName()
{
	m_sResultFile = ui.lineEditOutputFilePath->text();
}

void CFileMergerTab::UpdateProgressInfo()
{
	ui.progressBar->setValue(m_pFileMerger->GetProgressPercent());
	ui.labelStatus->setText(ss2qs(m_pFileMerger->GetProgressMessage()));
}

void CFileMergerTab::MergeFilesPressed()
{
	if (!m_bMergingStarted)
	{
		// start button was pressed
		if (m_vListOfFiles.size() < 2)
		{
			QMessageBox::warning(this, "Error", "List of files has to be contained at least two files for merging.");
			return;
		}
		m_sResultFile = ui.lineEditOutputFilePath->text();
		if (m_sResultFile.isEmpty())
		{
			QMessageBox::warning(this, "Error", "Results file is not selected.");
			return;
		}
		if (qs2ss(m_sResultFile) == m_pSystemStructure->GetFileName())
		{
			QMessageBox::warning(this, "Error", tr("File %1 is in use. Choose another result file or close the file that's open in the program.").arg(m_sResultFile));
			return;
		}
		if (m_sResultFile == m_SLastMergedFile)
		{
			if (QMessageBox::question(this, "Confirmation", tr("File %1 will be overwritten during the merging. Continue?").arg(m_sResultFile), QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
			return;
		}

		m_bMergingStarted = true;
		UpdateGUI(m_bMergingStarted);

		m_pFileMerger = new CFileMerger(m_pSystemStructure);
		m_pFileMerger->SetCurrentStatus(ERunningStatus::RUNNING);

		m_pFileMerger->SetListOfFiles(m_vListOfFiles);
		m_pFileMerger->SetResultFile(qs2ss(m_sResultFile));
		m_pFileMerger->SetFlagOfLoadingMergedFile(m_bLoadMergedFile);

		m_pFileMergerThread = new CFileMergerThread(m_pFileMerger);
		m_pQTThread = new QThread();
		connect(m_pQTThread, SIGNAL(started()), m_pFileMergerThread, SLOT(StartMerging()));
		connect(m_pFileMergerThread, SIGNAL(finished()), this, SLOT(MergingFinished()));
		m_pFileMergerThread->moveToThread(m_pQTThread);
		m_pQTThread->start();
		m_UpdateTimer.start(100); // start update timer
	}
	else
	{   // stop button was pressed
		m_bMergingStarted = false;
		m_pFileMergerThread->StopMerging();
		while (m_pFileMerger->GetCurrentStatus() != ERunningStatus::IDLE)
			m_pQTThread->wait(100);
		m_UpdateTimer.stop();
		ui.progressBar->setValue(0);
		QMessageBox::information(this, "Information", tr("Around %1% of data has been merged.").arg((unsigned)m_pFileMerger->GetProgressPercent()));
		MergingFinished();
	}
}

void CFileMergerTab::MergingFinished()
{
	m_bMergingStarted = false;
	UpdateGUI(m_bMergingStarted);

	m_UpdateTimer.stop();

	if (m_pQTThread != NULL)
	{
		m_pQTThread->exit();
		m_pQTThread = nullptr;
		delete m_pFileMergerThread;
		m_pFileMergerThread = nullptr;
	}

	if (ui.checkBoxLoadFile->isChecked())
		emit LoadMergedSystemStrcuture(m_sResultFile);

	if (m_pFileMerger->GetCurrentStatus() == ERunningStatus::RUNNING) // stop button wasn't pressed
	{
		m_pFileMerger->SetCurrentStatus(ERunningStatus::IDLE);
		if (!m_pFileMerger->GetErrorMessage().empty())
		{
			QMessageBox::warning(this, "Error", tr("%1").arg(ss2qs(m_pFileMerger->GetErrorMessage())));
			ui.labelStatus->setText("Merging error");
		}
		else ui.progressBar->setValue(100);
	}
	m_SLastMergedFile = ui.lineEditOutputFilePath->text();
}

void CFileMergerTab::UpdateGUI(bool _bIsMergingStarted)
{
	ui.listWidget->setDisabled(_bIsMergingStarted);

	ui.pushButtonAdd->setDisabled(_bIsMergingStarted);
	ui.pushButtonRemove->setDisabled(_bIsMergingStarted);
	ui.pushButtonMoveDown->setDisabled(_bIsMergingStarted);
	ui.pushButtonMoveUp->setDisabled(_bIsMergingStarted);
	ui.pushButtonRemoveAll->setDisabled(_bIsMergingStarted);
	ui.pushButtonCancel->setDisabled(_bIsMergingStarted);

	ui.lineEditOutputFilePath->setDisabled(_bIsMergingStarted);
	ui.toolButtonSetOuputtFile->setDisabled(_bIsMergingStarted);

	ui.checkBoxLoadFile->setDisabled(_bIsMergingStarted);

	if (_bIsMergingStarted)
	{
		ui.pushButtonMerge->setText("Stop");
		ui.pushButtonMerge->setIcon(QIcon(":/MusenGUI/Pictures/stop.png"));
		if (ui.checkBoxLoadFile->isChecked())
		{
			this->hide();
			this->setWindowModality(Qt::ApplicationModal);
			this->show();
		}
	}
	else
	{
		ui.pushButtonMerge->setText("Merge");
		ui.pushButtonMerge->setIcon(QIcon(":/MusenGUI/Pictures/play.png"));
		ui.labelStatus->setText("Merging finished");
		if (ui.checkBoxLoadFile->isChecked())
		{
			this->hide();
			this->setWindowModality(Qt::NonModal);
			this->show();
		}
	}
}