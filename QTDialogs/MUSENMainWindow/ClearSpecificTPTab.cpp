/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ClearSpecificTPTab.h"
#include "qtOperations.h"
#include <QKeyEvent>
#include <QMessageBox>
#include <QThread>

// Thread
CClearSpecificThread::CClearSpecificThread(CClearSpecificTimePoints* _pClearSpecificTimePoints, QObject* parent /*= 0*/) : QObject(parent)
{
	m_pClearSpecificTimePoints = _pClearSpecificTimePoints;
}

CClearSpecificThread::~CClearSpecificThread()
{
}

void CClearSpecificThread::StartRemoving()
{
	m_pClearSpecificTimePoints->Remove();
	emit finished();
}

// Tab
CClearSpecificTPTab::CClearSpecificTPTab(QWidget *parent): CMusenDialog(parent)
{
	ui.setupUi(this);

	dTimeStep = 0;
	m_nExpectedTimePoints = 0;

	m_vAllTimePoints.clear();
	m_vIndexesOfSelectedTPs.clear();

	m_pClearSpecificTimePoints = nullptr;;
	m_ClearSpecificThread = nullptr;
	m_pQTThread = nullptr;

	ui.listWidgetTimePoints->setSelectionMode(QAbstractItemView::ExtendedSelection);
	ui.progressBar->setValue(0);

	// Regular expression for floating point numbers
	QRegExp regExp("^[0-9]*[.]?[0-9]+(?:[eE][-+]?[0-9]+)?$");
	// Set regular expression for limitation of input in qlineEdits
	ui.lineEditTimeStep->setValidator(new QRegExpValidator(regExp, this));

	InitializeConnections();
}

CClearSpecificTPTab::~CClearSpecificTPTab()
{
}

void CClearSpecificTPTab::InitializeConnections()
{
	// buttons
	connect(ui.pushButtonRemove, &QPushButton::clicked, this, &CClearSpecificTPTab::RemoveButtonPressed);
	connect(ui.pushButtonCancel, &QPushButton::clicked, this, &CClearSpecificTPTab::reject);

	// tool buttons
	connect(ui.toolButtonSelectTimeStep, &QPushButton::clicked, this, &CClearSpecificTPTab::SelectTimeStep);
	connect(ui.toolButtonSelectEachSecond, &QPushButton::clicked, this, &CClearSpecificTPTab::SelectEachSecondTimePoint);

	// list
	connect(ui.listWidgetTimePoints, &QListWidget::itemSelectionChanged, this, &CClearSpecificTPTab::ChangeSelection);

	// timer
	connect(&m_UpdateTimer, &QTimer::timeout, this, &CClearSpecificTPTab::UpdateProgressInfo);
}

void CClearSpecificTPTab::keyPressEvent(QKeyEvent* event)
{
	switch (event->key())
	{
	case Qt::Key_F1:		OpenHelpFile();					break;
	case Qt::Key_Delete: 	RemoveButtonPressed();			break;
	default:				QDialog::keyPressEvent(event);	break;
	}
}


void CClearSpecificTPTab::UpdateWholeView()
{
	ui.listWidgetTimePoints->clear();
	m_vIndexesOfSelectedTPs.clear();
	m_vAllTimePoints = m_pSystemStructure->GetAllTimePoints();
	ui.labelNumberOfTP->setText(QString::number(m_vAllTimePoints.size()));
	ui.labelNumberOfTPAfter->setText(QString::number(m_vAllTimePoints.size()));
	ui.labelEndTime->setText(QString::number(m_pSystemStructure->GetMaxTime()));
	// fill list of time points
	for (auto i = 0; i < m_vAllTimePoints.size(); i++)
		ui.listWidgetTimePoints->addItem(QString::number(m_vAllTimePoints[i]));
}

void CClearSpecificTPTab::ChangeSelection()
{
	ui.labelNumberOfTPAfter->setText(QString::number(m_vAllTimePoints.size()-ui.listWidgetTimePoints->selectedItems().size()));
}

void CClearSpecificTPTab::SelectTimeStep()
{
	if (ui.lineEditTimeStep->text().isEmpty() || ui.lineEditTimeStep->text().toDouble() == 0)
	{
		QMessageBox::warning(this, "Error", "Value of time step for selection has to be set.");
		return;
	}

	if (m_vAllTimePoints.size() < 2)
	{
		QMessageBox::warning(this, "Error", "For this selection it is necessary to have at least two time points.");
		return;
	}

	double dTimeStep = ui.lineEditTimeStep->text().toDouble();	     // time step for cleaning
	double dTimeEnd = m_vAllTimePoints[m_vAllTimePoints.size() - 1]; // end time

	if (dTimeStep >= dTimeEnd)
	{
		QMessageBox::warning(this, "Error", "Value of time step for selection has to be smaller than end time value.");
		return;
	}

	SetDesiredTimeStepSectionEnabled(false);
	UpdateWholeView();
	ui.labelStatus->setText("Applying time step for selection...");

	size_t nCurrIter = 0;
	double dCurrTime = dTimeStep;

	while (dCurrTime <= dTimeEnd)
	{
		for (auto i = nCurrIter; i < m_vAllTimePoints.size(); i++)
		{
			if (m_vAllTimePoints[i] == dCurrTime || (m_vAllTimePoints[i] < dCurrTime && m_vAllTimePoints[i + 1] > dCurrTime))
			{
				nCurrIter = i + 1;
				break;
			}
			else
				if (i != 0) ui.listWidgetTimePoints->item(static_cast<int>(i))->setSelected(true);

		}
		dCurrTime = dCurrTime + dTimeStep;
	}

	ui.labelNumberOfTPAfter->setText(QString::number(m_vAllTimePoints.size() - ui.listWidgetTimePoints->selectedItems().size()));
	ui.labelStatus->setText("");
	SetDesiredTimeStepSectionEnabled(true);
}

void CClearSpecificTPTab::SelectEachSecondTimePoint()
{
	if (m_vAllTimePoints.size() < 2)
	{
		QMessageBox::warning(this, "Error", "For this selection it is necessary to have at least two time points.");
		return;
	}

	SetDesiredTimeStepSectionEnabled(false);
	UpdateWholeView();
	ui.labelStatus->setText("Selection of each second time point...");

	for (auto i = 1; i < m_vAllTimePoints.size(); i = i + 2)
		ui.listWidgetTimePoints->item(static_cast<int>(i))->setSelected(true);

	ui.labelNumberOfTPAfter->setText(QString::number(m_vAllTimePoints.size() - ui.listWidgetTimePoints->selectedItems().size()));
	ui.labelStatus->setText("");
	SetDesiredTimeStepSectionEnabled(true);
}

void CClearSpecificTPTab::SetDesiredTimeStepSectionEnabled(bool _bEnabled)
{
	ui.toolButtonSelectTimeStep->setEnabled(_bEnabled);
	ui.toolButtonSelectEachSecond->setEnabled(_bEnabled);
	ui.lineEditTimeStep->setEnabled(_bEnabled);
	ui.listWidgetTimePoints->setEnabled(_bEnabled);

	ui.pushButtonCancel->setEnabled(_bEnabled);
	ui.pushButtonRemove->setEnabled(_bEnabled);
}

void CClearSpecificTPTab::UpdateProgressInfo()
{
	ui.progressBar->setValue(m_pClearSpecificTimePoints->GetProgressPercent());
	ui.labelStatus->setText(ss2qs(m_pClearSpecificTimePoints->GetProgressMessage()));
}

void CClearSpecificTPTab::RemoveButtonPressed()
{
	// start button was pressed
	m_vIndexesOfSelectedTPs.clear();

	QList<QListWidgetItem*> listOfSelectedTP = ui.listWidgetTimePoints->selectedItems();
	m_vIndexesOfSelectedTPs.reserve(listOfSelectedTP.size());
	for (int i = 0; i < listOfSelectedTP.size(); i++)
		m_vIndexesOfSelectedTPs.push_back(ui.listWidgetTimePoints->row(listOfSelectedTP[i]));

	if (m_vIndexesOfSelectedTPs.size() == 0)
	{
		QMessageBox::warning(this, "Error", "At least one time point has to be selected for removing.");
		return;
	}

	if (std::find(m_vIndexesOfSelectedTPs.begin(), m_vIndexesOfSelectedTPs.end(), 0) != m_vIndexesOfSelectedTPs.end())
	{
		QMessageBox::warning(this, "Error", "Zero time point cannot be removed.");
		return;
	}

	if (QMessageBox::question(this, "Confirmation", "Selected time points will be removed from current scene. Continue?", QMessageBox::Yes | QMessageBox::No) == QMessageBox::No)
		return;


	SetDesiredTimeStepSectionEnabled(false);

	m_nExpectedTimePoints = ui.labelNumberOfTPAfter->text().toInt();

	m_pClearSpecificTimePoints = new CClearSpecificTimePoints(m_pSystemStructure);
	std::sort(m_vIndexesOfSelectedTPs.begin(), m_vIndexesOfSelectedTPs.end());
	m_pClearSpecificTimePoints->SetIndexesOfSelectedTPs(m_vIndexesOfSelectedTPs);
	m_ClearSpecificThread = new CClearSpecificThread(m_pClearSpecificTimePoints);
	m_pQTThread = new QThread();
	connect(m_pQTThread, SIGNAL(started()), m_ClearSpecificThread, SLOT(StartRemoving()));
	connect(m_ClearSpecificThread, SIGNAL(finished()), this, SLOT(RemovingFinished()));
	m_ClearSpecificThread->moveToThread(m_pQTThread);
	m_pQTThread->start();
	m_UpdateTimer.start(100); // start update timer
}

void CClearSpecificTPTab::RemovingFinished()
{
	m_UpdateTimer.stop();

	if (m_pQTThread != NULL)
	{
		m_pQTThread->exit();
		m_pQTThread = nullptr;
		delete m_ClearSpecificThread;
		m_ClearSpecificThread = nullptr;
	}

	if (!m_pClearSpecificTimePoints->GetErrorMessage().empty())
	{
		QMessageBox::warning(this, "Error", tr("%1").arg(ss2qs(m_pClearSpecificTimePoints->GetErrorMessage())));
		ui.labelStatus->setText("Error of the removing");
	}
	else
	{
		ui.progressBar->setValue(100);
		ui.labelStatus->setText("Removing finished");
		UpdateWholeView();
	}

	if (ui.labelNumberOfTP->text().toInt() != m_nExpectedTimePoints)
	{
		QMessageBox::warning(this, "Warning", "Some selected time points have not be removed.");
	}

	SetDesiredTimeStepSectionEnabled(true);
}
