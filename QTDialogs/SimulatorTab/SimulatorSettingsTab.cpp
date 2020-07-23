/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "SimulatorSettingsTab.h"

CSimulatorSettingsTab::CSimulatorSettingsTab(CSimulatorManager* _pSimulatorManager, QWidget *parent)
	: CMusenDialog(parent),
	m_pSimulatorManager(_pSimulatorManager)
{
	ui.setupUi(this);

	// regular expression for floating point numbers
	const QRegExp regExpFloat("^[0-9]*[.]?[0-9]+(?:[eE][-+]?[0-9]+)?$");
	// set regular expression for limitation of input in QLineEdits
	ui.lineEditVerletCoeff->setValidator(new QRegExpValidator(regExpFloat, this));

	connect(ui.listCPU,					&QListWidget::itemChanged,   this, &CSimulatorSettingsTab::ThreadPoolChanged);
	connect(ui.buttonBox,				&QDialogButtonBox::accepted, this, &CSimulatorSettingsTab::AcceptChanges);
	connect(ui.checkBoxStopBrokenBonds, &QCheckBox::stateChanged, [=](int _state) { ui.lineEditBrokenBonds->setEnabled(_state == Qt::CheckState::Checked); });
}

void CSimulatorSettingsTab::UpdateWholeView()
{
	m_bAvoidSignal = true;
	ui.spinBoxCellsNumber->setValue(m_pSimulatorManager->GetSimulatorPtr()->GetMaxCells());
	ui.lineEditVerletCoeff->setText(QString::number(m_pSimulatorManager->GetSimulatorPtr()->GetVerletCoeff()));
	ui.checkBoxAutoAdjust->setChecked(m_pSimulatorManager->GetSimulatorPtr()->GetAutoAdjustFlag());
	ui.groupBoxVariableTimeStep->setChecked(m_pSimulatorManager->GetSimulatorPtr()->GetVariableTimeStep());
	ui.lineEditPartMoveLimit->setText(QString::number(m_pSimulatorManager->GetSimulatorPtr()->GetPartMoveLimit()));
	ui.lineEditTimeStepFactor->setText(QString::number(m_pSimulatorManager->GetSimulatorPtr()->GetTimeStepFactor()));

	const bool stopByBrokenBonds = VectorContains(m_pSimulatorManager->GetSimulatorPtr()->GetStopCriteria(), CBaseSimulator::EStopCriteria::BROKEN_BONDS);
	ui.checkBoxStopBrokenBonds->setChecked(stopByBrokenBonds);
	ui.lineEditBrokenBonds->setText(QString::number(m_pSimulatorManager->GetSimulatorPtr()->GetStopValues().maxBrokenBonds));

	ui.checkBoxPartPartContact->setChecked(m_pSimulatorManager->GetSimulatorPtr()->GetModelManager()->GetConnectedPPContact());

	m_bAvoidSignal = false;
	UpdateCPUList();
}

void CSimulatorSettingsTab::UpdateCPUList() const
{
	const bool isIdle = m_pSimulatorManager->GetSimulatorPtr()->GetCurrentStatus() == ERunningStatus::IDLE; // is simulator idle
	ui.listCPU->setEnabled(isIdle); // can not change if simulation is running
	if (!isIdle) return;			// ...leave now

	QSignalBlocker block(ui.listCPU);
	ui.listCPU->clear();
	const auto systemCPUs = ThreadPool::CThreadPool::GetSystemCPUList();	// threads allowed by the system
	ui.listCPU->setEnabled(!systemCPUs.empty());							// no threads were allowed by the system - something went wrong - block it
	if (systemCPUs.empty()) return;											// ... and do nothing
	const auto userCPUs = ThreadPool::CThreadPool::GetUserCPUList();		// threads allowed by user
	const int threads = static_cast<int>(std::thread::hardware_concurrency());	// total number of threads in the system
	for (int i = 0; i < threads; ++i)											// for each available thread
	{
		QListWidgetItem *item = new QListWidgetItem(tr("CPU%1").arg(i));	// create an entry
		item->setFlags(item->flags() | Qt::ItemIsUserCheckable);			// make it checkable
		if (VectorContains(systemCPUs, i))	// if allowed by the system
			if (userCPUs.empty() || !userCPUs.empty() && VectorContains(userCPUs, i))	// if allowed by user or not specified
				item->setCheckState(Qt::Checked);										// turn it on
			else																		// explicitly disallowed by user
				item->setCheckState(Qt::Unchecked);										// turn it of
		else								// if not allowed by the system
		{
			item->setCheckState(Qt::Unchecked);					// turn it off
			item->setFlags(item->flags() ^ Qt::ItemIsEnabled);	// disable for user interaction
		}

		ui.listCPU->insertItem(i, item); // add this item to the list
	}
}

void CSimulatorSettingsTab::SetCPUList()
{
	if (!m_bThreadPoolChanged) return;
	if (!ui.listCPU->isEnabled()) return;

	// gather allowed CPUs
	std::vector<int> cpuList;
	for (int i = 0; i < ui.listCPU->count(); ++i)
		if (ui.listCPU->item(i)->checkState() == Qt::Checked)
			cpuList.push_back(i);

	// restart thread pool with new parameters
	ThreadPool::CThreadPool::SetUserCPUList(cpuList);
	RestartThreadPool();

	m_bThreadPoolChanged = false;
}

void CSimulatorSettingsTab::ThreadPoolChanged()
{
	m_bThreadPoolChanged = true;
}

void CSimulatorSettingsTab::AcceptChanges()
{
	if (m_bAvoidSignal) return;
	m_pSimulatorManager->GetSimulatorPtr()->SetMaxCells(ui.spinBoxCellsNumber->value());
	m_pSimulatorManager->GetSimulatorPtr()->SetVerletCoeff(ui.lineEditVerletCoeff->text().toDouble());
	m_pSimulatorManager->GetSimulatorPtr()->SetAutoAdjustFlag(ui.checkBoxAutoAdjust->isChecked());
	m_pSimulatorManager->GetSimulatorPtr()->SetVariableTimeStep(ui.groupBoxVariableTimeStep->isChecked());
	m_pSimulatorManager->GetSimulatorPtr()->SetPartMoveLimit(ui.lineEditPartMoveLimit->text().toDouble());
	m_pSimulatorManager->GetSimulatorPtr()->SetTimeStepFactor(ui.lineEditTimeStepFactor->text().toDouble());

	std::vector<CBaseSimulator::EStopCriteria> stopCriteria;
	if (ui.checkBoxStopBrokenBonds->isChecked())
		stopCriteria.push_back(CBaseSimulator::EStopCriteria::BROKEN_BONDS);
	CBaseSimulator::SStopValues values;
	values.maxBrokenBonds = ui.lineEditBrokenBonds->text().toUInt();
	m_pSimulatorManager->GetSimulatorPtr()->SetStopCriteria(stopCriteria);
	m_pSimulatorManager->GetSimulatorPtr()->SetStopValues(values);
	m_pSimulatorManager->GetSimulatorPtr()->GetModelManager()->SetConnectedPPContact(ui.checkBoxPartPartContact->isChecked());

	SetCPUList();
	accept();
}
