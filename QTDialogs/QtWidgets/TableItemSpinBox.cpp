/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "TableItemSpinBox.h"
#include "QtSignalBlocker.h"
#include "QtTable.h"

namespace DoubleSpinnerWrapper
{
	CTableEventFilter::CTableEventFilter(QWidget* _parent)
		: m_parent{ _parent }
	{
	}

	void CTableEventFilter::SetWidgetType(EWidgetType _type)
	{
		m_widget = _type;
	}

	bool CTableEventFilter::eventFilter(QObject* _obj, QEvent* _event)
	{
		// catch editing event
		if (auto* table = dynamic_cast<CQtTable*>(_obj))				// if the event comes from a proper table
		{
			const auto pos = table->CurrentCellPos();
			if (table->cellWidget(pos.first, pos.second) == m_parent)	// if a proper CItemSpinBox widget is selected
				if (_event->type() == QEvent::KeyPress)					// if it is a key event
				{
					const auto* keyEvent = dynamic_cast<QKeyEvent*>(_event);
					// do not enter in edit mode on copy/paste event
					if (m_widget == VIEWER && (keyEvent->matches(QKeySequence::Copy) || keyEvent->matches(QKeySequence::Paste)))
						return false;
					// escape during editing pressed
					if (m_widget == EDITOR && keyEvent->key() == Qt::Key_Escape)
					{
						emit EditDiscarded();
						return true;
					}
					// any valid symbol pressed
					if (keyEvent->key() >= Qt::Key_Plus && keyEvent->key() <= Qt::Key_Period || keyEvent->key() >= Qt::Key_0 && keyEvent->key() <= Qt::Key_9)
						emit TextEntered(keyEvent->text());
					// any symbol key pressed
					if (keyEvent->key() == Qt::Key_F2 || keyEvent->key() >= Qt::Key_Space && keyEvent->key() <= Qt::Key_AsciiTilde)
						emit EditTriggered();
					// F2 pressed
					if (keyEvent->key() == Qt::Key_F2)
						emit SelectRequested();
					// discard event if necessary
					if (keyEvent->key() == Qt::Key_F2 ||															// if F2 pressed...
						keyEvent->key() >= Qt::Key_Space && keyEvent->key() <= Qt::Key_AsciiTilde ||				// ...or any symbol...
						m_widget == EDITOR && (keyEvent->key() == Qt::Key_Up || keyEvent->key() == Qt::Key_Down))	// ...or up/down with active editor...
						return true;																				// ...discard event
				}
		}

		// standard event processing
		return false;
	}
}

using namespace DoubleSpinnerWrapper;

CTableItemSpinBox::CTableItemSpinBox(double _value, CQtTable* _parent)
	: QStackedWidget{ _parent },
	m_table{ _parent },
	m_tableEventFilter{ new CTableEventFilter{ this } },
	m_label{ new QLabel{ this } },
	m_editor{ new CQtDoubleSpinBox{ this } }
{
	addWidget(m_label);
	addWidget(m_editor);

	m_label->setMargin(3);
	m_table->installEventFilter(m_tableEventFilter);

	SetValue(_value);

	connect(m_editor,			&CQtDoubleSpinBox::TextChanged,      this, &CTableItemSpinBox::OnTextChanged);
	connect(m_editor,           &CQtDoubleSpinBox::ValueChanged,     this, &CTableItemSpinBox::EditingFinished);
	connect(m_editor,           &CQtDoubleSpinBox::editingFinished,  this, &CTableItemSpinBox::ActivateViewer);

	connect(m_tableEventFilter, &CTableEventFilter::EditTriggered,   this, &CTableItemSpinBox::ActivateEditor);
	connect(m_tableEventFilter, &CTableEventFilter::EditDiscarded,   this, &CTableItemSpinBox::DiscardEdit);
	connect(m_tableEventFilter, &CTableEventFilter::TextEntered,     this, &CTableItemSpinBox::EnterText);
	connect(m_tableEventFilter, &CTableEventFilter::SelectRequested, this, &CTableItemSpinBox::SelectAllText);

	setCurrentIndex(VIEWER);
}

CTableItemSpinBox::~CTableItemSpinBox()
{
	m_table->removeEventFilter(m_tableEventFilter);
}

void CTableItemSpinBox::focusOutEvent(QFocusEvent* _event)
{
	if (_event->lostFocus())
		ActivateViewer();
	QStackedWidget::focusOutEvent(_event);
}

void CTableItemSpinBox::mouseDoubleClickEvent(QMouseEvent* _event)
{
	ActivateEditor();
	QStackedWidget::mouseDoubleClickEvent(_event);
}

void CTableItemSpinBox::AllowOnlyPositive(bool _flag) const
{
	m_editor->AllowOnlyPositive(_flag);
}

void CTableItemSpinBox::ActivateViewer()
{
	if (currentIndex() == VIEWER) return;
	m_tableEventFilter->SetWidgetType(VIEWER);
	CQtSignalBlocker blocker{ m_editor };
	m_label->setText(m_editor->text());
	setCurrentIndex(VIEWER);
	m_table->setFocus(Qt::FocusReason::OtherFocusReason);	// return focus to parent table
}

void CTableItemSpinBox::ActivateEditor()
{
	if (currentIndex() == EDITOR) return;
	m_tableEventFilter->SetWidgetType(EDITOR);
	CQtSignalBlocker blocker{ m_editor };
	m_editor->SetText(m_label->text());
	setCurrentIndex(EDITOR);
	m_editor->setFocus(Qt::FocusReason::OtherFocusReason);	// set focus to editor
}

void CTableItemSpinBox::OnTextChanged()
{
	emit ValueChanged();
	if (currentIndex() == VIEWER)
		emit EditingFinished();
}

void CTableItemSpinBox::DiscardEdit()
{
	CQtSignalBlocker blocker{ m_editor };
	m_editor->SetText(m_label->text());
	ActivateViewer();
	emit ValueChanged();
}

void CTableItemSpinBox::SetValue(double _value) const
{
	m_editor->SetValue(_value);
	m_label->setText(m_editor->text());
}

double CTableItemSpinBox::GetValue() const
{
	return m_editor->GetValue();
}

void CTableItemSpinBox::EnterText(const QString& _text)
{
	ActivateEditor();
	m_editor->SetText(_text);
}

void CTableItemSpinBox::SelectAllText() const
{
	m_editor->selectAll();
}