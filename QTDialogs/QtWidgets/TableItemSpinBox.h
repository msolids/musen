/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "QtDoubleSpinBox.h"
#include <QLabel>
#include <QStackedWidget>

namespace DoubleSpinnerWrapper
{
	enum EWidgetType
	{
		VIEWER = 0,
		EDITOR = 1
	};

	// Filters key events of a parent CQtTable widget to pass them to a corresponding CItemSpinBox.
	class CTableEventFilter : public QObject
	{
		Q_OBJECT

		QWidget* m_parent;				// Pointer to parent CItemSpinBox widget.
		EWidgetType m_widget{ VIEWER };	// Widget currently active in CItemSpinBox.

	public:
		CTableEventFilter(QWidget* _parent);

		void SetWidgetType(EWidgetType _type);	// Sets type of the widget currently active in CItemSpinBox.

	protected:
		bool eventFilter(QObject* _obj, QEvent* _event) override;

	signals:
		void TextEntered(const QString& _text);	// Is emitted when any symbol key is pressed.
		void EditTriggered();					// Is emitted when any edit combination hit: F2 or any symbol pressed.
		void EditDiscarded();					// Is triggered if edit should be finished.
		void SelectRequested();					// Is emitted if the current text must be selected.
	};
}

class CQtTable;

// An inter-layer class to build CLineEditSpinner into CQtTable. Uses QLabel to mimic QTableWidgetItem behavior and CQtDoubleSpinner as editor.
class CTableItemSpinBox : public QStackedWidget
{
	Q_OBJECT

	CQtTable* m_table;												// Pointer to a parent CQtTable.
	DoubleSpinnerWrapper::CTableEventFilter* m_tableEventFilter;	// Event filter for parent CQtTable.

	QLabel* m_label;					// Viewer.
	CQtDoubleSpinBox* m_editor;			// Editor.

public:
	CTableItemSpinBox(double _value, CQtTable* _parent);
	CTableItemSpinBox(const CTableItemSpinBox& _other) = delete;
	CTableItemSpinBox(CTableItemSpinBox&& _other) noexcept = delete;
	CTableItemSpinBox& operator=(const CTableItemSpinBox& _other) = delete;
	CTableItemSpinBox& operator=(CTableItemSpinBox&& _other) noexcept = delete;
	~CTableItemSpinBox();

	void focusOutEvent(QFocusEvent* _event) override;
	void mouseDoubleClickEvent(QMouseEvent *_event) override;

	void AllowOnlyPositive(bool _flag) const;	// Whether to allow negative values.

	void SetValue(double _value) const;	// Sets new value.
	double GetValue() const;			// Returns current value.

private:
	void ActivateViewer();	// Activates viewer widget.
	void ActivateEditor();	// Activates editor widget.

	void OnTextChanged();					// Reacts on text change in editor.
	void DiscardEdit();						// Reacts on the escape operation in viewer mode.
	void EnterText(const QString& _text);	// Sets text to widgets.
	void SelectAllText() const;				// Selects all text in editor.

signals:
	void ValueChanged();	// Is emitted when value changes.
	void EditingFinished();	// Is called on editing is finished.
};