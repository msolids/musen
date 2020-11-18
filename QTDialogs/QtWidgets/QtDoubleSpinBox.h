/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <QAbstractSpinBox>
#include <QRegularExpression>

// Line editor for double values with a position-dependent spinner option.
class CQtDoubleSpinBox : public QAbstractSpinBox
{
	Q_OBJECT

	enum class EOperation { INCREASE, DECREASE };

	const QRegularExpression c_E{ "[(e|E)]" };	// Symbol of the exponent part.
	QString m_oldText;							// Is needed to know when to emit ValueChanged signal.

public:
	CQtDoubleSpinBox(QWidget* _parent);
	CQtDoubleSpinBox(double _value, QWidget* _parent);

	void AllowOnlyPositive(bool _flag);			// Whether to allow negative values.

	void SetValue(double _value);		// Sets new value.
	double GetValue() const;			// Returns current value.
	void SetText(const QString& _text);	// Sets the text if it passes validation.

private:
	void Initialize();	// Initial setup.

	void ValidateTextAndSet(const QString& _text) const;	// Sets the text if it passes validation.
	void ChangeValue(EOperation _type) const;				// Position-dependent increase/decrease of the value.

	static int SetPositiveSign(QString& _v, int _i);	// Ensures that the value (mantissa or exponent, depending on _i) is positive. Returns number of added(+)/removed(-) symbols.
	static int SetNegativeSign(QString& _v, int _i);	// Ensures that the value (mantissa or exponent, depending on _i) is negative. Returns number of added(+)/removed(-) symbols.

	static int Increment(QString& _v, int _i); // Recursively increments the string value starting from _v[_i] position. Returns number of added(+)/removed(-) symbols.
	static int Decrement(QString& _v, int _i); // Recursively decrements the string value starting from _v[_i] position. Returns number of added(+)/removed(-) symbols.

	int TrimZeros(QString& _v, int _i) const;	// Removes leading zeroes from mantissa and exponent. Returns number of symbols removed(-) after position _i.

	void stepBy(int _steps) override;				// Is called whenever the user triggers a step.
	StepEnabled	stepEnabled() const override;		// Determines whether stepping up and down is legal.
	void keyPressEvent(QKeyEvent* _event) override;

	void CheckEndEmitValueChanged();	// Emits the signal if the text had been changes since last emission.

signals:
	void TextChanged();
	void ValueChanged();
};


