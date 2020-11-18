/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "QtDoubleSpinBox.h"
#include <QRegExpValidator>
#include <QFocusEvent>
#include <QLineEdit>

CQtDoubleSpinBox::CQtDoubleSpinBox(QWidget* _parent)
	: QAbstractSpinBox{ _parent }
{
	Initialize();
}

CQtDoubleSpinBox::CQtDoubleSpinBox(double _value, QWidget* _parent)
	: QAbstractSpinBox{ _parent }
{
	Initialize();
	SetValue(_value);
}

void CQtDoubleSpinBox::Initialize()
{
	// set validator to limit input to floating point numbers only
	auto* validator = new QDoubleValidator{ this };
	QLocale loc{ QLocale::English }; // forces using dot '.' as a decimal separator
	loc.setNumberOptions(QLocale::RejectGroupSeparator);
	validator->setLocale(loc);
	lineEdit()->setValidator(validator);

	// connect signals
	connect(lineEdit(), &QLineEdit::textChanged,			this, &CQtDoubleSpinBox::TextChanged);
	connect(this,		&QAbstractSpinBox::editingFinished,	this, &CQtDoubleSpinBox::CheckEndEmitValueChanged);
}
void CQtDoubleSpinBox::AllowOnlyPositive(bool _flag)
{
	auto* validator = new QDoubleValidator{ this };
	if (_flag)
	{
		const auto* old = dynamic_cast<const QDoubleValidator*>(lineEdit()->validator());
		validator->setRange(0, std::numeric_limits<double>::max(), old->decimals());
		validator->setNotation(old->notation());
		validator->setLocale(old->locale());
	}
	lineEdit()->setValidator(validator);
}

void CQtDoubleSpinBox::SetValue(double _value)
{
	SetText(QString::number(_value));
}

double CQtDoubleSpinBox::GetValue() const
{
	return text().toDouble();
}

void CQtDoubleSpinBox::SetText(const QString& _text)
{
	const int pos = lineEdit()->cursorPosition();
	ValidateTextAndSet(_text);
	lineEdit()->setCursorPosition(pos);
	m_oldText = text();
}

void CQtDoubleSpinBox::ValidateTextAndSet(const QString& _text) const
{
	QString text = _text;
	int i = 0;
	if (lineEdit()->validator()->validate(text, i) != QValidator::Invalid)
		lineEdit()->setText(text);
}

void CQtDoubleSpinBox::ChangeValue(EOperation _type) const
{
	QString str = text();									// current value in text form
	int pos = lineEdit()->cursorPosition() - 1;				// position of selected symbol
	const QStringList parts = str.split(c_E);				// split into mantissa and exponent
	const bool mantissa = pos < parts[0].size();			// whether mantissa or exponent is selected
	const QString part = mantissa ? parts[0] : parts[1];	// get a selected part
	if (part.isEmpty()) return;
	const double value = part.toDouble();					// double value of a selected part
	pos += TrimZeros(str, pos);								// remove leading zeros
	if (mantissa &&															// if mantissa is selected...
		pos >= 0 && part[pos] == '0' && part.left(pos).toDouble() == 0.0 ||	// ...and all values to the left are zero...
		value == 0.0)														// ...or the whole selected value is zero...
	{
		pos += Increment(str, pos);											// ...increment it...
		switch (_type)
		{
		case EOperation::INCREASE: pos += SetPositiveSign(str, pos); break;	// ...and ensure it is positive
		case EOperation::DECREASE: pos += SetNegativeSign(str, pos); break;	// ...and ensure it is negative
		}
	}
	else if (value > 0)														// if the value is positive...
	{
		switch (_type)
		{
		case EOperation::INCREASE: pos += Increment(str, pos); break;		// ...increment it
		case EOperation::DECREASE: pos += Decrement(str, pos); break;		// ...decrement it
		}
	}
	else if (value < 0)														// if the value is negative...
	{
		switch (_type)
		{
		case EOperation::INCREASE: pos += Decrement(str, pos); break;		// ...decrement it
		case EOperation::DECREASE: pos += Increment(str, pos); break;		// ...increment it
		}
	}
	pos += TrimZeros(str, pos);				// remove leading zeros
	ValidateTextAndSet(str);				// set new value
	lineEdit()->setCursorPosition(pos + 1);	// set new cursor position
}

int CQtDoubleSpinBox::SetPositiveSign(QString& _v, int _i)
{
	while (_i >= 0)						// check each symbol from _i to the beginning of the string
	{
		if (_v[_i].toUpper() == 'E')	// either wrong position or the value is already positive
			return 0;
		if (_v[_i] == '-')				// value is negative
		{
			if (_i == 0)				// negative mantissa
			{
				_v.remove(0, 1);		// remove the minus sign
				return -1;
			}
			else						// negative exponent
			{
				_v[_i] = '+';			// add the plus sign
				return 0;
			}
		}
		_i--;
	}
	return 0;
}

int CQtDoubleSpinBox::SetNegativeSign(QString& _v, int _i)
{
	while (_i >= 0)						// check each symbol from _i to the beginning of the string
	{
		if (_v[_i].toUpper() == 'E' && _i + 1 < _v.size() && (_v[_i + 1] == '+' || _v[_i + 1] == '-')) // wrong position
			return 0;
		if (_v[_i] == '-')				// value is already negative
			return 0;
		if (_v[_i] == '+')				// positive mantissa or exponent
		{
			_v[_i] = '-';				// change to minus sign
			return 0;
		}
		if (_i == 0)					// positive mantissa
		{
			_v.insert(_i, '-');			// insert minus sign
			return 1;
		}
		if (_v[_i].toUpper() == 'E')	// positive exponent
		{
			_v.insert(_i + 1, '-');		// insert minus sign
			return 1;
		}
		_i--;
	}
	return 0;
}

int CQtDoubleSpinBox::Increment(QString& _v, int _i)
{
	if (_i < -1 || _i > _v.length())				// position out of range
		return 0;
	if (_i == -1 && (_v[0] == '+' || _v[0] == '-')) // beginning of the string before sign symbol
		return 0;
	if (_i == -1 || _v[_i] == '+' || _v[_i] == '-')	// beginning of the string or of its exponent
	{
		_v.insert(_i + 1, '1');						// add leading 1
		return 1;
	}
	if (_v[_i].toUpper() == 'E' &&
		_i + 1 < _v.size() &&
		_v[_i + 1] != '+' && _v[_i + 1] != '-')		// beginning of the exponent
	{
		_v.insert(_i + 1, '1');						// add leading 1
		return 1;
	}
	if (_v[_i] == '.' || _v[_i] == ',')				// decimal separator
		return Increment(_v, _i - 1);				// skip it
	if (_v[_i] >= '0' && _v[_i] <= '8')				// digit (0 - 8)
	{
		_v[_i].unicode() += 1;						// increment the digit
		return 0;
	}
	if (_v[_i] == '9')								// digit (9)
	{
		_v[_i] = '0';								// set the digit to 0 and increment previous position
		return Increment(_v, _i - 1);
	}
	return 0;
}

int CQtDoubleSpinBox::Decrement(QString& _v, int _i)
{
	if (_i < 0 || _i > _v.length())					// position out of range
		return 0;
	if (_v[_i] == '.' || _v[_i] == ',')				// decimal separator
		return Decrement(_v, _i - 1);				// skip it
	if (_v[_i] >= '1' && _v[_i] <= '9')				// digit (1 - 9)
	{
		_v[_i].unicode() -= 1;						// decrement the digit
		return 0;
	}
	if (_v[_i] == '0')								// digit (9)
	{
		_v[_i] = '9';								// set the digit to 0 and decrement previous position
		return Decrement(_v, _i - 1);
	}
	return 0;
}

int CQtDoubleSpinBox::TrimZeros(QString& _v, int _i) const
{
	// removes leading zeroes starting from _v[_i]
	const auto Trim = [](QString& _str, int _pos, int _initPos)
	{
		if (_str[_pos].unicode() == '+' || _str[_pos].unicode() == '-') // skip sign
			++_pos;
		int cnt = 0; // number of removed symbols
		while (_pos < _str.size() && _str[_pos].unicode() == '0' && _pos + 1 < _str.size() && _str[_pos + 1].isDigit())
		{
			_str.remove(_pos, 1);
			if (_pos < _initPos - cnt) // do not consider symbols after cursor position
				cnt++;
		}
		return cnt;
	};

	if (_v.size() < 1) return 0;	// string is too short
	int cnt = 0;					// number of removed symbols
	// trim mantissa
	if (_v[0].unicode() != '.' && _v[0].unicode() != ',')
		cnt += Trim(_v, 0, _i);
	// trim exponent
	const int index = _v.indexOf(c_E);
	if (index != -1)
		cnt += Trim(_v, index + 1, _i - cnt);
	return -cnt;
}

QAbstractSpinBox::StepEnabled CQtDoubleSpinBox::stepEnabled() const
{
	return StepUpEnabled | StepDownEnabled;
}

void CQtDoubleSpinBox::keyPressEvent(QKeyEvent* _event)
{
	if (_event->key() == Qt::Key_Enter)
	{
		// discard selection and put the cursor back
		const int pos = lineEdit()->cursorPosition();
		QAbstractSpinBox::keyPressEvent(_event);
		lineEdit()->setCursorPosition(pos);
	}
	else
		QAbstractSpinBox::keyPressEvent(_event);
}

void CQtDoubleSpinBox::CheckEndEmitValueChanged()
{
	if (text() == m_oldText) return;

	m_oldText = text();
	emit ValueChanged();
}

void CQtDoubleSpinBox::stepBy(int _steps)
{
	if (_steps > 0)
		for (int i = 0; i < _steps; ++i)
			ChangeValue(EOperation::INCREASE);
	else if (_steps < 0)
		for (int i = 0; i > _steps; --i)
			ChangeValue(EOperation::DECREASE);
	CheckEndEmitValueChanged();
}
