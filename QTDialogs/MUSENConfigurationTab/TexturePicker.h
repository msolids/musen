/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include <QDialog>
#include "ui_TexturePicker.h"

class CTexturePicker : public QDialog
{
	Q_OBJECT

public:
	CTexturePicker(QWidget *parent = Q_NULLPTR);

private:
	Ui::texturePicker ui;

public:
	QString SelectedTexture() const;
};
