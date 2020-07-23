/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "TexturePicker.h"

CTexturePicker::CTexturePicker(QWidget *parent)
	: QDialog(parent)
{
	ui.setupUi(this);

	for (unsigned i = 0;; ++i)
	{
		const QString name = tr(":/QT_GUI/Pictures/SphereTexture%1.png").arg(i);
		QIcon icon(name);
		if (icon.availableSizes().count() == 0) break;
		QListWidgetItem* item = new QListWidgetItem(icon, QString::number(i + 1));
		item->setData(Qt::UserRole, name);
		ui.listWidget->addItem(item);
	}

	connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &CTexturePicker::accept);
	connect(ui.buttonBox, &QDialogButtonBox::rejected, this, &CTexturePicker::reject);
}

QString CTexturePicker::SelectedTexture() const
{
	if (QListWidgetItem* item = ui.listWidget->currentItem())
		return item->data(Qt::UserRole).toString();
	return {};
}
