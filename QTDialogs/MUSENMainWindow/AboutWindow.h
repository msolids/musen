/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ui_AboutWindow.h"

class CAboutWindow : public QDialog
{
	Q_OBJECT

	struct S3rdParty
	{
		QString name;
		QString link;
		QString text;
		QString licenseName;
		QString licenseLink;
	};

	Ui::CAboutWindow ui{};

	QString m_headerProgramName;
	QString m_headerTeamName;
	QString m_headerContactPerson;
	QString m_headerContactEmail;
	QString m_headerUpdatesLink;

	std::vector<QString> m_mainDevelopers;
	std::vector<QString> m_otherDevelopers;

	std::vector<S3rdParty> m_libraries;

public:
	CAboutWindow(const QString& _buildVersion, QWidget* _parent = nullptr);

private:
	void InitializeConnections() const;

	void SetHeaderText(const QString& _buildVersion) const;
	void SetLicense() const;
	void SetContributors() const;
	void SetThirdParties() const;
};
