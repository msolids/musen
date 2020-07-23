/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "MusenGUI.h"
#include "BuildVersion.h"
#include <QtWidgets/QApplication>

//#define TUHH_DOMAIN_MEMBER

int main(int argc, char *argv[])
{
#ifdef _WIN32
	SetProcessDPIAware(); // deals with some high-DPI issues. maybe
#endif

	srand(time(nullptr));

	QApplication app(argc, argv);

	qRegisterMetaTypeStreamOperators<QList<int>>("QList<int>");

	QSurfaceFormat format;
	format.setDepthBufferSize(24);

	format.setVersion(3, 3);
	format.setProfile(QSurfaceFormat::CoreProfile);

	//format.setVersion(2, 0);
	//format.setProfile(QSurfaceFormat::CompatibilityProfile);

	QSurfaceFormat::setDefaultFormat(format);

	const QString stylePath = "/styles/musen_style1.qss";
	const QFileInfo styleInfo("." + stylePath);
	const QString styleFullPath = styleInfo.exists() && styleInfo.isFile() ? "." + stylePath : QCoreApplication::applicationDirPath() + stylePath;
	QFile styleFile(styleFullPath);
	styleFile.open(QFile::ReadOnly);
	const QString StyleSheet = QLatin1String(styleFile.readAll());
	app.setStyleSheet(StyleSheet);

#ifdef TUHH_DOMAIN_MEMBER
	const QSettings settings("HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Services\\Tcpip\\Parameters", QSettings::NativeFormat);
	const QString sKey = settings.value("NV Domain").toString();
	if (sKey != "tu-harburg.de")
	{
		QMessageBox::critical(nullptr, "Error", "This Software is licensed to run at the TUHH only.");
		return 0;
	}
#endif

	MusenGUI Musen(QString::fromStdString(CURRENT_BUILD_VERSION));
	QStringList args = app.arguments();
	if (args.size() > 1)
		Musen.LoadFromFile(args[1]);
	Musen.show();

	return app.exec();
}
