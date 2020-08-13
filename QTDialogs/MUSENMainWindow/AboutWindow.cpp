#include "AboutWindow.h"
#include "MUSENVersion.h"
#include <QDateTime>
#include <QFileDialog>
#include <QTextStream>

CAboutWindow::CAboutWindow(const QString& _buildVersion, QWidget* _parent)
	: QDialog(_parent)
{
	ui.setupUi(this);

	m_headerProgramName   = "MUSEN";
	m_headerTeamName      = "MUSEN Development Team";
	m_headerContactPerson = "Maksym Dosta";
	m_headerContactEmail  = "dosta@tuhh.de";
	m_headerUpdatesLink   = "https://cloud.tuhh.de/index.php/s/HFpwk4NWdpo3QSf";

	m_mainDevelopers  = { "Maksym Dosta", "Vasyl Skorych" };
	m_otherDevelopers = { "Nicolas Depta", "Serhii Dranyshnikov", "Kolja Jarolin", "Ihor Khlopov", "Vadim Yanushkevich" };

	m_libraries = {
		{ "Qt 5.12.3", "http://www.qt.io/", "Copyright 2019, The Qt Company", "LGPLv3 License", "https://doc.qt.io/qt-5/lgpl.html" },
		{ "Protocol Buffers v3.9.1", "https://developers.google.com/protocol-buffers/", "Copyright 2019, Google Inc.", "BSD License", "https://github.com/protocolbuffers/protobuf/blob/master/LICENSE" },
		{ "zlib 1.2.11", "http://www.zlib.net/", "Copyright 2017, Jean-loup Gailly and Mark Adler", "zlib License", "https://www.zlib.net/zlib_license.html" },
		{ "CUDA 10.0", "https://developer.nvidia.com/cuda-zone", "Copyright 2019, Nvidia Corporation", "NVIDIA License", "https://developer.download.nvidia.com/compute/cuda/10.0/Prod/docs/sidebar/EULA.pdf" },
		{ "Inno Setup 6.0.2", "http://www.jrsoftware.org/isinfo.php", "Copyright 2019, Jordan Russell", "Modified BSD license", "http://www.jrsoftware.org/files/is/license.txt" } };

	SetHeaderText(_buildVersion);
	SetLicense();
	SetContributors();
	SetThirdParties();
	InitializeConnections();
}

void CAboutWindow::InitializeConnections() const
{
	connect(ui.buttonBox, &QDialogButtonBox::accepted, this, &CAboutWindow::accept);
}

void CAboutWindow::SetHeaderText(const QString& _buildVersion) const
{
	const QString description("<b>" + m_headerProgramName + "</b><br/>"
		"<br/>"
		"Version " + MUSEN_VERSION_STR + "<br/>"
		"Build " + _buildVersion + "<br/>"
		"<br/>"
		"Copyright " + QString::number(QDate::currentDate().year()) + ", " + m_headerTeamName + "<br/>"
		"<br/>"
		"Contact: <a href=\"mailto:"+ m_headerContactEmail + R"("><span style=" text - decoration: underline; color:#0000ff; ">)" + m_headerContactPerson + "</span></a><br/>"
		"<br/>"
		"Check for updates <a href=\"" + m_headerUpdatesLink + R"("><span style=" text-decoration: underline; color:#0000ff;">here</span></a><br/>)"
	);

	ui.labelText->setText(description);
}

void CAboutWindow::SetLicense() const
{
	QFile file(":/QT_GUI/License");
	file.open(QIODevice::ReadOnly);
	ui.textBrowserLicense->append(QTextStream{ &file }.readAll());
	QTextCursor textCursor = ui.textBrowserLicense->textCursor();
	textCursor.movePosition(QTextCursor::Start, QTextCursor::MoveAnchor, 1);
	ui.textBrowserLicense->setTextCursor(textCursor);
	file.close();
}

void CAboutWindow::SetContributors() const
{
	const auto StyledString = [&](const QString& _s)
	{
		return R"(<p style="margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;"><span style="font-size:9pt;">)" + _s + "</span></p>";
	};

	QString text;
	text.append(StyledString("Development Team:"));
	for (const auto& name : m_mainDevelopers)
		text.append(StyledString("- " + name));
	text.append(StyledString("<br />"));
	text.append(StyledString("Other contributors:"));
	for (const auto& name : m_otherDevelopers)
		text.append(StyledString("- " + name));

	ui.textBrowserDevelopers->setHtml(text);
	QTextCursor textCursor = ui.textBrowserDevelopers->textCursor();
	textCursor.movePosition(QTextCursor::Start, QTextCursor::MoveAnchor, 1);
	ui.textBrowserDevelopers->setTextCursor(textCursor);
}

void CAboutWindow::SetThirdParties() const
{
	QString text;
	for (const auto& lib : m_libraries)
		text.append(
			"<p style=\"margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">"
			"<a href=\"" + lib.link + "\">"
			"<span style=\"text-decoration: underline; color:#0000ff;\">" + lib.name + "</span></a><br />"
			+ lib.text + "<br />"
			"<a href=\"" + lib.licenseLink + "\">"
			"<span style=\" text-decoration: underline; color:#0000ff;\">" + lib.licenseName + "</span></a></p>");

	ui.textBrowserLibraries->setHtml(text);
	QTextCursor textCursor = ui.textBrowserLibraries->textCursor();
	textCursor.movePosition(QTextCursor::Start, QTextCursor::MoveAnchor, 1);
	ui.textBrowserLibraries->setTextCursor(textCursor);
}
