/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "NewObjectPanel.h"

CNewObjectPanel::CNewObjectPanel(QWidget *parent)
	: QWidget(parent)
{
	ui.setupUi(this);

	InitializeConnections();
}

CNewObjectPanel::~CNewObjectPanel()
{
}

void CNewObjectPanel::SetPointers(CSystemStructure* _systemStructure, CMaterialsDatabase* _materialsDB, CUnitConvertor* _unitConvertor)
{
	m_systemStructure = _systemStructure;
	m_materialsDB     = _materialsDB;
	m_converer        = _unitConvertor;
}

void CNewObjectPanel::InitializeConnections() const
{
	connect(ui.radioParticle,	&QRadioButton::toggled, this, &CNewObjectPanel::ObjectTypeChanged);
	connect(ui.radioBond,		&QRadioButton::toggled, this, &CNewObjectPanel::ObjectTypeChanged);
	connect(ui.buttonAdd,		&QPushButton::clicked,	this, &CNewObjectPanel::AddObject);
}

void CNewObjectPanel::UpdateWholeView() const
{
	UpdateLabels();
	UpdateMaterials();
	UpdateVisibility();
}

QString CNewObjectPanel::StatusMessage() const
{
	return m_statusMessage;
}

void CNewObjectPanel::UpdateLabels() const
{
	ui.labelDiameter->setText("Diameter [" + QString::fromStdString(m_converer->GetSelectedUnit(EUnitType::PARTICLE_DIAMETER)) + "]");
	ui.labelContactDiameter->setText("Contact diameter [" + QString::fromStdString(m_converer->GetSelectedUnit(EUnitType::PARTICLE_DIAMETER)) + "]");
	ui.labelX->setText("X [" + QString::fromStdString(m_converer->GetSelectedUnit(EUnitType::LENGTH)) + "]");
	ui.labelY->setText("Y [" + QString::fromStdString(m_converer->GetSelectedUnit(EUnitType::LENGTH)) + "]");
	ui.labelZ->setText("Z [" + QString::fromStdString(m_converer->GetSelectedUnit(EUnitType::LENGTH)) + "]");
}

void CNewObjectPanel::UpdateMaterials() const
{
	if (!m_materialsDB) return;

	const int iOld = ui.comboMaterial->currentIndex();
	ui.comboMaterial->clear();
	for (size_t i = 0; i < m_materialsDB->CompoundsNumber(); ++i)
		ui.comboMaterial->insertItem(static_cast<int>(i), QString::fromStdString(m_materialsDB->GetCompoundName(i)));
	ui.comboMaterial->setCurrentIndex(iOld < static_cast<int>(m_materialsDB->CompoundsNumber()) ? iOld : -1);
}

void CNewObjectPanel::UpdateVisibility() const
{
	if (!m_systemStructure) return;

	ui.labelContactDiameter->setVisible(ObjectType() == EObjectTypes::PARTICLES && m_systemStructure->IsContactRadiusEnabled());
	ui.lineEditContactDiameter->setVisible(ObjectType() == EObjectTypes::PARTICLES && m_systemStructure->IsContactRadiusEnabled());
	ui.labelX->setVisible(ObjectType() == EObjectTypes::PARTICLES);
	ui.lineEditX->setVisible(ObjectType() == EObjectTypes::PARTICLES);
	ui.labelY->setVisible(ObjectType() == EObjectTypes::PARTICLES);
	ui.lineEditY->setVisible(ObjectType() == EObjectTypes::PARTICLES);
	ui.labelZ->setVisible(ObjectType() == EObjectTypes::PARTICLES);
	ui.lineEditZ->setVisible(ObjectType() == EObjectTypes::PARTICLES);
	ui.checkboxOverlap->setVisible(ObjectType() == EObjectTypes::PARTICLES);

	ui.labelID1->setVisible(ObjectType() == EObjectTypes::SOLID_BONDS);
	ui.lineEditID1->setVisible(ObjectType() == EObjectTypes::SOLID_BONDS);
	ui.labelID2->setVisible(ObjectType() == EObjectTypes::SOLID_BONDS);
	ui.lineEditID2->setVisible(ObjectType() == EObjectTypes::SOLID_BONDS);
}

void CNewObjectPanel::ObjectTypeChanged(bool _checked) const
{
	if (!_checked) return;
	UpdateVisibility();
}

void CNewObjectPanel::AddObject()
{
	m_statusMessage = CheckData();
	if (!m_statusMessage.isEmpty())
	{
		emit ObjectAdded();
		return;
	}

	unsigned newObjectID{};

	const size_t materialID = ui.comboMaterial->currentIndex();

	switch (ObjectType())
	{
	case EObjectTypes::PARTICLES:
	{
		const double diameter = m_converer->GetValueSI(EUnitType::PARTICLE_DIAMETER, ui.lineEditDiameter->text().toDouble());
		const double contactDiameter = m_systemStructure->IsContactRadiusEnabled()
			? m_converer->GetValueSI(EUnitType::PARTICLE_DIAMETER, ui.lineEditContactDiameter->text().toDouble())
			: diameter;
		const CVector3 coord{
			m_converer->GetValueSI(EUnitType::LENGTH, ui.lineEditX->text().toDouble()),
			m_converer->GetValueSI(EUnitType::LENGTH, ui.lineEditY->text().toDouble()),
			m_converer->GetValueSI(EUnitType::LENGTH, ui.lineEditZ->text().toDouble())
		};

		auto* part = dynamic_cast<CSphere*>(m_systemStructure->AddObject(SPHERE));
		part->SetStartActivityTime(0.0);
		part->SetEndActivityTime(DEFAULT_ACTIVITY_END);
		part->SetRadius(diameter / 2.0);
		part->SetContactRadius(contactDiameter / 2.0);
		part->SetCoordinates(0, coord);
		part->SetOrientation(0, CQuaternion{ 0, 1, 0, 0 });
		part->SetCompound(m_materialsDB->GetCompound(materialID));
		newObjectID = part->m_lObjectID;
		break;
	}
	case EObjectTypes::SOLID_BONDS:
	{
		const unsigned idL = ui.lineEditID1->text().toUInt();
		const unsigned idR = ui.lineEditID2->text().toUInt();
		const double diameter = m_converer->GetValueSI(EUnitType::PARTICLE_DIAMETER, ui.lineEditDiameter->text().toDouble());
		auto* sphere1 = dynamic_cast<CSphere*>(m_systemStructure->GetObjectByIndex(idL));
		auto* sphere2 = dynamic_cast<CSphere*>(m_systemStructure->GetObjectByIndex(idR));

		auto* bond = dynamic_cast<CSolidBond*>(m_systemStructure->AddObject(SOLID_BOND));
		bond->SetStartActivityTime(0.0);
		bond->SetEndActivityTime(DEFAULT_ACTIVITY_END);
		bond->SetDiameter(diameter);
		bond->m_nLeftObjectID = idL;
		bond->m_nRightObjectID = idR;
		bond->SetInitialLength(Length(GetSolidBond(sphere1->GetCoordinates(0), sphere2->GetCoordinates(0), m_systemStructure->GetPBC())));
		bond->SetCompound(m_materialsDB->GetCompound(materialID));
		newObjectID = bond->m_lObjectID;
		break;
	}
	}

	m_statusMessage = "Object successfully added. ID = " + QString::number(newObjectID);

	emit ObjectAdded();
}

QString CNewObjectPanel::CheckData() const
{
	const size_t materialID = ui.comboMaterial->currentIndex();
	if (materialID >= m_materialsDB->CompoundsNumber())
		return "Error: Wrong material selected!";

	switch (ObjectType())
	{
	case EObjectTypes::PARTICLES:
	{
		const bool allowOverlap = ui.checkboxOverlap->isChecked();
		if (!allowOverlap) // check on possible overlap with other particles
		{
			const double newContactDiameter = m_systemStructure->IsContactRadiusEnabled()
				? m_converer->GetValueSI(EUnitType::PARTICLE_DIAMETER, ui.lineEditContactDiameter->text().toDouble())
				: m_converer->GetValueSI(EUnitType::PARTICLE_DIAMETER, ui.lineEditDiameter->text().toDouble());
			const CVector3 newCoord{
				m_converer->GetValueSI(EUnitType::LENGTH, ui.lineEditX->text().toDouble()),
				m_converer->GetValueSI(EUnitType::LENGTH, ui.lineEditY->text().toDouble()),
				m_converer->GetValueSI(EUnitType::LENGTH, ui.lineEditZ->text().toDouble())
			};

			for (const auto& p : m_systemStructure->GetAllSpheres(0))
				if (Length(p->GetCoordinates(0), newCoord) < p->GetContactRadius() + newContactDiameter / 2.0)
					return "Error: Overlap with particle ID = " + QString::number(p->m_lObjectID);
		}
		break;
	}
	case EObjectTypes::SOLID_BONDS:
	{
		const unsigned idL = ui.lineEditID1->text().toUInt();
		const unsigned idR = ui.lineEditID2->text().toUInt();
		const double diameter = m_converer->GetValueSI(EUnitType::PARTICLE_DIAMETER, ui.lineEditDiameter->text().toDouble());

		if (idL == idR)
			return "Error: Same ID of contact particles!";
		if (!m_systemStructure->GetObjectByIndex(idL))
			return "Error: No particle with ID = " + QString::number(idL) + "!";
		if (!m_systemStructure->GetObjectByIndex(idR))
			return "Error: No particle with ID = " + QString::number(idR) + "!";
		if (m_systemStructure->GetObjectByIndex(idL)->GetObjectType() != SPHERE)
			return "Error: Object with ID = " + QString::number(idL) + " is not a particle!";
		if (m_systemStructure->GetObjectByIndex(idR)->GetObjectType() != SPHERE)
			return "Error: Object with ID = " + QString::number(idR) + " is not a particle!";

		auto* sphere1 = dynamic_cast<CSphere*>(m_systemStructure->GetObjectByIndex(idL));
		auto* sphere2 = dynamic_cast<CSphere*>(m_systemStructure->GetObjectByIndex(idR));
		if (diameter > sphere1->GetRadius() * 2 || diameter > sphere2->GetRadius() * 2)
			return "Error: Bond diameter is larger than the particles' diameter!";

		break;
	}
	}

	return {};
}

CNewObjectPanel::EObjectTypes CNewObjectPanel::ObjectType() const
{
	if (ui.radioParticle->isChecked())  return EObjectTypes::PARTICLES;
	if (ui.radioBond->isChecked())		return EObjectTypes::SOLID_BONDS;
	return EObjectTypes::PARTICLES;
}
