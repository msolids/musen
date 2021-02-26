/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "BondsGenerator.h"

bool CBondsGenerator::IsDataCorrect() const
{
	if (m_generators.empty())
	{
		m_errorMessage = "No generators defined";
		return false;
	}

	for (const auto& g : m_generators)
	{
		if (!g.isActive) continue;

		if (g.compoundKey.empty())
		{
			m_errorMessage = "Error in class '" + g.name + "': material not specified";
			return false;
		}
		if (!m_pSystemStructure->m_MaterialDatabase.GetCompound(g.compoundKey))
		{
			m_errorMessage = "Error in class '" + g.name + "': the selected compound was not found in the database";
			return false;
		}
		if (g.maxDistance < g.minDistance)
		{
			m_errorMessage = "Error in class '" + g.name + "': max distance is less than min distance";
			return false;
		}
	}

	m_errorMessage.clear();
	return true;
}

void CBondsGenerator::Clear()
{
	m_generators.clear();
}

void CBondsGenerator::StartGeneration()
{
	struct SBond
	{
		unsigned leftID{};
		unsigned rightID{};
		double diameter{};
		double length{};
	};

	struct SParticle
	{
		unsigned id{};
		double radius{};
		CVector3 position;
		std::string compound;
	};

	m_status = ERunningStatus::RUNNING;
	if (!m_pSystemStructure) return;

	// gather information about all existing particles
	std::vector<SParticle> particles; // all particles on the scene
	for(const auto& p : m_pSystemStructure->GetAllSpheres(0))
		particles.push_back(SParticle{ p->m_lObjectID, p->GetRadius(), p->GetCoordinates(0), p->GetCompoundKey() });

	for (auto& generator : m_generators)
	{
		if (m_status == ERunningStatus::TO_BE_STOPPED) break;
		if(!generator.isActive) continue;

		// gather information about all particles that are already connected by bonds to avoid their reconnection with new bonds
		std::vector<std::vector<unsigned>> connectedParticles(m_pSystemStructure->GetTotalObjectsCount()); // all connected particles
		for (const auto& b : m_pSystemStructure->GetAllSolidBonds(0))
		{
			connectedParticles[b->m_nLeftObjectID].push_back(b->m_nRightObjectID);
			connectedParticles[b->m_nRightObjectID].push_back(b->m_nLeftObjectID);
		}

		CContactCalculator calculator;
		std::vector<unsigned> ID1, ID2;
		std::vector<double> overlaps;
		for (unsigned i = 0; i < particles.size(); ++i)
			calculator.AddParticle(i, particles[i].position, particles[i].radius + generator.maxDistance / 2);
		calculator.GetAllOverlaps(ID1, ID2, overlaps, m_pSystemStructure->GetPBC());

		std::vector<SBond> bonds(overlaps.size());
		ParallelFor(overlaps.size(), [&](size_t i)
		{
			const size_t i1 = ID1[i];
			const size_t i2 = ID2[i];
			const double r1 = particles[i1].radius;
			const double r2 = particles[i2].radius;

			// check the distance condition of the solid bond
			const double surfDistance = generator.maxDistance - overlaps[i];
			if (surfDistance <= generator.maxDistance && surfDistance >= generator.minDistance)
			{
				const double diameter = std::min({ generator.diameter, 2 * r1, 2 * r2 }); // limitation of the bond diameter

				// check compounds
				if (generator.isCompoundSpecific)
				{
					const bool bCmp1InList1 = VectorContains(generator.compoundsLists.first,  particles[i1].compound);
					const bool bCmp2InList1 = VectorContains(generator.compoundsLists.first,  particles[i2].compound);
					const bool bCmp1InList2 = VectorContains(generator.compoundsLists.second, particles[i1].compound);
					const bool bCmp2InList2 = VectorContains(generator.compoundsLists.second, particles[i2].compound);
					if (!(bCmp1InList1 && bCmp2InList2 || bCmp2InList1 && bCmp1InList2)) return;
				}

				// check overlay
				if (!generator.isOverlayAllowed)
				{
					size_t j;
					for (j = 0; j < connectedParticles[particles[i1].id].size(); ++j)
						if (connectedParticles[particles[i1].id][j] == particles[i2].id)
							break;
					if (j < connectedParticles[particles[i1].id].size()) return;
				}

				bonds[i] = SBond{ particles[i1].id, particles[i2].id, diameter, surfDistance + r1 + r2 };
			}
		});

		// remove empty bonds
		bonds.erase(std::remove_if(bonds.begin(), bonds.end(), [](const SBond& b) { return b.leftID == b.rightID; }), bonds.end());

		std::vector<CPhysicalObject*> objects = m_pSystemStructure->AddSeveralObjects(SOLID_BOND, bonds.size());
		for (size_t i = 0; i < objects.size(); ++i)
		{
			auto* bond = dynamic_cast<CSolidBond*>(objects[i]);
			bond->SetDiameter(bonds[i].diameter);
			bond->SetInitialLength(bonds[i].length);
			bond->m_nLeftObjectID = bonds[i].leftID;
			bond->m_nRightObjectID = bonds[i].rightID;
			bond->SetCompoundKey(generator.compoundKey);
			bond->SetTangentialOverlap(0.0, CVector3{ 0.0 });
			generator.completeness = i * 100. / bonds.size();
			generator.generatedBonds++;
		}
		generator.completeness = 100.;
	}
	m_pSystemStructure->UpdateAllObjectsCompoundsProperties();
	m_status = ERunningStatus::IDLE;
}

void CBondsGenerator::LoadConfiguration()
{
	const ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	m_generators.clear();
	for (int i = 0; i < protoMessage.bonds_generator().generators_size(); ++i)
	{
		const ProtoBondsGenerator& gen = protoMessage.bonds_generator().generators(i);
		m_generators.emplace_back();
		SBondClass& bond = m_generators.back();
		bond.name = gen.name();
		bond.compoundKey = gen.material_key();
		bond.minDistance = gen.min_length();
		bond.maxDistance = gen.max_length();
		bond.diameter = gen.property1();
		bond.isOverlayAllowed = gen.allow_overlap();
		bond.isCompoundSpecific = gen.compound_specific();
		for (int j = 0; j < gen.partner_compounds1_size(); ++j)
			bond.compoundsLists.first.push_back(gen.partner_compounds1(j));
		for (int j = 0; j < gen.partner_compounds2_size(); ++j)
			bond.compoundsLists.second.push_back(gen.partner_compounds2(j));
		bond.isActive = gen.activity();
	}
}

void CBondsGenerator::SaveConfiguration()
{
	ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	protoMessage.mutable_bonds_generator()->clear_generators();
	for (const auto& generator : m_generators)
	{
		ProtoBondsGenerator* gen = protoMessage.mutable_bonds_generator()->add_generators();
		gen->set_name(generator.name);
		gen->set_material_key(generator.compoundKey);
		gen->set_min_length(generator.minDistance);
		gen->set_max_length(generator.maxDistance);
		gen->set_property1(generator.diameter);
		gen->set_allow_overlap(generator.isOverlayAllowed);
		gen->set_compound_specific(generator.isCompoundSpecific);
		for (const auto& key : generator.compoundsLists.first)
			gen->add_partner_compounds1(key);
		for (const auto& key : generator.compoundsLists.second)
			gen->add_partner_compounds2(key);
		gen->set_activity(generator.isActive);
	}
}
