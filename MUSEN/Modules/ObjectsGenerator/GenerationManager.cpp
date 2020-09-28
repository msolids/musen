#include "GenerationManager.h"

CGenerationManager::CGenerationManager() : CMusenComponent()
{
	m_pAgglomDB = nullptr;
}

CGenerationManager::~CGenerationManager()
{
	DeleteAllGenerators();
}

void CGenerationManager::DeleteAllGenerators()
{
	for ( unsigned i = 0; i< m_vGenerators.size(); i++ )
		delete m_vGenerators[ i ];
	m_vGenerators.clear();
}

CObjectsGenerator* CGenerationManager::GetGenerator( size_t _nIndex )
{
	if ( _nIndex >= m_vGenerators.size() ) return NULL;
	return m_vGenerators[ _nIndex ];
}

const CObjectsGenerator* CGenerationManager::GetGenerator(size_t _nIndex) const
{
	if (_nIndex >= m_vGenerators.size()) return NULL;
	return m_vGenerators[_nIndex];
}

void CGenerationManager::CreateNewGenerator()
{
	m_vGenerators.push_back(new CObjectsGenerator(m_pAgglomDB, &m_pSystemStructure->m_MaterialDatabase));
	m_vGenerators.back()->m_sName = "ObjectsGenerator " + std::to_string(m_vGenerators.size());
}

void CGenerationManager::DeleteGenerator( size_t _nIndex )
{
	if ( _nIndex >=  m_vGenerators.size() ) return;
	delete m_vGenerators[ _nIndex ];
	m_vGenerators.erase( m_vGenerators.begin() + _nIndex );
}

void CGenerationManager::LoadConfiguration()
{
	const ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	DeleteAllGenerators();
	for (int i = 0; i < protoMessage.objects_generator().generators_size(); ++i)
	{
		const ProtoObjectsGenerator& protoGen = protoMessage.objects_generator().generators(i);
		CreateNewGenerator();
		CObjectsGenerator* gen = m_vGenerators.back();
		const int version = protoGen.version();
		gen->m_sName = protoGen.name();
		gen->m_sVolumeKey = protoGen.volume_key();
		gen->m_bRandomVelocity = protoGen.random_velocity_flag();
		gen->m_vObjInitVel = Proto2Val(protoGen.init_velocity());
		gen->m_dVelMagnitude = protoGen.velocity_magnitude();
		gen->m_sMixtureKey = protoGen.mixture_key();
		gen->m_bGenerateMixture = (protoGen.obj_type() == ProtoObjectsGenerator_ObjectsType_cSphere);
		gen->m_sAgglomerateKey = protoGen.aggl_key();
		gen->m_dStartGenerationTime = protoGen.start_generation_time();
		gen->m_dEndGenerationTime = protoGen.end_generation_time();
		gen->m_dUpdateStep = protoGen.update_time_step();
		gen->m_bInsideGeometries = protoGen.inside_geometries();
		for (int j = 0; j < protoGen.aggl_part_materials_alias_size(); ++j)
			gen->m_partMaterials[protoGen.aggl_part_materials_alias(j)] = protoGen.aggl_part_materials_key(j);
		for (int j = 0; j < protoGen.aggl_bond_materials_alias_size(); ++j)
			gen->m_bondMaterials[protoGen.aggl_bond_materials_alias(j)] = protoGen.aggl_bond_materials_key(j);
		m_vGenerators[i]->m_dAgglomerateScaleFactor = protoGen.scaling_factor();
		m_vGenerators[i]->m_bActive = protoGen.activity();
		if (version == 0) // compatibility with older versions
		{
			m_vGenerators[i]->m_rateType = CObjectsGenerator::ERateType::GENERATION_RATE;
			m_vGenerators[i]->m_rateValue = protoGen.generation_rate();
			m_vGenerators[i]->m_maxIterations = 30;
		}
		else
		{
			m_vGenerators[i]->m_rateType = static_cast<CObjectsGenerator::ERateType>(protoGen.rate_type());
			m_vGenerators[i]->m_rateValue = protoGen.rate_value();
			m_vGenerators[i]->m_maxIterations = protoGen.max_iterations();
		}
	}
}

void CGenerationManager::SaveConfiguration()
{
	ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	protoMessage.mutable_objects_generator()->clear_generators();
	for (const auto& gen : m_vGenerators)
	{
		ProtoObjectsGenerator* protoGen = protoMessage.mutable_objects_generator()->add_generators();
		protoGen->set_version(1);
		protoGen->set_name(gen->m_sName);
		protoGen->set_volume_key(gen->m_sVolumeKey);
		protoGen->set_random_velocity_flag(gen->m_bRandomVelocity);
		Val2Proto(protoGen->mutable_init_velocity(), gen->m_vObjInitVel);
		protoGen->set_velocity_magnitude(gen->m_dVelMagnitude);
		protoGen->set_mixture_key(gen->m_sMixtureKey);
		if (gen->m_bGenerateMixture)
			protoGen->set_obj_type(ProtoObjectsGenerator_ObjectsType_cSphere);
		else
			protoGen->set_obj_type(ProtoObjectsGenerator_ObjectsType_cAgglomerate);
		protoGen->set_aggl_key(gen->m_sAgglomerateKey);
		protoGen->set_start_generation_time(gen->m_dStartGenerationTime);
		protoGen->set_end_generation_time(gen->m_dEndGenerationTime);
		protoGen->set_update_time_step(gen->m_dUpdateStep);
		protoGen->set_inside_geometries(gen->m_bInsideGeometries);
		for (auto& material : gen->m_partMaterials)
		{
			protoGen->add_aggl_part_materials_alias(material.first);
			protoGen->add_aggl_part_materials_key(material.second);
		}
		for (auto& material : gen->m_bondMaterials)
		{
			protoGen->add_aggl_bond_materials_alias(material.first);
			protoGen->add_aggl_bond_materials_key(material.second);
		}
		protoGen->set_scaling_factor(gen->m_dAgglomerateScaleFactor);
		protoGen->set_activity(gen->m_bActive);
		protoGen->set_max_iterations(gen->m_maxIterations);
		protoGen->set_rate_type(E2I(gen->m_rateType));
		protoGen->set_rate_value(gen->m_rateValue);
	}
}

std::string CGenerationManager::IsDataCorrect() const
{
	for (size_t i = 0; i < m_vGenerators.size(); ++i)
	{
		if ( !m_vGenerators[ i ]->m_bActive ) continue; // consider only active objects generators

		CAnalysisVolume* pGenVolume = m_pSystemStructure->AnalysisVolume( m_vGenerators[ i ]->m_sVolumeKey );
		if (!pGenVolume)
			return "No generation volume was specified for dynamic generator - " + m_vGenerators[ i ]->m_sName;

		if (m_vGenerators[i]->m_bGenerateMixture)
		{
			std::string sError = m_pSystemStructure->m_MaterialDatabase.IsMixtureCorrect(m_vGenerators[i]->m_sMixtureKey);
			if ( !sError.empty() )
				return "Dynamic generator " + m_vGenerators[i]->m_sName + ": "+ sError;
		}
		else
		{
			SAgglomerate *pAgglo = m_pAgglomDB->GetAgglomerate(m_vGenerators[i]->m_sAgglomerateKey);
			if (!pAgglo)
				return "No correct agglomerate has been specified for dynamic generator - " + m_vGenerators[i]->m_sName;
			for (size_t j = 0; j < pAgglo->vParticles.size(); ++j)
				if (!m_pSystemStructure->m_MaterialDatabase.GetCompound(m_vGenerators[i]->m_partMaterials[pAgglo->vParticles[j].sCompoundAlias]))
					return "No correct particle materials have been specified for dynamic generator - " + m_vGenerators[i]->m_sName;
			for (size_t j = 0; j < pAgglo->vBonds.size(); ++j)
				if (!m_pSystemStructure->m_MaterialDatabase.GetCompound(m_vGenerators[i]->m_bondMaterials[pAgglo->vBonds[j].sCompoundAlias]))
					return "No correct bond materials have been specified for dynamic generator - " + m_vGenerators[i]->m_sName;
		}
	}
	return "";
}

size_t CGenerationManager::GenerateObjects(double _dTime, CSimplifiedScene& _Scene)
{
	size_t objectsCounter = 0;
	for (auto& generator : m_vGenerators)
		if (generator->m_bActive)
			objectsCounter += generator->Generate(_dTime, m_pSystemStructure, _Scene);
	return objectsCounter;
}

bool CGenerationManager::IsNeedToBeGenerated(double _dTime) const
{
	for (size_t i = 0; i < m_vGenerators.size(); ++i)
		if (m_vGenerators[i]->IsNeedToBeGenerated(_dTime))
			return true;
	return false;
}

void CGenerationManager::Initialize()
{
	for ( unsigned i = 0; i < m_vGenerators.size(); i++ )
		m_vGenerators[ i ]->Initialize();
}

size_t CGenerationManager::GetGeneratorsNumber() const
{
	return m_vGenerators.size();
}

size_t CGenerationManager::GetActiveGeneratorsNumber() const
{
	size_t nNumber = 0;
	for (size_t i = 0; i < m_vGenerators.size(); i++)
		if (m_vGenerators[i]->m_bActive)
			nNumber++;
	return nNumber;
}

void CGenerationManager::SetAgglomeratesDatabase(CAgglomeratesDatabase* _pDatabase)
{
	m_pAgglomDB = _pDatabase;
}