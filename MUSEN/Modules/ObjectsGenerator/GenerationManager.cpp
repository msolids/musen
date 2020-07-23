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
	const ProtoModulesData& _pProtoMessage = *m_pSystemStructure->GetProtoModulesData();
	DeleteAllGenerators();
	for (int i = 0; i < _pProtoMessage.objects_generator().generators_size(); ++i)
	{
		const ProtoObjectsGenerator& genProto = _pProtoMessage.objects_generator().generators(i);
		CreateNewGenerator();
		CObjectsGenerator* pGen = m_vGenerators.back();
		pGen->m_sName = genProto.name();
		pGen->m_sVolumeKey = genProto.volume_key();
		pGen->m_bRandomVelocity = genProto.random_velocity_flag();
		pGen->m_vObjInitVel = ProtoVectorToVector(genProto.init_velocity());
		pGen->m_dVelMagnitude = genProto.velocity_magnitude();
		pGen->m_sMixtureKey = genProto.mixture_key();
		pGen->m_bGenerateMixture = (genProto.obj_type() == ProtoObjectsGenerator_ObjectsType_cSphere);
		pGen->m_sAgglomerateKey = genProto.aggl_key();
		pGen->m_dStartGenerationTime = genProto.start_generation_time();
		pGen->m_dEndGenerationTime = genProto.end_generation_time();
		pGen->m_dUpdateStep = genProto.update_time_step();
		pGen->m_dGenerationRate = genProto.generation_rate();
		pGen->m_bInsideGeometries = genProto.inside_geometries();
		for (int j = 0; j < genProto.aggl_part_materials_alias_size(); ++j)
			pGen->m_partMaterials[genProto.aggl_part_materials_alias(j)] = genProto.aggl_part_materials_key(j);
		for (int j = 0; j < genProto.aggl_bond_materials_alias_size(); ++j)
			pGen->m_bondMaterials[genProto.aggl_bond_materials_alias(j)] = genProto.aggl_bond_materials_key(j);
		m_vGenerators[i]->m_dAgglomerateScaleFactor = genProto.scaling_factor();
		m_vGenerators[i]->m_bActive = genProto.activity();
	}
}

void CGenerationManager::SaveConfiguration()
{
	ProtoModulesData& _pProtoMessage = *m_pSystemStructure->GetProtoModulesData();
	_pProtoMessage.mutable_objects_generator()->clear_generators();
	for (size_t i = 0; i < m_vGenerators.size(); ++i)
	{
		ProtoObjectsGenerator* pGen = _pProtoMessage.mutable_objects_generator()->add_generators();
		pGen->set_name(m_vGenerators[i]->m_sName);
		pGen->set_volume_key(m_vGenerators[i]->m_sVolumeKey);
		pGen->set_random_velocity_flag(m_vGenerators[i]->m_bRandomVelocity);
		VectorToProtoVector(pGen->mutable_init_velocity(), m_vGenerators[i]->m_vObjInitVel);
		pGen->set_velocity_magnitude(m_vGenerators[i]->m_dVelMagnitude);
		pGen->set_mixture_key(m_vGenerators[i]->m_sMixtureKey);
		if (m_vGenerators[i]->m_bGenerateMixture)
			pGen->set_obj_type(ProtoObjectsGenerator_ObjectsType_cSphere);
		else
			pGen->set_obj_type(ProtoObjectsGenerator_ObjectsType_cAgglomerate);
		pGen->set_aggl_key(m_vGenerators[i]->m_sAgglomerateKey);
		pGen->set_start_generation_time(m_vGenerators[i]->m_dStartGenerationTime);
		pGen->set_end_generation_time(m_vGenerators[i]->m_dEndGenerationTime);
		pGen->set_update_time_step(m_vGenerators[i]->m_dUpdateStep);
		pGen->set_generation_rate(m_vGenerators[i]->m_dGenerationRate);
		pGen->set_inside_geometries(m_vGenerators[i]->m_bInsideGeometries);
		for (auto it = m_vGenerators[i]->m_partMaterials.begin(); it != m_vGenerators[i]->m_partMaterials.end(); ++it)
		{
			pGen->add_aggl_part_materials_alias(it->first);
			pGen->add_aggl_part_materials_key(it->second);
		}
		for (auto it = m_vGenerators[i]->m_bondMaterials.begin(); it != m_vGenerators[i]->m_bondMaterials.end(); ++it)
		{
			pGen->add_aggl_bond_materials_alias(it->first);
			pGen->add_aggl_bond_materials_key(it->second);
		}
		pGen->set_scaling_factor(m_vGenerators[i]->m_dAgglomerateScaleFactor);
		pGen->set_activity(m_vGenerators[i]->m_bActive);
	}
}

std::string CGenerationManager::IsDataCorrect() const
{
	for (size_t i = 0; i < m_vGenerators.size(); ++i)
	{
		if ( !m_vGenerators[ i ]->m_bActive ) continue; // consider only active objects generators

		CAnalysisVolume* pGenVolume = m_pSystemStructure->GetAnalysisVolume( m_vGenerators[ i ]->m_sVolumeKey );
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