/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelManager.h"
#include "MUSENFileFunctions.h"

#include "../MUSEN/Models/ParticleParticle/HeatConduction/ModelPPHeatConduction.h"
#include "../MUSEN/Models/ParticleParticle/Hertz/ModelPPHertz.h"
#include "../MUSEN/Models/ParticleParticle/HertzMindlin/ModelPPHertzMindlin.h"
#include "../MUSEN/Models/ParticleParticle/ChealNess/ModelPPChealNess.h"
#include "../MUSEN/Models/ParticleParticle/HertzMindlinLiquid/ModelPPHertzMindlinLiquid.h"
#include "../MUSEN/Models/ParticleParticle/JKR/ModelPPJKR.h"
#include "../MUSEN/Models/ParticleParticle/LinearElastic/ModelPPLinearElastic.h"
#include "../MUSEN/Models/ParticleParticle/PopovJKR/ModelPPPopovJKR.h"
#include "../MUSEN/Models/ParticleParticle/SimpleViscoElastic/ModelPPSimpleViscoElastic.h"
#include "../MUSEN/Models/ParticleParticle/TestSinteringModel/ModelPPSintering.h"
#include "../MUSEN/Models/ParticleParticle/SinteringTemperature/ModelPPSinteringTemperature.h"

#include "../MUSEN/Models/ParticleWall/PWHertzMindlin/ModelPWHertzMindlin.h"
#include "../MUSEN/Models/ParticleWall/PWHertzMindlinLiquid/ModelPWHertzMindlinLiquid.h"
#include "../MUSEN/Models/ParticleWall/PWJKR/ModelPWJKR.h"
#include "../MUSEN/Models/ParticleWall/PWPopovJKR/ModelPWPopovJKR.h"
#include "../MUSEN/Models/ParticleWall/PWSimpleViscoElastic/ModelPWSimpleViscoElastic.h"

#include "../MUSEN/Models/SolidBonds/BondModelAerogel/ModelSBAerogel.h"
#include "../MUSEN/Models/SolidBonds/BondModelElastic/ModelSBElastic.h"
#include "../MUSEN/Models/SolidBonds/BondModelElasticPerfectlyPlastic/ModelSBElasticPerfectlyPlastic.h"
#include "../MUSEN/Models/SolidBonds/BondModelCreep/ModelSBCreep.h"
#include "../MUSEN/Models/SolidBonds/BondModelKelvin/ModelSBKelvin.h"
#include "../MUSEN/Models/SolidBonds/BondModelLinearPlastic/ModelSBLinearPlastic.h"
#include "../MUSEN/Models/SolidBonds/BondModelThermal/ModelSBThermal.h"
#include "../MUSEN/Models/SolidBonds/BondModelWeakening/ModelSBWeakening.h"

#include "../MUSEN/Models/LiquidBonds/CapilaryViscous/ModelLBCapilarViscous.h"

#include "../MUSEN/Models/ExternalForce/CentrifugalCasting/ModelEFCentrifugalCasting.h"
#include "../MUSEN/Models/ExternalForce/ViscousField/ModelEFViscousField.h"
#include "../MUSEN/Models/ExternalForce/HeatTransfer/ModelEFHeatTransfer.h"



namespace StaticLibs
{
	struct SModule
	{
		std::string name;
		CreateModelFunction function;
	};

	template<class T>
	class Constructor
	{
		static CAbstractDEMModel* alloc()
		{
			return new T();
		}

	public:
		static SModule get()
		{
			SModule m;
			m.function = alloc;
			m.name = typeid(T).name();
			while (m.name.compare(0, 5, "Model"))
				m.name = m.name.substr(1, m.name.size() - 1);
			return m;
		}
	};

	SModule all_modules[] =
	{
		Constructor<CModelPPHeatConduction>::get(),
		Constructor<CModelPPHertz>::get(),
		Constructor<CModelPPHertzMindlin>::get(),
		Constructor<CModelPPHertzMindlinLiquid>::get(),
		Constructor<CModelPPChealNess>::get(),
		Constructor<CModelPPJKR>::get(),
		Constructor<CModelPPLinearElastic>::get(),
		Constructor<CModelPPPopovJKR>::get(),
		Constructor<CModelPPSimpleViscoElastic>::get(),
		Constructor<CModelPPSintering>::get(),
		Constructor<CModelPPSinteringTemperature>::get(),

		Constructor<CModelPWHertzMindlin>::get(),
		Constructor<CModelPWHertzMindlinLiquid>::get(),
		Constructor<CModelPWJKR>::get(),
		Constructor<CModelPWPopovJKR>::get(),
		Constructor<CModelPWSimpleViscoElastic>::get(),

		Constructor<CModelSBAerogel>::get(),
		Constructor<CModelSBElastic>::get(),
		Constructor<CModelSBElasticPerfectlyPlastic>::get(),
		Constructor<CModelSBCreep>::get(),
		Constructor<CModelSBKelvin>::get(),
		Constructor<CModelSBLinearPlastic>::get(),
		Constructor<CModelSBThermal>::get(),
		Constructor<CModelSBWeakening>::get(),

		Constructor<CModelLBCapilarViscous>::get(),

		Constructor<CModelEFCentrifugalCasting>::get(),
		Constructor<CModelEFViscousField>::get(),
		Constructor<CModelEFHeatTransfer>::get(),
	};

	SModule* LoadLibrary_static(const std::string& _sName)
	{
		for (size_t i = 0; i < sizeof(all_modules) / sizeof(SModule); ++i)
			if(_sName == all_modules[i].name)
				return &all_modules[i];
		return nullptr;
	}
}



CModelDescriptor::CModelDescriptor(std::string _error, ELibType _libType)
	: sError{ std::move(_error) }
	, libType{ _libType }
{}

CModelDescriptor::CModelDescriptor(std::unique_ptr<CAbstractDEMModel> _model, std::string _path, ELibType _libType)
	: pModel{ std::move(_model) }
	, sPath{ std::move(_path) }
	, libType{ _libType }
{}

const CAbstractDEMModel* CModelDescriptor::GetModel() const
{
	return pModel.get();
}

CAbstractDEMModel* CModelDescriptor::GetModel()
{
	return pModel.get();
}

std::string CModelDescriptor::GetName() const
{
	return sPath;
}

std::string CModelDescriptor::GetError() const
{
	return sError;
}

ELibType CModelDescriptor::GetLibType() const
{
	return libType;
}


CModelManager::CModelManager()
{
	UpdateAvailableModels();
}

CModelManager::~CModelManager()
{
	ClearAllActiveModels();

	//for (size_t i = 0; i < m_vAvailableModels.size(); ++i)
	//	delete m_vAvailableModels[i].pModel;
}

std::vector<std::string> CModelManager::GetDirs() const
{
	return m_vDirList;
}

void CModelManager::SetDirs(const std::vector<std::string>& _vDirs)
{
	m_vDirList.clear();

	for (size_t i = 0; i < _vDirs.size(); ++i)
		m_vDirList.push_back(unifyPath(_vDirs[i]));
	UpdateAvailableModels();
}

void CModelManager::AddDir(const std::string& _sDir)
{
	if (_sDir.empty()) return;
	bool bNewDir = true;
	for (size_t i = 0; i < m_vDirList.size(); ++i)
		if (std::find(m_vDirList.begin(), m_vDirList.end(), _sDir) != m_vDirList.end()) // already in the list
		{
			bNewDir = false;
			break;
		}
	if(bNewDir)
	{
		m_vDirList.push_back(unifyPath(_sDir));
		UpdateAvailableModels();
	}
}

void CModelManager::RemoveDir(size_t _index)
{
	if (_index >= m_vDirList.size()) return;
	m_vDirList.erase(m_vDirList.begin() + _index);
	UpdateAvailableModels();
}

void CModelManager::UpDir(size_t _index)
{
	if ((_index < m_vDirList.size()) && (_index != 0))
		std::iter_swap(m_vDirList.begin() + _index, m_vDirList.begin() + _index - 1);
}

void CModelManager::DownDir(size_t _index)
{
	if ((_index < m_vDirList.size()) && (_index != m_vDirList.size() - 1))
		std::iter_swap(m_vDirList.begin() + _index, m_vDirList.begin() + _index + 1);
}

std::vector<const CModelDescriptor*> CModelManager::GetAvailableModelsDescriptors() const
{
	auto res = ReservedVector<const CModelDescriptor*>(m_vAvailableModels.size());
	for (const auto& descriptor : m_vAvailableModels)
		res.push_back(&descriptor);
	return res;
}

std::vector<const CModelDescriptor*> CModelManager::GetAvailableModelsDescriptors(const EMusenModelType& _type) const
{
	std::vector<const CModelDescriptor*> res;
	for (const auto& descriptor : m_vAvailableModels)
		if (descriptor.GetModel() && descriptor.GetModel()->GetType() == _type)
			res.push_back(&descriptor);
	return res;
}

std::vector<CModelDescriptor*> CModelManager::GetAvailableModelsDescriptors(const EMusenModelType& _type)
{
	std::vector<CModelDescriptor*> res;
	for (auto& descriptor : m_vAvailableModels)
		if (descriptor.GetModel() && descriptor.GetModel()->GetType() == _type)
			res.push_back(&descriptor);
	return res;
}

std::vector<const CModelDescriptor*> CModelManager::GetModelsDescriptors() const
{
	auto res = ReservedVector<const CModelDescriptor*>(m_vCurrentModels.size());
	for (const auto& descriptor : m_vCurrentModels)
		res.push_back(&descriptor);
	return res;
}

std::vector<CAbstractDEMModel*> CModelManager::GetAllActiveModels() const
{
	auto res = ReservedVector<CAbstractDEMModel*>(m_vCurrentModels.size());
	for (const auto& descriptor : m_vCurrentModels)
		res.push_back(descriptor.pModel.get());
	return res;
}

bool CModelManager::IsModelActive(const EMusenModelType& _modelType) const
{
	return std::any_of(m_vCurrentModels.begin(), m_vCurrentModels.end(), [&](const CModelDescriptor& _info) { return _info.pModel->GetType() == _modelType; });
}

bool CModelManager::IsModelActive(const std::string& _name) const
{
	return std::any_of(m_vCurrentModels.begin(), m_vCurrentModels.end(), [&](const CModelDescriptor& _info) { return _info.sPath == _name; });
}

std::vector<const CModelDescriptor*> CModelManager::GetModelsDescriptors(const EMusenModelType& _type) const
{
	std::vector<const CModelDescriptor*> res;
	for (const auto& descriptor : m_vCurrentModels)
		if (descriptor.GetModel() && descriptor.GetModel()->GetType() == _type)
			res.push_back(&descriptor);
	return res;
}

std::vector<CModelDescriptor*> CModelManager::GetModelsDescriptors(const EMusenModelType& _type)
{
	std::vector<CModelDescriptor*> res;
	for (auto& descriptor : m_vCurrentModels)
		if (descriptor.GetModel() && descriptor.GetModel()->GetType() == _type)
			res.push_back(&descriptor);
	return res;
}

//bool CModelManager::IsModelGPUCompatible(const EMusenModelType& _modelType) const
//{
//	if (m_vCurrentModels.at(_modelType).pModel)
//		return m_vCurrentModels.at(_modelType).pModel->HasGPUSupport();
//	else
//		return false;
//}
//
//CAbstractDEMModel* CModelManager::GetModel(const EMusenModelType& _modelType)
//{
//	return m_vCurrentModels.at(_modelType).pModel;
//}
//
//const CAbstractDEMModel* CModelManager::GetModel(const EMusenModelType& _modelType) const
//{
//	return m_vCurrentModels.at(_modelType).pModel;
//}
//
//std::string CModelManager::GetModelPath(const EMusenModelType& _modelType) const
//{
//	if (m_vCurrentModels.at(_modelType).pModel)
//		return m_vCurrentModels.at(_modelType).sPath;
//	else
//		return "";
//}
//
//void CModelManager::SetModelPath(const EMusenModelType& _modelType, const std::string& _sPath)
//{
//	std::string sNewPath = unifyPath(_sPath);
//	if (m_vCurrentModels[_modelType].sPath == sNewPath)	// this model is already loaded
//		return;
//
//	ClearModel(_modelType);	// delete old model
//
//	m_vCurrentModels[_modelType] = LoadModelByName(sNewPath, _modelType);
//}

CModelDescriptor* CModelManager::AddActiveModel(const std::string& _name)
{
	const std::string name = unifyPath(_name);
	if (IsModelActive(name)) // this model is already loaded
		return {};

	CModelDescriptor descriptor{ LoadModelByName(name) };
	if (!descriptor.pModel) return {};

	m_vCurrentModels.push_back(std::move(descriptor));
	return &m_vCurrentModels.back();
}

void CModelManager::RemoveActiveModel(const std::string& _name)
{
	const std::string name = unifyPath(_name);
	const size_t i = VectorFind(m_vCurrentModels, [&](const CModelDescriptor& _descriptor) { return _descriptor.sPath == name; });
	if (i >= m_vCurrentModels.size()) return;
	m_vCurrentModels.erase(m_vCurrentModels.begin() + i);
}

CModelDescriptor* CModelManager::ReplaceActiveModel(const std::string& _oldName, const std::string& _newName)
{
	const std::string oldName = unifyPath(_oldName);
	const size_t i = VectorFind(m_vCurrentModels, [&](const CModelDescriptor& _descriptor) { return _descriptor.sPath == oldName; });
	if (i < m_vCurrentModels.size())
		m_vCurrentModels.erase(m_vCurrentModels.begin() + i);

	const std::string newName = unifyPath(_newName);
	if (IsModelActive(newName)) return {};
	CModelDescriptor descriptor{ LoadModelByName(newName) };
	if (!descriptor.pModel) return {};

	if (i < m_vCurrentModels.size()) // replace
	{
		m_vCurrentModels.insert(m_vCurrentModels.begin() + i, std::move(descriptor));
		return &m_vCurrentModels[i];
	}
	else // just add new
	{
		m_vCurrentModels.push_back(std::move(descriptor));
		return &m_vCurrentModels.back();
	}
}

SOptionalVariables CModelManager::GetUtilizedVariables() const
{
	SOptionalVariables activeVariables; // result variable
	for (const auto& info : m_vCurrentModels)
	{
		if (info.pModel)
			activeVariables |= info.pModel->GetUtilizedVariables();
	}
	return activeVariables;
}

//std::string CModelManager::GetModelParameters(const EMusenModelType& _modelType) const
//{
//	if (m_vCurrentModels.at(_modelType).pModel)
//		return m_vCurrentModels.at(_modelType).pModel->GetParametersStr();
//	else
//		return "";
//}

void CModelManager::SetModelParameters(const std::string& _name, const std::string& _params) const
{
	const size_t i = VectorFind(m_vCurrentModels, [&](const CModelDescriptor& _modelInfo) { return _modelInfo.sPath == _name; });
	if (i >= m_vCurrentModels.size()) return;
	m_vCurrentModels[i].pModel->SetParametersStr(_params);
}

//void CModelManager::SetModelDefaultParameters(const EMusenModelType& _modelType)
//{
//	if (m_vCurrentModels[_modelType].pModel)
//		m_vCurrentModels[_modelType].pModel->SetDefaultValues();
//}
//
//std::string CModelManager::GetModelError(const EMusenModelType& _modelType) const
//{
//	return m_vCurrentModels.at(_modelType).sError;
//}

void CModelManager::LoadConfiguration()
{
	const auto LoadModel = [&](const ProtoMusenModel& _protoModel, const EMusenModelType& _modelType)
	{
		if (_protoModel.key().empty()) return; // no model was previously saved

		CModelDescriptor res = LoadModelByName(_protoModel.path(), _modelType); // try to load by path
		if (!res.pModel)
			res = LoadModelByKey(_protoModel.key(), _modelType); // try to load by key

		if (res.pModel) // model loaded
		{
			res.pModel->SetParametersStr(_protoModel.params());
			m_vCurrentModels.push_back(std::move(res));
		}
	};

	ClearAllActiveModels();

	const ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	if (!protoMessage.has_model_manager()) return; // for old versions of file
	const ProtoModuleModelManager& mm = protoMessage.model_manager();
	const uint32_t version = mm.version();

	if (version == 0) // compatibility with older versions
	{
		LoadModel(mm.pp_model()   , EMusenModelType::PP);
		LoadModel(mm.pw_model()   , EMusenModelType::PW);
		LoadModel(mm.sb_model()   , EMusenModelType::SB);
		LoadModel(mm.lb_model()   , EMusenModelType::LB);
		LoadModel(mm.ef_model()   , EMusenModelType::EF);
		LoadModel(mm.ht_pp_model(), EMusenModelType::PP);
	}
	else
	{
		for (const ProtoMusenModel& protoModel : mm.models())
		{
			LoadModel(protoModel, EMusenModelType::UNSPECIFIED);
		}
	}
	m_bConnectedPPContact = mm.connected_pp_contact();
}

void CModelManager::SaveConfiguration()
{
	ProtoModuleModelManager* protoMM = m_pSystemStructure->GetProtoModulesData()->mutable_model_manager();
	protoMM->set_version(1);
	protoMM->clear_models();
	for (const auto& model : m_vCurrentModels)
	{
		auto* protoModel = protoMM->add_models();
		protoModel->set_key(model.pModel ? model.pModel->GetUniqueKey() : "");
		protoModel->set_path(model.pModel ? model.sPath : "");
		protoModel->set_params(model.pModel ? model.pModel->GetParametersStr() : "");
	}
	protoMM->set_connected_pp_contact(m_bConnectedPPContact);
}

void CModelManager::LoadModelConfiguration(const ProtoMusenModel& _protoModel, const EMusenModelType& _modelType, CModelDescriptor* _pModelInfo)
{
	if (_protoModel.key().empty()) // no model was previously saved
		return;

	*_pModelInfo = LoadModelByName(_protoModel.path(), _modelType); // try to load by path
	if (!_pModelInfo->pModel)
		*_pModelInfo = LoadModelByKey(_protoModel.key(), _modelType); // try to load by key

	if (_pModelInfo->pModel) // model loaded
		_pModelInfo->pModel->SetParametersStr(_protoModel.params());
}

void CModelManager::UpdateAvailableModels()
{
	// clear old models
	//for (size_t i = 0; i < m_vAvailableModels.size(); ++i)
	//	delete m_vAvailableModels[i].pModel;
	m_vAvailableModels.clear();

	// static models
	for (size_t i = 0; i < sizeof(StaticLibs::all_modules) / sizeof(StaticLibs::SModule); ++i)
		m_vAvailableModels.push_back(LoadStaticModel(StaticLibs::all_modules[i].name));

	// dynamic models
	for (size_t i = 0; i < m_vDirList.size(); ++i)
	{
		std::vector<std::string> vDLLs = MUSENFileFunctions::filesList(m_vDirList[i], "*.dll");
		for (size_t j = 0; j < vDLLs.size(); ++j)
		{
			CModelDescriptor model = LoadDynamicModel(vDLLs[j]);
			if (model.pModel)
				m_vAvailableModels.push_back(std::move(model));
		}
	}
}

CModelDescriptor CModelManager::LoadDynamicModel(const std::string& _modelPath, const EMusenModelType& _modelType /*= EMusenModelType::UNSPECIFIED*/)
{
#ifdef _WIN32
	const std::string path = unifyPath(_modelPath);

	if (path.empty()) // just an empty model (will not be used during the simulation)
		return {};

	// TODO: there is no call of FreeLibrary(instLibrary) at successful program end
	// try to load library
	const HINSTANCE instLibrary = LoadLibraryA(windowsPath(path).c_str());
	if (!instLibrary)
		return CModelDescriptor{ BuildErrorDescription(_modelPath, EErrorType::WRONG_PATH), ELibType::DYNAMIC };

	// try to get constructor
	const auto createModelFunc = reinterpret_cast<CreateModelFunction>(GetProcAddress(instLibrary, MUSEN_CREATE_MODEL_FUN_NAME));
	if (!createModelFunc)
	{
		FreeLibrary(instLibrary);
		return CModelDescriptor{ BuildErrorDescription(_modelPath, EErrorType::WRONG_DLL_INTERFACE), ELibType::DYNAMIC };
	}

	// try to create model
	std::unique_ptr<CAbstractDEMModel> loadedModel;
	try
	{
		loadedModel.reset(createModelFunc());
	}
	catch (...)
	{
		FreeLibrary(instLibrary);
		return CModelDescriptor{ BuildErrorDescription(_modelPath, EErrorType::WRONG_DLL_VERSION), ELibType::DYNAMIC };
	}

	// check model is of defined type
	if (loadedModel->GetType() == EMusenModelType::UNSPECIFIED)
	{
		FreeLibrary(instLibrary);
		return CModelDescriptor{ BuildErrorDescription(_modelPath, EErrorType::UNSPEC_MODEL_TYPE), ELibType::DYNAMIC };
	}

	// check model type
	if (_modelType != EMusenModelType::UNSPECIFIED && loadedModel->GetType() != _modelType)
	{
		FreeLibrary(instLibrary);
		return CModelDescriptor{ BuildErrorDescription(_modelPath, EErrorType::WRONG_MODEL_TYPE), ELibType::DYNAMIC };
	}

	return CModelDescriptor{ std::move(loadedModel), path, ELibType::DYNAMIC };

#else
	// no implementation of dynamic models for Linux
	return SModelInfo{ nullptr, "_WIN32", ELibType::DYNAMIC };
#endif
}

CModelDescriptor CModelManager::LoadStaticModel(const std::string& _modelName, const EMusenModelType& _modelType /*= EMusenModelType::UNSPECIFIED*/)
{
	if (_modelName.empty())	// just an empty model (will not be used during the simulation)
		return {};

	// try to load library
	const StaticLibs::SModule* instLibrary = StaticLibs::LoadLibrary_static(_modelName);
	if (!instLibrary)
		return CModelDescriptor{ BuildErrorDescription(_modelName, EErrorType::WRONG_PATH), ELibType::STATIC };

	// instantiate model
	std::unique_ptr<CAbstractDEMModel> loadedModel{ instLibrary->function() };

	// check model is of defined type
	if (loadedModel->GetType() == EMusenModelType::UNSPECIFIED)
		return CModelDescriptor{ BuildErrorDescription(_modelName, EErrorType::UNSPEC_MODEL_TYPE), ELibType::STATIC };

	// check model type
	if (_modelType != EMusenModelType::UNSPECIFIED && loadedModel->GetType() != _modelType)
		return CModelDescriptor{ BuildErrorDescription(_modelName, EErrorType::WRONG_MODEL_TYPE), ELibType::STATIC };

	return CModelDescriptor{ std::move(loadedModel), _modelName, ELibType::STATIC };
}

CModelDescriptor CModelManager::LoadModelByName(const std::string& _modelName, const EMusenModelType& _modelType /*= EMusenModelType::UNSPECIFIED*/)
{
	// try to load static model
	CModelDescriptor model = LoadStaticModel(_modelName, _modelType);
	if (model.pModel) return model; // found static
	// try to load dynamic model
	model = LoadDynamicModel(_modelName, _modelType);
	if (model.pModel) return model; // found dynamic
	return {}; // not found
}

CModelDescriptor CModelManager::LoadModelByKey(const std::string& _modelKey, const EMusenModelType& _modelType /*= EMusenModelType::UNSPECIFIED*/) const
{
	for (const auto& modelInfo : m_vAvailableModels)
		if (modelInfo.pModel->GetUniqueKey() == _modelKey)
		{
			CModelDescriptor model = LoadModelByName(modelInfo.sPath, _modelType);
			if (model.pModel) return model;
		}
	return {};
}

std::string CModelManager::BuildErrorDescription(const std::string& _model, const EErrorType& _type)
{
	std::string message;

	switch (_type)
	{
	case EErrorType::WRONG_PATH:
		message = "Cannot find '" + _model + "' model";
		break;
	case EErrorType::WRONG_DLL_INTERFACE:
		message = "Wrong format of '" + _model + "' model";
		break;
	case EErrorType::WRONG_DLL_VERSION:
		message = "Unsupported version of '" + _model + "' model";
		break;
	case EErrorType::UNSPEC_MODEL_TYPE:
		message = "Unspecified model type of '" + _model + "' model";
		break;
	case EErrorType::WRONG_MODEL_TYPE:
		message = "Wrong model type of '" + _model + "' model";
		break;
	}

	return message;
}

//void CModelManager::ClearModel(const EMusenModelType& _modelType)
//{
//	if (m_vCurrentModels[_modelType].pModel)
//	{
//		delete m_vCurrentModels[_modelType].pModel;
//		m_vCurrentModels[_modelType].pModel = nullptr;
//		m_vCurrentModels[_modelType].sPath.clear();
//	}
//}

// TODO: remove
void CModelManager::ClearAllActiveModels()
{
	m_vCurrentModels.clear();
}

