/* Copyright (c) 2013-2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ModelManager.h"
#include "MUSENFileFunctions.h"
#ifdef _WIN32
#include "SafeWindowsHeader.h"
#else
#include <dlfcn.h>
#endif


#include "../Models/ParticleParticle/HeatConduction/ModelPPHeatConduction.h"
#include "../Models/ParticleParticle/Hertz/ModelPPHertz.h"
#include "../Models/ParticleParticle/HertzMindlin/ModelPPHertzMindlin.h"
#include "../Models/ParticleParticle/ChealNess/ModelPPChealNess.h"
#include "../Models/ParticleParticle/HertzMindlinLiquid/ModelPPHertzMindlinLiquid.h"
#include "../Models/ParticleParticle/JKR/ModelPPJKR.h"
#include "../Models/ParticleParticle/LinearElastic/ModelPPLinearElastic.h"
#include "../Models/ParticleParticle/PopovJKR/ModelPPPopovJKR.h"
#include "../Models/ParticleParticle/SimpleViscoElastic/ModelPPSimpleViscoElastic.h"
#include "../Models/ParticleParticle/TestSinteringModel/ModelPPSintering.h"
#include "../Models/ParticleParticle/SinteringTemperature/ModelPPSinteringTemperature.h"
#include "../Models/ParticleParticle/HertzMindlinVdW/ModelPPHertzMindlinVdW.h"

#include "../Models/ParticleWall/PWHeatTransfer/ModelPWHeatTransfer.h"
#include "../Models/ParticleWall/PWHertzMindlin/ModelPWHertzMindlin.h"
#include "../Models/ParticleWall/PWHertzMindlinLiquid/ModelPWHertzMindlinLiquid.h"
#include "../Models/ParticleWall/PWJKR/ModelPWJKR.h"
#include "../Models/ParticleWall/PWPopovJKR/ModelPWPopovJKR.h"
#include "../Models/ParticleWall/PWSimpleViscoElastic/ModelPWSimpleViscoElastic.h"

#include "../Models/SolidBonds/BondModelAerogel/ModelSBAerogel.h"
#include "../Models/SolidBonds/BondModelElastic/ModelSBElastic.h"
#include "../Models/SolidBonds/BondModelElasticPerfectlyPlastic/ModelSBElasticPerfectlyPlastic.h"
#include "../Models/SolidBonds/BondModelCreep/ModelSBCreep.h"
#include "../Models/SolidBonds/BondModelHeatConduction/ModelSBHeatConduction.h"
#include "../Models/SolidBonds/BondModelKelvin/ModelSBKelvin.h"
#include "../Models/SolidBonds/BondModelLinearPlastic/ModelSBLinearPlastic.h"
#include "../Models/SolidBonds/BondModelThermal/ModelSBThermal.h"
#include "../Models/SolidBonds/BondModelWeakening/ModelSBWeakening.h"
#include "../Models/SolidBonds/BondModelPlasticConcrete/ModelSBPlasticConcrete.h"

#include "../Models/LiquidBonds/CapilaryViscous/ModelLBCapilarViscous.h"

#include "../Models/ExternalForce/CentrifugalCasting/ModelEFCentrifugalCasting.h"
#include "../Models/ExternalForce/ViscousField/ModelEFViscousField.h"
#include "../Models/ExternalForce/HeatTransfer/ModelEFHeatTransfer.h"
#include "../Models/ExternalForce/LiquidDiffusion/ModelEFLiquidDiffusion.h"


/*
 * Holds all statically loaded libraries and functions to load them.
 */
namespace StaticLibs
{
	// Descriptor of the static model library.
	struct SModule
	{
		std::string name;               // Name of the class of the model.
		CreateModelFunction allocate{}; // Function to allocate the model.
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
			SModule m{ typeid(T).name(), alloc };
			while (m.name.compare(0, 5, "Model"))
				m.name = m.name.substr(1, m.name.size() - 1);
			return m;
		}
	};

	// Pointers to all available statically loaded models.
	std::vector<SModule> allStaticModels
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
		Constructor<CModelPPHertzMindlinVdW>::get(),

		Constructor<CModelPWHeatTransfer>::get(),
		Constructor<CModelPWHertzMindlin>::get(),
		Constructor<CModelPWHertzMindlinLiquid>::get(),
		Constructor<CModelPWJKR>::get(),
		Constructor<CModelPWPopovJKR>::get(),
		Constructor<CModelPWSimpleViscoElastic>::get(),

		Constructor<CModelSBAerogel>::get(),
		Constructor<CModelSBElastic>::get(),
		Constructor<CModelSBElasticPerfectlyPlastic>::get(),
		Constructor<CModelSBCreep>::get(),
		Constructor<CModelSBHeatConduction>::get(),
		Constructor<CModelSBKelvin>::get(),
		Constructor<CModelSBLinearPlastic>::get(),
		Constructor<CModelSBThermal>::get(),
		Constructor<CModelSBWeakening>::get(),
		Constructor<CModelSBPlasticConcrete>::get(),

		Constructor<CModelLBCapilarViscous>::get(),

		Constructor<CModelEFCentrifugalCasting>::get(),
		Constructor<CModelEFViscousField>::get(),
		Constructor<CModelEFHeatTransfer>::get(),
		Constructor<CModelEFLiquidDiffusion>::get(),
	};

	// Provides a descriptor of the given static model.
	SModule* LoadLibraryStatic(const std::string& _name)
	{
		const auto it = std::find_if(allStaticModels.begin(), allStaticModels.end(), [&](const SModule& _m) { return _m.name == _name; });
		if (it == allStaticModels.end()) return nullptr;
		return &*it;
	}
}


void CModelDescriptor::lib_deleter_t::operator()(void* _handle) const
{
#ifdef _WIN32
	FreeLibrary(static_cast<HMODULE>(_handle));
#else
	dlclose(_handle);
#endif
}

CModelDescriptor::CModelDescriptor(std::string _error, ELibType _libType)
	: errorMessage{ std::move(_error) }
	, libType{ _libType }
{}

CModelDescriptor::CModelDescriptor(std::unique_ptr<CAbstractDEMModel> _model, std::string _path, ELibType _libType)
	: model{ std::move(_model) }
	, path{ std::move(_path) }
	, libType{ _libType }
{}

CModelDescriptor::CModelDescriptor(std::unique_ptr<CAbstractDEMModel> _model, std::string _path, lib_ptr_t _library)
	: library{ std::move(_library) }
	, model{ std::move(_model) }
	, path{ std::move(_path) }
	, libType{ ELibType::DYNAMIC }
{}

const CAbstractDEMModel* CModelDescriptor::GetModel() const
{
	return model.get();
}

CAbstractDEMModel* CModelDescriptor::GetModel()
{
	return model.get();
}

std::string CModelDescriptor::GetPath() const
{
	return path;
}

std::string CModelDescriptor::GetError() const
{
	return errorMessage;
}

ELibType CModelDescriptor::GetLibType() const
{
	return libType;
}


#ifdef _WIN32
const std::string CModelManager::c_libraryFileExtension{ "dll" };
#else
const std::string CModelManager::c_libraryFileExtension{ "so" };
#endif

CModelManager::CModelManager()
{
	UpdateAvailableModels();
}

std::vector<std::string> CModelManager::GetDirs() const
{
	return m_dirs;
}

void CModelManager::SetDirs(const std::vector<std::string>& _dirs)
{
	m_dirs.clear();
	for (const auto& dir : _dirs)
		m_dirs.push_back(unifyPath(dir));
	UpdateAvailableModels();
}

void CModelManager::AddDir(const std::string& _dir)
{
	if (_dir.empty()) return;
	const auto dir = unifyPath(_dir);
	if (std::find(m_dirs.begin(), m_dirs.end(), dir) != m_dirs.end()) return; // already in the list
	m_dirs.push_back(dir);
	UpdateAvailableModels();
}

void CModelManager::RemoveDir(size_t _index)
{
	if (_index >= m_dirs.size()) return;
	m_dirs.erase(m_dirs.begin() + _index);
	UpdateAvailableModels();
}

void CModelManager::UpDir(size_t _index)
{
	if (_index < m_dirs.size() && _index != 0)
		std::iter_swap(m_dirs.begin() + _index, m_dirs.begin() + _index - 1);
}

void CModelManager::DownDir(size_t _index)
{
	if (_index < m_dirs.size() && _index != m_dirs.size() - 1)
		std::iter_swap(m_dirs.begin() + _index, m_dirs.begin() + _index + 1);
}

std::vector<const CModelDescriptor*> CModelManager::GetAvailableModelsDescriptors() const
{
	auto res = ReservedVector<const CModelDescriptor*>(m_availableModels.size());
	for (const auto& descriptor : m_availableModels)
		res.push_back(descriptor.get());
	return res;
}

std::vector<const CModelDescriptor*> CModelManager::GetAvailableModelsDescriptors(const EMusenModelType& _type) const
{
	std::vector<const CModelDescriptor*> res;
	for (const auto& descriptor : m_availableModels)
		if (descriptor->GetModel() && descriptor->GetModel()->GetType() == _type)
			res.push_back(descriptor.get());
	return res;
}

std::vector<CModelDescriptor*> CModelManager::GetAvailableModelsDescriptors(const EMusenModelType& _type)
{
	std::vector<CModelDescriptor*> res;
	for (auto& descriptor : m_availableModels)
		if (descriptor->GetModel() && descriptor->GetModel()->GetType() == _type)
			res.push_back(descriptor.get());
	return res;
}

std::vector<const CModelDescriptor*> CModelManager::GetActiveModelsDescriptors() const
{
	auto res = ReservedVector<const CModelDescriptor*>(m_activeModels.size());
	for (const auto& descriptor : m_activeModels)
		res.push_back(descriptor.get());
	return res;
}

std::vector<const CModelDescriptor*> CModelManager::GetActiveModelsDescriptors(const EMusenModelType& _type) const
{
	std::vector<const CModelDescriptor*> res;
	for (const auto& descriptor : m_activeModels)
		if (descriptor->GetModel() && descriptor->GetModel()->GetType() == _type)
			res.push_back(descriptor.get());
	return res;
}

std::vector<CModelDescriptor*> CModelManager::GetActiveModelsDescriptors(const EMusenModelType& _type)
{
	std::vector<CModelDescriptor*> res;
	for (auto& descriptor : m_activeModels)
		if (descriptor->GetModel() && descriptor->GetModel()->GetType() == _type)
			res.push_back(descriptor.get());
	return res;
}

std::vector<CAbstractDEMModel*> CModelManager::GetAllActiveModels() const
{
	auto res = ReservedVector<CAbstractDEMModel*>(m_activeModels.size());
	for (const auto& descriptor : m_activeModels)
		res.push_back(descriptor->model.get());
	return res;
}

bool CModelManager::IsModelActive(const EMusenModelType& _modelType) const
{
	return std::any_of(m_activeModels.begin(), m_activeModels.end(), [&](const auto& _info) { return _info->model->GetType() == _modelType; });
}

bool CModelManager::IsModelActive(const std::string& _name) const
{
	return std::any_of(m_activeModels.begin(), m_activeModels.end(), [&](const auto& _info) { return _info->path == _name; });
}

CModelDescriptor* CModelManager::AddActiveModel(const std::string& _name)
{
	const std::string name = unifyPath(_name);
	if (IsModelActive(name)) // this model is already loaded
		return {};

	std::unique_ptr<CModelDescriptor> descriptor{ LoadModelByName(name) };
	if (!descriptor->model) return {};

	m_activeModels.push_back(std::move(descriptor));
	return m_activeModels.back().get();
}

void CModelManager::RemoveActiveModel(const std::string& _name)
{
	const std::string name = unifyPath(_name);
	m_activeModels.erase(std::remove_if(m_activeModels.begin(), m_activeModels.end(),
		[&](const auto& _descriptor) { return _descriptor->path == name; }), m_activeModels.end());
}

CModelDescriptor* CModelManager::ReplaceActiveModel(const std::string& _oldName, const std::string& _newName)
{
	const std::string oldName = unifyPath(_oldName);
	const auto it = std::find_if(m_activeModels.begin(), m_activeModels.end(), [&](const auto& _descriptor) { return _descriptor->path == oldName; });
	const auto pos = std::distance(m_activeModels.begin(), it);
	if (it != m_activeModels.end())
		m_activeModels.erase(it);

	const std::string newName = unifyPath(_newName);
	if (IsModelActive(newName)) return {};
	std::unique_ptr<CModelDescriptor> descriptor{ LoadModelByName(newName) };
	if (!descriptor->model) return {};

	return m_activeModels.insert(std::next(m_activeModels.begin(), pos), std::move(descriptor))->get();
}

void CModelManager::SetModelParameters(const std::string& _name, const std::string& _params) const
{
	const auto it = std::find_if(m_activeModels.begin(), m_activeModels.end(),
		[&](const auto& _descriptor) { return _descriptor->path == _name; });
	if (it == m_activeModels.end() || !it->get()->GetModel()) return;
	it->get()->model->SetParametersStr(_params);
}

SOptionalVariables CModelManager::GetUtilizedVariables() const
{
	SOptionalVariables activeVariables; // result variable
	for (const auto& info : m_activeModels)
	{
		if (info->model)
			activeVariables |= info->model->GetUtilizedVariables();
	}
	return activeVariables;
}

void CModelManager::LoadConfiguration()
{
	const auto LoadModel = [&](const ProtoMusenModel& _protoModel, const EMusenModelType& _modelType)
	{
		if (_protoModel.key().empty()) return; // no model was previously saved

		auto res = LoadModelByName(_protoModel.path(), _modelType); // try to load by path
		if (!res->model)
			res = LoadModelByKey(_protoModel.key(), _modelType); // try to load by key

		if (res->model) // model loaded
		{
			res->model->SetParametersStr(_protoModel.params());
			m_activeModels.push_back(std::move(res));
		}
	};

	m_activeModels.clear();

	const ProtoModulesData& protoMessage = *m_pSystemStructure->GetProtoModulesData();
	if (!protoMessage.has_model_manager()) return; // for old versions of file
	const ProtoModuleModelManager& protoManager = protoMessage.model_manager();
	const uint32_t version = protoManager.version();

	if (version == 0) // compatibility with older versions
	{
		LoadModel(protoManager.pp_model()   , EMusenModelType::PP);
		LoadModel(protoManager.pw_model()   , EMusenModelType::PW);
		LoadModel(protoManager.sb_model()   , EMusenModelType::SB);
		LoadModel(protoManager.lb_model()   , EMusenModelType::LB);
		LoadModel(protoManager.ef_model()   , EMusenModelType::EF);
		LoadModel(protoManager.ht_pp_model(), EMusenModelType::PP);
	}
	else // actual version saving
	{
		for (const ProtoMusenModel& protoModel : protoManager.models())
		{
			LoadModel(protoModel, EMusenModelType::UNSPECIFIED);
		}
	}
	m_connectedPPContact = protoManager.connected_pp_contact();
}

void CModelManager::SaveConfiguration()
{
	ProtoModuleModelManager* protoManager = m_pSystemStructure->GetProtoModulesData()->mutable_model_manager();
	protoManager->set_version(1);
	protoManager->clear_models();
	for (const auto& model : m_activeModels)
	{
		auto* protoModel = protoManager->add_models();
		protoModel->set_key(model->model ? model->model->GetUniqueKey() : "");
		protoModel->set_path(model->model ? model->path : "");
		protoModel->set_params(model->model ? model->model->GetParametersStr() : "");
	}
	protoManager->set_connected_pp_contact(m_connectedPPContact);
}

void CModelManager::UpdateAvailableModels()
{
	// clear old models
	m_availableModels.clear();

	// static models
	for (const auto& descriptor : StaticLibs::allStaticModels)
		m_availableModels.push_back(LoadStaticModel(descriptor.name));

	// dynamic models
	for (const auto& dir : m_dirs)
	{
		for (const auto& dll : MUSENFileFunctions::filesList(dir, "*." + c_libraryFileExtension))
		{
			auto descriptor = LoadDynamicModel(dll);
			if (descriptor->model)
				m_availableModels.push_back(std::move(descriptor));
		}
	}
}

std::unique_ptr<CModelDescriptor> CModelManager::LoadModelByName(const std::string& _modelName, const EMusenModelType& _modelType /*= EMusenModelType::UNSPECIFIED*/)
{
	// try to load static model
	auto descriptor = LoadStaticModel(_modelName, _modelType);
	if (descriptor->model) return descriptor; // found static
	// try to load dynamic model
	descriptor = LoadDynamicModel(_modelName, _modelType);
	if (descriptor->model) return descriptor; // found dynamic
	return {};
}

std::unique_ptr<CModelDescriptor> CModelManager::LoadModelByKey(const std::string& _modelKey, const EMusenModelType& _modelType /*= EMusenModelType::UNSPECIFIED*/) const
{
	for (const auto& descriptor : m_availableModels)
		if (descriptor->model->GetUniqueKey() == _modelKey)
		{
			auto model = LoadModelByName(descriptor->path, _modelType);
			if (model->model) return model;
		}
	return {};
}

std::unique_ptr<CModelDescriptor> CModelManager::LoadDynamicModel(const std::string& _modelPath, const EMusenModelType& _modelType /*= EMusenModelType::UNSPECIFIED*/)
{
	const std::string path = unifyPath(_modelPath);

	if (path.empty()) return {};

	// try to load library
	CModelDescriptor::lib_ptr_t instLibrary(LoadLibraryFromFile(path));
	if (!instLibrary)
		return std::make_unique<CModelDescriptor>(BuildErrorDescription(path, EErrorType::WRONG_PATH), ELibType::DYNAMIC);

	// try to get constructor
	const auto createModelFunc = reinterpret_cast<CreateModelFunction>(LoadModelConstructor(instLibrary.get()));
	if (!createModelFunc)
		return std::make_unique<CModelDescriptor>(BuildErrorDescription(path, EErrorType::WRONG_DLL_INTERFACE), ELibType::DYNAMIC);

	// try to create model
	std::unique_ptr<CAbstractDEMModel> loadedModel;
	try
	{
		loadedModel.reset(createModelFunc());
	}
	catch (...)
	{
		return std::make_unique<CModelDescriptor>(BuildErrorDescription(path, EErrorType::WRONG_DLL_VERSION), ELibType::DYNAMIC);
	}

	// check model is of defined type
	if (loadedModel->GetType() == EMusenModelType::UNSPECIFIED)
		return std::make_unique<CModelDescriptor>(BuildErrorDescription(path, EErrorType::UNSPEC_MODEL_TYPE), ELibType::DYNAMIC);

	// check model type
	if (_modelType != EMusenModelType::UNSPECIFIED && loadedModel->GetType() != _modelType)
		return std::make_unique<CModelDescriptor>(BuildErrorDescription(path, EErrorType::WRONG_MODEL_TYPE), ELibType::DYNAMIC);

	return std::make_unique<CModelDescriptor>(std::move(loadedModel), path, std::move(instLibrary));
}

std::unique_ptr<CModelDescriptor> CModelManager::LoadStaticModel(const std::string& _modelName, const EMusenModelType& _modelType /*= EMusenModelType::UNSPECIFIED*/)
{
	if (_modelName.empty()) return {};

	// try to load library
	const StaticLibs::SModule* instLibrary = StaticLibs::LoadLibraryStatic(_modelName);
	if (!instLibrary)
		return std::make_unique<CModelDescriptor>(BuildErrorDescription(_modelName, EErrorType::WRONG_PATH), ELibType::STATIC);

	// instantiate model
	std::unique_ptr<CAbstractDEMModel> loadedModel{ instLibrary->allocate() };
	if (!loadedModel)
		return std::make_unique<CModelDescriptor>(BuildErrorDescription(_modelName, EErrorType::CANNOT_ALLOCATE), ELibType::STATIC);

	// check model is of defined type
	if (loadedModel->GetType() == EMusenModelType::UNSPECIFIED)
		return std::make_unique<CModelDescriptor>(BuildErrorDescription(_modelName, EErrorType::UNSPEC_MODEL_TYPE), ELibType::STATIC);

	// check model type
	if (_modelType != EMusenModelType::UNSPECIFIED && loadedModel->GetType() != _modelType)
		return std::make_unique<CModelDescriptor>(BuildErrorDescription(_modelName, EErrorType::WRONG_MODEL_TYPE), ELibType::STATIC);

	return std::make_unique<CModelDescriptor>(std::move(loadedModel), _modelName, ELibType::STATIC);
}

void* CModelManager::LoadLibraryFromFile(const std::string& _libPath)
{
#ifdef _WIN32
	return LoadLibrary(UnicodePath(windowsPath(_libPath)).c_str());
#else
	return dlopen(_libPath.c_str(), RTLD_LAZY);
#endif
}

void* CModelManager::LoadModelConstructor(void* _lib)
{
#ifdef _WIN32
	return static_cast<void*>(GetProcAddress(static_cast<HMODULE>(_lib), MUSEN_CREATE_MODEL_FUN_NAME));
#else
	return dlsym(_lib, MUSEN_CREATE_MODEL_FUN_NAME);
#endif
}

std::string CModelManager::BuildErrorDescription(const std::string& _modelName, const EErrorType& _errorType)
{
	std::string message;

	switch (_errorType)
	{
	case EErrorType::WRONG_PATH:			message = "Cannot find '" + _modelName + "' model";				break;
	case EErrorType::WRONG_DLL_INTERFACE:	message = "Wrong format of '" + _modelName + "' model";			break;
	case EErrorType::WRONG_DLL_VERSION:		message = "Unsupported version of '" + _modelName + "' model";	break;
	case EErrorType::UNSPEC_MODEL_TYPE:		message = "Unknown type of '" + _modelName + "' model";			break;
	case EErrorType::WRONG_MODEL_TYPE:		message = "Wrong type of '" + _modelName + "' model";			break;
	case EErrorType::CANNOT_ALLOCATE:		message = "Can not allocate '" + _modelName + "' model";		break;
	}

	return message;
}
