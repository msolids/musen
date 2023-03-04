/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "MUSENComponent.h"
#include "AbstractDEMModel.h"

class CModelManager : public CMusenComponent
{
public:
	enum class ELibType { STATIC, DYNAMIC };
	struct SModelInfo
	{
		CAbstractDEMModel* pModel;	// Pointer to a model.
		std::string sPath;			// Full path to a model.
		std::string sError;			// Description of occurred error during loading.
		ELibType libType;			// Type of underlying library: static or dynamic.
		SModelInfo() :
			pModel(nullptr), sPath(""), sError(""), libType(ELibType::STATIC) {}
		SModelInfo(CAbstractDEMModel* _pModel, const std::string& _sPath, const std::string& _sError, ELibType _libType) :
			pModel(_pModel), sPath(_sPath), sError(_sError), libType(_libType) {}
	};

private:
	enum class EErrorType
	{
		WRONG_PATH, WRONG_DLL_INTERFACE, WRONG_DLL_VERSION, UNSPEC_MODEL_TYPE, WRONG_MODEL_TYPE
	};
	std::vector<std::string> m_vDirList;					// List of directories, where to look for DLLs.
	std::vector<SModelInfo> m_vAvailableModels;				// List of all models from all directories.
	std::map<EMusenModelType, SModelInfo> m_vCurrentModels;	// List of selected/loaded models.
	bool m_bConnectedPPContact{ true };						// Whether to calculate contacts between particles directly connected with bond.

public:
	CModelManager();
	~CModelManager();
	CModelManager(const CModelManager&) = delete;				// not allowed because destructor deletes m_vAvailableModels[:].pModel
	CModelManager& operator=(const CModelManager&) = delete;	// not allowed because destructor deletes m_vAvailableModels[:].pModel

	// Returns list of folders, used to look for DLLs.
	std::vector<std::string> GetDirs() const;
	// Sets list of folders, where to look for DLLs.
	void SetDirs(const std::vector<std::string>& _vDirs);
	// Add new path, where to look for DLLs.
	void AddDir(const std::string& _sDir);
	// Removes specified path from the list of directories, used to look for DLLs.
	void RemoveDir(size_t _index);
	// Moves path upwards in the list.
	void UpDir(size_t _index);
	// Moves path downwards in the list.
	void DownDir(size_t _index);

	// Returns all models found in all added directories.
	std::vector<SModelInfo> GetAllAvailableModels() const;
	// Returns true if model with specified type was already loaded.
	bool IsModelDefined(const EMusenModelType& _modelType) const;
	// Returns true if model was reimplemented for GPU.
	bool IsModelGPUCompatible(const EMusenModelType& _modelType) const;
	// Returns loaded model of specified type. If no model of this type was loaded, returns nullptr.
	CAbstractDEMModel* GetModel(const EMusenModelType& _modelType);
	// Returns loaded model of specified type. If no model of this type was loaded, returns nullptr.
	const CAbstractDEMModel* GetModel(const EMusenModelType& _modelType) const;
	// Returns full path to the DLL of a specified model type, which is currently loaded. If no model loaded, returns empty string.
	std::string GetModelPath(const EMusenModelType& _modelType) const;
	// Sets full path to the DLL, which will be loaded as a model of a specified type.
	void SetModelPath(const EMusenModelType& _modelType, const std::string& _sPath);
	// Returns parameters of a loaded model of specified type. Returns empty string if no model of this type was loaded.
	std::string GetModelParameters(const EMusenModelType& _modelType) const;
	// Sets parameters to a loaded model of specified type.
	void SetModelParameters(const EMusenModelType& _modelType, const std::string& _sParams);
	// Sets default parameters to a loaded model of specified type.
	void SetModelDefaultParameters(const EMusenModelType& _modelType);
	// Returns description of error occurred during loading of model with specified type. If no errors occurred, returns empty string.
	std::string GetModelError(const EMusenModelType& _modelType) const;

	// Returns structure with bools for optional variables used by active models
	const SOptionalVariables GetUtilizedVariables();
	// set and get analysis that between interconnected particles contact should be calculated
	bool GetConnectedPPContact() const { return m_bConnectedPPContact; }
	void SetConnectedPPContact(bool _bPPContact) { m_bConnectedPPContact = _bPPContact; }

	// Uses the same file as system structure to load configuration.
	void LoadConfiguration() override;
	// Uses the same file as system structure to store configuration.
	void SaveConfiguration() override;

private:
	void LoadModelConfiguration(const ProtoMusenModel& _protoModel, const EMusenModelType& _modelType, SModelInfo* _pModelInfo);
	void SaveModelConfiguration(ProtoMusenModel* _pProtoModel, SModelInfo& _model);

	// Updates m_vAvailableModels according to m_vDirList.
	void UpdateAvailableModels();

	/// Loads static or dynamic model by specified path.
	/// If some error occurs during loading, writes error message to m_sError and returns nullptr as a model.
	/// If interface version (for dynamic) or model type (for static and dynamic) are specified, they will also be considered during the error analysis.
	SModelInfo LoadModelByPath(const std::string& _sModelPath, const EMusenModelType& _modelType = EMusenModelType::UNSPECIFIED);
	/// Loads model by specified key, looking among all available static models and dynamic models in all added folders.
	/// If some error occurs during loading, writes error message to m_sError and returns nullptr as a model.
	/// If interface version (for dynamic) or model type (for static and dynamic) are specified, they will also be considered during the error analysis.
	SModelInfo LoadModelByKey(const std::string& _sModelKey, const EMusenModelType& _modelType = EMusenModelType::UNSPECIFIED);

	/// Loads dynamic model from specified full path. If some error occurs during loading, writes error message to m_sError and returns nullptr.
	/// If interface version or model type are specified, they will also be considered during the error analysis.
	SModelInfo LoadDynamicModel(const std::string& _sModelPath, const EMusenModelType& _modelType = EMusenModelType::UNSPECIFIED);
	/// Loads static model by specified name. If some error occurs during loading, writes error message to m_sError and returns nullptr.
	/// If model type is specified, it will also be considered during the error analysis.
	SModelInfo LoadStaticModel(const std::string& _sModelName, const EMusenModelType& _modelType = EMusenModelType::UNSPECIFIED);

	// Returns description of occurred error.
	std::string BuildErrorDescription(const EMusenModelType& _modelType, const EErrorType& _type) const;

	// Clears model of specified type.
	void ClearModel(const EMusenModelType& _modelType);
	// Clears all models.
	void ClearAllModels();
};

