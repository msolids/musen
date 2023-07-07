/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "MUSENComponent.h"
#include "AbstractDEMModel.h"

enum class ELibType { STATIC, DYNAMIC };

class CModelDescriptor
{
	friend class CModelManager;

	// TODO: rename variables.
	std::unique_ptr<CAbstractDEMModel> pModel{}; // Pointer to a model.
	// TODO: rename to name.
	std::string sPath;                           // Full path to a model (dynamic) or its name (static).
	// TODO: remove if unused.
	std::string sError; // Description of occurred error during loading.
	ELibType libType{ ELibType::STATIC }; // Type of underlying library: static or dynamic.

public:
	CModelDescriptor() = default;
	CModelDescriptor(std::string _error, ELibType _libType);
	CModelDescriptor(std::unique_ptr<CAbstractDEMModel> _model, std::string _path, ELibType _libType);
	CModelDescriptor(const CModelDescriptor&) = delete; // can not copy unique pointer to the model
	CModelDescriptor(CModelDescriptor&&) noexcept = default;
	~CModelDescriptor() = default;
	CModelDescriptor& operator=(const CModelDescriptor& _other) = delete; // can not copy unique pointer to the model
	CModelDescriptor& operator=(CModelDescriptor&& _other) noexcept = default;

	// Returns a const pointer to the model itself. If the model was not loaded, returns nullptr.
	[[nodiscard]] const CAbstractDEMModel* GetModel() const;
	// Returns a pointer to the model itself. If the model was not loaded, returns nullptr.
	[[nodiscard]] CAbstractDEMModel* GetModel();
	// TODO: rename name to path and adjust comments
	// Returns name or full path to the DLL of the model.
	[[nodiscard]] std::string GetName() const;
	// Returns description of the error occurred during loading of the model. If no errors occurred, returns empty string.
	[[nodiscard]] std::string GetError() const;
	// Returns type of the model's library.
	[[nodiscard]] ELibType GetLibType() const;
};

class CModelManager : public CMusenComponent
{
private:
	enum class EErrorType
	{
		WRONG_PATH, WRONG_DLL_INTERFACE, WRONG_DLL_VERSION, UNSPEC_MODEL_TYPE, WRONG_MODEL_TYPE
	};
	std::vector<std::string> m_vDirList;					// List of directories, where to look for DLLs.
	std::vector<CModelDescriptor> m_vAvailableModels;				// List of all models from all directories.
	// TODO: rename to active
	std::vector<CModelDescriptor> m_vCurrentModels;	            // List of selected/loaded models.
	bool m_bConnectedPPContact{ true };						// Whether to calculate contacts between particles directly connected with bond.

public:
	CModelManager();
	~CModelManager() override;
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

	// Returns descriptors of all models of all types found in all added directories.
	[[nodiscard]] std::vector<const CModelDescriptor*> GetAvailableModelsDescriptors() const;
	// Returns descriptors of all models of the given type found in all added directories.
	[[nodiscard]] std::vector<const CModelDescriptor*> GetAvailableModelsDescriptors(const EMusenModelType& _type) const;
	// Returns descriptors of all models of the given type found in all added directories.
	[[nodiscard]] std::vector<CModelDescriptor*> GetAvailableModelsDescriptors(const EMusenModelType& _type);
	// Returns descriptors of all models selected for simulation.
	[[nodiscard]] std::vector<const CModelDescriptor*> GetModelsDescriptors() const;
	// Returns pointers to all models selected for simulation.
	std::vector<CAbstractDEMModel*> GetAllActiveModels() const;
	// TODO: consistently use loaded/active/selected.
	// Returns true if model with specified type was already loaded.
	bool IsModelActive(const EMusenModelType& _modelType) const;
	// Returns true if model with specified name/path was already loaded.
	bool IsModelActive(const std::string& _name) const;
	// Returns true if model was reimplemented for GPU.
	//bool IsModelGPUCompatible(const EMusenModelType& _modelType) const;
	// Returns loaded model of specified type. If no model of this type was loaded, returns nullptr.
	//CAbstractDEMModel* GetModel(const EMusenModelType& _modelType);
	// Returns loaded model of specified type. If no model of this type was loaded, returns nullptr.
	//const CAbstractDEMModel* GetModel(const EMusenModelType& _modelType) const;
	// Returns all loaded models of the specified type.
	std::vector<const CModelDescriptor*> GetModelsDescriptors(const EMusenModelType& _type) const;
	// Returns all loaded models of the specified type.
	std::vector<CModelDescriptor*> GetModelsDescriptors(const EMusenModelType& _type);
	// Returns full path to the DLL of a specified model type, which is currently loaded. If no model loaded, returns empty string.
	//std::string GetModelPath(const EMusenModelType& _modelType) const;

	// Sets full path to the DLL, which will be loaded as a model of a specified type.
	//void SetModelPath(const EMusenModelType& _modelType, const std::string& _sPath);
	// Adds a model with the given name/full path to the list of active ones.
	// If such model does not exist, does nothing.
	// Returns a pointer to the model's descriptor.
	CModelDescriptor* AddActiveModel(const std::string& _name);
	// Removes a model with the given name/full path from the list of active ones.
	void RemoveActiveModel(const std::string& _name);
	// Replaces an active model with the specified name/full path with a new one.
	// If old model does not exist, just adds the new one. If new model does not exist, only removes the old one.
	// Returns a pointer to the new model's descriptor.
	CModelDescriptor* ReplaceActiveModel(const std::string& _oldName, const std::string& _newName);
	// Returns parameters of a loaded model of specified type. Returns empty string if no model of this type was loaded.
	//std::string GetModelParameters(const EMusenModelType& _modelType) const;
	// Sets parameters to a loaded model.
	void SetModelParameters(const std::string& _name, const std::string& _params) const;
	// Sets default parameters to a loaded model of specified type.
	//void SetModelDefaultParameters(const EMusenModelType& _modelType);
	// Returns description of error occurred during loading of model with specified type. If no errors occurred, returns empty string.
	//std::string GetModelError(const EMusenModelType& _modelType) const;

	// Returns structure with bools for optional variables used by active models
	SOptionalVariables GetUtilizedVariables() const;
	// set and get analysis that between interconnected particles contact should be calculated
	bool GetConnectedPPContact() const { return m_bConnectedPPContact; }
	void SetConnectedPPContact(bool _bPPContact) { m_bConnectedPPContact = _bPPContact; }

	// Uses the same file as system structure to load configuration.
	void LoadConfiguration() override;
	// Uses the same file as system structure to store configuration.
	void SaveConfiguration() override;

private:
	void LoadModelConfiguration(const ProtoMusenModel& _protoModel, const EMusenModelType& _modelType, CModelDescriptor* _pModelInfo);

	// Updates m_vAvailableModels according to m_vDirList.
	void UpdateAvailableModels();

	/// Loads static or dynamic model by specified name/full path.
	/// For dynamic models, name is equivalent to model's full path.
	/// If some error occurs during loading, writes error message to sError of the SModelInfo and returns nullptr as a model.
	/// If model type are specified, they will also be considered during the error analysis.
	static CModelDescriptor LoadModelByName(const std::string& _modelName, const EMusenModelType& _modelType = EMusenModelType::UNSPECIFIED);
	/// Loads model by specified key, looking among all available static models and dynamic models in all added folders.
	/// If some error occurs during loading, writes error message to sError of the SModelInfo and returns nullptr as a model.
	/// If model type are specified, they will also be considered during the error analysis.
	CModelDescriptor LoadModelByKey(const std::string& _modelKey, const EMusenModelType& _modelType = EMusenModelType::UNSPECIFIED) const;

	/// Loads dynamic model from specified full path.
	/// If some error occurs during loading, writes error message to sError of the SModelInfo and returns nullptr as a model.
	/// If interface version or model type are specified, they will also be considered during the error analysis.
	static CModelDescriptor LoadDynamicModel(const std::string& _modelPath, const EMusenModelType& _modelType = EMusenModelType::UNSPECIFIED);
	/// Loads static model by specified name. If some error occurs during loading, writes error message to sError of the SModelInfo and returns nullptr as a model.
	/// If model type is specified, it will also be considered during the error analysis.
	static CModelDescriptor LoadStaticModel(const std::string& _modelName, const EMusenModelType& _modelType = EMusenModelType::UNSPECIFIED);

	// Returns description of occurred error.
	static std::string BuildErrorDescription(const std::string& _model, const EErrorType& _type);

	// Clears model of specified type.
	//void ClearModel(const EMusenModelType& _modelType);
	// Clears all currently loaded/active models.
	void ClearAllActiveModels();
};

