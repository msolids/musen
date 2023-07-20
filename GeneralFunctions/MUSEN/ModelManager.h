/* Copyright (c) 2013-2023, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "MUSENComponent.h"
#include "AbstractDEMModel.h"

enum class ELibType { STATIC, DYNAMIC };

/*
 * A descriptor needed to work with model's libraries, their loading and unloading.
 */
class CModelDescriptor
{
	friend class CModelManager;

	// Deleter of dynamic libraries
	struct lib_deleter_t
	{
		void operator()(void* _handle) const;
	};

	using lib_ptr_t = std::unique_ptr<std::remove_pointer_t<void*>, lib_deleter_t>; // Automatically frees the library on removal.

	lib_ptr_t library{};                        // Handler of a dynamic model library.
	std::unique_ptr<CAbstractDEMModel> model{}; // Pointer to a model.
	std::string path;                           // Full path to a file with model (dynamic) or its name (static).
	std::string errorMessage;                   // Description of the error that occurred while loading the model.
	ELibType libType{ ELibType::STATIC };       // Type of underlying library: static or dynamic.

public:
	CModelDescriptor() = default;
	explicit CModelDescriptor(std::string _error, ELibType _libType);
	explicit CModelDescriptor(std::unique_ptr<CAbstractDEMModel> _model, std::string _path, ELibType _libType);
	explicit CModelDescriptor(std::unique_ptr<CAbstractDEMModel> _model, std::string _path, lib_ptr_t _library);
	CModelDescriptor(const CModelDescriptor&) = delete; // forbidden to copy unique pointer to the model
	CModelDescriptor(CModelDescriptor&&) noexcept = default;
	~CModelDescriptor() = default;
	CModelDescriptor& operator=(const CModelDescriptor& _other) = delete; // forbidden to copy unique pointer to the model
	CModelDescriptor& operator=(CModelDescriptor&& _other) noexcept = default;

	// Returns a const pointer to the model itself. If the model was not loaded, returns nullptr.
	[[nodiscard]] const CAbstractDEMModel* GetModel() const;
	// Returns a pointer to the model itself. If the model was not loaded, returns nullptr.
	[[nodiscard]] CAbstractDEMModel* GetModel();
	// Returns full path to the model's file (dynamic) or its name.
	[[nodiscard]] std::string GetPath() const;
	// Returns description of the error occurred during loading of the model. If no errors occurred, returns empty string.
	[[nodiscard]] std::string GetError() const;
	// Returns type of the model's library.
	[[nodiscard]] ELibType GetLibType() const;
};

class CModelManager : public CMusenComponent
{
	enum class EErrorType
	{
		WRONG_PATH, WRONG_DLL_INTERFACE, WRONG_DLL_VERSION, UNSPEC_MODEL_TYPE, WRONG_MODEL_TYPE, CANNOT_ALLOCATE
	};

	static const std::string c_libraryFileExtension;                  // OS-specific file extension of shared libraries.

	std::vector<std::string> m_dirs;                                  // List of directories, where to look for shared libraries.
	std::vector<std::unique_ptr<CModelDescriptor>> m_availableModels; // List of all available models from all directories.
	std::vector<std::unique_ptr<CModelDescriptor>> m_activeModels;    // List of active (currently selected for simulation) models.
	// TODO: move it from here
	bool m_connectedPPContact{ true };                                // Whether to calculate contacts between particles directly connected with a bond.

public:
	CModelManager();

	// Returns a list of directories, used to look for shared libraries.
	[[nodiscard]] std::vector<std::string> GetDirs() const;
	// Sets list of folders, where to look for shared libraries.
	void SetDirs(const std::vector<std::string>& _dirs);
	// Add a new path, where to look for shared libraries.
	void AddDir(const std::string& _dir);
	// Removes specified path from the list of directories, used to look for shared libraries.
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
	[[nodiscard]] std::vector<const CModelDescriptor*> GetActiveModelsDescriptors() const;
	// Returns all models selected for simulation of the specified type.
	[[nodiscard]] std::vector<const CModelDescriptor*> GetActiveModelsDescriptors(const EMusenModelType& _type) const;
	// Returns all models selected for simulation of the specified type.
	[[nodiscard]] std::vector<CModelDescriptor*> GetActiveModelsDescriptors(const EMusenModelType& _type);
	// Returns pointers to all models selected for simulation.
	[[nodiscard]] std::vector<CAbstractDEMModel*> GetAllActiveModels() const;
	// Returns true if a model with the specified type was already selected for simulation.
	[[nodiscard]] bool IsModelActive(const EMusenModelType& _modelType) const;
	// Returns true if a model with the specified name/path was already selected for simulation.
	[[nodiscard]] bool IsModelActive(const std::string& _name) const;

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
	// Returns parameters of a model selected for simulation of specified type. Returns empty string if no model of this type was selected for simulation.
	//std::string GetModelParameters(const EMusenModelType& _modelType) const;
	// Sets parameters to a model selected for simulation.
	void SetModelParameters(const std::string& _name, const std::string& _params) const;
	// Returns optional variables required by active models.
	[[nodiscard]] SOptionalVariables GetUtilizedVariables() const;

	// Get flag that between interconnected particles contact should be calculated.
	[[nodiscard]] bool GetConnectedPPContact() const { return m_connectedPPContact; }
	// Set flag that between interconnected particles contact should be calculated.
	void SetConnectedPPContact(bool _enabled) { m_connectedPPContact = _enabled; }

	// Uses the same file as system structure to load configuration.
	void LoadConfiguration() override;
	// Uses the same file as system structure to store configuration.
	void SaveConfiguration() override;

private:
	// Loads information about all currently available models.
	void UpdateAvailableModels();

	// Loads static or dynamic model by specified name (static) or full path (dynamic).
	// If some error occurs during loading, writes error message to the descriptor and returns nullptr as a pointer to the model.
	// If model type is specified, it will also be considered during the error analysis.
	static std::unique_ptr<CModelDescriptor> LoadModelByName(const std::string& _modelName, const EMusenModelType& _modelType = EMusenModelType::UNSPECIFIED);
	// Loads model by specified key, looking among all available static and dynamic models.
	// If some error occurs during loading, writes error message to the descriptor and returns nullptr as a pointer to the model.
	// If model type is specified, it will also be considered during the error analysis.
	[[nodiscard]] std::unique_ptr<CModelDescriptor> LoadModelByKey(const std::string& _modelKey, const EMusenModelType& _modelType = EMusenModelType::UNSPECIFIED) const;

	// Loads a dynamic model from specified full path to the library.
	// If some error occurs during loading, writes an error message to the descriptor and returns nullptr as a pointer to the model.
	// If model type is specified, it will also be considered during the error analysis.
	static std::unique_ptr<CModelDescriptor> LoadDynamicModel(const std::string& _modelPath, const EMusenModelType& _modelType = EMusenModelType::UNSPECIFIED);
	// Loads a static model by the specified name.
	// If some error occurs during loading, writes an error message to the descriptor and returns nullptr as a pointer to the model.
	// If model type is specified, it will also be considered during the error analysis.
	static std::unique_ptr<CModelDescriptor> LoadStaticModel(const std::string& _modelName, const EMusenModelType& _modelType = EMusenModelType::UNSPECIFIED);

	// Loads library from a file with the specified path.
	static void* LoadLibraryFromFile(const std::string& _libPath);
	// Returns address of the model constructor from the provided library.
	static void* LoadModelConstructor(void* _lib);

	// Returns a description of the occurred error.
	static std::string BuildErrorDescription(const std::string& _modelName, const EErrorType& _errorType);
};

