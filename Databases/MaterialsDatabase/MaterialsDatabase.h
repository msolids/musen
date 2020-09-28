/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "Compound.h"
#include "Interaction.h"
#include "Mixture.h"
#include <fstream>
#include <cfloat>

class CMaterialsDatabase
{
private:
	std::string m_sFileName;
	std::vector<CCompound*> m_vCompounds;
	std::vector<CInteraction*> m_vInteractions;
	std::vector<CMixture*> m_vMixtures;

public:
	CMaterialsDatabase();
	~CMaterialsDatabase();

	/// Copies data from other database.
	CMaterialsDatabase& operator=(const CMaterialsDatabase& _mdb);

	/// Returns name of the current database file.
	std::string GetFileName() const;

	/// Creates new database by removing information about compounds and file name.
	void CreateNewDatabase();

	/// Initializes all compounds and properties with specified static conditions _dT and _dP. Then GetPropertyValue(Type)/GetValue(void) functions can be called instead
	/// of GetTPPropertyValue(Type,T,P)/GetValue(T,P) for Compounds and TPProperties to return data for specified _dT and _dP. Calling GetTPPropertyValue(Type,newT,newP)/GetValue(newT,newP)
	/// on CCompound or CTPProperty after that will reinitialize this Compound or Property with newT and newP for all next calls of GetPropertyValue(Type)/GetValue(void).
	void InitializeConditions(double _dT, double _dP);

	/// Removes information about Compounds, Interactions and Mixtures.
	void ClearData();

	/// Saves database to protobuf-file with specified name. If name is not specified, data is written to default file.
	void SaveToFile(const std::string& _sFileName = "");
	/// Loads database from protobuf-file with specified name. If name is not specified, data is loaded from default file.
	void LoadFromFile(const std::string& _sFileName = "");

	/// Saves database to provided protobuf file.
	void SaveToProtobufFile(ProtoMaterialsDatabase& _protoMDB);
	/// Loads database from provided protobuf file.
	void LoadFromProtobufFile(const ProtoMaterialsDatabase& _protoMDB);

	//////////////////////////////////////////////////////////////////////////
	/// Functions to work with compounds

	/// Returns number of defined compounds.
	size_t CompoundsNumber() const;
	/// Creates new compound and returns pointer to it.  If no key is specified or the key already exists in the database, new unique key is generated.
	CCompound* AddCompound(const std::string& _sCompoundKey = "");
	/// Creates new compound as a copy of specified one. Returns pointer to created compound.
	CCompound* AddCompound(const CCompound& _compound);
	/// Returns pointer to a compound with specified index. Returns nullptr if such compound has not been defined.
	CCompound* GetCompound(size_t _iCompound);
	/// Returns const pointer to a compound with specified index. Returns nullptr if such compound has not been defined.
	const CCompound* GetCompound(size_t _iCompound) const;
	/// Returns pointer to a compound with specified name. Returns nullptr if such compound has not been defined.
	CCompound* GetCompoundByName(const std::string& _sCompoundName);
	/// Returns const pointer to a compound with specified name. Returns nullptr if such compound has not been defined.
	const CCompound* GetCompoundByName(const std::string& _sCompoundName) const;
	/// Returns pointer to a compound with specified key. Returns nullptr if such compound has not been defined.
	CCompound* GetCompound(const std::string& _sCompoundKey);
	/// Returns const pointer to a compound with specified key. Returns nullptr if such compound has not been defined.
	const CCompound* GetCompound(const std::string& _sCompoundKey) const;
	/// Returns pointers to all compounds.
	std::vector<CCompound*> GetCompounds();
	/// Returns const pointers to all compounds.
	std::vector<const CCompound*> GetCompounds() const;
	/// Returns index of a compound with specified key. If not found -1 is returned.
	int GetCompoundIndex(const std::string& _sCompoundKey) const;
	/// Returns unique key of a compound with specified index. If not found empty string is returned.
	std::string GetCompoundKey(size_t _iCompound) const;
	/// Returns the list of unique keys of all defined compounds.
	std::vector<std::string> GetCompoundsKeys() const;
	/// Returns name of a compound with specified index. If not found empty string is returned.
	std::string GetCompoundName(size_t _iCompound) const;
	/// Returns name of a compound with specified index. If not found empty string is returned.
	std::string GetCompoundName(const std::string& _sCompoundKey) const;
	/// Returns the list of names of all defined compounds.
	std::vector<std::string> GetCompoundsNames() const;
	/// Removes compound with specified index from database.
	void RemoveCompound(size_t _iCompound);
	/// Removes compound with specified key from database.
	void RemoveCompound(const std::string& _sCompoundKey);
	/// Moves selected compound upwards in the list of compounds.
	void UpCompound(size_t _iCompound);
	/// Moves selected compound downwards in the list of compounds.
	void DownCompound(size_t _iCompound);

	//////////////////////////////////////////////////////////////////////////
	/// Functions to work with properties values

	/// Returns value of a constant property for specified compound. Returns NaN if such property doesn't exist.
	double GetConstPropertyValue(const std::string& _sCompoundKey, unsigned _nConstPropType) const;
	/// Returns value of a constant property for specified compound. Returns NaN if such property doesn't exist.
	double GetConstPropertyValue(size_t _iCompound, unsigned _nConstPropType) const;
	/// Returns value of a TP-property by specified temperature [K] and pressure [Pa] for specified compound.
	/// Returns NaN if such property doesn't exist or correlation for specified T and P has not been defined.
	double GetTPPropertyValue(const std::string& _sCompoundKey, unsigned _nTPPropType, double _dT, double _dP) const;
	/// Returns value of a TP-property by specified temperature [K] and pressure [Pa] for specified compound.
	/// Returns NaN if such property doesn't exist or correlation for specified T and P has not been defined.
	double GetTPPropertyValue(size_t _iCompound, unsigned _nTPPropType, double _dT, double _dP) const;
	/// Returns value of a TP-property  for specified compound by last used temperature and pressure (or for normal conditions for the first call) or value of a constant property.
	/// Returns NaN if such property doesn't exist or correlation for specified T and P has not been defined.
	double GetPropertyValue(const std::string& _sCompoundKey, unsigned _nPropType) const;
	/// Returns value of a TP-property  for specified compound by last used temperature and pressure (or for normal conditions for the first call) or value of a constant property.
	/// Returns NaN if such property doesn't exist or correlation for specified T and P has not been defined.
	double GetPropertyValue(size_t _iCompound, unsigned _nPropType) const;

	/// Sets value of a constant property for specified compound.
	void SetConstPropertyValue(const std::string& _sCompoundKey, unsigned _nConstPropType, double _dValue);
	/// Sets value of a constant property for specified compound.
	void SetConstPropertyValue(size_t _iCompound, unsigned _nConstPropType, double _dValue);
	///  Sets correlation of a TP-property by specified temperature and pressure interval for specified compound.
	void SetTPPropertyCorrelation(const std::string& _sCompoundKey, unsigned _nTPPropType, double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nCorrType, const std::vector<double>& _vParams);
	///  Sets correlation of a TP-property by specified temperature and pressure interval for specified compound.
	void SetTPPropertyCorrelation(size_t _iCompound, unsigned _nTPPropType, double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nCorrType, const std::vector<double>& _vParams);
	/// Sets value of a constant property or constant correlation of a TP-property for an entire temperature and pressure interval for specified compound.
	void SetPropertyValue(const std::string& _sCompoundKey, unsigned _nPropType, double _dValue);
	/// Sets value of a constant property or constant correlation of a TP-property for an entire temperature and pressure interval for specified compound.
	void SetPropertyValue(size_t _iCompound, unsigned _nPropType, double _dValue);

	//////////////////////////////////////////////////////////////////////////
	/// Functions to work with interactions

	/// Returns number of defined interactions.
	size_t InteractionsNumber() const;
	/// Returns pointer to an interaction with specified index. Returns nullptr if such interaction has not been defined.
	CInteraction* GetInteraction(size_t _iInteraction);
	/// Returns const pointer to an interaction with specified index. Returns nullptr if such interaction has not been defined.
	const CInteraction* GetInteraction(size_t _iInteraction) const;
	/// Returns pointer to an interaction between compounds with specified keys. Returns nullptr if such interaction has not been defined.
	CInteraction* GetInteraction(const std::string& _sCompoundKey1, const std::string& _sCompoundKey2);
	/// Returns const pointer to an interaction between compounds with specified keys. Returns nullptr if such interaction has not been defined.
	const CInteraction* GetInteraction(const std::string& _sCompoundKey1, const std::string& _sCompoundKey2) const;
	/// Returns index of an interaction between compounds with specified keys. If not found -1 is returned.
	int GetInteractionIndex(const std::string& _sCompoundKey1, const std::string& _sCompoundKey2) const;

	//////////////////////////////////////////////////////////////////////////
	/// Functions to work with interactions values

	/// Returns value of an interaction property for specified compounds. Returns NaN if such property doesn't exist or correlation for specified T and P has not been defined.
	double GetInteractionValue(const std::string& _sCompKey1, const std::string& _sCompKey2, unsigned _nIntType, double _dT, double _dP) const;
	/// Returns value of an interaction property for specified compounds by last used temperature and pressure (or for normal conditions for the first call).
	/// Returns NaN if such property doesn't exist or correlation for specified T and P has not been defined.
	double GetInteractionValue(const std::string& _sCompKey1, const std::string& _sCompKey2, unsigned _nIntType) const;

	/// Sets correlation of an interaction by specified temperature and pressure interval.
	void SetInteractionCorrelation(const std::string& _sCompKey1, const std::string& _sCompKey2, unsigned _nPropType, double _dT1, double _dT2, double _dP1, double _dP2, ECorrelationTypes _nCorrType, const std::vector<double>& _vParams);
	/// Sets constant correlation of an interaction for an entire temperature and pressure interval.
	void SetInteractionValue(const std::string& _sCompKey1, const std::string& _sCompKey2, unsigned _nPropType, double _dValue);

	//////////////////////////////////////////////////////////////////////////
	/// Functions to work with mixtures

	/// Returns number of defined mixtures.
	size_t MixturesNumber() const;
	/// Creates new mixture and returns pointer to it. If no key is specified or the key already exists in the database, new unique key is generated.
	CMixture* AddMixture(const std::string& _sMixtureKey = "");
	/// Creates new mixture as a copy of specified one. Returns pointer to created mixture.
	CMixture* AddMixture(const CMixture& _mixture);
	/// Returns index of a mixture with specified key. Returns -1 if such mixture has not been defined.
	int GetMixtureIndex(const std::string& _sMixtureKey) const;
	/// Returns name of the mixture.
	std::string GetMixtureName(size_t _iMixture) const;
	/// Returns name of the mixture.
	std::string GetMixtureName(const std::string& _sMixtureKey) const;
	/// Returns key of the specified mixture.
	std::string GetMixtureKey(size_t _iMixture) const;
	/// Returns pointer to a specified mixture. Returns nullptr if such mixture has not been defined.
	CMixture* GetMixture(size_t _iMixture);
	/// Returns constant pointer to a specified mixture. Returns nullptr if such mixture has not been defined.
	const CMixture* GetMixture(size_t _iMixture) const;
	/// Returns pointer to a specified mixture. Returns nullptr if such mixture has not been defined.
	CMixture* GetMixture(const std::string& _sMixtureKey);
	/// Returns constant pointer to a specified mixture. Returns nullptr if such mixture has not been defined.
	const CMixture* GetMixture(const std::string& _sMixtureKey) const;
	/// Returns pointer to a specified mixture. Returns nullptr if such mixture has not been defined.
	CMixture* GetMixtureByName(const std::string& _mixtureName);
	/// Returns constant pointer to a specified mixture. Returns nullptr if such mixture has not been defined.
	const CMixture* GetMixtureByName(const std::string& _mixtureName) const;
	/// Removes mixture with specified index from database.
	void RemoveMixture(size_t _iMixture);
	/// Removes mixture with specified key from database.
	void RemoveMixture(const std::string& _sMixtureKey);
	/// Analyzes that specified mixture is defined correctly. If not - returns error message.
	std::string IsMixtureCorrect(const std::string& _sMixtureKey) const;
	/// Moves selected mixture upwards in the list of mixtures.
	void UpMixture(size_t _iMixture);
	/// Moves selected mixture downwards in the list of mixtures.
	void DownMixture(size_t _iMixture);

	// Checks whether all entered data are valid and returns an error message.
	std::string IsDataCorrect() const;

private:
	/// Creates new Interaction between two compounds and returns pointer to it. If such interaction already exists, pointer to it will be returned.
	CInteraction* AddInteraction(const std::string& _sCompoundKey1, const std::string& _sCompoundKey2);
	/// Removes Interaction between compounds with specified keys from database.
	void RemoveInteraction(const std::string& _sCompoundKey1, const std::string& _sCompoundKey2);
	/// Removes Interaction with specified index from database.
	void RemoveInteraction(size_t _iInteraction);
	/// Analyzes the list of all defined compounds and check if all interactions between compounds are defined. If some interactions are not included, than the default interaction is generated.
	/// Interactions with nonexistent compounds are removed.
	void ConformInteractions();
	/// Adds default interactions with specified compound.
	void ConformInteractionsAdd(const std::string& _sCompoundKey);
	/// Removes interactions with specified compound.
	void ConformInteractionsRemove(const std::string& _sCompoundKey);
	/// Removes specified compound from mixtures fractions
	void ConformMixturesRemove(const std::string& _sCompoundKey);

	/// Returns list of mixtures keys that have been defined in the database.
	std::vector<std::string> GetMixturesKeys() const;
};