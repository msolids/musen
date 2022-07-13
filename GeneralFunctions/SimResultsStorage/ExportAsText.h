/* Copyright (c) 2013-2022, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "ExportAsTextMacro.h"
#include "Quaternion.h"
#include "ByteStream.h"
#include <cassert>
#include <set>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <utility>

// Defines flags and initializes them with true.
#define DEFINE_FLAGS(...) FOR_EACH_VA_DEF(bool,true,__VA_ARGS__)
// Defines flags and a constructor, which puts pointers to them to a std::vector<bool*> flags of SBaseFlags.
#define CREATE_FLAGS(name, ...) DEFINE_FLAGS(__VA_ARGS__) name() { FOR_EACH_VA_REF(flags.push_back,__VA_ARGS__) }

class CPackageGenerator;
class CBondsGenerator;
class CSystemStructure;
class CConstraints;

struct SBaseFlags
{
protected:
	SBaseFlags() = default;
	~SBaseFlags() = default;
public:
	std::vector<bool*> flags;
	SBaseFlags(const SBaseFlags&) = delete;
	SBaseFlags(SBaseFlags&&) = delete;
	SBaseFlags& operator=(const SBaseFlags&) { return *this; }
	SBaseFlags& operator=(SBaseFlags&&) = delete;
	[[nodiscard]] bool AllOn() const { return std::all_of(flags.begin(), flags.end(), [](const bool* _f) { return *_f; }); }
	[[nodiscard]] bool AllOff() const { return std::all_of(flags.begin(), flags.end(), [](const bool* _f) { return !*_f; }); }
	void SetAll(bool _value) { for (auto* f : flags) *f = _value; }
	void SetFlags(const std::vector<bool>& _flags)
	{
		assert(flags.size() == _flags.size());
		for (size_t i = 0; i < flags.size(); ++i)
			*flags[i] = _flags[i];
	}
	void Copy(const SBaseFlags& _other)
	{
		assert(flags.size() == _other.flags.size());
		for (size_t i = 0; i < flags.size(); ++i)
			*flags[i] = *_other.flags[i];
	}
};

class CExportAsText
{
public:
	struct SObjectTypeFlags : SBaseFlags
	{
		CREATE_FLAGS(SObjectTypeFlags, particles, bonds, walls)
	};
	struct SConstPropsFlags : SBaseFlags
	{
		CREATE_FLAGS(SConstPropsFlags, id, type, geometry, material, activityInterval)
	};
	struct STDPartPropsFlags : SBaseFlags
	{
		CREATE_FLAGS(STDPartPropsFlags, angVel, coord, force, forceAmpl, orient, princStress, stressTensor, temperature, velocity)
	};
	struct STDBondPropsFlags : SBaseFlags
	{
		CREATE_FLAGS(STDBondPropsFlags, coord, force, forceAmpl, tangOverlap, temperature, totTorque, velocity)
	};
	struct STDWallPropsFlags : SBaseFlags
	{
		CREATE_FLAGS(STDWallPropsFlags, coord, force, forceAmpl, velocity)
	};
	struct SSceneInfoFlags : SBaseFlags
	{
		CREATE_FLAGS(SSceneInfoFlags, domain, pbc, anisotropy, contactRadius)
	};
	struct SGeometriesFlags : SBaseFlags
	{
		CREATE_FLAGS(SGeometriesFlags, baseInfo, tdProperties, wallsList, analysisVolumes)
	};
	struct SMaterialsFlags : SBaseFlags
	{
		CREATE_FLAGS(SMaterialsFlags, compounds, interactions, mixtures)
	};
	struct SGeneratorsFlags : SBaseFlags
	{
		CREATE_FLAGS(SGeneratorsFlags, packageGenerator, bondsGenerator)
	};
	/*
	 * All flags to select elements and their properties to be exported.
	 */
	struct SExportSelector
	{
		SObjectTypeFlags objectTypes;  // Object type flags.
		SSceneInfoFlags sceneInfo;     // Scene info flags.
		SConstPropsFlags constProps;   // Constant properties flags.
		STDPartPropsFlags tdPropsPart; // TD properties flags for particles.
		STDBondPropsFlags tdPropsBond; // TD properties flags for bonds.
		STDWallPropsFlags tdPropsWall; // TD properties flags for walls.
		SGeometriesFlags geometries;   // Geometry flags.
		SMaterialsFlags materials;     // Materials flags.
		SGeneratorsFlags generators;   // Generators flags.
	private:
		std::vector<SBaseFlags*> allFlags; // Vector of pointers to all flags.
	public:
		SExportSelector() : allFlags{ &objectTypes, &sceneInfo, &constProps, &tdPropsPart, &tdPropsBond, &tdPropsWall, &geometries, &materials, &generators } {}
		SExportSelector(const SExportSelector& _rhs) : SExportSelector() { for (size_t i = 0; i < allFlags.size(); ++i) *allFlags[i] = *_rhs.allFlags[i]; }
		SExportSelector& operator=(const SExportSelector& _rhs) { for (size_t i = 0; i < allFlags.size(); ++i) allFlags[i]->Copy(*_rhs.allFlags[i]); return *this; }
		SExportSelector(SExportSelector&&) = delete;
		SExportSelector& operator=(SExportSelector&&) = delete;
		~SExportSelector() = default;
		void SetAll(bool _value) { for (auto* f : allFlags) f->SetAll(_value); }
		[[nodiscard]] bool AllOn() const { return std::all_of(allFlags.begin(), allFlags.end(), [](const auto& _flags) { return _flags->AllOn(); }); }
	};
	/*
	 * Layout descriptor of the temporary file with time-dependent data.
	 * Binary layout:
	 * part1_(tp1 tp2 ... tpN) part2_(tp1 tp2 ... tpN) ... partNp_(tp1 tp2 ... tpN)
	 * bond1_(tp1 tp2 ... tpN) bond2_(tp1 tp2 ... tpN) ... bondNp_(tp1 tp2 ... tpN)
	 * wall1_(tp1 tp2 ... tpN) wall2_(tp1 tp2 ... tpN) ... wallNp_(tp1 tp2 ... tpN)
	 * For easier access, all time points are stored, regardless their activity.
	 */
	struct SBinDescriptor
	{
		size_t objNum{};     // Number of objects of this type.
		size_t entryLen{};   // Length in bytes of an entry with parameters of a single time point of a single object of this type.
		uint64_t fullLen{};  // Size of all data for objects of this type.
		uint64_t startPos{}; // Position in file of the first record with objects of this type.
	};

private:
	/*
	 * Constant data for each object.
	 */
	struct SConstantData
	{
		// TODO: do not fill optional if not required
		size_t id{};						// ID. Required.
		unsigned type{};					// Type: SPHERE, BOND, etc. Required.
		std::string geometry;				// Geometry, depending on type: radius, length, etc. Optional.
		std::string compound;				// Compound key. Optional.
		std::pair<double, double> activity; // Activity time interval. Optional.
		SConstantData(size_t _id, unsigned _type, std::string _geometry, std::string _compound, std::pair<double, double> _activity)
			: id{ _id }, type{ _type }, geometry{ std::move(_geometry) }, compound{ std::move(_compound) }, activity{ std::move(_activity) } {}
	};
	/*
	 * Data for extracting time-dependent properties.
	 */
	struct STDData
	{
		std::filesystem::path binFileName;	                    // Name of temporary binary file for time-dependent data.
		std::ofstream binFileW;                                 // Handler of temporary binary file for writing time-dependent data.
		std::ifstream binFileR;                                 // Handler of temporary binary file for reading time-dependent data.
		std::map<unsigned, SBinDescriptor> binFileLayout;       // Layout information of temporary binary file for time-dependent data, <object_type, data>.
		// TODO: consider store in file; for 30M objects and 1K TP will use about 3.6 GiB RAM
		std::unordered_map<size_t, std::vector<bool>> tpActive; // For time-dependent data, flags whether the object is to be exported in specific time point, <object ID><whether to export a point>.
	};

	SExportSelector m_selectors; // Selection of objects and properties to export.

	const CSystemStructure* m_systemStructure{};   // Pointer to system structure.
	const CConstraints* m_constraints{};           // Pointer to defined constraints.
	const CPackageGenerator* m_packageGenerator{}; // Pointer to package generator.
	const CBondsGenerator* m_bondsGenerator{};     // Pointer to bonds generator.

	std::filesystem::path m_resFileName; // Name of resulting text file.
	std::ofstream m_resFile;             // Resulting text file.

	std::vector<double> m_timePoints;                     // List of time points, which should be considered.
	std::streamsize m_precision{ std::cout.precision() }; // Decimal precision to be used to format floating-point values of time-dependent properties.

	ERunningStatus m_status{ ERunningStatus::IDLE }; // Current running status of the exporter: IDLE, RUNNING, etc.
	std::string m_statusMessage;	                 // Text description of the current status of the exporter.
	double m_progress{};                             // Exporting progress in percent.

	std::vector<SConstantData> m_constData; // Gathered constant data for each object, prepared for export.
	STDData m_tdData;                       // Data needed to handle time-dependent object properties.

public:
	// Sets pointers to required objects. Must be called before export.
	void SetPointers(const CSystemStructure* _systemStructure, const CConstraints* _constraints, const CPackageGenerator* _pakageGenerator, const CBondsGenerator* _bondsGenerator);

	// Sets exporting settings.
	void SetSelectors(const SExportSelector& _selectors);
	// Sets name of text file for data export.
	void SetFileName(const std::filesystem::path& _name);
	// Sets vector of time points which should be considered.
	void SetTimePoints(const std::vector<double>& _timePoints);

	// Sets decimal precision to be used to format floating-point values of time-dependent properties.
	void SetPrecision(int _precision);
	// Returns decimal precision to be used to format floating-point values of time-dependent properties.
	[[nodiscard]] int GetPrecision() const;

	// Returns current status of data export (IDLE, RUNNING, ...).
	[[nodiscard]] ERunningStatus GetStatus() const;

	// Returns exporting progress in percent.
	[[nodiscard]] double GetProgress() const;
	// Returns a string with a description of the current job.
	[[nodiscard]] std::string GetStatusMessage() const;

	// Requests export stop by setting current status to TO_BE_STOPPED.
	void RequestStop();
	// Requests export stop with the given message and sets current status to FAILED.
	void RequestErrorStop(const std::string& _message);
	// Writes status message and sets status.
	void SetStatus(const std::string& _message, ERunningStatus _status = ERunningStatus::RUNNING);

	// Exports data with the given settings into the text file.
	void Export();

private:
	// Applies all selected object-related filters and returns IDs that need to be considered.
	[[nodiscard]] std::set<size_t> FilterObjects(const std::set<size_t>& _ids);
	// Filters IDs by selected object type.
	[[nodiscard]] std::set<size_t> FilterObjectsByType(const std::set<size_t>& _ids);
	// Filters out objects that are not active during the whole selected time interval.
	[[nodiscard]] std::set<size_t> FilterObjectsByActivity(const std::set<size_t>& _ids);
	// Filters IDs by selected constraints.
	[[nodiscard]]//std::set<size_t> FilterObjectsByConstraints(const std::set<size_t>& _ids);
	// Filters IDs by selected material constraints.
	[[nodiscard]] std::set<size_t> FilterObjectsByMaterial(const std::set<size_t>& _ids);
	// Filters IDs by selected diameter constraints.
	[[nodiscard]] std::set<size_t> FilterObjectsByDiameter(const std::set<size_t>& _ids);
	// Filters IDs by selected volume constraints. Also fills time-dependent activity flags.
	[[nodiscard]] std::set<size_t> FilterObjectsByVolume(const std::set<size_t>& _ids);

	// Gathers constant object parameters into in-memory structure for faster access during export. On error, writes to status.
	void PrepareConstData(const std::set<size_t>& _objectsID);
	// Gathers time-dependent object parameters into temporary binary file fro faster access during export. On error, writes to status.
	void PrepareTDData();

	// Writes objects data into results file.
	void WriteObjectsData();
	// Writes scene data into results file.
	void WriteSceneData();
	// Writes geometries data into results file.
	void WriteGeometriesData();
	// Writes materials data into results file.
	void WriteMaterialsData();
	// Writes generators data into results file.
	void WriteGeneratorsData();

	// Tries to open the file given by _name for writing, associating it with _file, in the given opening _mode. On error, writes to status.
	void TryOpenFileW(std::ofstream& _file, const std::filesystem::path& _name, std::ios::openmode _mode = static_cast<std::ios::openmode>(0));
	// Tries to open the file given by _name for reading, associating it with _file, in the given opening _mode. On error, writes to status.
	void TryOpenFileR(std::ifstream& _file, const std::filesystem::path& _name, std::ios::openmode _mode = static_cast<std::ios::openmode>(0));
	// Checks current status and returns true if it does not allow to proceed export, e.g. TO_BE_STOPPED or ERROR.
	[[nodiscard]] bool ToBeStopped() const;
	// Cleans up everything before exiting export.
	void Finalize();

	// Calculates values for binary file layout, filling m_tdFileLayout.
	void CalculateBinFileLayout();

	// Returns a stream with filtered time-dependent data in binary form for a particle. _id must be an index of a particle in m_constData.
	[[nodiscard]] CByteStream GetBinDataPart(size_t _id) const;
	// Returns a stream with filtered time-dependent data in binary form for a bond. _id must be an index of a bond in m_constData.
	[[nodiscard]] CByteStream GetBinDataBond(size_t _id) const;
	// Returns a stream with filtered time-dependent data in binary form for a wall. _id must be an index of a wall in m_constData.
	[[nodiscard]] CByteStream GetBinDataWall(size_t _id) const;
	// Returns a stream with filtered time-dependent data in binary form for an object of a given type. _id must be an index of an object of that type in m_constData.
	[[nodiscard]] CByteStream GetBinData(size_t _id) const;

	// Writes time-dependent particle data from binary stream to result text file.
	void WriteFromBinDataPart(CByteStream& _stream);
	// Writes time-dependent bond data from binary stream to result text file.
	void WriteFromBinDataBond(CByteStream& _stream);
	// Writes time-dependent wall data from binary stream to result text file.
	void WriteFromBinDataWall(CByteStream& _stream);
	// Writes time-dependent object data from binary stream to result text file.
	void WriteFromBinData(unsigned _type, CByteStream& _stream);

	// Writes value(s) to results file with proper format.
	template <typename T, typename... Ts>
	void WriteValue(T&& _val, Ts&&... _vals);
	// Writes value(s) to results file with proper format.
	template <typename T, typename... Ts>
	void WriteValue(ETXTCommands _key, T&& _val, Ts&&... _vals);
	// Writes new line.
	void WriteLine();
	// Writes value(s) to results file with proper format and adds line end.
	template <typename T, typename... Ts>
	void WriteLine(T&& _val, Ts&&... _vals);
	// Writes value(s) to results file with proper format and adds line end.
	template <typename T, typename... Ts>
	void WriteLine(ETXTCommands _key, T&& _val, Ts&&... _vals);
};
