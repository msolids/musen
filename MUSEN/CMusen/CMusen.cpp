/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include <iostream>
#include "BuildVersion.h"
#include "ScriptAnalyzer.h"
#include "ScriptRunner.h"
#include "ArgumentsParser.h"
#include "MUSENVersion.h"

void PrintArgumentsInfo()
{
	std::cout << "Usage: cmusen -key[=value] [-key[=value]] ..." << std::endl;
	std::cout << std::endl;
	std::cout << "Mandatory arguments to start simulation:" << std::endl;
	std::cout << "-s, -script     path to script file" << std::endl;
	std::cout << std::endl;
	std::cout << "Optional arguments:" << std::endl;
	std::cout << "-t, -threads    maximum number of threads available for the program" << std::endl;
	std::cout << "-a, -affinity   hexadecimal mask of cores pin threads to them" << std::endl;
	std::cout << std::endl;
	std::cout << "Information:" << std::endl;
	std::cout << "-v, -version    print information about current version" << std::endl;
	std::cout << "-m, -models     print information about available models and their parameters" << std::endl;
}

void PrintModelsInfo()
{
	CModelManager modelManager;
#ifndef STATIC_MODULES
	modelManager.AddDir(".");	// add current directory
#endif
	std::vector<CModelManager::SModelInfo> models = modelManager.GetAllAvailableModels();
	std::cout << " ===== Available models ===== " << std::endl;
	std::cout << "Particle-particle contacts: " << std::endl;
	for (const auto& model : models)
		if (model.pModel->GetType() == EMusenModelType::PP)
			std::cout << "  " << model.sPath << " (" << model.pModel->GetName() << ") " << model.pModel->GetParametersStr() << std::endl;
	std::cout << "Particle-wall contacts: " << std::endl;
	for (const auto& model : models)
		if (model.pModel->GetType() == EMusenModelType::PW)
			std::cout << "  " << model.sPath << " (" << model.pModel->GetName() << ") " << model.pModel->GetParametersStr() << std::endl;
	std::cout << "Solid bonds: " << std::endl;
	for (const auto& model : models)
		if (model.pModel->GetType() == EMusenModelType::SB)
			std::cout << "  " << model.sPath << " (" << model.pModel->GetName() << ") " << model.pModel->GetParametersStr() << std::endl;
	std::cout << "Liquid bonds: " << std::endl;
	for (const auto& model : models)
		if (model.pModel->GetType() == EMusenModelType::LB)
			std::cout << "  " << model.sPath << " (" << model.pModel->GetName() << ") " << model.pModel->GetParametersStr() << std::endl;
	std::cout << "External force: " << std::endl;
	for (const auto& model : models)
		if (model.pModel->GetType() == EMusenModelType::EF)
			std::cout << "  " << model.sPath << " (" << model.pModel->GetName() << ") " << model.pModel->GetParametersStr() << std::endl;
}

void PrintVersionInfo()
{
	std::cout << "Version: " << MUSEN_VERSION_STR << std::endl;
	std::cout << "Build: " << CURRENT_BUILD_VERSION << std::endl;
}

void SetMaxThreads(const std::string& _arg)
{
	std::stringstream ss{ _arg };
	const auto threads = GetValueFromStream<size_t>(&ss);
	ThreadPool::CThreadPool::SetMaxThreadsNumber(threads);
}

void SetThreadsList(const std::string& _arg)
{
	const auto Hex2Bin = [](const std::string& _hex)
	{
		std::string res;
		for (auto c : _hex)
		{
			uint8_t n;
			if (c >= '0' && c <= '9')	n = c - '0';
			else						n = 10 + c - 'A';
			for (int8_t j = 3; j >= 0; --j)
				res.push_back(n & 1 << j ? '1' : '0');
		}
		return res;
	};

	std::string mask = Hex2Bin(_arg);
	std::reverse(mask.begin(), mask.end());

	std::vector<int> listCPUs;
	for (int i = 0; i < static_cast<int>(mask.size()) && i < static_cast<int>(std::thread::hardware_concurrency()); ++i)
		if (mask[i] == '1')
			listCPUs.push_back(i);
	ThreadPool::CThreadPool::SetUserCPUList(listCPUs);
}

void RunMusen(const std::string& _arg)
{
	InitializeThreadPool();

	std::cout << "Script name:    " << _arg << std::endl;
	const CScriptAnalyzer ScriptAnalyzer(_arg);
	std::cout << "Number of jobs: " << ScriptAnalyzer.JobsCount() << std::endl;
	if (!ScriptAnalyzer.JobsCount())
	{
		std::cout << "Error: The script file cannot be opened, contains errors, or has no jobs." << std::endl;
		return;
	}

	size_t counter = 0;
	CScriptRunner ScriptRunner;
	for (const auto& job : ScriptAnalyzer.Jobs())
	{
		std::cout << std::endl << "//////////////////// Start processing job: " << 1 + counter++ << " ////////////////////" << std::endl;
		ScriptRunner.RunScriptJob(job);
	}
}

int main(int argc, const char *argv[])
{
	const CArgumentsParser parser(argc, argv);

	if (parser.ArgumentsNumber() == 0)
	{
		PrintArgumentsInfo();
		return 0;
	}

	if (parser.IsArgumentExist("models") || parser.IsArgumentExist("m"))
		PrintModelsInfo();
	if (parser.IsArgumentExist("version") || parser.IsArgumentExist("v"))
		PrintVersionInfo();
	if (parser.IsArgumentExist("threads") || parser.IsArgumentExist("t"))
		SetMaxThreads(parser.IsArgumentExist("t") ? parser.GetArgument("t") : parser.GetArgument("threads"));
	if (parser.IsArgumentExist("affinity") || parser.IsArgumentExist("a"))
		SetThreadsList(parser.IsArgumentExist("a") ? parser.GetArgument("a") : parser.GetArgument("affinity"));
	if (parser.IsArgumentExist("script") || parser.IsArgumentExist("s"))
		RunMusen(parser.IsArgumentExist("s") ? parser.GetArgument("s") : parser.GetArgument("script"));

	return 0;
}
