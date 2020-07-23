/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once
#include <string>
#include <map>

/// Parses input arguments. If only one argument is available, it will be treated as -script
class CArgumentsParser
{
	const char m_kSeparator = '=';
	const std::string m_kKeySigns = "-";

	std::map<std::string, std::string> m_tokens;

public:
	CArgumentsParser(int _argc, const char** _argv);

	// Returns total number of unique arguments.
	size_t ArgumentsNumber() const;
	// Determines whether the argument exist.
	bool IsArgumentExist(const std::string& _key) const;
	// Returns argument by its key.
	std::string GetArgument(const std::string& _key) const;

private:
	// Parses argument returning a key and a value.
	std::pair<std::string, std::string> ParseArgument(const std::string& _argument) const;

	// Removes leading m_kKeySigns.
	std::string RemoveSign(const std::string& _str) const;
	// Removes leading and trailing quotes.
	static std::string RemoveQuotes(const std::string& _str);
};

