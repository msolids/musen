/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#include "ArgumentsParser.h"
#include "MUSENVectorFunctions.h"
#include "MUSENStringFunctions.h"

CArgumentsParser::CArgumentsParser(int _argc, const char** _argv)
{
	for (int i = 1; i < _argc; ++i)
	{
		const auto argument = ParseArgument(std::string{ _argv[i] });
		if (i == 1 && argument.first.empty()) // special case for script file without key
			m_tokens["script"] = argument.second;
		else
			m_tokens[argument.first] = argument.second;
	}
}

size_t CArgumentsParser::ArgumentsNumber() const
{
	return m_tokens.size();
}

bool CArgumentsParser::IsArgumentExist(const std::string& _key) const
{
	return MapContainsKey(m_tokens, _key);
}

std::string CArgumentsParser::GetArgument(const std::string& _key) const
{
	return m_tokens.at(_key);
}

std::pair<std::string, std::string> CArgumentsParser::ParseArgument(const std::string& _argument) const
{
	const auto parts = SplitString(_argument, m_kSeparator);
	if (parts.empty())
		return {};
	if (parts.size() == 1)
	{
		if (StringContains(m_kKeySigns, parts.front().front())) // a key without value
			return { ToLowerCase(RemoveSign(parts[0])), "" };
		return { "", RemoveQuotes(parts.front()) }; // a value without key
	}
	return { ToLowerCase(RemoveSign(parts[0])), RemoveQuotes(parts[1]) };
}

std::string CArgumentsParser::RemoveSign(const std::string& _str) const
{
	std::string res{ _str };
	if (!res.empty() && StringContains(m_kKeySigns, res.front()))
		res.erase(0, 1);
	return res;
}

std::string CArgumentsParser::RemoveQuotes(const std::string& _str)
{
	std::string res{ _str };
	if (!res.empty() && res.front() == '"')
		res.erase(0, 1);
	if (!res.empty() && res.back() == '"')
		res.erase(res.size() - 1, 1);
	return res;
}
