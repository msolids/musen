/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <iterator>

#ifdef PATH_CONFIGURED
std::string inline UnicodePath(const std::string& _sPath)
{
	return _sPath;
}
#else
#include <xlocbuf>
#include <codecvt>
std::wstring inline UnicodePath(const std::string& _sPath)
{
	return std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t>().from_bytes(_sPath);
}
#endif

std::string inline GenerateKey(size_t _length = 10)
{
	std::string result;
	static const char symbols[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	for (size_t i = 0; i < _length; ++i)
		result += symbols[std::rand() % (sizeof(symbols) - 1)];
	return result;
}

// Creates a key that is not already in the provided list.
std::string inline GenerateUniqueKey(const std::vector<std::string>& _existing, size_t _length = 10)
{
	while (true)
	{
		std::string res = GenerateKey(_length);
		if (std::find(_existing.begin(), _existing.end(), res) == _existing.end())
			return res;
	}
}

// Creates a key that is not already in the provided list.
std::string inline GenerateUniqueKey(const std::string& _init, const std::vector<std::string>& _existing, size_t _length = 10)
{
	if (!_init.empty() && std::find(_existing.begin(), _existing.end(), _init) == _existing.end())
		return _init;
	return GenerateUniqueKey(_existing, _length);
}

// compare two file names and if they are same return true
bool inline FileNamesAreSame(std::string _sFileName1, std::string _sFileName2)
{
	size_t index = 0;
	while (true)
	{
		index = _sFileName1.find("//", index); 		/* Locate the substring to replace. */
		if (index == std::string::npos) break;
		_sFileName1.replace(index, 2, "\\"); 		/* Make the replacement. */
		index += 2;	/* Advance index forward so the next iteration doesn't pick it up as well. */
	}
	index = 0;
	while (true)
	{
		index = _sFileName1.find('/', index); 		/* Locate the substring to replace. */
		if (index == std::string::npos) break;
		_sFileName1.replace(index, 1, "\\"); 		/* Make the replacement. */
		index += 2;	/* Advance index forward so the next iteration doesn't pick it up as well. */
	}
	index = 0;
	while (true)
	{
		index = _sFileName1.find("\\\\", index); 	/* Locate the substring to replace. */
		if (index == std::string::npos) break;
		_sFileName1.replace(index, 2, "\\"); 		/* Make the replacement. */
		index += 2;	/* Advance index forward so the next iteration doesn't pick it up as well. */
	}
	index = 0;
	while (true)
	{
		index = _sFileName2.find("//", index); 		/* Locate the substring to replace. */
		if (index == std::string::npos) break;
		_sFileName2.replace(index, 2, "\\"); 		/* Make the replacement. */
		index += 2;	/* Advance index forward so the next iteration doesn't pick it up as well. */
	}
	index = 0;
	while (true)
	{
		index = _sFileName2.find('/', index); 		/* Locate the substring to replace. */
		if (index == std::string::npos) break;
		_sFileName2.replace(index, 1, "\\"); 		/* Make the replacement. */
		index += 2;	/* Advance index forward so the next iteration doesn't pick it up as well. */
	}
	index = 0;
	while (true)
	{
		index = _sFileName2.find("\\\\", index); 	/* Locate the substring to replace. */
		if (index == std::string::npos) break;
		_sFileName2.replace(index, 2, "\\"); 		/* Make the replacement. */
		index += 2;	/* Advance index forward so the next iteration doesn't pick it up as well. */
	}
	if (_sFileName1 == _sFileName2)
		return true;
	else
		return false;
}

inline std::istream& safeGetLine( std::istream& is, std::string& t )
{
	t.clear();
	std::streambuf* sb = is.rdbuf();

	for ( ;; ) {
		int c = sb->sbumpc();
		switch ( c ) {
		case '\n':
			return is;
		case '\r':
			if ( sb->sgetc() == '\n' )
				sb->sbumpc();
			return is;
		case EOF:
			// Also handle the case when the last line has no line ending
			if ( t.empty() )
				is.setstate( std::ios::eofbit );
			return is;
		default:
			t += (char)c;
		}
	}
}

// Returns the next value from the stream and advances stream's iterator correspondingly.
template<typename T> T GetValueFromStream(std::istream* _is)
{
	T v;
	*_is >> v;
	return v;
}

inline size_t findStringAfter(std::istream& _file, const std::string& _string, std::string& _sRetString)
{
	std::string line;
	while (!_file.eof() && !_file.fail())
	{
		safeGetLine(_file, line);

		size_t nPos = line.find(_string);
		if (nPos != std::string::npos)
		{
			_sRetString = line;
			_sRetString.erase(0, nPos + _string.size());
			return nPos + _string.size();
		}
	}
	return std::string::npos;
}

/// Looks through the text file _stream to find first _sKey in it.
/// If found takes current line after _sKey as a sequence of values divided by space and returns this sequence in vector.
/// Used only to import data from EDEM file.
template<typename T> inline void getVecFromFile(std::istream& _stream, const std::string& _sKey, std::vector<T>& _vResult)
{
	_vResult.clear();
	std::stringstream tempStream;
	std::string sTemp;
	T tempVal;
	if ((findStringAfter(_stream, _sKey, sTemp) == std::string::npos) || (sTemp == "no data")) return;
	tempStream << sTemp;
	while (tempStream.good())
	{
		tempStream >> tempVal;
		_vResult.push_back(tempVal);
	}
}

inline std::string ToLowerCase(const std::string& _s)
{
	std::string res;
	std::transform(_s.begin(), _s.end(), std::back_inserter(res), ::tolower);
	return res;
}

inline std::string ToUpperCase(const std::string& _s)
{
	std::string res;
	std::transform(_s.begin(), _s.end(), std::back_inserter(res), ::toupper);
	return res;
}

// Replaces all occurrences of _sOld with _sNew in _sStr.
inline void replaceAll(std::string& _sStr, const std::string& _sOld, const std::string& _sNew)
{
	if (_sOld.empty() || _sStr.empty()) return;
	size_t iPos = 0;
	while ((iPos = _sStr.find(_sOld, iPos)) != std::string::npos)
	{
		_sStr.replace(iPos, _sOld.length(), _sNew);
		iPos += _sNew.length();
	}
}

// Brings the path to a unified view (slashes, backslashes)
inline std::string unifyPath(const std::string& _sPath)
{
	std::string s = _sPath;
	replaceAll(s, "\\", "/");
	replaceAll(s, "//", "/");
	return s;
}

// Brings the path to a windows specific view (slashes, backslashes)
inline std::string windowsPath(const std::string& _sPath)
{
	std::string s = _sPath;
	replaceAll(s, "//", "/");
	replaceAll(s, "/", "\\");
	replaceAll(s, "\\\\", "\\");
	return s;
}

// Splits the string according to delimiter.
inline std::vector<std::string> SplitString(const std::string& _s, char _delim)
{
	std::vector<std::string> res;
	std::stringstream ss(_s);
	std::string line;
	while (std::getline(ss, line, _delim))
		res.push_back(std::move(line));
	return res;
}

inline bool StringContains(const std::string& _s, char _v)
{
	return _s.find(_v) != std::string::npos;
}

inline void TrimWhitespaces(std::string& _s)
{
	_s.erase(_s.begin(), std::find_if(_s.begin(), _s.end(), [](char ch) { return !std::isspace(ch, std::locale::classic()); }));
	_s.erase(std::find_if(_s.rbegin(), _s.rend(), [](char ch) { return !std::isspace(ch, std::locale::classic()); }).base(), _s.end());
}

inline std::string GetRestOfLine(std::istream* _is)
{
	const std::istreambuf_iterator<char> eos;
	std::string res(std::istreambuf_iterator<char>(*_is), eos);
	TrimWhitespaces(res);
	return res;
}

inline std::string Double2String(double _v)
{
	std::ostringstream os;
	os << _v;
	return os.str();
}