/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <vector>
#include <set>
#include <algorithm>
#include <map>
#include "Vector3.h"

/// Uses linear interpolation to find the correct value of Y in point X.
template<typename T> T InterpolatedValue(const std::vector<double>& _xValues, const std::vector<T>& _yValues, double _x)
{
	if (_xValues.empty()) return T{ 0 };
	if (_xValues.size() != _yValues.size()) return T{ 0 };
	if (_xValues.size() == 1) return _yValues.front();

	size_t iRight = 1;
	while (iRight < _xValues.size() - 1 && _xValues[iRight] <= _x)
		++iRight;
	if (_xValues[iRight] == _xValues[iRight - 1]) return _yValues[iRight];

	const size_t iLeft = iRight - 1;
	return _yValues[iLeft] + (_x - _xValues[iLeft]) * (_yValues[iRight] - _yValues[iLeft]) / (_xValues[iRight] - _xValues[iLeft]);
}

// return the sum of all elements of vector
template<typename T> T VectorSum(const std::vector<T>& _vVec)
{
	T sum = 0;
	for (size_t i = 0; i < _vVec.size(); ++i)
		sum += _vVec[i];
	return sum;
}

// Return the lowest element in vector.
template<typename T> T VectorMin(const std::vector<T>& _vec)
{
	if (_vec.empty()) return T{};
	return *std::min_element(_vec.begin(), _vec.end());
}

// Return the greatest element in vector.
template<typename T> T VectorMax(const std::vector<T>& _vec)
{
	if (_vec.empty()) return T{};
	return *std::max_element(_vec.begin(), _vec.end());
}

// Returns a vector with reserved size.
template<typename T> std::vector<T> ReservedVector(size_t _size)
{
	std::vector<T> res;
	res.reserve(_size);
	return res;
}

// return the index of maximal value in the vector
unsigned inline VectorMaxIndex( const std::vector<double>& _vVec )
{
	if ( _vVec.empty() ) return 0;
	double dMax = _vVec[ 0 ];
	unsigned nMaxIndex = 0;
	for ( unsigned i=1; i < _vVec.size(); i++ )
		if ( _vVec[ i ] > dMax )
		{
			dMax = _vVec[i];
			nMaxIndex = i;
		}
	return nMaxIndex;
}

// Returns true if vector contains specified value at leas once.
template<typename T> bool VectorContains(const std::vector<T>& _vec, T _val)
{
	return std::find(_vec.begin(), _vec.end(), _val) != _vec.end();
}

template<typename T> std::vector<T> VectorDifference(const std::vector<T>& _v1, const std::vector<T>& _v2)
{
	std::vector<T> res;
	std::set<T> set1(_v1.begin(), _v1.end());
	std::set<T> set2(_v2.begin(), _v2.end());
	std::set_difference(set1.begin(), set1.end(), set2.begin(), set2.end(), std::back_inserter(res));
	return res;
}

template<typename T> std::vector<T> VectorIntersection(const std::vector<T>& _v1, const std::vector<T>& _v2)
{
	std::vector<T> res;
	std::set<T> set1(_v1.begin(), _v1.end());
	std::set<T> set2(_v2.begin(), _v2.end());
	std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), std::back_inserter(res));
	return res;
}

template<typename T> std::vector<T> VectorIntersection(std::initializer_list<std::vector<T>> _vectors)
{
	if (_vectors.size() == 0) return {};
	std::vector<T> v0 = *_vectors.begin();
	std::set<T> res(v0.begin(), v0.end());
	std::set<T> temp;
	for (size_t i = 1; i < _vectors.size(); ++i)
	{
		std::vector<T> v = *(_vectors.begin() + i);
		std::set_intersection(res.begin(), res.end(), v.begin(), v.end(), std::inserter(temp, temp.begin()));
		std::swap(res, temp);
		temp.clear();
	}
	return std::vector<T>(res.begin(), res.end());
}

template<typename T> std::vector<T> VectorUnion(const std::vector<T>& _v1, const std::vector<T>& _v2)
{
	std::vector<T> res;
	std::set<T> set1(_v1.begin(), _v1.end());
	std::set<T> set2(_v2.begin(), _v2.end());
	std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(), std::back_inserter(res));
	return res;
}

template<typename T> bool inline IsSubset(const std::vector<T>& _set, const std::vector<T>& _subset)
{
	return VectorIntersection(_set, _subset).size() == _subset.size();
}

// Returns true if set contains the specified value at leas once.
template<typename T> bool SetContains(const std::set<T>& _set, T _val)
{
	return std::find(_set.begin(), _set.end(), _val) != _set.end();
}

template<typename T> std::set<T> SetDifference(const std::set<T>& _set1, const std::set<T>& _set2)
{
	std::set<T> res;
	std::set_difference(_set1.begin(), _set1.end(), _set2.begin(), _set2.end(), std::inserter(res, res.begin()));
	return res;
}

template<typename T> std::set<T> SetIntersection(const std::set<T>& _set1, const std::set<T>& _set2)
{
	std::set<T> res;
	std::set_intersection(_set1.begin(), _set1.end(), _set2.begin(), _set2.end(), std::inserter(res, res.begin()));
	return res;
}

template<typename T> std::set<T> SetUnion(const std::set<T>& _set1, const std::set<T>& _set2)
{
	std::set<T> res;
	std::set_union(_set1.begin(), _set1.end(), _set2.begin(), _set2.end(), std::inserter(res, res.begin()));
	return res;
}

template<typename K, typename V>
bool MapContainsKey(const std::map<K, V>& _map, K _key)
{
	return _map.find(_key) != _map.end();
}