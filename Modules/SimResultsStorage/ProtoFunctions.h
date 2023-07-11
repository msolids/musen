/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "Triangle.h"
#include "DisableWarningHelper.h"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include "SimulationDescription.pb.h"
PRAGMA_WARNING_POP

// Sets data from CVector3 to ProtoVector.
inline void Val2Proto(ProtoVector* _proto, const CVector3& _vector)
{
	_proto->set_x(_vector.x);
	_proto->set_y(_vector.y);
	_proto->set_z(_vector.z);
}

// Sets data from ProtoVector to CVector3.
inline CVector3 Proto2Val(const ProtoVector& _proto)
{
	return CVector3{ _proto.x(), _proto.y(), _proto.z() };
}

// Sets data from CMatrix3 to ProtoMatrix.
inline void Val2Proto(ProtoMatrix* _proto, const CMatrix3& _matrix)
{
	_proto->mutable_v1()->set_x(_matrix.values[0][0]);
	_proto->mutable_v1()->set_y(_matrix.values[0][1]);
	_proto->mutable_v1()->set_z(_matrix.values[0][2]);
	_proto->mutable_v2()->set_x(_matrix.values[1][0]);
	_proto->mutable_v2()->set_y(_matrix.values[1][1]);
	_proto->mutable_v2()->set_z(_matrix.values[1][2]);
	_proto->mutable_v3()->set_x(_matrix.values[2][0]);
	_proto->mutable_v3()->set_y(_matrix.values[2][1]);
	_proto->mutable_v3()->set_z(_matrix.values[2][2]);
}

// Sets data from ProtoMatrix to CMatrix3.
inline CMatrix3 Proto2Val(const ProtoMatrix& _proto)
{
	return CMatrix3{
		_proto.v1().x(), _proto.v1().y(), _proto.v1().z(),
		_proto.v2().x(), _proto.v2().y(), _proto.v2().z(),
		_proto.v3().x(), _proto.v3().y(), _proto.v3().z() };
}

// Sets data from CTriangle to ProtoTriangle.
void inline Val2Proto(ProtoTriangle* _proto, const CTriangle& _triangle)
{
	Val2Proto(_proto->mutable_vert1(), _triangle.p1);
	Val2Proto(_proto->mutable_vert2(), _triangle.p2);
	Val2Proto(_proto->mutable_vert3(), _triangle.p3);
}

// Sets data from ProtoTriangle to CTriangle.
CTriangle inline Proto2Val(const ProtoTriangle& _proto)
{
	return CTriangle{ Proto2Val(_proto.vert1()), Proto2Val(_proto.vert2()), Proto2Val(_proto.vert3()) };
}

// Sets data from a proto array of P to a vector of T.
template<typename T, typename P>
std::vector<T> Proto2Val(const google::protobuf::RepeatedPtrField<P>& _proto)
{
	std::vector<T> values(_proto.size());
	for (int i = 0; i < _proto.size(); ++i)
		values[i] = Proto2Val(_proto[i]);
	return values;
}

// Sets data from a proto array of P to a vector of T.
template<typename T, typename P>
std::vector<T> Proto2Val(const google::protobuf::RepeatedField<P>& _proto)
{
	std::vector<T> values(_proto.size());
	for (int i = 0; i < _proto.size(); ++i)
		values[i] = _proto[i];
	return values;
}

// Sets data from CColor to ProtoColor.
void inline Val2Proto(ProtoColor* _proto, const CColor& _color)
{
	_proto->set_r(_color.r);
	_proto->set_g(_color.g);
	_proto->set_b(_color.b);
	_proto->set_a(_color.a);
}

// Sets data from ProtoColor to CColor.
CColor inline Proto2Val(const ProtoColor& _proto)
{
	return CColor{ _proto.r(), _proto.g(), _proto.b(), _proto.a() };
}