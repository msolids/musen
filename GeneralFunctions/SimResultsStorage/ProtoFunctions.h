/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include "Vector3.h"
#pragma warning(push)
#pragma warning(disable: 26495)
#include "GeneratedFiles/SimulationDescription.pb.h"
#pragma warning(pop)

// Sets data from CVector3 to ProtoVector.
void inline VectorToProtoVector(ProtoVector* _pProto, const CVector3& _vector)
{
	_pProto->set_x(_vector.x);
	_pProto->set_y(_vector.y);
	_pProto->set_z(_vector.z);
}

// Sets data from ProtoVector to CVector3.
CVector3 inline ProtoVectorToVector(const ProtoVector& _proto)
{
	return CVector3( _proto.x(), _proto.y(), _proto.z() );
}

// Sets data from CMatrix3 to ProtoMatrix.
void inline MatrixToProtoMatrix(ProtoMatrix* _pProto, const CMatrix3& _matrix)
{
	_pProto->mutable_v1()->set_x(_matrix.values[0][0]);
	_pProto->mutable_v1()->set_y(_matrix.values[0][1]);
	_pProto->mutable_v1()->set_z(_matrix.values[0][2]);
	_pProto->mutable_v2()->set_x(_matrix.values[1][0]);
	_pProto->mutable_v2()->set_y(_matrix.values[1][1]);
	_pProto->mutable_v2()->set_z(_matrix.values[1][2]);
	_pProto->mutable_v3()->set_x(_matrix.values[2][0]);
	_pProto->mutable_v3()->set_y(_matrix.values[2][1]);
	_pProto->mutable_v3()->set_z(_matrix.values[2][2]);
}

// Sets data from ProtoMatrix to CMatrix3.
CMatrix3 inline ProtoMatrixToMatrix(const ProtoMatrix& _proto)
{
	return CMatrix3(
		_proto.v1().x(), _proto.v1().y(), _proto.v1().z(),
		_proto.v2().x(), _proto.v2().y(), _proto.v2().z(),
		_proto.v3().x(), _proto.v3().y(), _proto.v3().z());
}

// Sets data from STriangleType to ProtoTriangle.
void inline TriangleToProtoTriangle(ProtoTriangle* _pProto, const STriangleType& _vector)
{
	VectorToProtoVector(_pProto->mutable_vert1(), _vector.p1);
	VectorToProtoVector(_pProto->mutable_vert2(), _vector.p2);
	VectorToProtoVector(_pProto->mutable_vert3(), _vector.p3);
}

// Sets data from ProtoTriangle to STriangleType.
STriangleType inline ProtoTriangleToTriangle(const ProtoTriangle& _proto)
{
	return STriangleType(ProtoVectorToVector(_proto.vert1()), ProtoVectorToVector(_proto.vert2()), ProtoVectorToVector(_proto.vert3()));
}

// Sets data from CColor to ProtoColor.
void inline ColorToProtoColor(ProtoColor* _pProto, const CColor& _color)
{
	_pProto->set_r(_color.r);
	_pProto->set_g(_color.g);
	_pProto->set_b(_color.b);
	_pProto->set_a(_color.a);
}

// Sets data from ProtoColor to CColor.
CColor inline ProtoColorToColor(const ProtoColor& _proto)
{
	return CColor(_proto.r(), _proto.g(), _proto.b(), _proto.a());
}