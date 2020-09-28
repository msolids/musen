/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

//constant for calculation radians from angle
#define PI_180  0.0174532925
#define _180_PI  57.295779579

#include "MixedFunctions.h"
#include "BasicTypes.h"
#include "SceneTypes.h"
#include "Triangle.h"
#include <algorithm>

// calculates the projection of a point onto a plane which is given by a one point and normal vector
CVector3 inline ProjectionPointToPlane( const CVector3& _SourcePoint, const CVector3& _PointOfPlane, const CVector3& _NormalVector )
{
	return _SourcePoint- _NormalVector*DotProduct( _SourcePoint-_PointOfPlane, _NormalVector );
}

bool inline IsLineIntersectTriangle( const CVector3& _linePoint1, const CVector3& _linePoint2, const CVector3& _planePoint1,
								  const CVector3& _planePoint2, const CVector3& _planePoint3)
{
	CVector3 u =_planePoint3-_planePoint1;       // second edge
	CVector3 v =_planePoint2-_planePoint1;       // first edge
	CVector3 n = u*v;
	CVector3 dir = _linePoint2 - _linePoint1;
	CVector3 w0 = _linePoint1 - _planePoint1;
	double a = -DotProduct(n, w0);
	double b = DotProduct(n, dir);


	if (fabs(b) <= 0 )
		return false;

	// get intersect point of ray with triangle plane
	double r = a / b;
	if (r < 0.0)                    // ray goes away from triangle
		return false;                   // => no intersect
	// for a segment, also test if (r > 1.0) => no intersect
	CVector3 I = _linePoint1 + dir*r;            // intersect point of ray and plane

	// is I inside T?
	double uu, uv, vv, wu, wv, D;
	uu = DotProduct(u, u);
	uv = DotProduct(u, v);
	vv = DotProduct(v,v);
	CVector3 w = I - _planePoint1;
	wu = DotProduct(w, u);
	wv = DotProduct(w, v);
	D = uv * uv - uu * vv;

	// get and test parametric coordinates
	double s, t;
	s = (uv * wv - vv * wu) / D;
	if (s < 0.0 || s > 1.0)         // I is outside T
		return false;
	t = (uv * wu - uu * wv) / D;
	if (t < 0.0 || (s + t) > 1.0)  // I is outside T
		return false;

	return true;                       // I is in T

}

// Intersects ray r = p + td, |d| = 1, with sphere s and, if intersecting, returns t value of intersection and intersection point q.
// where P is the ray origin, d - normalized direction vector, t - length of the segment
bool inline IntersectRaySphere(const CVector3& _P, const CVector3& _D, const CVector3& _SphereCenter, double _dRadius, double* _pLength, CVector3* _pIntersection )
{
	CVector3 m;
	m = _P - _SphereCenter;
	double b = DotProduct(m, _D);
	double dC = DotProduct(m, m) - _dRadius*_dRadius;
	// Exit if r�s origin outside s (c > 0)and r pointing away from s (b > 0)
	if (( dC > 0) && (b > 0)) return false;
	double dDiscr = b*b - dC;
	// A negative discriminant corresponds to ray missing sphere
	if ( dDiscr < 0 ) return false;
	// Ray now found to intersect sphere, compute smallest t value of intersection
	*_pLength = - b; // - sqrt( dDiscr );
	// If t is negative, ray started inside sphere so clamp t to zero
	if ( *_pLength < 0 ) *_pLength = 0;
	*_pIntersection = _P + _D*(*_pLength);
	return true;
}

// return the volume of intersection of two spheres
double inline SpheresIntersectionVolume( double _dRadius1, double _dRadius2, double _dDistance )
{
	if (( _dRadius1 <= 0 ) || ( _dRadius2 <= 0 ) || ( _dDistance < 0 )) return 0;
	double dR, dr; // dR > dr
	if ( _dRadius1 > _dRadius2 )
	{
		dR = _dRadius1;
		dr = _dRadius2;
	}
	else
	{
		dr = _dRadius1;
		dR = _dRadius2;
	}

	if ( _dDistance == 0 ) return 4.0/3.0*PI*pow( dr, 3 );
	if ( _dDistance + dr < dR ) return 4.0/3.0*PI*pow( dr, 3 ); // one particle is totally in the volume of another

	double db = (dr*dr - dR*dR + pow(_dDistance,2))/(2*_dDistance);
	if ( db >= 0 ) // sum of two segments
	{
		double dh1 = dr-db;
		double dh2 = dR-_dDistance+db;
		return PI/3.0*dh1*dh1*(3*dr-dh1)+PI/3.0*dh2*dh2*(3*dR-dh2);
	}
	else
	{
		double dh1 = dr-fabs(db);
		double dh2 = dR-_dDistance-fabs(db);
		return 4.0/3.0*PI*pow( dr, 3 ) - PI/3.0*dh1*dh1*(3*dr-dh1)+PI/3.0*dh2*dh2*(3*dR-dh2);
	}
}

double inline DistanceFromPointToSegment( const CVector3& _vecPoint, const CVector3& _vVert1, const CVector3& _vVert2 )
{
	return Length((_vecPoint - _vVert1)*(_vecPoint - _vVert2)) / Length(_vVert2 - _vVert1);
}

CVector3 inline LineLineIntersection(const CVector3& _p1L1, const CVector3& _p2L1, const CVector3& _p1L2, const CVector3& _p2L2)
{
	CVector3 dirVecL1 = _p2L1 - _p1L1;
	CVector3 dirVecL2 = _p2L2 - _p1L2;
	double denomX = dirVecL1.y*dirVecL2.x - dirVecL2.y*dirVecL1.x;
	double denomY = dirVecL1.x*dirVecL2.y - dirVecL2.x*dirVecL1.y;
	double denomZ = dirVecL1.y*dirVecL2.z - dirVecL2.y*dirVecL1.z;
	double x = (_p1L1.x*dirVecL1.y*dirVecL2.x - _p1L2.x*dirVecL2.y*dirVecL1.x - _p1L1.y*dirVecL1.x*dirVecL2.x + _p1L2.y*dirVecL2.x*dirVecL1.x);
	double y = (_p1L1.y*dirVecL1.x*dirVecL2.y - _p1L2.y*dirVecL2.x*dirVecL1.y - _p1L1.x*dirVecL1.y*dirVecL2.y + _p1L2.x*dirVecL2.y*dirVecL1.y);
	double z = (_p1L1.z*dirVecL1.y*dirVecL2.z - _p1L2.z*dirVecL2.y*dirVecL1.z - _p1L1.y*dirVecL1.z*dirVecL2.z + _p1L2.y*dirVecL2.z*dirVecL1.z);
	if (denomX == 0)
		if (x == 0)	x = 0;
		else		return CVector3(std::numeric_limits<double>::quiet_NaN());
	else x /= denomX;
	if (denomY == 0)
		if (y == 0)	y = 0;
		else		return CVector3(std::numeric_limits<double>::quiet_NaN());
	else y /= denomY;
	if (denomZ == 0)
		if (z == 0)	z = 0;
		else		return CVector3(std::numeric_limits<double>::quiet_NaN());
	else z /= denomZ;
	return CVector3(x, y, z);
}

// calculates distance between specified point and triangle
double inline DistanceFromPointToTriangle( const CVector3& _vecPoint, const CVector3& _vVert1, const CVector3& _vVert2, const CVector3& _vVert3 )
{
	CVector3 vEdge0 = _vVert2 - _vVert1;
	CVector3 vEdge1 = _vVert3 - _vVert1;
	CVector3 vD = _vVert1 - _vecPoint;

	double a = DotProduct(vEdge0, vEdge0);
	double b = DotProduct(vEdge0, vEdge1);
	double c = DotProduct(vEdge1, vEdge1);
	double d = DotProduct(vEdge0, vD);
	double e = DotProduct(vEdge1, vD);
	double f = DotProduct(vD, vD);

	double det = a*c - b*b;
	double s = b*e - c*d;
	double t = b*d - a*e;

	if ( s + t < det )
	{
		if ( s < 0 )
		{
			if ( t < 0 )
			{
				if ( d < 0 )
				{
					s = ClampFunction( -d/a, 0, 1 );
					t = 0;
				}
				else
				{
					s = 0;
					t = ClampFunction( -e/c, 0, 1 );
				}
			}
			else
			{
				s = 0;
				t = ClampFunction( -e/c, 0, 1 );
			}
		}
		else if ( t < 0 )
		{
			s = ClampFunction( -d/a, 0, 1 );
			t = 0;
		}
		else
		{
			double invDet = 1.0 / det;
			s *= invDet;
			t *= invDet;
		}
	}
	else
	{
		if ( s < 0 )
		{
			double tmp0 = b+d;
			double tmp1 = c+e;
			if ( tmp1 > tmp0 )
			{
				double numer = tmp1 - tmp0;
				double denom = a-2*b+c;
				s = ClampFunction( numer/denom, 0, 1 );
				t = 1-s;
			}
			else
			{
				t = ClampFunction( -e/c, 0, 1 );
				s = 0;
			}
		}
		else if ( t < 0.f )
		{
			if ( a+d > b+e )
			{
				double numer = c+e-b-d;
				double denom = a-2*b+c;
				s = ClampFunction( numer/denom, 0, 1 );
				t = 1-s;
			}
			else
			{
				s = ClampFunction( -e/c, 0, 1 );
				t = 0;
			}
		}
		else
		{
			double numer = c+e-b-d;
			double denom = a-2*b+c;
			s = ClampFunction( numer/denom, 0, 1 );
			t = 1.0 - s;
		}
	}

	return a*s*s + 2 * b*s*t + c*t*t + 2 * d*s + 2 * e*t + DotProduct(vD, vD);
	//+ SMALL;
	//return _vVert1 + vEdge0*s + vEdge1*t;
}

enum class EIntersectionType : int8_t
{
	NO_CONTACT = 0,
	FACE_CONTACT = 1,
	EDGE_CONTACT = 2,
	VERTEX_CONTACT = 3,
};

// return 0 - no contact, 1 - face contact, 2 - edge contact, 3 - vertices contact
// from publication of Su et al. Discrete element simulation of particle flow
inline std::pair<EIntersectionType, CVector3> IsSphereIntersectTriangle(const SWallStruct::SCoordinates& _wallCoords, const CVector3& _wallNormalVec,
	const CVector3& _partCoord, const double _partRadius)
{
	if (_partCoord.x <= _wallCoords.minCoord.x - _partRadius
	 || _partCoord.y <= _wallCoords.minCoord.y - _partRadius
	 || _partCoord.z <= _wallCoords.minCoord.z - _partRadius
	 || _partCoord.x >= _wallCoords.maxCoord.x + _partRadius
	 || _partCoord.y >= _wallCoords.maxCoord.y + _partRadius
	 || _partCoord.z >= _wallCoords.maxCoord.z + _partRadius)
		return { EIntersectionType::NO_CONTACT, {} };

	const CVector3 center = (_wallCoords.vert1 + _wallCoords.vert2 + _wallCoords.vert3) / 3.0;	// center of the plane
	const double ppd = DotProduct(_partCoord - center, _wallNormalVec);							//particle projection point distance
	if (std::fabs(ppd) >= _partRadius)
		return { EIntersectionType::NO_CONTACT, {} };

	const CVector3 A = _partCoord - _wallNormalVec * ppd; // projection point

	const CVector3 edge21 = _wallCoords.vert2 - _wallCoords.vert1;
	const CVector3 edge31 = _wallCoords.vert3 - _wallCoords.vert1;
	const CVector3 W = A - _wallCoords.vert1;

	const double d00 = DotProduct(edge21, edge21);
	const double d01 = DotProduct(edge21, edge31);
	const double d11 = DotProduct(edge31, edge31);
	const double d20 = DotProduct(W, edge21);
	const double d21 = DotProduct(W, edge31);
	const double invDenom = 1.0 / (d00 * d11 - d01 * d01);
	const double gamma = (d11 * d20 - d01 * d21) * invDenom;
	const double betta = (d00 * d21 - d01 * d20) * invDenom;
	const double alpha = 1.0f - gamma - betta;

	if ((gamma > 0 && gamma < 1) && (alpha > 0 && alpha < 1) && (betta > 0 && betta < 1))
		return { EIntersectionType::FACE_CONTACT, A };
	else // A is outside polygon
	{
		const CVector3 edge32 = _wallCoords.vert3 - _wallCoords.vert2;
		const CVector3 edge13 = _wallCoords.vert1 - _wallCoords.vert3;
		const double lc1 = std::min(std::max(DotProduct(A - _wallCoords.vert1, edge21) / SquaredLength(edge21), 0.), 1.);
		const double lc2 = std::min(std::max(DotProduct(A - _wallCoords.vert2, edge32) / SquaredLength(edge32), 0.), 1.);
		const double lc3 = std::min(std::max(DotProduct(A - _wallCoords.vert3, edge13) / SquaredLength(edge13), 0.), 1.);
		const bool C1IsVertice = lc1 == 0 || lc1 == 1;
		const bool C2IsVertice = lc2 == 0 || lc2 == 1;
		const bool C3IsVertice = lc3 == 0 || lc3 == 1;
		const CVector3 C1 = _wallCoords.vert1 + lc1 * edge21; // mistake in publication
		const CVector3 C2 = _wallCoords.vert2 + lc2 * edge32;
		const CVector3 C3 = _wallCoords.vert3 + lc3 * edge13;
		const double sqrLength1 = SquaredLength(C1 - _partCoord);
		const double sqrLength2 = SquaredLength(C2 - _partCoord);
		const double sqrLength3 = SquaredLength(C3 - _partCoord);
		if (std::min({ sqrLength1, sqrLength2, sqrLength3 }) >= _partRadius * _partRadius)
			return { EIntersectionType::NO_CONTACT, {} };
		if (sqrLength1 <= sqrLength2 && sqrLength1 <= sqrLength3)
			return { C1IsVertice ? EIntersectionType::VERTEX_CONTACT : EIntersectionType::EDGE_CONTACT, C1 };
		if (sqrLength2 <= sqrLength3)
			return { C2IsVertice ? EIntersectionType::VERTEX_CONTACT : EIntersectionType::EDGE_CONTACT, C2 };
		else
			return { C3IsVertice ? EIntersectionType::VERTEX_CONTACT : EIntersectionType::EDGE_CONTACT, C3 };
	}
}

namespace
{
	CUDA_DEVICE bool IsPointInDomain(const SVolumeType& _vDomain, const CVector3& _vPoint)
	{
		if (_vDomain.coordBeg.x > _vPoint.x) return false;
		if (_vDomain.coordBeg.y > _vPoint.y) return false;
		if (_vDomain.coordBeg.z > _vPoint.z) return false;
		if (_vDomain.coordEnd.x < _vPoint.x) return false;
		if (_vDomain.coordEnd.y < _vPoint.y) return false;
		if (_vDomain.coordEnd.z < _vPoint.z) return false;

		return true;
	}
}

inline bool CheckVolumesIntersection( const SVolumeType& _v1, const SVolumeType& _v2 )
{
	if ( _v1.coordBeg.x >= _v2.coordEnd.x ) return false;
	if ( _v2.coordBeg.x >= _v1.coordEnd.x ) return false;
	if ( _v1.coordBeg.y >= _v2.coordEnd.y ) return false;
	if ( _v2.coordBeg.y >= _v1.coordEnd.y ) return false;
	if ( _v1.coordBeg.z >= _v2.coordEnd.z ) return false;
	if ( _v2.coordBeg.z >= _v1.coordEnd.z ) return false;
	return true;
}

inline CVector3 SpheresContactPoint(const CVector3& _P1, const CVector3& _P2, double _R1, double _R2)
{
	return _P1 + (_P2 - _P1)*_R1 / (_R1 + _R2);
}

inline SVolumeType GetBoundingBox(const std::vector<CTriangle>& _vTriangles)
{
	SVolumeType bb;
	bb.coordBeg.Init(0);
	bb.coordEnd.Init(0);

	if (_vTriangles.empty()) return bb;
	bb.coordEnd = bb.coordBeg = _vTriangles.front().p1;
	for (size_t i = 0; i < _vTriangles.size(); ++i)
	{
		bb.coordBeg = Min(bb.coordBeg, _vTriangles[i].p1);
		bb.coordBeg = Min(bb.coordBeg, _vTriangles[i].p2);
		bb.coordBeg = Min(bb.coordBeg, _vTriangles[i].p3);

		bb.coordEnd = Max(bb.coordEnd, _vTriangles[i].p1);
		bb.coordEnd = Max(bb.coordEnd, _vTriangles[i].p2);
		bb.coordEnd = Max(bb.coordEnd, _vTriangles[i].p3);
	}
	return bb;
}

// used Rodriges algorithm to obtain rotation matrix which used to transform source vector into destination vector
inline CMatrix3 GetRotationMatrix(CVector3* _pVecSrc, CVector3* _pVecDest, bool bPositive)
{
	CMatrix3 resultMatrix;
	CMatrix3 EyeMatrix;
	CVector3 _InitVec, vecTemp;


	// Rodriges formula
	_InitVec = _pVecSrc->Normalized(); // Vector normalization
	vecTemp = _InitVec*(*_pVecDest);
	double cosTheta = DotProduct(*_pVecDest, _InitVec);

	resultMatrix.values[0][0] = 0;
	resultMatrix.values[0][1] = -vecTemp.z;
	resultMatrix.values[0][2] = vecTemp.y;

	resultMatrix.values[1][0] = vecTemp.z;
	resultMatrix.values[1][1] = 0;
	resultMatrix.values[1][2] = -vecTemp.x;

	resultMatrix.values[2][0] = -vecTemp.y;
	resultMatrix.values[2][1] = vecTemp.x;
	resultMatrix.values[2][2] = 0;

	EyeMatrix.values[0][0] = 1; EyeMatrix.values[0][1] = 0; EyeMatrix.values[0][2] = 0;
	EyeMatrix.values[1][0] = 0; EyeMatrix.values[1][1] = 1; EyeMatrix.values[1][2] = 0;
	EyeMatrix.values[2][0] = 0; EyeMatrix.values[2][1] = 0; EyeMatrix.values[2][2] = 1;

	if (vecTemp.Length() > 0)
		resultMatrix = cosTheta*EyeMatrix + resultMatrix +
		CMatrix3::GetFromVecMult(vecTemp)*(1 - cosTheta)*(1 / (vecTemp.Length()*vecTemp.Length()));
	else
	{
		if (bPositive != true)
		{
			EyeMatrix.values[0][0] = -1; EyeMatrix.values[0][1] = 0; EyeMatrix.values[0][2] = 0;
			EyeMatrix.values[1][0] = 0; EyeMatrix.values[1][1] = -1; EyeMatrix.values[1][2] = 0;
			EyeMatrix.values[2][0] = 0; EyeMatrix.values[2][1] = 0; EyeMatrix.values[2][2] = -1;
		}
		resultMatrix = EyeMatrix;
	}
	return resultMatrix;
}

inline CVector3 fabs(const CVector3& _vInput)
{
	return CVector3{ fabs(_vInput.x), fabs(_vInput.y), fabs(_vInput.z) };
}