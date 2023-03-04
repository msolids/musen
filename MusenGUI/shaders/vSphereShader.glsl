/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

attribute vec3 SphereCoordinate;
attribute vec3 SphereRadius;
attribute vec3 SphereColor;

uniform float SphereScale;   // scale to calculate size in pixels
uniform mat4 MatrixP;
uniform mat4 MatrixMV;
uniform mat4 MatrixMVP;

varying vec3 RealSphereColor;
varying float RealSphereRadius; // sphere radius for fragment shader
varying vec3 EyeSpaceSphereCenterPosition; // sphere center position in eye space coordinates for fragment shader


void main()
{
	EyeSpaceSphereCenterPosition = vec3( MatrixMV * vec4(SphereCoordinate, 1.0) ); // vec4( gl_Vertex.xyz, 1.0 )
	float EyeSpaceDistanceFromEyeToSphereCenter = length(EyeSpaceSphereCenterPosition); // camera (eye) in eye space is in position (0; 0; 0)
	RealSphereRadius = SphereRadius;
	gl_FrontColor = vec4(SphereColor, 1.0);
	gl_Position = MatrixMVP * vec4(SphereCoordinate, 1.0);
	gl_PointSize = SphereRadius * ( SphereScale / EyeSpaceDistanceFromEyeToSphereCenter );
	gl_TexCoord[0] = vec4(0.0, 0.0, 0.0, 0.0);
	RealSphereColor = SphereColor;
};