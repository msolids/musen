/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

uniform sampler2D Texture;
uniform mat4 MatrixP;

varying vec3 RealSphereColor;
varying float RealSphereRadius;
varying vec3 EyeSpaceSphereCenterPosition;

void main()
{
	vec3 Normal;
	Normal.xy = gl_TexCoord[0].xy * 2.0 - 1.0;
	float xx_plus_yy = dot( Normal.xy, Normal.xy );
	if ( xx_plus_yy > 1.0 )
		discard; // kill pixels outside circle

	Normal.z = sqrt( 1.0 - xx_plus_yy );

	// calculating depth:
	vec4 EyeSpaceCurrentPixelPosition = vec4(EyeSpaceSphereCenterPosition + Normal * RealSphereRadius, 1.0);
	vec4 ClipSpaceCurrentPixelPosition = MatrixP * EyeSpaceCurrentPixelPosition;
	gl_FragDepth = ClipSpaceCurrentPixelPosition.z / ClipSpaceCurrentPixelPosition.w * 0.5 + 0.5; // [-1; 1] -> [0; 1]

	gl_FragColor = vec4(RealSphereColor, 1.0) * texture2D(Texture, gl_TexCoord[0].xy);
};