/* Copyright (c) 2013-2020, MUSEN Development Team.
 * Copyright (c) 2026, DyssolTEC GmbH.
 * All rights reserved. This file is part of MUSEN framework. See LICENSE file for license and warranty information. */

#version 150 compatibility

uniform sampler2D u_texture;	// texture
uniform highp mat4 u_matrix_p;	// projection matrix

in highp vec3 g_position;	// particle center position in eye space coordinates
in highp vec4 g_color;		// color of particle
in highp float g_radius;	// radius of particle
in highp vec2 g_texcoord;	// texture coordinates

void main()
{
	highp vec3 normal;
	normal.xy = g_texcoord * 2.0 - 1.0;
	highp float mag = dot(normal.xy, normal.xy);
	if (mag > 1.0) discard; // kill pixels outside circle
	normal.z = sqrt(1.0 - mag);

	// calculate depth
	highp vec4 pixelPos = u_matrix_p * vec4(g_position + normal * g_radius, 1.0);
	gl_FragDepth = pixelPos.z / pixelPos.w * 0.5 + 0.5; // [-1; 1] -> [0; 1]

	gl_FragColor = g_color * texture(u_texture, g_texcoord);
};
