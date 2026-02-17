/* Copyright (c) 2026, DyssolTEC GmbH.
 * All rights reserved. This file is part of MUSEN framework. See LICENSE file for license and warranty information. */

#version 330

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform highp mat4 u_matrix_p;	// projection matrix

in highp vec3 v_position[];	// eye-space center from vertex shader
in highp vec4 v_color[];	// color from vertex shader
in highp float v_radius[];	// radius from vertex shader

out highp vec3 g_position;	// eye-space center for fragment shader
out highp vec4 g_color;		// color for fragment shader
out highp float g_radius;	// radius for fragment shader
out highp vec2 g_texcoord;	// texture coordinates for fragment shader

void main()
{
	vec3 center = v_position[0];
	float r = v_radius[0];

	// billboard axes in eye space (camera looks along -z, right=+x, up=+y)
	vec3 right = vec3(r, 0.0, 0.0);
	vec3 up    = vec3(0.0, r, 0.0);

	// per-particle values shared by all vertices
	g_position = center;
	g_color = v_color[0];
	g_radius = r;

	// emit quad as triangle strip
	// bottom-left
	g_texcoord = vec2(0.0, 1.0);
	gl_Position = u_matrix_p * vec4(center - right - up, 1.0);
	EmitVertex();
	// bottom-right
	g_texcoord = vec2(1.0, 1.0);
	gl_Position = u_matrix_p * vec4(center + right - up, 1.0);
	EmitVertex();
	// top-left
	g_texcoord = vec2(0.0, 0.0);
	gl_Position = u_matrix_p * vec4(center - right + up, 1.0);
	EmitVertex();
	// top-right
	g_texcoord = vec2(1.0, 0.0);
	gl_Position = u_matrix_p * vec4(center + right + up, 1.0);
	EmitVertex();

	EndPrimitive();
}
