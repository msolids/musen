/* Copyright (c) 2013-2020, MUSEN Development Team.
 * Copyright (c) 2026, DyssolTEC GmbH.
 * All rights reserved. This file is part of MUSEN framework. See LICENSE file for license and warranty information. */

#version 330

uniform highp mat4 u_matrix_mv;	// model-view matrix

in highp vec3 a_position;	// coordinates of particle
in highp vec4 a_color;		// color of particle
in highp float a_radius;	// radius of particle

out highp vec3 v_position;	// eye-space center for geometry shader
out highp vec4 v_color;		// color for geometry shader
out highp float v_radius;	// radius for geometry shader

void main()
{
	v_position = vec3(u_matrix_mv * vec4(a_position, 1.0));
	v_color    = a_color;
	v_radius   = a_radius;
};
