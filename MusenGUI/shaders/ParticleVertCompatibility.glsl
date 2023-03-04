/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#version 130

uniform highp float u_scale;		// scale to calculate size in pixels
uniform highp mat4 u_matrix_mv;		// model-view matrix
uniform highp mat4 u_matrix_mvp;	// model-view-projection matrix

attribute highp vec3 a_position;	// coordinates of particle
attribute highp vec4 a_color;		// color of particle
attribute highp float a_radius;		// radius of particle

varying highp vec3 v_position;		// particle center position in eye space coordinates for fragment shader
varying highp vec4 v_color;			// color of particle for fragment shader
varying highp float v_radius;		// radius of particle for fragment shader

void main()
{
	v_position = vec3(u_matrix_mv * vec4(a_position, 1.0));
	v_color    = a_color;
	v_radius   = a_radius;

	gl_Position    = u_matrix_mvp * vec4(a_position, 1.0);
	gl_PointSize   = a_radius * (u_scale / length(v_position));
};