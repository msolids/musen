/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#version 130

uniform highp mat4 u_matrix_mvp;	// model-view-projection matrix

attribute highp vec3 a_position;	// position of vertex
attribute highp vec4 a_color;		// color of vertex

varying highp vec4 v_color;			// color of vertex for fragment shader

void main()
{
	v_color     = a_color;

	gl_Position = u_matrix_mvp * vec4(a_position, 1.0);
}
