/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#version 130

uniform highp mat3 u_matrix_normal;	// normal matrix from model-view transformation
uniform highp mat4 u_matrix_mv;		// model-view matrix
uniform highp mat4 u_matrix_rot;	// rotation matrix
uniform highp float u_scaling;		// resize coefficient
uniform highp int u_win_width;		// window width
uniform highp int u_win_height;		// window height

attribute highp vec3 a_position;	// position of vertex
attribute highp vec3 a_normal;		// normal vector for vertex
attribute highp float a_type;		// type of axis (X=0/Y=1/Z=2)

varying highp vec3 v_position;		// vertex coordinates for fragment shader
varying highp vec3 v_normal;		// normal vector for vertex
varying highp vec4 v_color;			// color of vertex for fragment shader

void main()
{
	v_position = vec3(u_matrix_mv * vec4(a_position, 1.0));
	v_normal   = u_matrix_normal * a_normal;
	if (a_type == 0)		v_color = vec4(1.0, 0.0, 0.0, 1.0);
	else if (a_type == 1)	v_color = vec4(0.0, 1.0, 0.0, 1.0);
	else					v_color = vec4(0.0, 0.0, 1.0, 1.0);

	highp vec4 pos   = u_matrix_rot * vec4(a_position, 1.0);
	highp vec4 scale = vec4(pos.x / (u_win_width / u_scaling), pos.y / (u_win_height / u_scaling), pos.zw);
	highp vec4 shift = vec4(scale.x - 1 + (u_scaling * 1.3 / u_win_width), scale.y - 1 + (u_scaling * 1.3 / u_win_height), scale.zw);

	gl_Position = shift;
}
