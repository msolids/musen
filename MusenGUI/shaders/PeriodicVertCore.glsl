/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#version 330

uniform highp mat4 u_matrix_mvp;	// model-view-projection matrix

in highp vec3 a_position;			// position of vertex
in highp float a_type;		        // type of boundary (X = 0 / Y = 1 / Z = 2)

out highp vec4 v_color;				// color of vertex for fragment shader

void main()
{
	if (a_type == 0)		v_color = vec4(1.0, 0.0, 0.0, 0.2);
	else if (a_type == 1)	v_color = vec4(0.0, 1.0, 0.0, 0.2);
	else					v_color = vec4(0.0, 0.0, 1.0, 0.2);

	gl_Position = u_matrix_mvp * vec4(a_position, 1.0);
}
