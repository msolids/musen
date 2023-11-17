/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#version 130

varying highp vec4 v_color;		// color of disc
varying highp vec2 v_local;		// local coordinates of each vertex

void main()
{
	highp float dist = sqrt(dot(v_local, v_local));
	if (dist >= 1) discard; // kill pixels outside circle

	gl_FragColor = v_color;
};