/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#version 130

varying highp vec4 v_color;		// color of vertex

void main()
{
	gl_FragColor = v_color;
}
