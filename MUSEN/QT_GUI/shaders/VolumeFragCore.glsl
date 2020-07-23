/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#version 330

in highp vec4 v_color;		// color of vertex

out highp vec4 o_color;		// output value of color

void main()
{
	o_color = v_color;
}
