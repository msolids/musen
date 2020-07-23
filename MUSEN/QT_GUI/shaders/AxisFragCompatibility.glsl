/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#version 130

varying highp vec3 v_position;	// vertex coordinates
varying highp vec3 v_normal;	// normal vector
varying highp vec4 v_color;		// color of vertex

void main()
{
	highp vec3 viewPos            = vec3( 0.0,  0.0, 0.0);
	highp vec3 lightPos           = vec3(-1.0, -1.0, 1.0);  // differs from others!
	highp vec3 lightColor         = vec3( 1.0,  1.0, 1.0);
	highp vec3 lightSpecularColor = vec3( 0.4,  0.4, 0.4);

	highp float ambientStrength   = 0.25;
	highp vec3 ambient            = ambientStrength * lightColor;

	highp float diffuseStrength   = 1.0;
	highp vec3 norm               = normalize(v_normal);
	highp vec3 lightDir           = normalize(lightPos - v_position);
	highp float diff              = max(dot(norm, lightDir), 0.0);
	highp vec3 diffuse            = diffuseStrength * diff * lightColor;

	highp float specularStrength  = 0.4;
	highp vec3 viewDir            = normalize(viewPos - v_position);
	highp vec3 reflectDir         = reflect(-lightDir, norm);
	highp float spec              = pow(max(dot(viewDir, reflectDir), 0.0), 16);
	highp vec3 specular           = specularStrength * spec * lightSpecularColor;

	gl_FragColor = vec4(ambient + diffuse + specular, 1.0) * v_color;
}
