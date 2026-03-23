/* Copyright (c) 2026, DyssolTEC GmbH.
   All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
   See LICENSE file for license and warranty information. */

#pragma once

class IPackageGeneratorHelper;
struct SGPUParticles;

/// Factory to create a GPU package generator helper.
/// @param _particles Pointer to particles on GPU.
/// @return Pointer to the created GPU package generator helper.
IPackageGeneratorHelper* CreatePackageGeneratorHelperGPU(SGPUParticles* _particles);
