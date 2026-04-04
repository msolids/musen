#!/bin/sh

# Copyright (c) 2023, MUSEN Development Team.
# Copyright (c) 2026, DyssolTEC GmbH.
# All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
# See LICENSE file for license and warranty information.

# Copy sources to /mnt/musen_src on the container's native filesystem.

SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

mkdir -p /mnt/musen_src
tar -cf - -C "${SCRIPTPATH}/.." \
    --exclude=.git \
    --exclude=.vs \
    --exclude=build \
    --exclude=ExternalLibraries \
    --exclude=Installers/Compiler \
    --exclude=install \
    --exclude=Installers/Installers \
    --exclude=MUSEN_Linux \
    --exclude=x64 \
    . | tar -xf - -C /mnt/musen_src
