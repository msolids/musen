#!/bin/sh

# Copyright (c) 2023, MUSEN Development Team. All rights reserved. 
# Copyright (c) 2026, DyssolTEC GmbH. All rights reserved. 
# This file is part of MUSEN framework http://msolids.net/musen. See LICENSE file for license and warranty information.

# Absolute path to this script, e.g. /home/user/bin/script.sh
SCRIPT=$(readlink -f "$0")
# Absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
# Run copy
rsync -av --exclude=.git --exclude=.vs --exclude=Documentation --exclude=ExternalLibraries --exclude=Installers/Compiler --exclude=Installers/Installers --exclude=MUSEN_Linux --exclude=x64 ${SCRIPTPATH}/../ ~/musen/
