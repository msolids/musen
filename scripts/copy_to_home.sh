#!/bin/bash

# Copyright (c) 2023, MUSEN Development Team. All rights reserved. This file is part of MUSEN framework http://msolids.net/musen. See LICENSE file for license and warranty information.

rsync -av --exclude=.git --exclude=.vs --exclude=ExternalLibraries --exclude=Installers/Compiler --exclude=Installers/Installers --exclude=MUSEN_Linux --exclude=x64 /mnt/d/Codes/musen ~/