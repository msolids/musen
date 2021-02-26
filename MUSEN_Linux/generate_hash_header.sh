#!/bin/bash

#######################################################################################
# Creates file BuildHash.h with hash and name of the current git-brach.
# The generated file is used to construct a full vesion info of the executable.
#######################################################################################

if ! [ -x "$(command -v git)" ] || [ "$(git rev-parse --is-inside-work-tree 2>/dev/null)" != "true" ]; then
	printf "#pragma once\nconst std::string _BUILD_HASH = \"default\";" > BuildHash.h
else
	GIT_HASH=$(git rev-parse --short HEAD)
	GIT_NAME=$(git rev-parse --abbrev-ref HEAD)

	printf "#pragma once\nconst std::string _BUILD_HASH = \"$GIT_HASH.$GIT_NAME\";" > BuildHash.h
fi
