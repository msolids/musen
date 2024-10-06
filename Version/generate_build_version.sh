#!/bin/bash

# Copyright (c) 2023, MUSEN Development Team. All rights reserved. This file is part of MUSEN framework http://msolids.net/musen. See LICENSE file for license and warranty information.

# get current date and time
date_time=$(date +"%g%m%d.%H%M%S")

# check if git is installed and accessible
is_git_installed=true
if ! [ -x "$(command -v git)" ] ; then
	is_git_installed=false
	echo "Git not found. No additional version information will be generated."
fi

# check if it is a git repository
is_git_repo=false
if [ "$is_git_installed" = true ] ; then
	rev_parse_output=$(git rev-parse --is-inside-work-tree)
	if [ "$rev_parse_output" = true ] ; then
		is_git_repo=true
	else
		echo "Not a git repository. No additional version information will be generated."
	fi
fi

# get hash of the current git commit and branch name
hash_branch=""
if [ "$is_git_repo" = true ] ; then
	commit_hash=$(git rev-parse --short HEAD)
	branch=$(git rev-parse --abbrev-ref HEAD)
	hash_branch="${commit_hash}"."${branch}"
fi

# write build time
echo "#pragma once" > $(dirname "$0")/BuildTime.h
echo "const std::string _BUILD_TIME = \""${date_time}"\";" >> $(dirname "$0")/BuildTime.h

# write hash and branch
echo "#pragma once" > $(dirname "$0")/BuildHash.h
echo "const std::string _BUILD_HASH = \""${hash_branch}"\";" >> $(dirname "$0")/BuildHash.h
