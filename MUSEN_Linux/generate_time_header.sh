#!/bin/bash

#######################################################################################
# Creates file BuildTime.h with current date and time.
# The generated file is used to construct a full vesion info of the executable.
#######################################################################################

DATE_TIME=$(date +%y%m%d.%H%M%S)

echo "#pragma once
const std::string _BUILD_TIME = \"$DATE_TIME\";" > BuildTime.h
