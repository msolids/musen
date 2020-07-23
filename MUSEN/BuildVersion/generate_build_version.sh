#!/bin/bash

DATE_TIME=$(date +%y%m%d.%H%M%S)
GIT_HASH=$(git rev-parse --short HEAD)
GIT_NAME=$(git rev-parse --abbrev-ref HEAD)

echo "#pragma once
const std::string _BUILD_TIME = \"$DATE_TIME\";" > BuildTime.h

echo "#pragma once
const std::string _BUILD_HASH = \"$GIT_HASH.$GIT_NAME\";" > BuildHash.h
