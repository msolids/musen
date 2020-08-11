#!/bin/bash

DATE_TIME=$(date +%y%m%d.%H%M%S)

echo "#pragma once
const std::string _BUILD_TIME = \"$DATE_TIME\";" > ./MUSEN_src/MUSEN/BuildVersion/BuildTime.h
