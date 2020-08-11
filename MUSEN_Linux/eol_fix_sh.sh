#!/bin/bash

# Fix end of lines (EOL) for linux after windows modification.
find . -name '*.sh' -exec dos2unix '{}' \;