:: Copyright (c) 2026, DyssolTEC GmbH.
:: All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
:: See LICENSE file for license and warranty information.

docker run -it --rm --volume=%cd%/../../:/mnt/src_host:ro musen.ubuntu18:latest bash -c "/mnt/src_host/scripts/copy_src.sh && exec bash"
pause
