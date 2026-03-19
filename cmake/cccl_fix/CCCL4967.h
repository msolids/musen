/* Copyright (c) 2026, DyssolTEC GmbH.
   All rights reserved. This file is part of MUSEN framework https://github.com/msolids/musen.
   See LICENSE file for license and warranty information. */

/* Workaround for CCCL bug https://github.com/NVIDIA/cccl/issues/4967
   It breaks compilation when >= 16 CUDA architectures are compiled. */

#include <cuda/std/__cccl/preprocessor.h>

#if !defined(_CCCL_PP_SPLICE_WITH_IMPL20)
#  define _CCCL_PP_SPLICE_WITH_IMPL20(SEP, P1, ...) \
     _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL19(SEP, __VA_ARGS__))
#  undef  _CCCL_PP_SPLICE_WITH_IMPL21
#  define _CCCL_PP_SPLICE_WITH_IMPL21(SEP, P1, ...) \
     _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL20(SEP, __VA_ARGS__))
#endif
