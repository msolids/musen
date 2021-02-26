/* Copyright (c) 2013-2020, MUSEN Development Team. All rights reserved.
   This file is part of MUSEN framework http://msolids.net/musen.
   See LICENSE file for license and warranty information. */

#pragma once

#include <string>
#include "DisableWarningHelper.h"
PRAGMA_WARNING_PUSH
PRAGMA_WARNING_DISABLE
#include <driver_types.h>
#include <cuda_runtime.h>
PRAGMA_WARNING_POP

class CCUDADefines
{
public:
	unsigned CUDA_BLOCKS_NUM{};
	unsigned CUDA_THREADS_PER_BLOCK{};

	CCUDADefines()
	{
		const auto settings = GetSettings(0);
		CUDA_BLOCKS_NUM = settings.first;
		CUDA_THREADS_PER_BLOCK = settings.second;
	}

	static std::pair<unsigned, unsigned> GetSettings(int _iDevice)
	{
		cudaDeviceProp prop{};
		cudaGetDeviceProperties(&prop, _iDevice);
		const int blocksNumber = prop.multiProcessorCount;
		const int threadsPerBlock = prop.singleToDoublePrecisionPerfRatio > 4 ? 256 : 512;
		return { blocksNumber, threadsPerBlock };
	}
};

/// Definition of atomicAdd() for doubles for SM < 6.0. In newer versions it is already present.
#if defined(__CUDACC__) && __CUDA_ARCH__ < 600
namespace CUDALegacy
{
	static __inline__ __device__ double myAtomicAdd(double* address, double val)
	{
		unsigned long long int* address_as_ull = (unsigned long long int*)address;
		unsigned long long int old = *address_as_ull, assumed;
		do {
			assumed = old;
			old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
		} while (assumed != old);
		return __longlong_as_double(old);
	}
}
#endif

#ifdef __CUDACC__
	#define CUDA_HOST_DEVICE __host__ __device__
	#define CUDA_DEVICE __device__
	#define CUDA_KERNEL_ARGS2_DEFAULT(FUN, ...) do { FUN<<<m_cudaDefines->CUDA_BLOCKS_NUM, m_cudaDefines->CUDA_THREADS_PER_BLOCK>>>(__VA_ARGS__); KERNEL_CHECK_ERROR; } while(0)
	#define CUDA_KERNEL_ARGS4_DEFAULT(FUN, stream, ...) do { FUN<<<m_cudaDefines->CUDA_BLOCKS_NUM, m_cudaDefines->CUDA_THREADS_PER_BLOCK, 0, stream>>>(__VA_ARGS__); KERNEL_CHECK_ERROR; } while(0)
	#define CUDA_KERNEL_ARGS2(FUN, blocks, threadsPerBlock, ...) do { FUN<<<blocks, threadsPerBlock>>>(__VA_ARGS__); KERNEL_CHECK_ERROR; } while(0)
	#define CUDA_KERNEL_ARGS3(FUN, blocks, threadsPerBlock, sharedMemPerBlock, ...) do { FUN<<<blocks, threadsPerBlock, sharedMemPerBlock>>>(__VA_ARGS__); KERNEL_CHECK_ERROR; } while(0)
	#define CUDA_KERNEL_ARGS4(FUN, blocks, threadsPerBlock, sharedMemPerBlock, stream, ...) do { FUN<<<blocks, threadsPerBlock, sharedMemPerBlock, stream>>>(__VA_ARGS__); KERNEL_CHECK_ERROR; } while(0)
	#define CUDA_MEMCOPY_TO_SYMBOL(symbol, src, size) CUDA_SAFE_LAUNCH(cudaMemcpyToSymbol(symbol, &src, size))
	#define CUDA_SYNCTHREADS __syncthreads()
#if __CUDA_ARCH__ < 600
	#define CUDA_VECTOR3_ATOMIC_ADD(vector3, value) { \
		CUDALegacy::myAtomicAdd(&vector3.x, value.x); \
		CUDALegacy::myAtomicAdd(&vector3.y, value.y); \
		CUDALegacy::myAtomicAdd(&vector3.z, value.z); \
	}
	#define CUDA_VECTOR3_ATOMIC_SUB(vector3, value) { \
		CUDALegacy::myAtomicAdd(&vector3.x, -value.x); \
		CUDALegacy::myAtomicAdd(&vector3.y, -value.y); \
		CUDALegacy::myAtomicAdd(&vector3.z, -value.z); \
	}
#else
	#define CUDA_VECTOR3_ATOMIC_ADD(vector3, value) { \
		atomicAdd(&vector3.x, value.x); \
		atomicAdd(&vector3.y, value.y); \
		atomicAdd(&vector3.z, value.z); \
	}
	#define CUDA_VECTOR3_ATOMIC_SUB(vector3, value) { \
		atomicAdd(&vector3.x, -value.x); \
		atomicAdd(&vector3.y, -value.y); \
		atomicAdd(&vector3.z, -value.z); \
	}
#endif
#if CUDART_VERSION < 11000
	#define CUDA_CUB_FLAGGED(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items) thrust::cuda_cub::cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, static_cast<int>(num_items))
#else
	#include <cub/device/device_select.cuh>
	#define CUDA_CUB_FLAGGED(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items) cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, static_cast<int>(num_items))
#endif
#else
	#define CUDA_HOST_DEVICE inline
	#define CUDA_DEVICE inline
	#define CUDA_KERNEL_ARGS2_DEFAULT(FUN, ...) ((void)(__VA_ARGS__))
	#define CUDA_KERNEL_ARGS4_DEFAULT(FUN, stream, ...) ((void)(__VA_ARGS__))
	#define CUDA_KERNEL_ARGS2(FUN, blocks, threadsPerBlock, ...) ((void)(__VA_ARGS__))
	#define CUDA_KERNEL_ARGS3(FUN, blocks, threadsPerBlock, sharedMemPerBlock, ...) ((void)(__VA_ARGS__))
	#define CUDA_KERNEL_ARGS4(FUN, blocks, threadsPerBlock, sharedMemPerBlock, stream, ...) ((void)(__VA_ARGS__))
	#define CUDA_MEMCOPY_TO_SYMBOL(symbol, src, size)
	#define CUDA_SYNCTHREADS
	#define CUDA_VECTOR3_ATOMIC_ADD(vector3, value) vector3 += value
	#define CUDA_VECTOR3_ATOMIC_SUB(vector3, value) vector3 -= value
	#define CUDA_CUB_FLAGGED(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items)
#endif

//#define CUDA_CHECK_ERRORS 1

#ifdef CUDA_CHECK_ERRORS
#pragma warning (disable : 4996 26812)
	#include <ctime>
	#include <chrono>
	#include <string>
	#include <iostream>

	#define KERNEL_CHECK_ERROR { CudaErrorCheck(cudaPeekAtLastError(), __FILE__, __LINE__); CudaErrorCheck(cudaDeviceSynchronize(), __FILE__, __LINE__); }
	#define CUDA_SAFE_LAUNCH(FUN) CudaErrorCheck((FUN), __FILE__, __LINE__)

	inline void CudaErrorCheck(cudaError_t _code, const char* _file, int _line)
	{
		if (_code)
		{
			const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			std::string nowStr = std::ctime(&now);
			nowStr.pop_back();
			std::cerr << nowStr << " " << _file << ":" << _line << " " << cudaGetErrorName(_code) << "(" << _code << ")" << std::endl;
		}
	}
#else
	#define KERNEL_CHECK_ERROR
	#define CUDA_SAFE_LAUNCH
#endif

#define CUDA_MEMCPY_H2H(DST, SRC, COUNT) CUDA_SAFE_LAUNCH(cudaMemcpy(DST, SRC, COUNT, cudaMemcpyHostToHost))
#define CUDA_MEMCPY_H2D(DST, SRC, COUNT) CUDA_SAFE_LAUNCH(cudaMemcpy(DST, SRC, COUNT, cudaMemcpyHostToDevice))
#define CUDA_MEMCPY_D2H(DST, SRC, COUNT) CUDA_SAFE_LAUNCH(cudaMemcpy(DST, SRC, COUNT, cudaMemcpyDeviceToHost))
#define CUDA_MEMCPY_D2D(DST, SRC, COUNT) CUDA_SAFE_LAUNCH(cudaMemcpy(DST, SRC, COUNT, cudaMemcpyDeviceToDevice))
#define CUDA_MALLOC_H(PTRPTR, SIZE) CUDA_SAFE_LAUNCH(cudaMallocHost(PTRPTR, SIZE, cudaHostAllocDefault))
#define CUDA_MALLOC_D(PTRPTR, SIZE) CUDA_SAFE_LAUNCH(cudaMalloc(PTRPTR, SIZE))
#define CUDA_FREE_H(PTR) CUDA_SAFE_LAUNCH(cudaFreeHost(PTR))
#define CUDA_FREE_D(PTR) CUDA_SAFE_LAUNCH(cudaFree(PTR))
#define CUDA_MEMSET(PTR, VAL, SIZE) CUDA_SAFE_LAUNCH(cudaMemset(PTR, VAL, SIZE))
#define CUDA_MEMSET_ASYNC(PTR, VAL, SIZE) CUDA_SAFE_LAUNCH(cudaMemsetAsync(PTR, VAL, SIZE))

#define CUDA_REDUCE_CALLER(function, size, src, temp, dst) do { \
	const unsigned nBlocksNum = (static_cast<unsigned>(size) + m_cudaDefines->CUDA_THREADS_PER_BLOCK - 1) / m_cudaDefines->CUDA_THREADS_PER_BLOCK; \
	if (nBlocksNum > 1) { \
		CUDA_KERNEL_ARGS3(function, nBlocksNum, m_cudaDefines->CUDA_THREADS_PER_BLOCK, sizeof((dst)[0])*m_cudaDefines->CUDA_THREADS_PER_BLOCK, static_cast<unsigned>(size), src,  temp); \
		CUDA_KERNEL_ARGS3(function,          1, m_cudaDefines->CUDA_THREADS_PER_BLOCK, sizeof((dst)[0])*m_cudaDefines->CUDA_THREADS_PER_BLOCK,                  nBlocksNum, temp, dst); \
	} \
	else \
		CUDA_KERNEL_ARGS3(function, nBlocksNum, m_cudaDefines->CUDA_THREADS_PER_BLOCK, sizeof((dst)[0])*m_cudaDefines->CUDA_THREADS_PER_BLOCK, static_cast<unsigned>(size), src,  dst); \
} while (0);
