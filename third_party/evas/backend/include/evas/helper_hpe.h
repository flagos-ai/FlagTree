/* Copyright (c) 2024, EVAS Technology Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of EVAS Technology Inc nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
///////////////////////////////////////////////////////////////////////////////
// These are Helper functions for initialization and error checking

#ifndef _COMMON_HELPER_H
#define _COMMON_HELPER_H


// #ifdef __AC_HOST_COMPILE__
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <hpe.h>

union gem_float {
	struct fp {
		unsigned int mantissa : 23;
		unsigned int exp : 8;
		unsigned int sign : 1;
	} B;
	float A;
};

static inline int ReadBinFile(const char *fileName, void **inputBuff,
							  unsigned int *fileSize) {
	FILE *binFile;
	long binFileBufferLen;

	// 打开文件
	binFile = fopen(fileName, "rb");
	if (binFile == NULL) {
		printf("open file %s failed\n", fileName);
		return -1;
	}

	// 移动文件指针到文件末尾，获取文件大小
	fseek(binFile, 0, SEEK_END);
	binFileBufferLen = ftell(binFile);
	if (binFileBufferLen == 0) {
		printf("binfile is empty, filename is %s\n", fileName);
		fclose(binFile);
		return -1;
	}
	rewind(binFile); // 重置文件指针到文件开始位置

	// 分配缓冲区
	*inputBuff = malloc((size_t)binFileBufferLen);
	if (*inputBuff == NULL) {
		printf("malloc device buffer failed. size is %lu\n", binFileBufferLen);
		fclose(binFile);
		return -1;
	}

	// 读取文件内容到缓冲区
	fread(*inputBuff, sizeof(char), (size_t)binFileBufferLen, binFile);
	fclose(binFile); // 关闭文件

	if (fileSize != NULL) {
		*fileSize = (unsigned int)binFileBufferLen;
	}
	return 0;
}

// Runtime error messages
template <typename T>
void check(T result, char const *const func, const char *const file,
		   int const line) {
	if (result) {
		fprintf(stderr, "Runtime error at %s:%d code=%d(%s) \"%s\" \n", file,
				line, static_cast<unsigned int>(result), evGetErrorName(result),
				func);
		exit(EXIT_FAILURE);
	}
}

// This will output the proper runtime error strings in the event
// that a host call returns an error
#define checkErrors(val) check((val), #val, __FILE__, __LINE__)

#define checkEqual(val, cmp)                                                   \
	if ((val) != (cmp)) {                                                      \
		fprintf(stderr, "Runtime error at %s:%d code=%d \n", __FILE__,         \
				__LINE__, static_cast<unsigned int>(val));                     \
		exit(EXIT_FAILURE);                                                    \
	}
#define checkNotEqual(val, cmp)                                                \
	if ((val) == (cmp)) {                                                      \
		fprintf(stderr, "Runtime error at %s:%d code=%d \n", __FILE__,         \
				__LINE__, static_cast<unsigned int>(val));                     \
		exit(EXIT_FAILURE);                                                    \
	}

// This will output the proper error string when calling evGetLastError
#define getLastError(msg) __getLastError(msg, __FILE__, __LINE__)

inline void __getLastError(const char *errorMessage, const char *file,
						   const int line) {
	evError_t err = evGetLastError();

	if (evSuccess != err) {
		fprintf(stderr,
				"%s(%i) : getLastError() error :"
				" %s : (%d) %s.\n",
				file, line, errorMessage, static_cast<int>(err),
				evGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// This will only print the proper error string when calling evGetLastError
// but not exit program incase error detected.
#define printLastError(msg) __printLastError(msg, __FILE__, __LINE__)

inline void __printLastError(const char *errorMessage, const char *file,
							 const int line) {
	evError_t err = evGetLastError();

	if (evSuccess != err) {
		fprintf(stderr,
				"%s(%i) : getLastError() error :"
				" %s : (%d) %s.\n",
				file, line, errorMessage, static_cast<int>(err),
				evGetErrorString(err));
	}
}

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#ifndef MIN
#define MIN(a, b) (a < b ? a : b)
#endif

// Float To Int conversion
inline int ftoi(float value) {
	return (value >= 0 ? static_cast<int>(value + 0.5)
					   : static_cast<int>(value - 0.5));
}

// General NPU Device Initialization
inline int npuDeviceInit(int devID) {
	int device_count;
	checkErrors(evGetDeviceCount(&device_count));

	if (device_count == 0) {
		fprintf(stderr, "npuDeviceInit() error: "
						"no devices supporting runtime.\n");
		exit(EXIT_FAILURE);
	}

	if (devID < 0) {
		devID = 0;
	}

	if (devID > device_count - 1) {
		fprintf(stderr, "\n");
		fprintf(stderr, ">> %d capable NPU device(s) detected. <<\n",
				device_count);
		fprintf(stderr,
				">> npuDeviceInit (-device=%d) is not a valid"
				" NPU device. <<\n",
				devID);
		fprintf(stderr, "\n");
		return -devID;
	}

	int major = -1;
	checkErrors(evGetDeviceAttribute(&major, evDevAttrChipHWVersion, devID));

	if (major < 0) {
		fprintf(
			stderr,
			"npuDeviceInit(): NPU device [%d] does not support runtime. %d\n",
			devID, major);
		exit(EXIT_FAILURE);
	}

	checkErrors(evSetDevice(devID));
	printf("npuDeviceInit() Device [%d]: %d\n", devID, major);

	return devID;
}

#define ELAPSED_TIME_START()                                                   \
	struct timespec start;                                                     \
	do {                                                                       \
		clock_gettime(CLOCK_MONOTONIC, &start);                                \
	} while (0)

#define ELAPSED_TIME_END()                                                     \
	do {                                                                       \
		struct timespec end;                                                   \
		long elapsed_time;                                                     \
		clock_gettime(CLOCK_MONOTONIC, &end);                                  \
		elapsed_time =                                                         \
			(end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9; \
		printf("%s elapsed time: %ld seconds\n", __FILE__, elapsed_time);      \
	} while (0)
// #endif
#define EXTERN(var, dtype, nums) extern dtype var[nums]

#define INCBIN(sym, file, section, align)                                      \
	asm(".section " #section ", \"a\", @progbits\n\t"                          \
		".type " #sym ", @object\n\t"                                          \
		".global " #sym "\n\t" #sym ":\n\t"                                    \
		".align " #align "\n\t"                                                \
		".incbin " #file "\n\t"                                                \
		".align " #align "\n\t" #sym "_end:\n\t");

typedef struct {
	int cls_nr;
	int core_nr;
} kernel_configs;

static bool init = false;

static inline int get_rand_clusters(void) {
	if (!init) {
		srand(time(NULL));
		init = true;
	}
	return rand() % 8 + 1;
}

static inline int get_rand_cores(void) {
	if (!init) {
		srand(time(NULL));
		init = true;
	}
	return rand() % 4 + 1;
}
#ifdef __AC_DEVICE_COMPILE__
#define MFN                                                                    \
	__attribute__((                                                            \
		target("no-zve32x,no-zve32f,no-zve64x,no-zve64f,no-zvfh,no-zvl32b,no-" \
			   "zvl64b,no-zvl128b,no-zvl256b,no-zvl512b,no-zvl1024b")))
#else
#define MFN
#endif

#endif // COMMON_HELPER_H_
