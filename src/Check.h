// MOA Renderer - error checking helpers for CUDA and OptiX calls.
// Any failure throws a std::runtime_error with the file, line and a readable message.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace moa {

[[noreturn]] inline void throwError(const std::string& what, const char* file, int line)
{
    std::ostringstream ss;
    ss << what << "\n    at " << file << ":" << line;
    throw std::runtime_error(ss.str());
}

}  // namespace moa

#define CUDA_CHECK(call)                                                                     \
    do                                                                                       \
    {                                                                                        \
        const cudaError_t moaErr_ = (call);                                                  \
        if (moaErr_ != cudaSuccess)                                                          \
            ::moa::throwError(std::string("CUDA error in ") + #call + ": " +                 \
                                  cudaGetErrorName(moaErr_) + " - " + cudaGetErrorString(moaErr_), \
                              __FILE__, __LINE__);                                           \
    } while (0)

#define OPTIX_CHECK(call)                                                                    \
    do                                                                                       \
    {                                                                                        \
        const OptixResult moaRes_ = (call);                                                  \
        if (moaRes_ != OPTIX_SUCCESS)                                                        \
            ::moa::throwError(std::string("OptiX error in ") + #call + ": " +                \
                                  optixGetErrorName(moaRes_) + " - " + optixGetErrorString(moaRes_), \
                              __FILE__, __LINE__);                                           \
    } while (0)

// For OptiX calls that fill a log buffer (module, program group and pipeline creation).
// Expects `char log[]` and `size_t logSize` in scope.
#define OPTIX_CHECK_LOG(call)                                                                \
    do                                                                                       \
    {                                                                                        \
        logSize = sizeof(log);                                                               \
        const OptixResult moaRes_ = (call);                                                  \
        if (moaRes_ != OPTIX_SUCCESS)                                                        \
            ::moa::throwError(std::string("OptiX error in ") + #call + ": " +                \
                                  optixGetErrorName(moaRes_) + "\n" + std::string(log, logSize > 1 ? logSize - 1 : 0), \
                              __FILE__, __LINE__);                                           \
    } while (0)
