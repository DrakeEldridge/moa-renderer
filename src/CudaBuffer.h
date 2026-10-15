// MOA Renderer - a minimal owning wrapper around a block of GPU memory.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "Check.h"

#include <cuda.h>  // CUdeviceptr
#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace moa {

class CudaBuffer
{
public:
    CudaBuffer() = default;
    ~CudaBuffer() { release(); }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    CudaBuffer(CudaBuffer&& other) noexcept : m_ptr(other.m_ptr), m_size(other.m_size)
    {
        other.m_ptr = nullptr;
        other.m_size = 0;
    }
    CudaBuffer& operator=(CudaBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            m_ptr = other.m_ptr;
            m_size = other.m_size;
            other.m_ptr = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    // Allocates (or re-allocates) the buffer. Contents are undefined.
    void resize(size_t bytes)
    {
        if (bytes == m_size)
            return;
        release();
        if (bytes > 0)
        {
            CUDA_CHECK(cudaMalloc(&m_ptr, bytes));
            m_size = bytes;
        }
    }

    void release()
    {
        if (m_ptr)
            cudaFree(m_ptr);  // no check: may run during shutdown or stack unwinding
        m_ptr = nullptr;
        m_size = 0;
    }

    void upload(const void* src, size_t bytes)
    {
        resize(bytes);
        if (bytes > 0)
            CUDA_CHECK(cudaMemcpy(m_ptr, src, bytes, cudaMemcpyHostToDevice));
    }

    template <typename T>
    void upload(const std::vector<T>& v)
    {
        upload(v.data(), v.size() * sizeof(T));
    }

    template <typename T>
    void download(std::vector<T>& v) const
    {
        v.resize(m_size / sizeof(T));
        if (m_size > 0)
            CUDA_CHECK(cudaMemcpy(v.data(), m_ptr, v.size() * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void clear()
    {
        if (m_size > 0)
            CUDA_CHECK(cudaMemset(m_ptr, 0, m_size));
    }

    size_t size() const { return m_size; }
    void* get() const { return m_ptr; }
    CUdeviceptr dptr() const { return reinterpret_cast<CUdeviceptr>(m_ptr); }
    template <typename T>
    T* as() const { return static_cast<T*>(m_ptr); }

private:
    void* m_ptr = nullptr;
    size_t m_size = 0;
};

}  // namespace moa
