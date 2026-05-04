#pragma once
#include <chrono>
#include <functional>

#include "cuda.h"
#include "cuda_runtime.h"

namespace SPIN {
    template <typename Func, typename... Args>
    double TimeCheck(Func func, Args&&... args) {
        auto _check_begin = std::chrono::high_resolution_clock::now();
        func(std::forward<Args>(args)...);
        auto _check_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> Dur = _check_end - _check_begin;

        return Dur.count() / 1000.;
    }

    template <typename Func, typename... Args>
    double TimeCheckCUDA(Func func, Args&&... args) {
        cudaEvent_t _check_begin, _check_end;
        cudaEventCreate(&_check_begin); cudaEventCreate(&_check_end);
        cudaEventRecord(_check_begin);
        func(std::forward<Args>(args)...);
        cudaEventRecord(_check_end);
        cudaEventSynchronize(_check_end);
        float Dur = 0;
        cudaEventElapsedTime(&Dur, _check_begin, _check_end);
        cudaEventDestroy(_check_begin); cudaEventDestroy(_check_end);
        return (double)Dur;
    }
}