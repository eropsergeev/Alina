#pragma once

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <bit>
#include "fastrnn/tensor.hpp"

template<size_t n>
void do_fft(fastrnn::Tensor<std::complex<float>, n> &a) {
    if constexpr (n == 1) {
        return;
    } else {
        static_assert(n % 2 == 0);
        do_fft(a.template view<2, n / 2>()[0]);
        do_fft(a.template view<2, n / 2>()[1]);
        std::complex<float> w0(cos(2 * M_PI / n), sin(2 * M_PI / n)), w(1);
        #pragma GCC ivdep
        for (size_t i = 0; i < n / 2; ++i) {
            std::complex<float> x = a[i];
            std::complex<float> y = a[n / 2 + i];
            a[i] = x + w * y;
            a[n / 2 + i] = x - w * y;
            w *= w0;
        }
    }
}

template<size_t n>
void fft(fastrnn::Tensor<std::complex<float>, n> &a) {
    constexpr size_t p = std::countr_zero(n);
    auto rev = [](size_t x) {
        size_t ans = 0;
        for (size_t i = 0; i < p; ++i) {
            ans |= (x >> i & 1) << (p - 1 - i);
        }
        return ans;
    };
    for (size_t i = 0; i < n; ++i) {
        auto r = rev(i);
        if (r > i) {
            swap(a[i], a[r]);
        }
    }
    do_fft(a);
}
