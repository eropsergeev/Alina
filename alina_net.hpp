#pragma once

#include <cinttypes>
#include "fastrnn/tensor.hpp"

constexpr size_t code_size = 40, hidden_size = 128, linear_size = 128;

float apply_once(const fastrnn::Tensor<float, code_size> &x, fastrnn::Tensor<float, hidden_size> &h);

extern "C" {

void init(uint32_t seed);

void add_data(float *arr, size_t s, bool y);

void shuffle();

void train_epoch(size_t n, size_t seq, float *losses);

float apply_to(float *arr, size_t s, float *out);

void save_to_file(const char *name);

void load_from_file(const char *name);

};
