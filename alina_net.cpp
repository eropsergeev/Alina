#include "alina_net.hpp"

#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <utility>
#include <algorithm>
#include <ctime>
#include <fstream>
#include "fastrnn/variable.hpp"
#include "fastrnn/executer.hpp"
#include "fastrnn/gru.hpp"
#include "fastrnn/allocator.hpp"
#include "fastrnn/optimizer.hpp"
#include "fastrnn/linear.hpp"

using namespace fastrnn;

TensorAllocator<float, (size_t) 1 << 28, true> alloc;

template<size_t n, class Executer = StaticExecuter<1>>
void add_cross_entropy_loss(
    Variable<Tensor<float, n>> x,
    size_t i,
    float w,
    Variable<Tensor<float>> l,
    GradientCalculator &calc,
    Executer &exe = Executer::object)
{
    size_t max_index = std::max_element(x.data->begin(), x.data->end()) - x.data->begin();
    Variable x_norm(*alloc.allocate<n>(), *alloc.allocate<n>());
    calc.add_(x, x_norm);
    for (size_t j = 0; j < n; ++j)
        calc.sub_(x[max_index], x_norm[j]);
    Variable exp_x(*alloc.allocate<n>(), *alloc.allocate<n>());
    calc.exp(x_norm, exp_x, exe);
    Variable sum_exp(*alloc.allocate<>(), *alloc.allocate<>());
    calc.sum(exp_x, sum_exp);
    calc.apply_func(
        sum_exp, x_norm[i], l,
        [w](auto sum_exp, auto x, auto &l) {
            l += (std::log(sum_exp) - x) * w;
        }, [w](auto sum_exp, auto x, auto, auto &sum_exp_grad, auto &x_grad, auto l_grad) {
            sum_exp_grad += w * l_grad / sum_exp;
            x_grad -= w * l_grad;
        }
    );
}

std::mt19937 rnd;

float frand() {
    return ((float) rnd() / (rnd.max() / 2) - 1) * 0.05;
}

#ifndef THREADS
#define THREADS 1
#endif

StaticExecuter<THREADS> *exe;

Linear<float, code_size, linear_size, true> l1;
Linear<float, linear_size, linear_size, true> l2;
Linear<float, linear_size, linear_size, true> l3;
GRUCell<float, linear_size, hidden_size, true> cell;
Linear<float, hidden_size, linear_size, true> l4;
Linear<float, linear_size, linear_size, true> l5;
Linear<float, linear_size, 2, true> l6;

std::vector<std::pair<std::vector<Tensor<float, code_size>>, bool>> dataset;

RMSPropOptimizer<float> *opt = nullptr;

float apply_once(const Tensor<float, code_size> &x, Tensor<float, hidden_size> &h) {
    Tensor<float, linear_size> o1, o2, o3, o4, o5;
    Tensor<float, 2> o6;
    Tensor<float, hidden_size> nh;
    l1.no_grad()(x, o1);
    relu(o1, o1);
    l2.no_grad()(o1, o2);
    relu(o2, o2);
    l3.no_grad()(o2, o3);
    relu(o3, o3);
    cell.no_grad()(o3, h, nh);
    h = nh;
    l4.no_grad()(h, o4);
    relu(o4, o4);
    l5.no_grad()(o4, o5);
    relu(o5, o5);
    l6.no_grad()(o5, o6);
    o6 -= *std::max_element(o6.begin(), o6.end());
    exp(o6, o6);
    return o6[1] / (o6[0] + o6[1]);
}

extern "C" {

void init(uint32_t seed) {
    rnd.seed(seed);
    delete opt;
    opt = new RMSPropOptimizer<float>(1e-3);
    cell.register_in_optimizer(*opt);
    l1.register_in_optimizer(*opt);
    l2.register_in_optimizer(*opt);
    l3.register_in_optimizer(*opt);
    l4.register_in_optimizer(*opt);
    l5.register_in_optimizer(*opt);
    l6.register_in_optimizer(*opt);
    cell = GRUCell<float, linear_size, hidden_size, true>(frand);
    l1 = Linear<float, code_size, linear_size, true>(frand);
    l2 = Linear<float, linear_size, linear_size, true>(frand);
    l3 = Linear<float, linear_size, linear_size, true>(frand);
    l4 = Linear<float, hidden_size, linear_size, true>(frand);
    l5 = Linear<float, linear_size, linear_size, true>(frand);
    l6 = Linear<float, linear_size, 2, true>(frand);
    dataset.clear();
}

void add_data(float *arr, size_t s, bool y) {
    std::vector<Tensor<float, code_size>> x(s);
    for (size_t i = 0; i < s; ++i) {
        memcpy(x[i].data(), arr + i * code_size, sizeof(x[i]));
    }
    dataset.emplace_back(std::move(x), y);
}

void shuffle() {
    std::shuffle(dataset.begin(), dataset.end(), rnd);
}

void train_epoch(size_t n, size_t seq, float *losses) {
    exe = new StaticExecuter<THREADS>;
    if (n == 0) {
        n = dataset.size();
    }
    GradientCalculator calc;
    alloc.reset();
    Variable h(*alloc.allocate<hidden_size>(), *alloc.allocate<hidden_size>());
    Tensor<float> l(0), l_(0);
    Variable var_l(l, l_);
    int cnt = 0;
    for (size_t i = 0; i < n; ++i) {
        if (i % seq == 0) {
            alloc.reset();
            opt->zero_grad();
            h = Variable(*alloc.allocate<hidden_size>(), *alloc.allocate<hidden_size>());
            l = 0;
            cnt = 0;
        }
        auto &X = dataset[i].first;
        auto y = dataset[i].second;
        for (size_t j = 0; j < X.size(); ++j) {
            auto &x = X[j];
            Variable o1(*alloc.allocate<linear_size>(), *alloc.allocate<linear_size>());
            Variable o1r(*alloc.allocate<linear_size>(), *alloc.allocate<linear_size>());
            Variable o2(*alloc.allocate<linear_size>(), *alloc.allocate<linear_size>());
            Variable o2r(*alloc.allocate<linear_size>(), *alloc.allocate<linear_size>());
            Variable o3(*alloc.allocate<linear_size>(), *alloc.allocate<linear_size>());
            Variable o3r(*alloc.allocate<linear_size>(), *alloc.allocate<linear_size>());
            Variable new_h(*alloc.allocate<hidden_size>(), *alloc.allocate<hidden_size>());
            l1(x, o1, calc, *exe);
            calc.relu(o1, o1r, *exe);
            l2(o1r, o2, calc, *exe);
            calc.relu(o2, o2r, *exe);
            l3(o2r, o3, calc, *exe);
            calc.relu(o3, o3r, *exe);
            cell(o3r, h, new_h, calc, alloc, *exe);
            h = new_h;
            if (!y || j + 50 >= X.size()) {
                Variable o4(*alloc.allocate<linear_size>(), *alloc.allocate<linear_size>());
                Variable o4r(*alloc.allocate<linear_size>(), *alloc.allocate<linear_size>());
                Variable o5(*alloc.allocate<linear_size>(), *alloc.allocate<linear_size>());
                Variable o5r(*alloc.allocate<linear_size>(), *alloc.allocate<linear_size>());
                Variable o6(*alloc.allocate<2>(), *alloc.allocate<2>());
                l4(h, o4, calc, *exe);
                calc.relu(o4, o4r, *exe);
                l5(o4r, o5, calc, *exe);
                calc.relu(o5, o5r, *exe);
                l6(o5r, o6, calc, *exe);
                ++cnt;
                // std::cout << *o6[0].data << " " << *o6[1].data << "\n";
                add_cross_entropy_loss(o6, y, y ? 100 : 1, var_l, calc, *exe);
            }
        }
        if (i % seq == (seq - 1)) {
            Variable mean_l(*alloc.allocate<>(), *alloc.allocate<>());
            if (cnt) {
                calc.apply_func(var_l, mean_l, [cnt](auto sum, auto &mean) {
                    mean += sum / cnt;
                }, [cnt](auto, auto, auto &sum_, auto mean_) {
                    sum_ += mean_ / cnt;
                });
            }
            calc.backward(mean_l);
            *losses++ = l / cnt;
            opt->step();
        }
    }
    delete exe;
}

float apply_to(float *arr, size_t s, float *out) {
    Tensor<float, code_size> x;
    Tensor<float, hidden_size> h(0);
    float ans = 0;
    for (size_t i = 0; i < s; ++i) {
        memcpy(x.data(), arr + i * code_size, sizeof(x));
        Tensor<float, linear_size> o1, o2, o3, o4, o5;
        Tensor<float, 2> o6;
        Tensor<float, hidden_size> nh;
        l1.no_grad()(x, o1);
        relu(o1, o1);
        l2.no_grad()(o1, o2);
        relu(o2, o2);
        l3.no_grad()(o2, o3);
        relu(o3, o3);
        cell.no_grad()(o3, h, nh);
        h = nh;
        l4.no_grad()(h, o4);
        relu(o4, o4);
        l5.no_grad()(o4, o5);
        relu(o5, o5);
        l6.no_grad()(o5, o6);
        // float mx = *std::max_element(o6.begin(), o6.end());
        float mx = *std::max_element(o6.begin(), o6.end());
        o6[0] -= mx;
        o6[1] -= mx;
        exp(o6, o6);
        if (out) {
            *out++ = o6[1] / (o6[0] + o6[1]);
        }
        ans = std::max(ans, o6[1] / (o6[0] + o6[1]));
    }
    return ans;
}

void save_to_file(const char *name) {
    std::ofstream out(name, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<char *>(l1.W.data()), sizeof(l1.W));
    out.write(reinterpret_cast<char *>(l1.b.data()), sizeof(l1.b));
    out.write(reinterpret_cast<char *>(l2.W.data()), sizeof(l2.W));
    out.write(reinterpret_cast<char *>(l2.b.data()), sizeof(l2.b));
    out.write(reinterpret_cast<char *>(l3.W.data()), sizeof(l3.W));
    out.write(reinterpret_cast<char *>(l3.b.data()), sizeof(l3.b));
    out.write(reinterpret_cast<char *>(l4.W.data()), sizeof(l4.W));
    out.write(reinterpret_cast<char *>(l4.b.data()), sizeof(l4.b));
    out.write(reinterpret_cast<char *>(l5.W.data()), sizeof(l5.W));
    out.write(reinterpret_cast<char *>(l5.b.data()), sizeof(l5.b));
    out.write(reinterpret_cast<char *>(l6.W.data()), sizeof(l6.W));
    out.write(reinterpret_cast<char *>(l6.b.data()), sizeof(l6.b));

    out.write(reinterpret_cast<char *>(cell.Wr.data()), sizeof(cell.Wr));
    out.write(reinterpret_cast<char *>(cell.Ur.data()), sizeof(cell.Ur));
    out.write(reinterpret_cast<char *>(cell.br.data()), sizeof(cell.br));
    out.write(reinterpret_cast<char *>(cell.Wz.data()), sizeof(cell.Wz));
    out.write(reinterpret_cast<char *>(cell.Uz.data()), sizeof(cell.Uz));
    out.write(reinterpret_cast<char *>(cell.bz.data()), sizeof(cell.bz));
    out.write(reinterpret_cast<char *>(cell.Wh.data()), sizeof(cell.Wh));
    out.write(reinterpret_cast<char *>(cell.Uh.data()), sizeof(cell.Uh));
    out.write(reinterpret_cast<char *>(cell.bh.data()), sizeof(cell.bh));
}

void load_from_file(const char *name) {
    std::ifstream in(name, std::ios::in | std::ios::binary);
    in.read(reinterpret_cast<char *>(l1.W.data()), sizeof(l1.W));
    in.read(reinterpret_cast<char *>(l1.b.data()), sizeof(l1.b));
    in.read(reinterpret_cast<char *>(l2.W.data()), sizeof(l2.W));
    in.read(reinterpret_cast<char *>(l2.b.data()), sizeof(l2.b));
    in.read(reinterpret_cast<char *>(l3.W.data()), sizeof(l3.W));
    in.read(reinterpret_cast<char *>(l3.b.data()), sizeof(l3.b));
    in.read(reinterpret_cast<char *>(l4.W.data()), sizeof(l4.W));
    in.read(reinterpret_cast<char *>(l4.b.data()), sizeof(l4.b));
    in.read(reinterpret_cast<char *>(l5.W.data()), sizeof(l5.W));
    in.read(reinterpret_cast<char *>(l5.b.data()), sizeof(l5.b));
    in.read(reinterpret_cast<char *>(l6.W.data()), sizeof(l6.W));
    in.read(reinterpret_cast<char *>(l6.b.data()), sizeof(l6.b));

    in.read(reinterpret_cast<char *>(cell.Wr.data()), sizeof(cell.Wr));
    in.read(reinterpret_cast<char *>(cell.Ur.data()), sizeof(cell.Ur));
    in.read(reinterpret_cast<char *>(cell.br.data()), sizeof(cell.br));
    in.read(reinterpret_cast<char *>(cell.Wz.data()), sizeof(cell.Wz));
    in.read(reinterpret_cast<char *>(cell.Uz.data()), sizeof(cell.Uz));
    in.read(reinterpret_cast<char *>(cell.bz.data()), sizeof(cell.bz));
    in.read(reinterpret_cast<char *>(cell.Wh.data()), sizeof(cell.Wh));
    in.read(reinterpret_cast<char *>(cell.Uh.data()), sizeof(cell.Uh));
    in.read(reinterpret_cast<char *>(cell.bh.data()), sizeof(cell.bh));
}

};
