#include <iostream>
#include <fstream>
#include <complex>
#include <algorithm>
#include <vector>
#include <random>
#include <type_traits>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <AudioFile.h>
#include "fft.hpp"
#include "fastrnn/tensor.hpp"
#include "alina_net.hpp"

using namespace std;
using namespace fastrnn;

const unsigned SAMPLE_RATE = 16000;
const unsigned WINDOW_SIZE = 128;
const unsigned TRAIN_SERIES_LEN = 20;
const unsigned FREQ_FROM = 3, FREQ_TO = 43;

template<unsigned freq_from, unsigned freq_to, class BidirIt, class OutIt>
void spectrogram(BidirIt first, BidirIt last, OutIt out) {
    static_assert(remove_reference<decltype(*out)>::type::static_size == freq_to - freq_from);
    while (last - first > WINDOW_SIZE) {
        Tensor<complex<float>, WINDOW_SIZE> window;
        copy(first, first + WINDOW_SIZE, window.begin());
        fft(window);
        transform(window.begin() + freq_from, window.begin() + freq_to, out->begin(), [](auto &x) { return abs(*x.data()); });
        ++out;
        first += WINDOW_SIZE / 2;
    }
}

float get_treshold(vector<float> powers) {
    auto low = powers.begin() + powers.size() / 10;
    auto high = powers.begin() + powers.size() / 10 * 9;
    nth_element(powers.begin(), low, powers.end());
    nth_element(powers.begin(), high, powers.end());
    return *low + (*high - *low) * 0.2;
}

void split(const vector<float> &samples, vector<vector<Tensor<float, FREQ_TO - FREQ_FROM>>> &ans) {
    vector<Tensor<float, FREQ_TO>> spect(samples.size() / (WINDOW_SIZE / 2) - 1);
    spectrogram<0, FREQ_TO>(samples.begin(), samples.end(), spect.begin());
    vector<float> powers(spect.size());
    transform(spect.begin(), spect.end(), powers.begin(), [](const auto &tensor) {
        return accumulate(tensor.begin(), tensor.end(), 0.0f);
    });
    auto tres = get_treshold(powers);
    int cur_sum = 0;
    for (size_t i = 0; i < 100; ++i) {
        cur_sum += (powers[i] > tres);
    }
    for (size_t i = 100; i < 110; ++i) {
        cur_sum -= 100 * (powers[i] > tres);
    }
    size_t last = 0;
    for (size_t i = 110; i < powers.size(); ++i) {
        cur_sum -= 100 * (powers[i] > tres);
        cur_sum += 101 * (powers[i - 10] > tres);
        cur_sum -= (powers[i - 110] > tres);
        if (cur_sum > 45 && i - last >= 70) {
            ans.emplace_back(i - last + 30);
            transform(spect.begin() + last, spect.begin() + i, ans.back().begin(), [](auto &x) {
                return x.template subtensor<FREQ_FROM, FREQ_TO>();
            });
            last = i;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        cerr << "Specify dataset directory, output weights files pattern and epochs count\n";
        return 1;
    }
    int epochs = strtol(argv[3], nullptr, 10);
    auto meta = nlohmann::json::parse(ifstream(string(argv[1]) + "meta.json"));
    vector<vector<Tensor<float, FREQ_TO - FREQ_FROM>>> positive, negative;
    for (auto &x : meta.items()) {
        auto &vec = (x.key().substr(0, 3) == "pos" ? positive : negative);
        for (auto &file_meta : x.value()) {
            AudioFile<float> file;
            file.load(string(argv[1]) + "/" + file_meta["path"].get<string>());
            assert(file.getSampleRate() == SAMPLE_RATE);
            if (!file_meta["regions"].is_null()) {
                for (auto reg : file_meta["regions"]) {
                    vec.emplace_back((reg[1].get<int>() - reg[0].get<int>()) / (WINDOW_SIZE / 2) - 1);
                    spectrogram<FREQ_FROM, FREQ_TO>(file.samples[0].begin() + reg[0], file.samples[0].begin() + reg[1], vec.back().begin());
                }
            } else {
                split(file.samples[0], vec);
            }
        }
    }
    cout << positive.size() << " positive and " << negative.size() << "negative samples" << endl;
    vector<vector<Tensor<float, FREQ_TO - FREQ_FROM>>> X_val, X_train;
    vector<bool> y_val, y_train;
    mt19937 rnd(42);
    for (auto &v : positive) {
        for (auto &x : v) {
            auto s = accumulate(x.begin(), x.end(), 0.0f);
            if (s < 1e-5)
                s = 1e-5;
            x /= s;
        }
    }
    for (auto &v : negative) {
        for (auto &x : v) {
            auto s = accumulate(x.begin(), x.end(), 0.0f);
            if (s < 1e-5)
                s = 1e-5;
            x /= s;
        }
    }
    for (auto &x : positive) {
        if (rnd() % 10) {
            X_train.emplace_back(move(x));
            y_train.emplace_back(1);
        } else {
            X_val.emplace_back(move(x));
            y_val.emplace_back(1);
        }
    }
    for (auto &x : negative) {
        if (rnd() % 10) {
            X_train.emplace_back(move(x));
            y_train.emplace_back(0);
        } else {
            X_val.emplace_back(move(x));
            y_val.emplace_back(0);
        }
    }
    init(777);
    for (size_t i = 0; i < X_train.size(); ++i) {
        add_data(X_train[i][0].data(), X_train[i].size(), y_train[i]);
    }
    nlohmann::json report;
    for (int i = 0; i < epochs; ++i) {
        shuffle();
        size_t iters = X_train.size() / TRAIN_SERIES_LEN;
        float *losses = new float[iters];
        train_epoch(0, TRAIN_SERIES_LEN, losses);
        nlohmann::json iteration_report;
        iteration_report["train_loss"] = accumulate(losses, losses + iters, 0.0) / iters;
        cerr << "Epoch #" << i << ":\n";
        cerr << "train loss = " << iteration_report["train_loss"].get<double>() << "\n";
        report.push_back(iteration_report);
    }
    cout << report.dump() << "\n";
}
