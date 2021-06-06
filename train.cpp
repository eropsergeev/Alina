#include <iostream>
#include <fstream>
#include <complex>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <AudioFile.h>
#include "fft.hpp"
#include "fastrnn/tensor.hpp"

using namespace std;
using namespace fastrnn;

const unsigned SAMPLE_RATE = 16000;
const unsigned WINDOW_SIZE = 128;
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
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << "Specify dataset directory and output weights files pattern\n";
        return 1;
    }
    auto meta = nlohmann::json::parse(ifstream(string(argv[1]) + "meta.json"));
    cout << meta << "\n";
    vector<vector<Tensor<float, FREQ_TO - FREQ_FROM>>> positive, negative;
    for (auto &x : meta.items()) {
        auto &vec = (x.key().substr(0, 3) == "pos" ? positive : negative);
        for (auto &file_meta : x.value()) {
            AudioFile<float> file;
            file.load(string(argv[1]) + "/" + file_meta["path"].get<string>());
            assert(file.getSampleRate() == SAMPLE_RATE);
            if (!file_meta["regions"].is_null()) {
                cout << file_meta["regions"] << endl;
                for (auto reg : file_meta["regions"]) {
                    vec.emplace_back((reg[1] - reg[0]) / (WINDOW_SIZE / 2) - 1);
                    spectrogram<FREQ_FROM, FREQ_TO>(file.samples[0].begin() + reg[0], file.samples[0].begin() + reg[1], vec.back().begin());
                }
            } else {
                split(file.samples[0], vec);
            }
        }
    }
    cout << positive.size() << " " << negative.size() << "\n";
}