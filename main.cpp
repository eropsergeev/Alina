#include <iostream>
#include <deque>
#include <fstream>
#include <numeric>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <cinttypes>
#include <limits>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <sys/types.h>
#include <dirent.h>
#include <vosk_api.h>
#include <nlohmann/json.hpp>
#include <regex>
#include "fft.hpp"
#include "sound_reader.hpp"
#include "skills.hpp"

using namespace std;
using namespace fastrnn;

extern "C" float apply_to(float *arr, size_t s, float *out);
extern "C" void load_from_file(const char *);

float apply_once(const Tensor<float, 40> &x, Tensor<float, 128> &h);

int main(int argc, char **argv) {
    if (argc < 4) {
        cerr << "Specify weights file, treshold and vosk model!\n";
        return 1;
    }
    load_from_file(argv[1]);
    float treshold = atof(argv[2]);

    const size_t HISTORY_LEN = 24000; // 1.5 sec
    const size_t MAX_SR_CHUNCK = 8000;
    SoundBuffer<64000, HISTORY_LEN> buffer;
    mutex buffer_mutex, state_mutex;
    condition_variable have_keyword, have_audio_history;
    size_t sr_offset = HISTORY_LEN;

    vector<unique_ptr<Skill>> skills;
    vector<string> files;

    auto dir = opendir("skills");
    for (auto it = readdir(dir); it; it = readdir(dir)) {
        if (it->d_type == DT_REG)
            files.emplace_back(it->d_name);
    }
    closedir(dir);

    sort(files.begin(), files.end());
    for (auto &x : files) {
        if (x.size() >= 3 && x.substr(x.size() - 3, 3) == ".re") {
            string re;
            ifstream in("skills/" + x);
            getline(in, re);
            cout << re << endl;
            if (binary_search(files.begin(), files.end(), x.substr(0, x.size() - 3))) {
                skills.emplace_back(make_unique<FileSkill>(regex(re), "skills/" + x.substr(0, x.size() - 3)));
            } else if (binary_search(files.begin(), files.end(), x.substr(0, x.size() - 3) + ".so")) {
                skills.emplace_back(make_unique<SoSkill>(regex(re), "skills/" + x.substr(0, x.size() - 3) + ".so"));
            }
        }
    }

    bool state = 0;

    thread([&]() {
        VoskModel *model = vosk_model_new(argv[3]);
        VoskRecognizer *recognizer = vosk_recognizer_new(model, 16000.0);
        vosk_recognizer_set_max_alternatives(recognizer, 5);
        cout << "Start!" << endl;
        while (1) {
            {
                std::unique_lock lock(state_mutex);
                have_keyword.wait(lock, [&state] { return state; });
            }
            bool final = 0;
            const size_t MAX_PHRASE_LEN = 10 * 16000;
            size_t phrase_len = 0;
            while (!final && phrase_len < MAX_PHRASE_LEN) {
                unique_lock lock(buffer_mutex);
                have_audio_history.wait(lock, [&sr_offset] { return sr_offset >= MAX_SR_CHUNCK; });
                auto [p, n] = buffer.get_history(sr_offset);
                sr_offset -= n;
                phrase_len += n;
                lock.unlock();
                final = vosk_recognizer_accept_waveform_s(recognizer, p, n);
            }
            auto result = nlohmann::json::parse(vosk_recognizer_final_result(recognizer));
            const string keyword = "алина";
            for (auto &x : result["alternatives"]) {
                if (x["text"].get<string>().substr(0, keyword.size()) == keyword) {
                    for (auto &s : skills)
                        s->check_and_apply(x["text"]);
                    break;
                }
            }
            state = 0;
        }
    }).detach();

    Tensor<float, 128> window;
    for (size_t i = 0; i < window.size();) {
        auto [p, n] = buffer.get_samples(window.size() - i);
        for (size_t j = 0; j < n; ++j) {
            window[i + j] = (float) p[j] / numeric_limits<int16_t>::max();
        }
        i += n;
    }
    Tensor<float, 128> h(0);
    const size_t POWER_HISTORY_LEN = 250;
    std::deque<float> powers;
    float power_sum = 0;
    while (1) {
        window.view<2, 64>()[0] = window.view<2, 64>()[1];
        for (size_t i = 64; i < window.size();) {
            buffer_mutex.lock();
            auto [p, n] = buffer.get_samples(window.size() - i);
            sr_offset = min(HISTORY_LEN, sr_offset + n);
            if (sr_offset >= MAX_SR_CHUNCK) {
                have_audio_history.notify_one();
            }
            buffer_mutex.unlock();
            for (size_t j = 0; j < n; ++j) {
                window[i + j] = (float) p[j] / numeric_limits<int16_t>::max();
            }
            i += n;
        }
        Tensor<complex<float>, 128> to_fft;
        for (size_t i = 0; i < window.size(); ++i) {
            to_fft[i] = complex<float>(window[i]);
        }
        fft(to_fft);
        Tensor<float, 40> spect;
        for (size_t i = 0; i < spect.size(); ++i) {
            spect[i] = abs((complex<float> &) to_fft[i + 3]);
        }
        float s = accumulate(spect.begin(), spect.end(), 0.0);
        powers.emplace_back(s);
        power_sum += s;
        if (powers.size() > POWER_HISTORY_LEN) {
            power_sum -= powers.front();
            powers.pop_front();
        }
        s /= 40;
        if (s < 1e-5) {
            s = 1e-5;
        }
        for (auto &x : spect) {
            x /= s;
        }
        float res = apply_once(spect, h);
        lock_guard lock(state_mutex);
        auto prev_state = state;
        state = res > treshold;
        if (state) {
            cout << "Alina! " << res << endl;
            have_keyword.notify_one();
        } else if (prev_state) {
            h = 0;
        }
    }
}