#pragma once

#include <cinttypes>
#include <type_traits>
#include <memory>
#include <stdexcept>
#include <string>
#include <alsa/asoundlib.h>

template<size_t size, size_t history, unsigned required_rate = 16000>
class SoundBuffer {
private:
    snd_pcm_t *handle;
    size_t start{0}, end{0};
    unsigned step, channels;
    int16_t buffer[size];

    void read_some() {
        snd_pcm_uframes_t offset = 0, frames = (size - history - max_samples()) * step / channels;
        if (frames == 0) {
            throw std::runtime_error("Too large buffer is required");
        }
        const snd_pcm_channel_area_t *area;
        snd_pcm_wait(handle, -1);
        if (int err = snd_pcm_avail_update(handle); err < 0) {
            throw std::runtime_error(std::string("Update error: ") + snd_strerror(err));
        }
        if (int err = snd_pcm_mmap_begin(handle, &area, &offset, &frames)) {
            throw std::runtime_error(std::string("MMAP error: ") + snd_strerror(err));
        }
        assert(area->first % 16 == 0);
        assert(area->step == channels * 16);
        for (size_t i = 0; i < frames * channels / step; ++i) {
            buffer[end] = (reinterpret_cast<int16_t *>(area->addr) + area->first / 16 + offset * channels)[i * step];
            end = (end + 1) % size;
        }
        if (int err = snd_pcm_mmap_commit(handle, offset, frames); err < 0) {
            throw std::runtime_error(std::string("Commit error: ") + snd_strerror(err));
        }
    }

public:
    SoundBuffer() {
        if (int err = snd_pcm_open(&handle, "hw:1,0", SND_PCM_STREAM_CAPTURE, 0)) {
            throw std::runtime_error(std::string("Open error: ") + snd_strerror(err));
        }
        snd_pcm_hw_params_t *params;
        snd_pcm_hw_params_malloc(&params);
        snd_pcm_hw_params_any(handle, params);
        if (snd_pcm_hw_params_test_format(handle, params, SND_PCM_FORMAT_S16_LE)) {
            throw std::runtime_error("Foramt s16le not supported\n");
        }
        unsigned rate_max, rate_min;
        int dir;
        if (int err = snd_pcm_hw_params_get_channels(params, &channels)) {
            throw std::runtime_error(std::string("Channels get error: ") + snd_strerror(err));
        }
        if (int err = snd_pcm_hw_params_get_rate_max(params, &rate_max, &dir)) {
            throw std::runtime_error(std::string("Max rate get error: ") + snd_strerror(err));
        }
        if (int err = snd_pcm_hw_params_get_rate_min(params, &rate_min, &dir)) {
            throw std::runtime_error(std::string("Min rate get error: ") + snd_strerror(err));
        }
        snd_pcm_hw_params_free(params);
        unsigned rate = required_rate;
        while (rate < rate_min) {
            rate += required_rate;
        }
        if (rate > rate_max) {
            throw std::runtime_error(std::string("Required rate not supported"));
        }
        step = rate / required_rate * channels;
        if (int err = snd_pcm_set_params(
            handle,
            SND_PCM_FORMAT_S16_LE,
            SND_PCM_ACCESS_MMAP_INTERLEAVED,
            channels,
            rate,
            0,
            200'000)) {
            throw std::runtime_error(std::string("Setup error: ") + snd_strerror(err));
        }
        if (int err = snd_pcm_start(handle)) {
            throw std::runtime_error(std::string("Start error: ") + snd_strerror(err));
        }
    }
    ~SoundBuffer() {
        snd_pcm_close(handle);
    }
    size_t max_samples() {
        return (end - start + size) % size;
    }
    std::pair<const int16_t*, size_t> get_samples(size_t n) {
        while (max_samples() < n) {
            read_some();
        }
        if (size - start > n) {
            start += n;
            return {buffer + start - n, n};
        } else {
            std::pair<int16_t*, size_t> ret(buffer + start, size - start);
            start = 0;
            return ret;
        }
    }
    std::pair<const int16_t*, size_t> get_history(size_t n) const {
        size_t h_start = (start - n + size) % size;
        return {buffer + h_start, h_start < start ? n : size - h_start};
    }
};
