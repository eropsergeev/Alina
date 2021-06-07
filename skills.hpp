#pragma once

#include <regex>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <dlfcn.h>

class Skill {
public:
    Skill(std::regex re): re(std::move(re)) {}

    bool check_and_apply(const std::string &str) {
        std::match_results<const char *> results;
        if (!std::regex_match(str.c_str(), results, re)) {
            return 0;
        } else {
            apply(results);
            return 1;
        }
    }
protected:
    virtual void apply(const std::match_results<const char *> &results) = 0;
private:
    std::regex re;
};

class FileSkill: public Skill {
public:
    FileSkill(std::regex re, std::string path): Skill(re), path(std::move(path)) {}
protected:
    void apply(const std::match_results<const char *> &results) {
        std::vector<char *> args;
        args.emplace_back(path.data());
        for (auto &sub : results) {
            auto s = sub.first;
            auto f = sub.second;
            auto arg = new char [f - s + 1];
            memcpy(arg, s, f - s);
            arg[f - s] = 0;
            args.emplace_back(arg);
        }
        args.emplace_back(nullptr);
        if (auto pid = fork(); !pid) {
            execvp(path.data(), args.data());
        } else {
            waitpid(pid, nullptr, 0);
        }
        for (auto it = std::next(args.begin()); it != std::prev(args.end()); ++it) {
            delete [] *it;
        }
    }
private:
    std::string path;
};

class SoSkill: public Skill {
public:
    SoSkill(std::regex re, const std::string &path): Skill(re) {
        auto lib = dlopen(path.data(), RTLD_LAZY);
        func = reinterpret_cast<decltype(func)>(dlsym(lib, "run"));
    }
protected:
    void apply(const std::match_results<const char *> &results) {
        std::vector<const char *> args;
        for (auto &sub : results) {
            auto s = sub.first;
            auto f = sub.second;
            auto arg = new char [f - s + 1];
            memcpy(arg, s, f - s);
            arg[f - s] = 0;
            args.emplace_back(arg);
        }
        args.emplace_back(nullptr);
        func(args.data());
        for (auto it = std::next(args.begin()); it != std::prev(args.end()); ++it) {
            delete [] *it;
        }
    }
private:
    void (*func)(const char **); 
};
