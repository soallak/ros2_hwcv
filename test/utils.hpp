#pragma once
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <chrono>

#define TRACE(...)                      \
  spdlog::trace("Enter " #__VA_ARGS__); \
  __VA_ARGS__;                          \
  spdlog::trace("Exit " #__VA_ARGS__)

constexpr unsigned int BENCHMARK_NRUNS = 1000;

#define BENCHMARK(...)                                                \
  {                                                                   \
    auto now = std::chrono::steady_clock::now();                      \
    for (unsigned int i = 0; i < BENCHMARK_NRUNS; i++) {              \
      __VA_ARGS__;                                                    \
    }                                                                 \
    auto end = std::chrono::steady_clock::now();                      \
    std::chrono::nanoseconds dur(end - now);                          \
    spdlog::trace(fmt::format(FMT_STRING(#__VA_ARGS__ " took {} ns"), \
                              dur.count() / BENCHMARK_NRUNS));        \
  }
