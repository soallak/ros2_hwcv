#pragma once
#include <spdlog/spdlog.h>
#define TRACE(...)                      \
  spdlog::trace("Enter " #__VA_ARGS__); \
  __VA_ARGS__;                          \
  spdlog::trace("Exit " #__VA_ARGS__)