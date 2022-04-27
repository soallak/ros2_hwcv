#pragma once
#include <spdlog/fmt/fmt.h>  // we use the fmt library bundelled with spdlog
#include <spdlog/spdlog.h>

namespace hwcv {

std::shared_ptr<spdlog::logger> GetLogger();

}