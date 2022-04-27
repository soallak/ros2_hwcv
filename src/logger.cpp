#include "logger.hpp"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/syslog_sink.h>

namespace hwcv {

namespace {
const char* const logger_name_ = "hwcv";
}  // namespace

void CreateLogger() {
  // create and configure sinks
  auto color_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto syslog_sink = std::make_shared<spdlog::sinks::syslog_sink<std::mutex>>(
      "", LOG_PID, LOG_USER, true);
  color_sink->set_level(spdlog::level::trace);
  syslog_sink->set_level(spdlog::level::trace);

  // create and register logger
  std::vector<spdlog::sink_ptr> sinks{color_sink, syslog_sink};
  auto logger = std::make_shared<spdlog::logger>(logger_name_, sinks.begin(),
                                                 sinks.end());
  logger->set_level(spdlog::level::debug);
  spdlog::register_logger(logger);
}

std::shared_ptr<spdlog::logger> GetLogger() {
  static std::shared_ptr<spdlog::logger> logger;
  if (!logger) {
    CreateLogger();
    logger = spdlog::get(logger_name_);
  }
  return logger;
}

}  // namespace hwcv