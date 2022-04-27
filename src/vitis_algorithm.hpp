#pragma once

#include <spdlog/fmt/fmt.h>
#include <spdlog/logger.h>

#include <string>
#include <vitis_common/xcl2/xcl2.hpp>

#include "logger.hpp"

namespace hwcv {

/**
 * @brief Base class for Algorithms intended to be compiled by Vitis Vpp
 *
 */
class VitisAlgorithm {
 public:
  /**
   * @brief Get the absolute path to the binary file object to be loaded
   *
   * @return std::string Binary filename
   */
  virtual std::string GetBinaryFile() = 0;

  /**
   * @brief Load the binary file
   *
   */
  virtual void LoadBinaryFile();

  /**
   * @brief Destroy the Vitis Algorithm object
   *
   */
  virtual ~VitisAlgorithm();

  VitisAlgorithm(VitisAlgorithm const&) = delete;
  VitisAlgorithm& operator=(VitisAlgorithm const&) = delete;
  VitisAlgorithm(VitisAlgorithm&&) = delete;
  VitisAlgorithm& operator=(VitisAlgorithm&&) = delete;

 protected:
  /**
   * @brief Construct a new Vitis Algorithm object
   *
   */
  VitisAlgorithm();

  inline bool IsBinaryLoaded() { return is_binary_loaded_; }

 protected:
  bool is_binary_loaded_{false};
  std::shared_ptr<spdlog::logger> logger_;
  cl::Device device_;
  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Program program_;
};

}  // namespace hwcv