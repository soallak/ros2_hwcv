#pragma once

#include <stdexcept>

namespace hwcv {

class NotImplemented : public std::runtime_error {
 public:
  explicit NotImplemented(std::string const& what_arg)
      : std::runtime_error(what_arg) {}
};

class InvalidArgument : public std::invalid_argument {
 public:
  explicit InvalidArgument(std::string const& what_arg)
      : std::invalid_argument(what_arg) {}
};

class CLError : public std::runtime_error {
 public:
  explicit CLError(std::string const& what_arg)
      : std::runtime_error(what_arg) {}
};

}  // namespace hwcv