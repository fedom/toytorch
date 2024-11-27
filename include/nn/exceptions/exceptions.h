#ifndef SRC_EXCEPTION_EXCEPTIONS_H__
#define SRC_EXCEPTION_EXCEPTIONS_H__
#include <exception>
#include <string>

namespace toytorch {

class ExceptionBase : public std::exception {
 public:
  ExceptionBase(const std::string message) : message_(std::move(message)) {}
  virtual const char* what() const noexcept { return message_.c_str(); }

 private:
  std::string message_;
};

class ExceptionNotImpl : public ExceptionBase {
 public:
  ExceptionNotImpl(const std::string& msg = "")
      : ExceptionBase("NotImplemented," + msg) {}
};

class ExceptionOpBackwardNotImplemented : public ExceptionNotImpl {
 public:
  ExceptionOpBackwardNotImplemented(
      const std::string& msg = "")
      : ExceptionNotImpl("OpBackwardNotImplemented,") {}
};

class ExceptionInvalidArgument : public ExceptionBase {
 public:
  ExceptionInvalidArgument(const std::string& msg = "")
      : ExceptionBase("Invalid arguments," + msg) {}
};

class ExceptionTensorShapeIncompatible : public ExceptionInvalidArgument {
 public:
  ExceptionTensorShapeIncompatible(const std::string& msg = "")
      : ExceptionInvalidArgument("Tensor shape incompatible," + msg) {}
};

}  // namespace toytorch

#endif  // SRC_EXCEPTION_EXCEPTIONS_H__