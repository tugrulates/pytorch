#include "ATen/ATen.h"

#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>

namespace at { namespace native {

Tensor& _clamp__cuda(Tensor& self, optional<Scalar> min, optional<Scalar> max) {
  return _clamp_out_cuda(self, self, min, max);
}

Tensor& _clamp_out_cuda(
    Tensor& result,
    const Tensor& self,
    optional<Scalar> min,
    optional<Scalar> max) {
  if (min && max) {
    _th_clamp_out(result, self, *min, *max);
  } else if (max) {
    _th_clamp_max_out(result, self, *max);
  } else if (min) {
    _th_clamp_min_out(result, self, *min);
  } else {
    AT_ERROR("At least one of 'min' or 'max' must not be None");
  }
  return result;
}

Tensor& _clamp_max__cuda(Tensor& self, Scalar max) {
  return _th_clamp_max_out(self, self, max);
}

Tensor& _clamp_max_out_cuda(Tensor& result, const Tensor& self, Scalar max) {
  return _th_clamp_max_out(result, self, max);
}

Tensor& _clamp_min__cuda(Tensor& self, Scalar min) {
  return _th_clamp_min_out(self, self, min);
}

Tensor& _clamp_min_out_cuda(Tensor& result, const Tensor& self, Scalar min) {
  return _th_clamp_min_out(result, self, min);
}

// These are just forwarding stubs

#define IMPLEMENT_UNARY_OP_PREQUEL(op)                           \
  Tensor& _##op##__cuda(Tensor& self) {                          \
    return at::_th_##op##_out(self, self);                       \
  }                                                              \
  Tensor& _##op##_out_cuda(Tensor& result, const Tensor& self) { \
    return at::_th_##op##_out(result, self);                     \
  }

#define IMPLEMENT_UNARY_OP(op)                                                 \
  Tensor& _##op##__cuda(Tensor& self) {                                        \
    auto result_ = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(       \
                       CUDATensorId(),                                         \
                       self.dtype(),                                           \
                       cuda::getCUDADeviceAllocator(),                         \
                       false)                                                  \
                       .release();                                             \
    auto out =                                                                 \
        Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>::reclaim(   \
            result_));                                                         \
    return _##op##_out_cuda(out, self);                                        \
  }                                                                            \
  Tensor& _##op##_out_cuda(Tensor& result, const Tensor& self) {               \
    auto result_ = checked_tensor_unwrap(                                      \
        result, "result", 0, false, Backend::CUDA, self.scalar_type());        \
    auto self_ = self.unsafeGetTensorImpl();                                          \
    AT_DISPATCH_FLOATING_TYPES(self.type(), #op, [&] {                         \
      THCudaDoubleTensor_##op(globalContext().getTHCState(), result_, self_);  \
    });                                                                        \
    result_->maybe_zero_dim(self_->dim() == 0);                                \
    return result;                                                             \
  }

IMPLEMENT_UNARY_OP_PREQUEL(abs)
IMPLEMENT_UNARY_OP_PREQUEL(acos)
IMPLEMENT_UNARY_OP_PREQUEL(asin)
IMPLEMENT_UNARY_OP_PREQUEL(atan)
IMPLEMENT_UNARY_OP_PREQUEL(ceil)
IMPLEMENT_UNARY_OP_PREQUEL(cos)
IMPLEMENT_UNARY_OP_PREQUEL(cosh)
IMPLEMENT_UNARY_OP_PREQUEL(erf)
IMPLEMENT_UNARY_OP_PREQUEL(erfc)
IMPLEMENT_UNARY_OP_PREQUEL(exp)
IMPLEMENT_UNARY_OP_PREQUEL(expm1)
IMPLEMENT_UNARY_OP_PREQUEL(floor)
IMPLEMENT_UNARY_OP_PREQUEL(log)
IMPLEMENT_UNARY_OP_PREQUEL(log10)
IMPLEMENT_UNARY_OP_PREQUEL(log1p)
IMPLEMENT_UNARY_OP_PREQUEL(log2)
IMPLEMENT_UNARY_OP_PREQUEL(round)
IMPLEMENT_UNARY_OP_PREQUEL(rsqrt)
IMPLEMENT_UNARY_OP_PREQUEL(sigmoid)
IMPLEMENT_UNARY_OP(sin)
IMPLEMENT_UNARY_OP_PREQUEL(sinh)
IMPLEMENT_UNARY_OP_PREQUEL(sqrt)
IMPLEMENT_UNARY_OP_PREQUEL(tan)
IMPLEMENT_UNARY_OP_PREQUEL(tanh)
IMPLEMENT_UNARY_OP_PREQUEL(trunc)

}}
