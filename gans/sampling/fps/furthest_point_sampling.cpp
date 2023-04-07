// adopted from https: // github.com/erikwijmans/Pointnet2_PyTorch

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IS_INT(x)                                                        \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Int,                          \
              #x " must be an int tensor")
#define CHECK_IS_FLOAT(x)                                                      \
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float,                        \
              #x " must be a float tensor");

void gather_points_kernel_wrapper(int b, int c, int n, int npoints,
                                  const float *points, const int *idx,
                                  float *out);
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints,
                                       const float *grad_out, const int *idx,
                                       float *grad_points);

void furthest_point_sampling_kernel_wrapper(int b, int n, int m,
                                            const float *dataset, float *temp,
                                            int *idxs);

at::Tensor gather_points(at::Tensor points, at::Tensor idx) {
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(points);
  CHECK_IS_INT(idx);

  if (points.is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({points.size(0), points.size(1), idx.size(1)},
                   at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.is_cuda()) {
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
    gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                 idx.size(1), points.data_ptr<float>(),
                                 idx.data_ptr<int>(), output.data_ptr<float>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx,
                              const int n) {
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(idx);
  CHECK_IS_FLOAT(grad_out);
  CHECK_IS_INT(idx);

  if (grad_out.is_cuda()) {
    CHECK_CUDA(idx);
  }

  at::Tensor output =
      torch::zeros({grad_out.size(0), grad_out.size(1), n},
                   at::device(grad_out.device()).dtype(at::ScalarType::Float));

  if (grad_out.is_cuda()) {
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(grad_out));
    gather_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n,
                                      idx.size(1), grad_out.data_ptr<float>(),
                                      idx.data_ptr<int>(),
                                      output.data_ptr<float>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}

at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples) {
  CHECK_CONTIGUOUS(points);
  CHECK_IS_FLOAT(points);

  at::Tensor output =
      torch::zeros({points.size(0), nsamples},
                   at::device(points.device()).dtype(at::ScalarType::Int));

  at::Tensor tmp =
      torch::full({points.size(0), points.size(1)}, 1e10,
                  at::device(points.device()).dtype(at::ScalarType::Float));

  if (points.is_cuda()) {
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(points));
    furthest_point_sampling_kernel_wrapper(
        points.size(0), points.size(1), nsamples, points.data_ptr<float>(),
        tmp.data_ptr<float>(), output.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("furthest_point_sampling", &furthest_point_sampling,
        "furthest point sampling");
  m.def("gather_points", &gather_points, "gather points");
  m.def("gather_points_grad", &gather_points_grad, "gather points grad");
}