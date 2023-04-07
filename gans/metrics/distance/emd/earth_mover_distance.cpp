#ifndef _EMD
#define _EMD

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void approxmatch(int b, int n, int m, const float *xyz1, const float *xyz2,
                 float *match, float *temp, cudaStream_t stream);
void matchcost(int b, int n, int m, const float *xyz1, const float *xyz2,
               float *match, float *out, cudaStream_t stream);
void matchcostgrad(int b, int n, int m, const float *xyz1, const float *xyz2,
                   const float *match, float *grad1, float *grad2,
                   cudaStream_t stream);

//  temp: TensorShape{b,(n+m)*2}
std::vector<at::Tensor> ApproxMatch(at::Tensor set_d, at::Tensor set_q) {
  // std::cout << "[ApproxMatch] Called." << std::endl;
  int64_t batch_size = set_d.size(0);
  int64_t n_dataset_points = set_d.size(1); // n
  int64_t n_query_points = set_q.size(1);   // m
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(set_d));
  // std::cout << "[ApproxMatch] batch_size:" << batch_size << std::endl;
  at::Tensor match = torch::empty(
      {batch_size, n_query_points, n_dataset_points},
      torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
  at::Tensor temp = torch::empty(
      {batch_size, (n_query_points + n_dataset_points) * 2},
      torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
  CHECK_INPUT(set_d);
  CHECK_INPUT(set_q);
  CHECK_INPUT(match);
  CHECK_INPUT(temp);

  approxmatch(batch_size, n_dataset_points, n_query_points, set_d.data<float>(),
              set_q.data<float>(), match.data<float>(), temp.data<float>(),
              at::cuda::getCurrentCUDAStream());
  return {match, temp};
}

at::Tensor MatchCost(at::Tensor set_d, at::Tensor set_q, at::Tensor match) {
  // std::cout << "[MatchCost] Called." << std::endl;
  int64_t batch_size = set_d.size(0);
  int64_t n_dataset_points = set_d.size(1); // n
  int64_t n_query_points = set_q.size(1);   // m
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(set_d));
  // std::cout << "[MatchCost] batch_size:" << batch_size << std::endl;
  at::Tensor out = torch::empty(
      {batch_size},
      torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
  CHECK_INPUT(set_d);
  CHECK_INPUT(set_q);
  CHECK_INPUT(match);
  CHECK_INPUT(out);
  matchcost(batch_size, n_dataset_points, n_query_points, set_d.data<float>(),
            set_q.data<float>(), match.data<float>(), out.data<float>(),
            at::cuda::getCurrentCUDAStream());
  return out;
}

std::vector<at::Tensor> MatchCostGrad(at::Tensor set_d, at::Tensor set_q,
                                      at::Tensor match) {
  // std::cout << "[MatchCostGrad] Called." << std::endl;
  int64_t batch_size = set_d.size(0);
  int64_t n_dataset_points = set_d.size(1); // n
  int64_t n_query_points = set_q.size(1);   // m
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(set_d));
  // std::cout << "[MatchCostGrad] batch_size:" << batch_size << std::endl;
  at::Tensor grad1 = torch::empty(
      {batch_size, n_dataset_points, 3},
      torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
  at::Tensor grad2 = torch::empty(
      {batch_size, n_query_points, 3},
      torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
  CHECK_INPUT(set_d);
  CHECK_INPUT(set_q);
  CHECK_INPUT(match);
  CHECK_INPUT(grad1);
  CHECK_INPUT(grad2);
  matchcostgrad(batch_size, n_dataset_points, n_query_points,
                set_d.data<float>(), set_q.data<float>(), match.data<float>(),
                grad1.data<float>(), grad2.data<float>(),
                at::cuda::getCurrentCUDAStream());
  return {grad1, grad2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("approxmatch_forward", &ApproxMatch, "ApproxMatch forward (CUDA)");
  m.def("matchcost_forward", &MatchCost, "MatchCost forward (CUDA)");
  m.def("matchcost_backward", &MatchCostGrad, "MatchCost backward (CUDA)");
}

#endif