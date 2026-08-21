#pragma once

#include "qknorm_rope.cuh"

namespace sglang {
namespace sensenova_u1_5 {

constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kAxisNormDim = 64;

template <uint32_t kDim, typename PackedFloat, std::size_t N>
SGL_DEVICE device::AlignedVector<PackedFloat, N> apply_norm_cast_before_weight(
    const device::AlignedVector<PackedFloat, N>& input,
    const device::AlignedVector<PackedFloat, N>& weight,
    float eps) {
  using namespace device;
  float sum_of_squares = 0.0f;
#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto values = cast<fp32x2_t>(input[i]);
    sum_of_squares += values.x * values.x + values.y * values.y;
  }
  const float norm_factor = math::rsqrt(warp::reduce_sum(sum_of_squares) / kDim + eps);
  device::AlignedVector<PackedFloat, N> output;
#pragma unroll
  for (auto i = 0u; i < N; ++i) {
    const auto values = cast<fp32x2_t>(input[i]);
    const auto weights = cast<fp32x2_t>(weight[i]);
    const auto normalized = cast<fp32x2_t, PackedFloat>(
        cast<PackedFloat, fp32x2_t>({values.x * norm_factor, values.y * norm_factor}));
    output[i] = cast<PackedFloat, fp32x2_t>({normalized.x * weights.x, normalized.y * weights.y});
  }
  return output;
}

struct QKNormRopeKVParams {
  void* __restrict__ q;
  const void* __restrict__ k;
  const void* __restrict__ v;
  void* __restrict__ cache_k;
  void* __restrict__ cache_v;
  const void* __restrict__ q_t_weight;
  const void* __restrict__ k_t_weight;
  const void* __restrict__ q_hw_weight;
  const void* __restrict__ k_hw_weight;
  const void* __restrict__ t_cos;
  const void* __restrict__ t_sin;
  const void* __restrict__ h_cos;
  const void* __restrict__ h_sin;
  const void* __restrict__ w_cos;
  const void* __restrict__ w_sin;
  int64_t q_batch_stride;
  int64_t q_token_stride;
  int64_t k_batch_stride;
  int64_t k_token_stride;
  int64_t v_batch_stride;
  int64_t v_token_stride;
  int64_t cache_batch_stride;
  int64_t cache_token_stride;
  int64_t head_stride;
  uint32_t batch_size;
  uint32_t num_tokens;
  uint32_t num_q_heads;
  uint32_t num_kv_heads;
  float eps;
};

template <uint32_t kFirstLane, uint32_t kLanes, typename Storage>
SGL_DEVICE void apply_neox_rope(
    Storage& values,
    const bf16_t* cos,
    const bf16_t* sin,
    uint32_t lane) {
  using namespace device;
  constexpr uint32_t kHalfLanes = kLanes / 2;
  constexpr uint32_t kMask =
      kLanes == 32 ? 0xffffffffu : (((1u << kLanes) - 1u) << kFirstLane);
  if (lane >= kFirstLane && lane < kFirstLane + kLanes) {
    const uint32_t local_lane = lane - kFirstLane;
    const uint32_t partner =
        lane + (local_lane < kHalfLanes ? kHalfLanes : -static_cast<int32_t>(kHalfLanes));
    auto partner_vec = values[0];
    auto partner_bits = reinterpret_cast<const uint32_t&>(partner_vec);
    partner_bits = __shfl_sync(kMask, partner_bits, partner);
    reinterpret_cast<uint32_t&>(partner_vec) = partner_bits;
    auto& x = unpack(values[0]);
    const auto& y = unpack(partner_vec);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
      const uint32_t cache_idx = (local_lane % kHalfLanes) * 2 + i;
      x[i] = local_lane < kHalfLanes ? rotary_sub(x[i], cos[cache_idx], y[i], sin[cache_idx])
                                     : rotary_add(x[i], cos[cache_idx], y[i], sin[cache_idx]);
    }
  }
}

__global__ void qknorm_rope_kv_kernel(const QKNormRopeKVParams __grid_constant__ p) {
  using namespace device;
  using Packed = packed_t<bf16_t>;
  using Storage = AlignedVector<Packed, 1>;

  const uint32_t lane = threadIdx.x % kWarpThreads;
  const uint32_t warp = threadIdx.x / kWarpThreads;
  const uint32_t first = blockIdx.x * kWarpsPerBlock + warp;
  const uint32_t workers = gridDim.x * kWarpsPerBlock;
  const uint32_t heads = p.num_q_heads + p.num_kv_heads;
  const uint32_t total = p.batch_size * p.num_tokens * heads;

  for (uint32_t work = first; work < total; work += workers) {
    const uint32_t head = work % heads;
    const uint32_t flat_token = work / heads;
    const uint32_t batch = flat_token / p.num_tokens;
    const uint32_t token = flat_token % p.num_tokens;
    const bool is_q = head < p.num_q_heads;
    const uint32_t local_head = is_q ? head : head - p.num_q_heads;
    const int64_t input_offset =
        batch * (is_q ? p.q_batch_stride : p.k_batch_stride) +
        token * (is_q ? p.q_token_stride : p.k_token_stride) + local_head * p.head_stride;
    const void* input = pointer::offset(is_q ? p.q : p.k, input_offset);
    void* output =
        is_q ? pointer::offset(p.q, input_offset)
             : pointer::offset(p.cache_k,
                               batch * p.cache_batch_stride + token * p.cache_token_stride +
                                   local_head * p.head_stride);
    const void* t_weight = is_q ? p.q_t_weight : p.k_t_weight;
    const void* hw_weight = is_q ? p.q_hw_weight : p.k_hw_weight;

    const auto t_input = load_as<Storage>(input, lane);
    const auto hw_input =
        load_as<Storage>(pointer::offset(input, kAxisNormDim * sizeof(bf16_t)), lane);
    const auto t_w = load_as<Storage>(t_weight, lane);
    const auto hw_w = load_as<Storage>(hw_weight, lane);
    auto t = apply_norm_cast_before_weight<kAxisNormDim>(t_input, t_w, p.eps);
    auto hw = apply_norm_cast_before_weight<kAxisNormDim>(hw_input, hw_w, p.eps);

    const auto* t_cos = static_cast<const bf16_t*>(p.t_cos) + token * kAxisNormDim;
    const auto* t_sin = static_cast<const bf16_t*>(p.t_sin) + token * kAxisNormDim;
    const auto* h_cos = static_cast<const bf16_t*>(p.h_cos) + token * (kAxisNormDim / 2);
    const auto* h_sin = static_cast<const bf16_t*>(p.h_sin) + token * (kAxisNormDim / 2);
    const auto* w_cos = static_cast<const bf16_t*>(p.w_cos) + token * (kAxisNormDim / 2);
    const auto* w_sin = static_cast<const bf16_t*>(p.w_sin) + token * (kAxisNormDim / 2);
    apply_neox_rope<0, 32>(t, t_cos, t_sin, lane);
    apply_neox_rope<0, 16>(hw, h_cos, h_sin, lane);
    apply_neox_rope<16, 16>(hw, w_cos, w_sin, lane);
    store_as<Storage>(output, t, lane);
    store_as<Storage>(pointer::offset(output, kAxisNormDim * sizeof(bf16_t)), hw, lane);

    if (!is_q) {
      const void* v_input = pointer::offset(
          p.v,
          batch * p.v_batch_stride + token * p.v_token_stride + local_head * p.head_stride);
      void* v_output = pointer::offset(
          p.cache_v,
          batch * p.cache_batch_stride + token * p.cache_token_stride + local_head * p.head_stride);
      const auto v_t = load_as<Storage>(v_input, lane);
      const auto v_hw = load_as<Storage>(
          pointer::offset(v_input, kAxisNormDim * sizeof(bf16_t)), lane);
      store_as<Storage>(v_output, v_t, lane);
      store_as<Storage>(pointer::offset(v_output, kAxisNormDim * sizeof(bf16_t)), v_hw, lane);
    }
  }
}

struct QKNormRopeKVKernel {
  static void run(
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView cache_k,
      const tvm::ffi::TensorView cache_v,
      const tvm::ffi::TensorView q_t_weight,
      const tvm::ffi::TensorView k_t_weight,
      const tvm::ffi::TensorView q_hw_weight,
      const tvm::ffi::TensorView k_hw_weight,
      const tvm::ffi::TensorView t_cos,
      const tvm::ffi::TensorView t_sin,
      const tvm::ffi::TensorView h_cos,
      const tvm::ffi::TensorView h_sin,
      const tvm::ffi::TensorView w_cos,
      const tvm::ffi::TensorView w_sin,
      float eps) {
    using namespace host;
    auto B = SymbolicSize{"batch"};
    auto N = SymbolicSize{"tokens"};
    auto Q = SymbolicSize{"q_heads"};
    auto K = SymbolicSize{"kv_heads"};
    auto D = SymbolicSize{"head_dim"};
    auto Bq = SymbolicSize{"q_batch_stride"};
    auto Nq = SymbolicSize{"q_token_stride"};
    auto Bk = SymbolicSize{"k_batch_stride"};
    auto Nk = SymbolicSize{"k_token_stride"};
    auto Bv = SymbolicSize{"v_batch_stride"};
    auto Nv = SymbolicSize{"v_token_stride"};
    auto Bo = SymbolicSize{"cache_batch_stride"};
    auto No = SymbolicSize{"cache_token_stride"};
    auto Hd = SymbolicSize{"head_stride"};
    auto device = SymbolicDevice{};
    D.set_value(kHeadDim);
    device.set_options<kDLCUDA>();
    TensorMatcher({B, N, Q, D}).with_strides({Bq, Nq, Hd, 1}).with_dtype<bf16_t>().with_device(device).verify(q);
    TensorMatcher({B, N, K, D}).with_strides({Bk, Nk, Hd, 1}).with_dtype<bf16_t>().with_device(device).verify(k);
    TensorMatcher({B, N, K, D}).with_strides({Bv, Nv, Hd, 1}).with_dtype<bf16_t>().with_device(device).verify(v);
    TensorMatcher({B, N, K, D}).with_strides({Bo, No, Hd, 1}).with_dtype<bf16_t>().with_device(device).verify(cache_k).verify(cache_v);
    TensorMatcher({kAxisNormDim}).with_dtype<bf16_t>().with_device(device)
        .verify(q_t_weight).verify(k_t_weight).verify(q_hw_weight).verify(k_hw_weight);
    TensorMatcher({N, kAxisNormDim}).with_dtype<bf16_t>().with_device(device).verify(t_cos).verify(t_sin);
    TensorMatcher({N, kAxisNormDim / 2}).with_dtype<bf16_t>().with_device(device)
        .verify(h_cos).verify(h_sin).verify(w_cos).verify(w_sin);

    const uint32_t batch = static_cast<uint32_t>(B.unwrap());
    const uint32_t tokens = static_cast<uint32_t>(N.unwrap());
    const uint32_t q_heads = static_cast<uint32_t>(Q.unwrap());
    const uint32_t kv_heads = static_cast<uint32_t>(K.unwrap());
    if (batch == 0 || tokens == 0 || (q_heads == 0 && kv_heads == 0)) return;
    constexpr int64_t kElementBytes = sizeof(bf16_t);
    QKNormRopeKVParams p{
        .q = q.data_ptr(), .k = k.data_ptr(), .v = v.data_ptr(),
        .cache_k = cache_k.data_ptr(), .cache_v = cache_v.data_ptr(),
        .q_t_weight = q_t_weight.data_ptr(), .k_t_weight = k_t_weight.data_ptr(),
        .q_hw_weight = q_hw_weight.data_ptr(), .k_hw_weight = k_hw_weight.data_ptr(),
        .t_cos = t_cos.data_ptr(), .t_sin = t_sin.data_ptr(),
        .h_cos = h_cos.data_ptr(), .h_sin = h_sin.data_ptr(),
        .w_cos = w_cos.data_ptr(), .w_sin = w_sin.data_ptr(),
        .q_batch_stride = Bq.unwrap() * kElementBytes, .q_token_stride = Nq.unwrap() * kElementBytes,
        .k_batch_stride = Bk.unwrap() * kElementBytes, .k_token_stride = Nk.unwrap() * kElementBytes,
        .v_batch_stride = Bv.unwrap() * kElementBytes, .v_token_stride = Nv.unwrap() * kElementBytes,
        .cache_batch_stride = Bo.unwrap() * kElementBytes, .cache_token_stride = No.unwrap() * kElementBytes,
        .head_stride = Hd.unwrap() * kElementBytes,
        .batch_size = batch, .num_tokens = tokens, .num_q_heads = q_heads, .num_kv_heads = kv_heads,
        .eps = eps,
    };
    const uint32_t works = batch * tokens * (q_heads + kv_heads);
    const uint32_t sm = runtime::get_sm_count(device.unwrap().device_id);
    const uint32_t max_blocks = runtime::get_blocks_per_sm(qknorm_rope_kv_kernel, kThreadsPerBlock) * sm;
    LaunchKernel(std::min(max_blocks, div_ceil(works, kWarpsPerBlock)), kThreadsPerBlock, device.unwrap())(
        qknorm_rope_kv_kernel, p);
  }
};

}  // namespace sensenova_u1_5
}  // namespace sglang
