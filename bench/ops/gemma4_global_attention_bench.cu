#include "ninfer/ops/softmax_attention.h"

#include "core/device.h"
#include "core/paged_kv_storage.h"
#include "ninfer_bench_common.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr int kD = 512;
constexpr int kQHeads = 32;
constexpr int kKVHeads = 4;
constexpr int kGroups = 8;
constexpr int kPage = 64;
constexpr float kScale = 1.0F;
constexpr ops::AttentionHeadGeometry kGeometry{kD, kQHeads, kKVHeads};

enum class Route : std::uint8_t { Optimized, Control, Both };

struct Options {
    std::vector<int> contexts{1024, 4096, 16384, 32768, 65536, 131072, 196608, 262144};
    std::vector<int> tokens{1};
    Route route = Route::Optimized;
    int warmup = 2;
    int repeat = 5;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\nusage: ninfer_gemma4_global_attention_bench "
                 "[--context 1024,4096,...] [--tokens 1,2,4,16,128,1024] "
                 "[--route optimized|control|both] [--warmup N] [--repeat N]\n",
                 message);
    std::exit(2);
}

int parse_int(std::string_view text, int minimum, int maximum, const char* flag) {
    const std::string value(text);
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < minimum ||
        parsed > maximum) {
        usage(flag);
    }
    return static_cast<int>(parsed);
}

std::vector<int> parse_list(const char* value, int minimum, int maximum, const char* flag) {
    std::vector<int> result;
    std::string_view remaining(value);
    while (!remaining.empty()) {
        const std::size_t comma = remaining.find(',');
        result.push_back(parse_int(remaining.substr(0, comma), minimum, maximum, flag));
        if (comma == std::string_view::npos) break;
        remaining.remove_prefix(comma + 1);
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* flag) {
            if (++index == argc) usage(flag);
            return argv[index];
        };
        if (argument == "--context") {
            options.contexts = parse_list(next("--context requires a value"), 0, 262144,
                                          "--context");
        } else if (argument == "--tokens") {
            options.tokens =
                parse_list(next("--tokens requires a value"), 1, 1024, "--tokens");
        } else if (argument == "--route") {
            const std::string_view value(next("--route requires a value"));
            if (value == "optimized")
                options.route = Route::Optimized;
            else if (value == "control")
                options.route = Route::Control;
            else if (value == "both")
                options.route = Route::Both;
            else
                usage("--route expects optimized, control, or both");
        } else if (argument == "--warmup") {
            options.warmup = parse_int(next("--warmup requires a value"), 0, 1000, "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = parse_int(next("--repeat requires a value"), 1, 1000, "--repeat");
        } else if (argument == "--help" || argument == "-h") {
            usage("help");
        } else {
            usage("unknown argument");
        }
    }
    for (int context : options.contexts) {
        if (context % kPage != 0) usage("--context values must be page aligned");
    }
    return options;
}

__global__ void initialize_cache_kernel(std::uint8_t* key, std::uint8_t* value,
                                        __half* key_scale, __half* value_scale,
                                        std::int64_t code_elements,
                                        std::int64_t scale_elements) {
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < code_elements; index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        key[index] = static_cast<std::uint8_t>(0x1fu + (index & 0x30u));
        value[index] = static_cast<std::uint8_t>(0x21u + (index & 0x20u));
    }
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < scale_elements; index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        key_scale[index] = __float2half_rn(0.015625f);
        value_scale[index] = __float2half_rn(0.0078125f);
    }
}

__global__ void control_kernel(const __nv_bfloat16* query, const __nv_bfloat16* key,
                               const __nv_bfloat16* value, int context, int tokens,
                               __nv_bfloat16* output) {
    extern __shared__ float probability[];
    const int query_token = static_cast<int>(blockIdx.x) / kQHeads;
    const int query_head = static_cast<int>(blockIdx.x) - query_token * kQHeads;
    const int kv_head = query_head / 8;
    const int position = context + query_token;
    const auto* q = query + (query_token * kQHeads + query_head) * kD;
    for (int key_token = threadIdx.x; key_token <= position; key_token += blockDim.x) {
        const auto* k = key + (key_token * kKVHeads + kv_head) * kD;
        float score = 0.0F;
        for (int d = 0; d < kD; ++d) score += __bfloat162float(q[d]) * __bfloat162float(k[d]);
        probability[key_token] = score;
    }
    __syncthreads();
    __shared__ float maximum;
    if (threadIdx.x == 0) {
        maximum = -CUDART_INF_F;
        for (int key_token = 0; key_token <= position; ++key_token) {
            maximum = fmaxf(maximum, probability[key_token]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float denominator = 0.0F;
        for (int key_token = 0; key_token <= position; ++key_token) {
            probability[key_token] = expf(probability[key_token] - maximum);
            denominator += probability[key_token];
        }
        for (int key_token = 0; key_token <= position; ++key_token) {
            probability[key_token] = __bfloat162float(
                __float2bfloat16(probability[key_token] / denominator));
        }
    }
    __syncthreads();
    auto* out = output + (query_token * kQHeads + query_head) * kD;
    for (int d = threadIdx.x; d < kD; d += blockDim.x) {
        float result = 0.0F;
        for (int key_token = 0; key_token <= position; ++key_token) {
            const auto* v = value + (key_token * kKVHeads + kv_head) * kD;
            result += probability[key_token] * __bfloat162float(v[d]);
        }
        out[d] = __float2bfloat16(result);
    }
}

class Case {
public:
    Case(int context, int tokens, bool with_control)
        : context_(context), tokens_(tokens), pages_((context + tokens + kPage - 1) / kPage),
          q_(bench::make_bf16(static_cast<std::size_t>(kD) * kQHeads * tokens)),
          k_(bench::make_bf16(static_cast<std::size_t>(kD) * kKVHeads * tokens)),
          v_(bench::make_bf16(static_cast<std::size_t>(kD) * kKVHeads * tokens)),
          positions_(static_cast<std::size_t>(tokens) * sizeof(std::int32_t)),
          output_(bench::make_zeros(static_cast<std::size_t>(kD) * kQHeads * tokens * 2)),
          cache_k_(bench::make_zeros(static_cast<std::size_t>(kD / 2) * kPage * kKVHeads * pages_)),
          cache_v_(bench::make_zeros(cache_k_.bytes)),
          scale_k_(bench::make_zeros(static_cast<std::size_t>(kGroups) * kPage * kKVHeads * pages_ * 2)),
          scale_v_(bench::make_zeros(scale_k_.bytes)),
          table_(static_cast<std::size_t>(pages_) * sizeof(std::int32_t)),
          control_k_(with_control ? bench::make_bf16(static_cast<std::size_t>(kD) * kKVHeads *
                                                      (context + tokens))
                                  : DeviceBuffer{}),
          control_v_(with_control ? bench::make_bf16(static_cast<std::size_t>(kD) * kKVHeads *
                                                      (context + tokens))
                                  : DeviceBuffer{}),
          q_tensor_(q_.p, DType::BF16, {kD, kQHeads, tokens}),
          k_tensor_(k_.p, DType::BF16, {kD, kKVHeads, tokens}),
          v_tensor_(v_.p, DType::BF16, {kD, kKVHeads, tokens}),
          positions_tensor_(positions_.p, DType::I32, {tokens}),
          out_tensor_(output_.p, DType::BF16, {kD, kQHeads, tokens}),
          workspace_bytes_(workspace_capacity()), workspace_(std::max<std::size_t>(workspace_bytes_, 256)) {
        std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
        for (int token = 0; token < tokens; ++token) positions[token] = context + token;
        std::vector<std::int32_t> table(static_cast<std::size_t>(pages_));
        for (int page = 0; page < pages_; ++page) table[page] = pages_ - 1 - page;
        CUDA_CHECK(cudaMemcpy(positions_.p, positions.data(), positions_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(table_.p, table.data(), table_.bytes, cudaMemcpyHostToDevice));
        const std::int64_t code_elements = static_cast<std::int64_t>(cache_k_.bytes);
        const std::int64_t scale_elements = static_cast<std::int64_t>(scale_k_.bytes / 2);
        initialize_cache_kernel<<<1024, 256>>>(
            static_cast<std::uint8_t*>(cache_k_.p), static_cast<std::uint8_t*>(cache_v_.p),
            static_cast<__half*>(scale_k_.p), static_cast<__half*>(scale_v_.p), code_elements,
            scale_elements);
        CUDA_CHECK(cudaGetLastError());
        cache_ = {
            .k_pages = Tensor(cache_k_.p, DType::U8, {kD / 2, kPage, kKVHeads, pages_}),
            .v_pages = Tensor(cache_v_.p, DType::U8, {kD / 2, kPage, kKVHeads, pages_}),
            .k_scale_pages = Tensor(scale_k_.p, DType::FP16, {kGroups, kPage, kKVHeads, pages_}),
            .v_scale_pages = Tensor(scale_v_.p, DType::FP16, {kGroups, kPage, kKVHeads, pages_}),
            .block_table = Tensor(table_.p, DType::I32, {pages_}),
            .head_dim = kD,
            .num_kv_heads = kKVHeads,
            .storage = KvCacheStorage::RK4V4E8,
        };
        if (with_control) {
            CUDA_CHECK(cudaFuncSetAttribute(control_kernel,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>((context + tokens) * sizeof(float))));
        }
    }

    void optimized(cudaStream_t stream) {
        for (int begin = 0; begin < tokens_; begin += kPage) {
            const int count = std::min(kPage, tokens_ - begin);
            const ops::CausalAttentionExecutionEnvelope envelope{
                static_cast<std::uint32_t>(context_ + begin + 1),
                static_cast<std::uint32_t>(context_ + begin + count)};
            Tensor q = q_tensor_.slice(2, begin, count);
            Tensor k = k_tensor_.slice(2, begin, count);
            Tensor v = v_tensor_.slice(2, begin, count);
            Tensor positions = positions_tensor_.slice(0, begin, count);
            Tensor out = out_tensor_.slice(2, begin, count);
            ops::causal_full_softmax_attention(q, k, v, positions, kGeometry, kScale, cache_,
                                               envelope, workspace_, out, stream);
        }
    }

    void control(cudaStream_t stream) {
        control_kernel<<<tokens_ * kQHeads, 256,
                         static_cast<std::size_t>(context_ + tokens_) * sizeof(float), stream>>>(
            static_cast<const __nv_bfloat16*>(q_.p),
            static_cast<const __nv_bfloat16*>(control_k_.p),
            static_cast<const __nv_bfloat16*>(control_v_.p), context_, tokens_,
            static_cast<__nv_bfloat16*>(output_.p));
        CUDA_CHECK(cudaGetLastError());
    }

    std::size_t workspace_bytes() const { return workspace_bytes_; }

private:
    std::size_t workspace_capacity() const {
        std::size_t maximum = 0;
        for (int begin = 0; begin < tokens_; begin += kPage) {
            const int count = std::min(kPage, tokens_ - begin);
            const ops::CausalAttentionExecutionEnvelope envelope{
                static_cast<std::uint32_t>(context_ + begin + 1),
                static_cast<std::uint32_t>(context_ + begin + count)};
            maximum = std::max(
                maximum, ops::causal_full_softmax_attention_workspace_capacity_bytes(
                             kGeometry, KvCacheStorage::RK4V4E8, envelope, count, count));
        }
        return maximum;
    }

    int context_;
    int tokens_;
    int pages_;
    DeviceBuffer q_;
    DeviceBuffer k_;
    DeviceBuffer v_;
    DeviceBuffer positions_;
    DeviceBuffer output_;
    DeviceBuffer cache_k_;
    DeviceBuffer cache_v_;
    DeviceBuffer scale_k_;
    DeviceBuffer scale_v_;
    DeviceBuffer table_;
    DeviceBuffer control_k_;
    DeviceBuffer control_v_;
    Tensor q_tensor_;
    Tensor k_tensor_;
    Tensor v_tensor_;
    Tensor positions_tensor_;
    Tensor out_tensor_;
    PagedKVLayerView cache_;
    std::size_t workspace_bytes_;
    DeviceArena workspace_;
};

double visible_pairs(int context, int tokens) {
    return static_cast<double>(tokens) * (context + 1) +
           static_cast<double>(tokens) * (tokens - 1) / 2.0;
}

template <class Launch>
void measure(const char* route, int context, int tokens, std::size_t workspace_bytes,
             Launch&& launch, int warmup, int repeat, cudaStream_t stream) {
    const auto timing = bench::measure_launch(launch, stream, warmup, repeat);
    const double seconds = timing.median_us * 1.0e-6;
    const double flops = visible_pairs(context, tokens) * kQHeads * (4.0 * kD);
    const double decode_bytes = static_cast<double>(context + tokens) * kKVHeads * 544.0;
    std::printf("route=%-9s C=%6d T=%4d calls=%2d median=%10.3f us p95=%10.3f us "
                "useful=%7.2f TFLOP/s decode_payload=%7.1f GB/s workspace=%zu\n",
                route, context, tokens, (tokens + kPage - 1) / kPage, timing.median_us,
                timing.p95_us, flops / seconds / 1.0e12,
                tokens <= 4 ? decode_bytes / seconds / 1.0e9 : 0.0, workspace_bytes);
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        for (int context : options.contexts) {
            for (int tokens : options.tokens) {
                const bool control_supported = context + tokens <= 16384;
                if (options.route == Route::Control && !control_supported) continue;
                Case data(context, tokens,
                          options.route != Route::Optimized && control_supported);
                if (options.route != Route::Control) {
                    measure("optimized", context, tokens, data.workspace_bytes(),
                            [&](cudaStream_t value) { data.optimized(value); }, options.warmup,
                            options.repeat, stream);
                }
                if (options.route != Route::Optimized && control_supported) {
                    measure("control", context, tokens, 0,
                            [&](cudaStream_t value) { data.control(value); }, options.warmup,
                            options.repeat, stream);
                }
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_gemma4_global_attention_bench: %s\n", error.what());
        return 1;
    }
}
