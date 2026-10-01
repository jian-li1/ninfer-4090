#include "ninfer/ops/softmax_attention.h"

#include "core/device.h"
#include "core/paged_kv_storage.h"
#include "ninfer_bench_common.h"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <cuda_runtime.h>

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

constexpr int kD = 256;
constexpr int kQHeads = 32;
constexpr int kKVHeads = 16;
constexpr int kWindow = 1024;
constexpr int kPage = 64;
constexpr int kRingPages = 17;
constexpr int kGroups = 4;
constexpr float kScale = 1.0F;
constexpr ops::AttentionHeadGeometry kGeometry{kD, kQHeads, kKVHeads};

enum class Route : std::uint8_t { Optimized, Control, Both };

struct Options {
    std::vector<int> tokens{128, 256, 512, 1024, 2048};
    Route route = Route::Both;
    int warmup = 3;
    int repeat = 10;
};

[[noreturn]] void usage(const char* message) {
    std::fprintf(stderr,
                 "error: %s\nusage: ninfer_gemma4_sliding_attention_bench "
                 "[--tokens 128,256,512,1024,2048] [--route optimized|control|both] "
                 "[--warmup N] [--repeat N]\n",
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

std::vector<int> parse_list(const char* value) {
    std::vector<int> result;
    std::string_view remaining(value);
    while (!remaining.empty()) {
        const std::size_t comma = remaining.find(',');
        result.push_back(parse_int(remaining.substr(0, comma), 1, 2048, "--tokens"));
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
        if (argument == "--tokens") {
            options.tokens = parse_list(next("--tokens requires a value"));
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
    return options;
}

__global__ void control_kernel(const __nv_bfloat16* query, const __nv_bfloat16* key,
                               const __nv_bfloat16* value, int tokens,
                               __nv_bfloat16* output) {
    extern __shared__ float probability[];
    const int query_token = static_cast<int>(blockIdx.x) / kQHeads;
    const int query_head = static_cast<int>(blockIdx.x) - query_token * kQHeads;
    const int kv_head = query_head / 2;
    const int first_key = max(0, query_token - kWindow + 1);
    const auto* q = query + (query_token * kQHeads + query_head) * kD;
    for (int key_token = first_key + threadIdx.x; key_token <= query_token;
         key_token += blockDim.x) {
        const auto* k = key + (key_token * kKVHeads + kv_head) * kD;
        float score = 0.0F;
        for (int d = 0; d < kD; ++d) score += __bfloat162float(q[d]) * __bfloat162float(k[d]);
        probability[key_token] = score * kScale;
    }
    __syncthreads();
    __shared__ float maximum;
    if (threadIdx.x == 0) {
        maximum = -CUDART_INF_F;
        for (int key_token = first_key; key_token <= query_token; ++key_token) {
            maximum = fmaxf(maximum, probability[key_token]);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0F;
        for (int key_token = first_key; key_token <= query_token; ++key_token) {
            probability[key_token] = expf(probability[key_token] - maximum);
            sum += probability[key_token];
        }
        const float inverse_sum = 1.0F / sum;
        for (int key_token = first_key; key_token <= query_token; ++key_token) {
            probability[key_token] = __bfloat162float(
                __float2bfloat16(probability[key_token] * inverse_sum));
        }
    }
    __syncthreads();
    auto* out = output + (query_token * kQHeads + query_head) * kD;
    for (int d = threadIdx.x; d < kD; d += blockDim.x) {
        float result = 0.0F;
        for (int key_token = first_key; key_token <= query_token; ++key_token) {
            const auto* v = value + (key_token * kKVHeads + kv_head) * kD;
            result += probability[key_token] * __bfloat162float(v[d]);
        }
        out[d] = __float2bfloat16(result);
    }
}

class Case {
public:
    explicit Case(int tokens)
        : tokens_(tokens),
          q_(bench::make_bf16(static_cast<std::size_t>(kD) * kQHeads * tokens)),
          k_(bench::make_bf16(static_cast<std::size_t>(kD) * kKVHeads * tokens)),
          v_(bench::make_bf16(static_cast<std::size_t>(kD) * kKVHeads * tokens)),
          positions_(static_cast<std::size_t>(tokens) * sizeof(std::int32_t)),
          output_(bench::make_zeros(static_cast<std::size_t>(kD) * kQHeads * tokens * 2)),
          cache_k_(bench::make_zeros(static_cast<std::size_t>(kD / 2) * kPage * kKVHeads *
                                     kRingPages)),
          cache_v_(bench::make_zeros(cache_k_.bytes)),
          scale_k_(bench::make_zeros(static_cast<std::size_t>(kGroups) * kPage * kKVHeads *
                                     kRingPages * 2)),
          scale_v_(bench::make_zeros(scale_k_.bytes)),
          table_(kRingPages * sizeof(std::int32_t)),
          q_tensor_(q_.p, DType::BF16, {kD, kQHeads, tokens}),
          k_tensor_(k_.p, DType::BF16, {kD, kKVHeads, tokens}),
          v_tensor_(v_.p, DType::BF16, {kD, kKVHeads, tokens}),
          positions_tensor_(positions_.p, DType::I32, {tokens}),
          out_tensor_(output_.p, DType::BF16, {kD, kQHeads, tokens}) {
        std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
        for (int token = 0; token < tokens; ++token) {
            positions[static_cast<std::size_t>(token)] = token;
        }
        std::vector<std::int32_t> table(kRingPages);
        for (int page = 0; page < kRingPages; ++page) table[static_cast<std::size_t>(page)] = page;
        CUDA_CHECK(cudaMemcpy(positions_.p, positions.data(), positions_.bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(table_.p, table.data(), table_.bytes, cudaMemcpyHostToDevice));
        cache_ = {
            .k_pages = Tensor(cache_k_.p, DType::U8, {kD / 2, kPage, kKVHeads, kRingPages}),
            .v_pages = Tensor(cache_v_.p, DType::U8, {kD / 2, kPage, kKVHeads, kRingPages}),
            .k_scale_pages = Tensor(scale_k_.p, DType::FP16,
                                    {kGroups, kPage, kKVHeads, kRingPages}),
            .v_scale_pages = Tensor(scale_v_.p, DType::FP16,
                                    {kGroups, kPage, kKVHeads, kRingPages}),
            .block_table = Tensor(table_.p, DType::I32, {kRingPages}),
            .head_dim = kD,
            .num_kv_heads = kKVHeads,
            .storage = KvCacheStorage::RK4V4E8,
        };
    }

    void optimized(cudaStream_t stream) {
        for (int begin = 0; begin < tokens_; begin += kPage) {
            optimized_segment(begin, std::min(kPage, tokens_ - begin), stream);
        }
    }

    void prepare_decode(cudaStream_t stream) {
        for (int begin = 0; begin < kWindow; begin += kPage) {
            optimized_segment(begin, kPage, stream);
        }
    }

    void decode(cudaStream_t stream) { optimized_segment(kWindow, 1, stream); }

    void control(cudaStream_t stream) {
        control_kernel<<<tokens_ * kQHeads, 256, static_cast<std::size_t>(tokens_) * sizeof(float),
                         stream>>>(static_cast<const __nv_bfloat16*>(q_.p),
                                   static_cast<const __nv_bfloat16*>(k_.p),
                                   static_cast<const __nv_bfloat16*>(v_.p), tokens_,
                                   static_cast<__nv_bfloat16*>(output_.p));
        CUDA_CHECK(cudaGetLastError());
    }

private:
    void optimized_segment(int begin, int count, cudaStream_t stream) {
        Tensor q = q_tensor_.slice(2, begin, count);
        Tensor k = k_tensor_.slice(2, begin, count);
        Tensor v = v_tensor_.slice(2, begin, count);
        Tensor positions = positions_tensor_.slice(0, begin, count);
        Tensor out = out_tensor_.slice(2, begin, count);
        ops::causal_sliding_softmax_attention(q, k, v, positions, kGeometry, kWindow, kScale,
                                              cache_, out, stream);
    }

    int tokens_;
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
    Tensor q_tensor_;
    Tensor k_tensor_;
    Tensor v_tensor_;
    Tensor positions_tensor_;
    Tensor out_tensor_;
    PagedKVLayerView cache_;
};

double visible_pairs(int tokens) {
    if (tokens <= kWindow) return static_cast<double>(tokens) * (tokens + 1) / 2.0;
    return static_cast<double>(kWindow) * (kWindow + 1) / 2.0 +
           static_cast<double>(tokens - kWindow) * kWindow;
}

template <class Launch>
void measure_route(const char* name, int tokens, Launch&& launch, int warmup, int repeat,
                   cudaStream_t stream) {
    const auto timing = bench::measure_launch(launch, stream, warmup, repeat);
    const double seconds = timing.median_us * 1.0e-6;
    const double flops = visible_pairs(tokens) * kQHeads * (4.0 * kD);
    std::printf("route=%-9s T=%4d calls=%2d median=%9.3f us p95=%9.3f us tokens=%9.1f/s "
                "useful=%7.2f TFLOP/s\n",
                name, tokens, (tokens + kPage - 1) / kPage, timing.median_us, timing.p95_us,
                tokens / seconds, flops / seconds / 1.0e12);
}

void measure_decode(Case& data, int warmup, int repeat, cudaStream_t stream) {
    data.prepare_decode(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto timing = bench::measure_launch(
        [&](cudaStream_t value) { data.decode(value); }, stream, warmup, repeat);
    const double flops = static_cast<double>(kWindow) * kQHeads * (4.0 * kD);
    std::printf(
        "route=optimized decode_context=%d median=%9.3f us p95=%9.3f us useful=%7.2f TFLOP/s\n",
                kWindow, timing.median_us, timing.p95_us,
                flops / (timing.median_us * 1.0e-6) / 1.0e12);
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
        for (int tokens : options.tokens) {
            Case data(tokens);
            if (options.route != Route::Control) {
                measure_route("optimized", tokens,
                              [&](cudaStream_t value) { data.optimized(value); }, options.warmup,
                              options.repeat, stream);
            }
            if (options.route != Route::Optimized) {
                measure_route("control", tokens,
                              [&](cudaStream_t value) { data.control(value); }, options.warmup,
                              options.repeat, stream);
            }
        }
        if (options.route != Route::Control) {
            Case decode(kWindow + 1);
            measure_decode(decode, options.warmup, options.repeat, stream);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_gemma4_sliding_attention_bench: %s\n", error.what());
        return 1;
    }
}
