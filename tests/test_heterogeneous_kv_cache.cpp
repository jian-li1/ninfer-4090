#include "core/device.h"
#include "core/decode_graph.h"
#include "core/heterogeneous_kv_cache.h"
#include "runtime/kv_groups.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::HeterogeneousKVCache;
using ninfer::HeterogeneousKVCacheLayout;
using ninfer::HeterogeneousKVHostImage;
using ninfer::KvGroupRetention;
using ninfer::KvGroupSpec;

[[noreturn]] void fail(std::string_view message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void require(bool condition, std::string_view message) {
    if (!condition) { fail(message); }
}

template <typename Exception, typename Fn>
void require_throws(Fn&& fn, std::string_view message) {
    try {
        fn();
    } catch (const Exception&) { return; }
    fail(message);
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::array<std::int32_t, 2> read_state(const ninfer::Tensor& state) {
    std::array<std::int32_t, 2> host{};
    CUDA_CHECK(cudaMemcpy(host.data(), state.data, sizeof(host), cudaMemcpyDeviceToHost));
    return host;
}

std::vector<std::int32_t> read_table(const ninfer::Tensor& table) {
    std::vector<std::int32_t> host(static_cast<std::size_t>(table.ne[0]));
    CUDA_CHECK(cudaMemcpy(host.data(), table.data, host.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    return host;
}

void fill_physical_page(ninfer::Tensor plane, std::int32_t page, unsigned char byte) {
    require(page >= 0 && page < plane.ne[3], "physical page index is invalid");
    CUDA_CHECK(cudaMemset(static_cast<unsigned char*>(plane.data) +
                              static_cast<std::int64_t>(page) * plane.nb[3],
                          byte, plane.nb[3]));
}

void require_physical_page(ninfer::Tensor plane, std::int32_t page, unsigned char byte,
                           std::string_view message) {
    std::vector<unsigned char> host(plane.nb[3]);
    CUDA_CHECK(cudaMemcpy(host.data(),
                          static_cast<unsigned char*>(plane.data) +
                              static_cast<std::int64_t>(page) * plane.nb[3],
                          host.size(), cudaMemcpyDeviceToHost));
    for (unsigned char value : host) {
        if (value != byte) { fail(message); }
    }
}

KvGroupSpec tiny_group(std::uint32_t id, KvGroupRetention retention,
                       std::uint32_t physical_pages) {
    return {
        .group_id = id,
        .retention = retention,
        .layer_count = id == 10 ? 50U : 10U,
        .maximum_context = 32768,
        .window_tokens = retention == KvGroupRetention::SlidingWindow ? 1024U : 0U,
        .maximum_transaction_tokens = 2048,
        .table_rows = 2,
        .physical_page_groups = physical_pages,
        .geometry =
            {
                .page_tokens = 64,
                .device_plane_order = ninfer::PagedKVPlaneOrder::PageMajor,
                .planes = {{ninfer::DType::U8, 4, 1, 256}},
            },
    };
}

void commit_to(HeterogeneousKVCache& cache, std::int32_t row, std::uint32_t target) {
    const std::uint32_t first = cache.frontier(row);
    require(target > first, "test append target must advance");
    auto transaction = cache.begin_transaction(row, first, target - first);
    transaction.commit();
}

void exercise_target_plan() {
    using namespace ninfer::targets::gemma4_31b_it;
    const auto specs = make_text_kv_group_specs({
        .maximum_context = 32768,
        .maximum_transaction_tokens = 2048,
        .global_resident_token_capacity = 32768,
        .table_rows = 2,
    });
    require(specs[0].group_id == kSlidingKvGroup &&
                specs[0].retention == KvGroupRetention::SlidingWindow &&
                specs[0].layer_count == 50 && specs[0].geometry.planes.size() == 200 &&
                specs[0].physical_page_groups == 67,
            "Gemma sliding group plan is incorrect");
    require(specs[1].group_id == kGlobalKvGroup &&
                specs[1].retention == KvGroupRetention::FullHistory &&
                specs[1].layer_count == 10 && specs[1].geometry.planes.size() == 40 &&
                specs[1].physical_page_groups == 515,
            "Gemma global group plan is incorrect");
    require(specs[0].geometry.planes[0].dtype == ninfer::DType::U8 &&
                specs[0].geometry.planes[1].dtype == ninfer::DType::U8 &&
                specs[0].geometry.planes[2].dtype == ninfer::DType::FP16 &&
                specs[0].geometry.planes[0].leading_extent == 128 &&
                specs[0].geometry.planes[2].leading_extent == 4 &&
                specs[0].geometry.planes[0].head_extent == 16,
            "Gemma sliding E8 geometry is incorrect");
    require(specs[1].geometry.planes[0].dtype == ninfer::DType::U8 &&
                specs[1].geometry.planes[2].dtype == ninfer::DType::FP16 &&
                specs[1].geometry.planes[0].leading_extent == 256 &&
                specs[1].geometry.planes[2].leading_extent == 8 &&
                specs[1].geometry.planes[0].head_extent == 4,
            "Gemma global E8 geometry is incorrect");
    require(text_kv_layer_address(0) == TextKvLayerAddress{kSlidingKvGroup, 0} &&
                text_kv_layer_address(5) == TextKvLayerAddress{kGlobalKvGroup, 0} &&
                text_kv_layer_address(6) == TextKvLayerAddress{kSlidingKvGroup, 5} &&
                text_kv_layer_address(59) == TextKvLayerAddress{kGlobalKvGroup, 9},
            "Gemma model-layer to group-layer mapping is incorrect");
    require_throws<std::out_of_range>([] { (void)text_kv_layer_address(60); },
                                      "Gemma accepted an out-of-range layer");

    ninfer::LayoutBuilder builder;
    const HeterogeneousKVCacheLayout layout =
        ninfer::plan_heterogeneous_kv_cache(builder, specs);
    require(layout.groups.size() == 2 && layout.groups[0].table_page_capacity == 17 &&
                layout.groups[1].table_page_capacity == 512,
            "Gemma heterogeneous table planning is incorrect");

    const auto full_specs = make_text_kv_group_specs({
        .maximum_context = 262144,
        .maximum_transaction_tokens = 64,
        .global_resident_token_capacity = 262144,
        .table_rows = 1,
    });
    ninfer::LayoutBuilder full_builder;
    const HeterogeneousKVCacheLayout full_layout =
        ninfer::plan_heterogeneous_kv_cache(full_builder, full_specs);
    require(full_layout.groups[0].table_page_capacity == 17 &&
                full_layout.groups[0].spec.physical_page_groups == 19 &&
                full_layout.groups[1].table_page_capacity == 4096 &&
                full_layout.groups[1].spec.physical_page_groups == 4098 &&
                full_layout.payload_bytes() == 5971640320ULL &&
                full_layout.metadata_bytes() == 32920,
            "Gemma 262K KV layout or memory contract changed");

    const std::vector<std::byte> encoded = ninfer::encode_kv_group_descriptors(specs);
    const auto decoded = ninfer::decode_kv_group_descriptors(encoded);
    require(decoded.size() == specs.size(), "KV descriptor group count changed");
    for (std::size_t index = 0; index < specs.size(); ++index) {
        require(decoded[index] == ninfer::kv_group_descriptor(specs[index]),
                "KV descriptor round trip changed a group");
    }
    std::vector<std::byte> trailing = encoded;
    trailing.push_back(std::byte{0});
    require_throws<std::invalid_argument>(
        [&] { (void)ninfer::decode_kv_group_descriptors(trailing); },
        "KV descriptor decoder accepted trailing bytes");
}

void exercise_device_transactions() {
    const std::array specs{tiny_group(10, KvGroupRetention::SlidingWindow, 128),
                           tiny_group(20, KvGroupRetention::FullHistory, 700)};
    ninfer::LayoutBuilder builder;
    const HeterogeneousKVCacheLayout layout =
        ninfer::plan_heterogeneous_kv_cache(builder, specs);
    ninfer::DeviceBuffer backing(builder.finish(256));
    backing.fill(0);
    HeterogeneousKVCache cache({backing.p, backing.bytes}, layout);
    cache.activate(0);

    const auto committed_local = cache.execution_view(0, 10);
    const auto committed_global = cache.execution_view(0, 20);
    const auto batch_local = cache.batch_execution_view(10);
    const auto staging_batch_local = cache.staging_batch_execution_view(10);
    require(committed_local.block_table.data == batch_local.block_tables.slice(1, 0, 1).data,
            "single-row and batch local tables do not share stable storage");
    require(staging_batch_local.block_tables.data != batch_local.block_tables.data &&
                staging_batch_local.block_tables.ne[1] == batch_local.block_tables.ne[1],
            "staging batch table is not distinct and shape-compatible");
    require(cache.table_page_capacity(10) == 17 && cache.table_page_capacity(20) == 512,
            "runtime table capacities are incorrect");

    commit_to(cache, 0, 10);
    std::int32_t committed_page = read_table(committed_local.block_table)[0];
    fill_physical_page(cache.plane(10, 0), committed_page, 0x5a);
    {
        auto transaction = cache.begin_transaction(0, 10, 1);
        require_throws<std::logic_error>(
            [&] { (void)cache.begin_transaction(0, 10, 1); },
            "row admitted overlapping heterogeneous transactions");
        const auto staged = transaction.execution_view(10);
        const std::int32_t staged_page = read_table(staged.block_table)[0];
        require(staged.block_table.data != committed_local.block_table.data,
                "transaction did not use a distinct stable table");
        require(staged_page != committed_page, "partial tail was not copy-on-write");
        require_physical_page(cache.plane(10, 0), staged_page, 0x5a,
                              "copy-on-write did not preserve tail payload");
        fill_physical_page(transaction.plane(10, 0), staged_page, 0xa5);
        transaction.rollback();
    }
    require(cache.frontier(0) == 10 && read_table(committed_local.block_table)[0] == committed_page,
            "rollback changed the committed frontier or page table");
    require_physical_page(cache.plane(10, 0), committed_page, 0x5a,
                          "rollback changed committed tail payload");

    {
        auto transaction = cache.begin_transaction(0, 10, 2);
        require_throws<std::invalid_argument>(
            [&] { transaction.commit_prefix(0); },
            "transaction accepted an empty committed prefix");
        require_throws<std::invalid_argument>(
            [&] { transaction.commit_prefix(3); },
            "transaction accepted a prefix beyond its target");
        const std::int32_t staged_page =
            read_table(transaction.execution_view(10).block_table)[0];
        fill_physical_page(transaction.plane(10, 0), staged_page, 0x6b);
        transaction.commit_prefix(1);
        require(cache.frontier(0) == 11 &&
                    read_table(committed_local.block_table)[0] == staged_page,
                "same-page prefix commit published the wrong frontier or table");
        require_physical_page(cache.plane(10, 0), staged_page, 0x6b,
                              "same-page prefix commit lost retained payload");
    }

    commit_to(cache, 0, 63);
    {
        const std::vector<std::int32_t> before = read_table(committed_local.block_table);
        auto transaction = cache.begin_transaction(0, 63, 2);
        const std::vector<std::int32_t> staged =
            read_table(transaction.execution_view(10).block_table);
        require(staged[0] >= 0 && staged[1] >= 0 && staged[0] != staged[1],
                "cross-page transaction did not reserve both suffix pages");
        fill_physical_page(transaction.plane(10, 0), staged[0], 0x7c);
        transaction.commit_prefix(1);
        const std::vector<std::int32_t> committed = read_table(committed_local.block_table);
        require(cache.frontier(0) == 64 && committed[0] == staged[0] &&
                    committed[1] == before[1] && cache.resident_pages(0, 10) == 1 &&
                    cache.resident_pages(0, 20) == 1,
                "cross-page prefix commit retained a rejected suffix page");
        require_physical_page(cache.plane(10, 0), staged[0], 0x7c,
                              "cross-page prefix commit lost accepted payload");
    }

    commit_to(cache, 0, 1023);
    require(cache.resident_pages(0, 10) == 16 && cache.resident_pages(0, 20) == 16,
            "1023-token residency is incorrect");
    commit_to(cache, 0, 1024);
    require(cache.resident_pages(0, 10) == 16 && cache.resident_pages(0, 20) == 16,
            "1024-token residency is incorrect");
    commit_to(cache, 0, 1025);
    require(cache.resident_pages(0, 10) == 17 && cache.resident_pages(0, 20) == 17 &&
                cache.visible_begin(0, 10) == 1 && cache.visible_begin(0, 20) == 0,
            "1025-token sliding/global boundary is incorrect");

    const std::uint32_t rollback_frontier = cache.frontier(0);
    const std::vector<std::int32_t> rollback_table = read_table(committed_local.block_table);
    {
        auto transaction = cache.begin_transaction(0, rollback_frontier, 2048);
        require(transaction.target_frontier() == rollback_frontier + 2048,
                "transaction target frontier is incorrect");
    }
    require(cache.frontier(0) == rollback_frontier &&
                read_table(committed_local.block_table) == rollback_table,
            "destructor rollback across a wrap changed committed state");

    for (std::uint32_t target : {3073U, 5121U, 7169U, 8193U}) { commit_to(cache, 0, target); }
    require(cache.resident_pages(0, 10) == 17 && cache.resident_pages(0, 20) == 129 &&
                cache.visible_begin(0, 10) == 7169,
            "multiple wraps did not bound local residency or grow global residency");
    require(cache.execution_view(0, 10).block_table.data == committed_local.block_table.data &&
                cache.execution_view(0, 20).block_table.data == committed_global.block_table.data,
            "committed execution pointers changed across wraps");
    require(read_state(committed_local.state) == std::array<std::int32_t, 2>{7169, 8193} &&
                read_state(committed_global.state) == std::array<std::int32_t, 2>{0, 8193},
            "device group state does not match wrapped frontier");

    const std::int32_t wrapped_page = read_table(committed_local.block_table)[128 % 17];
    fill_physical_page(cache.plane(10, 0), wrapped_page, 0x33);
    {
        auto checkpoint = cache.checkpoint(0);
        commit_to(cache, 0, 10000);
        cache.restore(0, checkpoint);
        require(cache.frontier(0) == 8193 && cache.visible_begin(0, 10) == 7169 &&
                    cache.resident_pages(0, 10) == 17 && cache.resident_pages(0, 20) == 129,
                "wrapped prefix restore changed group state");
        const std::int32_t restored_page = read_table(committed_local.block_table)[128 % 17];
        require_physical_page(cache.plane(10, 0), restored_page, 0x33,
                              "wrapped prefix restore changed page payload");
        commit_to(cache, 0, 8257);
        require(cache.frontier(0) == 8257 && cache.resident_pages(0, 10) == 17 &&
                    cache.resident_pages(0, 20) == 130,
                "append after wrapped restore failed");

        cache.activate(1);
        commit_to(cache, 1, 128);
        require(cache.frontier(0) == 8257 && cache.frontier(1) == 128 &&
                    cache.resident_pages(1, 10) == 2 && cache.resident_pages(1, 20) == 2,
                "lane isolation failed");
        cache.restore(1, checkpoint);
        require(cache.frontier(0) == 8257 && cache.frontier(1) == 8193,
                "prefix restore into another lane changed its source lane");
        commit_to(cache, 1, 8257);
        require(cache.frontier(0) == 8257 && cache.frontier(1) == 8257,
                "append after cross-lane prefix restore failed");

        const HeterogeneousKVHostImage host_image = cache.export_host(0);
        CUDA_CHECK(cudaDeviceSynchronize());
        require(host_image.groups.size() == 2 &&
                    host_image.groups[0].logical_blocks.front() == 113 &&
                    host_image.groups[0].logical_blocks.back() == 129 &&
                    host_image.groups[1].logical_blocks.front() == 0 &&
                    host_image.groups[1].logical_blocks.back() == 129,
                "wrapped host image did not preserve local/global logical page ranges");
        commit_to(cache, 1, 10000);
        cache.import_host(1, host_image);
        CUDA_CHECK(cudaDeviceSynchronize());
        const HeterogeneousKVHostImage restored_host_image = cache.export_host(1);
        CUDA_CHECK(cudaDeviceSynchronize());
        require(restored_host_image == host_image && cache.frontier(1) == 8257 &&
                    cache.visible_begin(1, 10) == 7233,
                "host import did not restore both KV groups byte-exactly");

        HeterogeneousKVHostImage bad_blocks = host_image;
        ++bad_blocks.groups[0].logical_blocks.front();
        require_throws<std::invalid_argument>(
            [&] { cache.import_host(1, bad_blocks); },
            "host import accepted a stale sliding-ring logical base");
        HeterogeneousKVHostImage bad_payload = host_image;
        bad_payload.groups[1].payload.pop_back();
        require_throws<std::invalid_argument>(
            [&] { cache.import_host(1, bad_payload); },
            "host import accepted a truncated global payload");
    }

    while (cache.frontier(0) < 32768) {
        const std::uint32_t next = std::min(cache.frontier(0) + 2048U, 32768U);
        commit_to(cache, 0, next);
    }
    require(cache.frontier(0) == 32768 && cache.visible_begin(0, 10) == 31744 &&
                cache.resident_pages(0, 10) >= 16 && cache.resident_pages(0, 10) <= 17 &&
                cache.resident_pages(0, 20) == 512,
            "32K execution did not preserve bounded local and growing global residency");
}

void exercise_graph_transactions(cudaStream_t stream) {
    const std::array specs{tiny_group(10, KvGroupRetention::SlidingWindow, 128),
                           tiny_group(20, KvGroupRetention::FullHistory, 700)};
    ninfer::LayoutBuilder builder;
    const HeterogeneousKVCacheLayout layout =
        ninfer::plan_heterogeneous_kv_cache(builder, specs);
    ninfer::DeviceBuffer backing(builder.finish(256));
    backing.fill(0);
    HeterogeneousKVCache cache({backing.p, backing.bytes}, layout);
    cache.activate(0, stream);

    constexpr std::size_t local_entries = 17;
    constexpr std::size_t global_entries = 512;
    constexpr std::size_t observed_entries = local_entries + global_entries + 1;
    ninfer::DeviceBuffer parameter(sizeof(std::int32_t));
    ninfer::DeviceBuffer observed(observed_entries * sizeof(std::int32_t));

    ninfer::DecodeGraphDefinition definition;
    {
        auto transaction = cache.begin_transaction(0, 0, 1, stream);
        const auto local = transaction.execution_view(10);
        const auto global = transaction.execution_view(20);
        definition.capture(stream, [&] {
            CUDA_CHECK(cudaMemcpyAsync(observed.p, local.block_table.data,
                                       local_entries * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<std::int32_t*>(observed.p) + local_entries,
                global.block_table.data, global_entries * sizeof(std::int32_t),
                cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<std::int32_t*>(observed.p) + local_entries + global_entries,
                parameter.p, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, stream));
        });
        transaction.rollback();
    }
    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(definition);
    executable.upload(stream);

    const auto replay = [&](std::uint32_t first, bool commit) {
        auto transaction = cache.begin_transaction(0, first, 1, stream);
        const auto staged_local = transaction.execution_view(10);
        const auto staged_global = transaction.execution_view(20);
        const std::vector<std::int32_t> expected_local = read_table(staged_local.block_table);
        const std::vector<std::int32_t> expected_global = read_table(staged_global.block_table);
        const auto committed_state = read_state(staged_local.state);
        require(committed_state[1] == static_cast<std::int32_t>(first),
                "graph transaction exposed a staging target as committed state");
        const std::int32_t logical_position = static_cast<std::int32_t>(first);
        CUDA_CHECK(cudaMemcpyAsync(parameter.p, &logical_position, sizeof(logical_position),
                                   cudaMemcpyHostToDevice, stream));
        executable.launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::int32_t> host(observed_entries);
        observed.copy_to_host(host.data(), observed.bytes);
        require(std::equal(expected_local.begin(), expected_local.end(), host.begin()),
                "graph replay observed a stale local staging table");
        require(std::equal(expected_global.begin(), expected_global.end(),
                           host.begin() + static_cast<std::ptrdiff_t>(local_entries)),
                "graph replay observed a stale global staging table");
        require(host.back() == logical_position,
                "graph replay observed a stale explicit logical position");
        if (commit) {
            transaction.commit(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            transaction.rollback();
        }
    };

    replay(0, true);
    commit_to(cache, 0, 1088);
    auto wrapped_prefix = cache.checkpoint(0);
    replay(1088, false);
    require(cache.frontier(0) == 1088,
            "graph transaction rollback changed the wrapped frontier");
    replay(1088, true);
    require(cache.frontier(0) == 1089 && cache.visible_begin(0, 10) == 65,
            "graph transaction commit failed across local ring wrap");
    commit_to(cache, 0, 1200);
    cache.restore(0, wrapped_prefix, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    replay(1088, true);
    require(cache.frontier(0) == 1089,
            "graph replay failed after restoring a wrapped prefix");
    while (cache.frontier(0) < 8193) {
        commit_to(cache, 0, std::min(cache.frontier(0) + 2048U, 8193U));
    }
    replay(8193, true);
    require(cache.frontier(0) == 8194 && cache.visible_begin(0, 20) == 0,
            "graph replay failed at deep global context");
}

} // namespace

int main() {
    exercise_target_plan();

    int device_count = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || (count_error == cudaSuccess && device_count == 0)) {
        std::cout << "SKIP: target plan passed; no usable CUDA device\n";
        return 77;
    }
    if (count_error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_error) << '\n';
        return 1;
    }
    try {
        ninfer::DeviceContext context(0);
        exercise_device_transactions();
        exercise_graph_transactions(context.stream);
        std::cout << "heterogeneous KV cache checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "heterogeneous KV cache test failed: " << error.what() << '\n';
        return 1;
    }
}
