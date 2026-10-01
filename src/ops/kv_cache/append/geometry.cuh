#pragma once

namespace ninfer::ops {

template <int HeadDimValue, int KVHeadsValue>
struct KVCacheAppendFullGeometry {
    static_assert((HeadDimValue == 256 &&
                   (KVHeadsValue == 16 || KVHeadsValue == 4 || KVHeadsValue == 2)) ||
                  (HeadDimValue == 512 && KVHeadsValue == 4));
    static constexpr int HeadDim = HeadDimValue;
    static constexpr int KVHeads = KVHeadsValue;
    static constexpr int Groups = HeadDim / 64;
};

using KVCacheAppendD256Kv16 = KVCacheAppendFullGeometry<256, 16>;
using KVCacheAppendD256Kv4 = KVCacheAppendFullGeometry<256, 4>;
using KVCacheAppendD256Kv2 = KVCacheAppendFullGeometry<256, 2>;
using KVCacheAppendD512Kv4 = KVCacheAppendFullGeometry<512, 4>;

} // namespace ninfer::ops
