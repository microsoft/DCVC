// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cutlass/cutlass.h>

#include "../common_cu.h"

struct Sm75 {
    using SmArch = cutlass::arch::Sm75;
    using ShapeThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
    using ShapeWarp = cutlass::gemm::GemmShape<64, 64, 32>;
    using ShapeOp = cutlass::gemm::GemmShape<16, 8, 8>;
    static constexpr int Stages = 2;
};

template <typename Func, bool Cutlass2Only>
__forceinline__ auto launch_cutlass_switcher(const int sm, Func func, const int hint)
{
    assert(sm >= 80);
    if constexpr (!Cutlass2Only) {
        switch (hint) {
#if CURRENT_DEVICE_SM == 90
        case 30:
            return func(Hint30{});
        case 31:
            return func(Hint31{});
        case 32:
            return func(Hint32{});
        case 33:
            return func(Hint33{});
        case 34:
            return func(Hint34{});
        case 35:
            return func(Hint35{});
        case 36:
            return func(Hint36{});
        case 37:
            return func(Hint37{});
#elif CURRENT_DEVICE_SM == 100
        case 38:
            return func(Hint38{});
        case 39:
            return func(Hint39{});
        case 40:
            return func(Hint40{});
        case 41:
            return func(Hint41{});
        case 42:
            return func(Hint42{});
        case 43:
            return func(Hint43{});
        case 44:
            return func(Hint44{});
        case 45:
            return func(Hint45{});
        case 46:
            return func(Hint46{});
        case 47:
            return func(Hint47{});
        case 48:
            return func(Hint48{});
        case 49:
            return func(Hint49{});
        case 50:
            return func(Hint50{});
        case 51:
            return func(Hint51{});
        case 52:
            return func(Hint52{});
        case 53:
            return func(Hint53{});
        case 54:
            return func(Hint54{});
        case 55:
            return func(Hint55{});
        case 56:
            return func(Hint56{});
        case 57:
            return func(Hint57{});
        case 58:
            return func(Hint58{});
        case 59:
            return func(Hint59{});
        case 60:
            return func(Hint60{});
        case 61:
            return func(Hint61{});
        case 62:
            return func(Hint62{});
        case 63:
            return func(Hint63{});
        case 64:
            return func(Hint64{});
        case 65:
            return func(Hint65{});
        case 66:
            return func(Hint66{});
        case 67:
            return func(Hint67{});
        case 68:
            return func(Hint68{});
        case 69:
            return func(Hint69{});
        case 70:
            return func(Hint70{});
        case 71:
            return func(Hint71{});
        case 72:
            return func(Hint72{});
        case 73:
            return func(Hint73{});
        case 74:
            return func(Hint74{});
        case 75:
            return func(Hint75{});
        case 76:
            return func(Hint76{});
        case 77:
            return func(Hint77{});
        case 78:
            return func(Hint78{});
        case 79:
            return func(Hint79{});
        case 80:
            return func(Hint80{});
        case 81:
            return func(Hint81{});
        case 82:
            return func(Hint82{});
        case 83:
            return func(Hint83{});
        case 84:
            return func(Hint84{});
        case 85:
            return func(Hint85{});
        case 86:
            return func(Hint86{});
        case 87:
            return func(Hint87{});
        case 88:
            return func(Hint88{});
        case 89:
            return func(Hint89{});
        case 90:
            return func(Hint90{});
        case 91:
            return func(Hint91{});
        case 92:
            return func(Hint92{});
        case 93:
            return func(Hint93{});
        case 94:
            return func(Hint94{});
        case 95:
            return func(Hint95{});
#endif
        }
    }
    switch (hint) {
#if CURRENT_DEVICE_SM >= 80
    case 0:
        return func(Hint0{});
    case 1:
        return func(Hint1{});
    case 2:
        return func(Hint2{});
    case 3:
        return func(Hint3{});
    case 4:
        return func(Hint4{});
    case 5:
        return func(Hint5{});
    case 6:
        return func(Hint6{});
    case 7:
        return func(Hint7{});
    case 8:
        return func(Hint8{});
    case 9:
        return func(Hint9{});
    case 10:
        return func(Hint10{});
    case 11:
        return func(Hint11{});
    case 12:
        return func(Hint12{});
    case 13:
        return func(Hint13{});
    case 14:
        return func(Hint14{});
    case 15:
        return func(Hint15{});
    case 16:
        return func(Hint16{});
    case 17:
        return func(Hint17{});
    case 18:
        return func(Hint18{});
    case 19:
        return func(Hint19{});
    case 20:
        return func(Hint20{});
    case 21:
        return func(Hint21{});
    case 22:
        return func(Hint22{});
    case 23:
        return func(Hint23{});
    case 24:
        return func(Hint24{});
    case 25:
        return func(Hint25{});
    case 26:
        return func(Hint26{});
    case 27:
        return func(Hint27{});
    case 28:
        return func(Hint28{});
    case 29:
        return func(Hint29{});
#endif
    default:
        using Type = decltype(func(Hint0{}));
        static_assert(std::is_same_v<Type, at::Tensor>);
        return at::Tensor();
    }
}

template <typename Func, bool Cutlass2Only = false>
__forceinline__ auto
launch_cutlass_helper(const int sm, Func func, std::unordered_map<uint64_t, int>& map,
                      const std::string& map_name, const int C2, const int C1, const int H,
                      const int W, const int kernel = 0, const int stride = 0)
{
    if (sm == 75) {
        return func(Sm75{});
    }
    const uint64_t key = conv_key(sm, W, H, C2, C1, stride, kernel);
    int best_hint = -1;
    auto it = map.find(key);
    if (it == map.end()) {
        return launch_cutlass_switcher<Func, Cutlass2Only>(sm, func, 14);
    } else {
        best_hint = it->second;
    }
    return launch_cutlass_switcher<Func, Cutlass2Only>(sm, func, best_hint);
}
