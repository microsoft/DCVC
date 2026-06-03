// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cutlass/array.h>
#include <cutlass/epilogue/thread/detail.hpp>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>
#include <type_traits>

#include "../common_cu.h"
#include "../def_cutlass.h"

template <int N>
__forceinline__ __device__ cutlass::Array<cutlass::int4b_t, N>
chunk_add(const cutlass::Array<float, N>& intermediate)
{
    cutlass::Array<cutlass::half_t, N / 4> y;
    cutlass::Array<cutlass::int4b_t, N> z;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
        y[i] = static_cast<cutlass::half_t>(intermediate[i * 4 + 0] + intermediate[i * 4 + 1]
                                            + intermediate[i * 4 + 2] + intermediate[i * 4 + 3]);
    }

    memcpy(z.data(), y.data(), sizeof(y));

    return z;
}

// 0 or 1 additional input. If WithQuant, the addtional input is quant, otherwise, it is shortcut.
template <int NumAdditonalInputs, bool WSiLU, bool WithQuant, bool ChunkAdd, typename ElementC_,
          typename ElementAccumulator_, typename ElementCompute_, int ElementsPerAccess>
class LinearCombination {
public:
    static bool const kIsSingleSource = true;

    using ElementOutput = ElementC_;
    using ElementC = ElementC_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    using ElementVector = ElementC_;
    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kCount = kElementsPerAccess;

    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
    using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
    using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>;
    using FragmentOutput = cutlass::Array<ElementOutput, kElementsPerAccess>;

    using ElementZ = ElementC_;
    using ElementT = cutlass::int4b_t;
    using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>;
    using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;

    static bool const kIsHeavy = WSiLU;
    static bool const kStoreZ = !ChunkAdd;
    static bool const kStoreT = ChunkAdd;

    /// Host-constructable parameters structure
    struct Params {

        CUTLASS_HOST_DEVICE
        Params() {}
    };

public:
    /// Constructor from Params
    CUTLASS_HOST_DEVICE
    LinearCombination(Params const&) {}

    /// The "source" tensor corresponds to the residual input
    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return NumAdditonalInputs > 0; }

    CUTLASS_HOST_DEVICE
    void operator()(FragmentZ& frag_Z, FragmentT& frag_T, FragmentAccumulator const& AB,
                    FragmentC const& residual, FragmentCompute const& bias) const
    {
        FragmentCompute tmp_Accum =
            cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
        FragmentCompute tmp_residual =
            cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(residual);

        cutlass::plus<FragmentCompute> plus_op;
        FragmentCompute intermediate = plus_op(tmp_Accum, bias);

        if constexpr (WSiLU) {
            WSiLUOp<FragmentCompute> wsilu_op;
            intermediate = wsilu_op(intermediate);
        }

        if constexpr (WithQuant) {
            cutlass::multiplies<FragmentCompute> multiply_op;
            intermediate = multiply_op(intermediate, tmp_residual);
        } else {
            intermediate = plus_op(intermediate, tmp_residual);
        }

        if constexpr (ChunkAdd) {
            frag_T = chunk_add<kElementsPerAccess>(intermediate);
        } else {
            cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
            frag_Z = convert_z(intermediate);
        }
    }

    CUTLASS_HOST_DEVICE
    void operator()(FragmentZ& frag_Z, FragmentT& frag_T, FragmentAccumulator const& AB,
                    FragmentCompute const& bias) const
    {
        FragmentCompute tmp_Accum =
            cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);

        cutlass::plus<FragmentCompute> plus_op;
        FragmentCompute intermediate = plus_op(tmp_Accum, bias);

        if constexpr (WSiLU) {
            WSiLUOp<FragmentCompute> wsilu_op;
            intermediate = wsilu_op(intermediate);
        }

        if constexpr (ChunkAdd) {
            frag_T = chunk_add<kElementsPerAccess>(intermediate);
        } else {
            cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
            frag_Z = convert_z(intermediate);
        }
    }

    /// Functionally required for serial reduction in the epilogue
    /// IMPORTANT: Split-k is supported only when ActivationOp is Identity.
    CUTLASS_HOST_DEVICE
    void set_k_partition(int, int) {}
};

// 2 additional inputs.
// If WithQuant, the second addtional input is quant, otherwise, it is shortcut.
// The first additonal input is always shortcut.
template <bool WSiLU, bool WithQuant, bool ChunkAdd, typename ElementC_,
          typename ElementAccumulator_, typename ElementCompute_, int ElementsPerAccess>
class LinearCombination<2, WSiLU, WithQuant, ChunkAdd, ElementC_, ElementAccumulator_,
                        ElementCompute_, ElementsPerAccess> {
public:
    static bool const kIsSingleSource = false;

    using ElementOutput = ElementC_;
    using ElementC = ElementC_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    using ElementVector = ElementC_;
    static int const kElementsPerAccess = ElementsPerAccess;
    static int const kCount = kElementsPerAccess;

    using FragmentAccumulator = cutlass::Array<ElementAccumulator, ElementsPerAccess>;
    using FragmentCompute = cutlass::Array<ElementCompute, ElementsPerAccess>;
    using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>;
    using FragmentOutput = cutlass::Array<ElementOutput, kElementsPerAccess>;

    using ElementZ = ElementC_;
    using ElementT = cutlass::int4b_t;
    using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>;
    using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;

    static bool const kIsHeavy = WSiLU;
    static bool const kStoreZ = !ChunkAdd;
    static bool const kStoreT = ChunkAdd;

    /// Host-constructable parameters structure
    struct Params {

        CUTLASS_HOST_DEVICE
        Params() {}
    };

public:
    /// Constructor from Params
    CUTLASS_HOST_DEVICE
    LinearCombination(Params const&) {}

    /// The "source" tensor corresponds to the residual input
    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return true; }

    CUTLASS_HOST_DEVICE
    void operator()(FragmentZ& frag_Z, FragmentT& frag_T, FragmentAccumulator const& AB,
                    FragmentC const& residual1, FragmentC const& residual2,
                    FragmentCompute const& bias) const
    {
        FragmentCompute tmp_Accum =
            cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
        FragmentCompute tmp_residual1 =
            cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(residual1);
        FragmentCompute tmp_residual2 =
            cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(residual2);

        cutlass::plus<FragmentCompute> plus_op;
        FragmentCompute intermediate = plus_op(tmp_Accum, bias);

        if constexpr (WSiLU) {
            WSiLUOp<FragmentCompute> wsilu_op;
            intermediate = wsilu_op(intermediate);
        }

        intermediate = plus_op(intermediate, tmp_residual1);

        if constexpr (WithQuant) {
            cutlass::multiplies<FragmentCompute> multiply_op;
            intermediate = multiply_op(intermediate, tmp_residual2);
        } else {
            intermediate = plus_op(intermediate, tmp_residual2);
        }

        if constexpr (ChunkAdd) {
            frag_T = chunk_add<kElementsPerAccess>(intermediate);
        } else {
            cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
            frag_Z = convert_z(intermediate);
        }
    }

    /// Should never be called
    CUTLASS_HOST_DEVICE
    void operator()(FragmentZ&, FragmentT&, FragmentAccumulator const&, FragmentCompute const&) const
    {
    }

    /// Functionally required for serial reduction in the epilogue
    /// IMPORTANT: Split-k is supported only when ActivationOp is Identity.
    CUTLASS_HOST_DEVICE
    void set_k_partition(int, int) {}
};
