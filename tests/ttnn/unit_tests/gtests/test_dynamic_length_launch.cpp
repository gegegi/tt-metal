// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <exception>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/dynamic_length.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::device_operation::test {
namespace {

using dynamic_length::DynamicAxis;

// Sized by the input's row count and, like the ops this guards, leaves the valid length out of its program hash, so
// every length below the capacity shares one cached program.
struct LengthOp {
    struct operation_attributes_t {
        std::size_t length = 0;
    };
    struct tensor_args_t {
        Tensor input;
        std::optional<Tensor> length_tensor;
    };
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct Factory {
        struct shared_variables_t {};
        using cached_program_t = CachedProgram<shared_variables_t>;

        static cached_program_t create(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
            tt::tt_metal::Program program;
            tt::tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
                CoreCoord{0, 0},
                tt::tt_metal::DataMovementConfig{});
            return cached_program_t(std::move(program), shared_variables_t{});
        }

        static void override_runtime_arguments(
            cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {}
    };
    using program_factory_t = std::variant<Factory>;

    static inline int outputs_created = 0;

    static std::vector<DynamicAxis> dynamic_axes(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        return {DynamicAxis{
            .name = "length",
            .capacity = tensor_args.input.logical_shape()[-2],
            .value = attrs.length,
            .tensor = tensor_args.length_tensor ? &*tensor_args.length_tensor : nullptr,
        }};
    }

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.input.tensor_spec();
    }

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        ++outputs_created;
        return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
    }

    static dynamic_length::DynamicKey dynamic_key(const operation_attributes_t& attrs, const tensor_args_t&) {
        dynamic_length::DynamicKey key;
        key.value(attrs.length);
        return key;
    }
};

// LengthOp's program keyed without the input's rows, so every height shares it.
struct RowsOp {
    using operation_attributes_t = LengthOp::operation_attributes_t;
    using tensor_args_t = LengthOp::tensor_args_t;
    using spec_return_value_t = LengthOp::spec_return_value_t;
    using tensor_return_value_t = LengthOp::tensor_return_value_t;
    using program_factory_t = LengthOp::program_factory_t;

    static dynamic_length::DynamicKey dynamic_key(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        dynamic_length::DynamicKey key;
        key.dim(tensor_args.input, -2);
        return key;
    }

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t& tensor_args) {
        return tensor_args.input.tensor_spec();
    }

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        return create_device_tensor(compute_output_specs(attrs, tensor_args), tensor_args.input.device());
    }
};

struct NoAxesOp {
    using operation_attributes_t = LengthOp::operation_attributes_t;
    using tensor_args_t = LengthOp::tensor_args_t;
};

static_assert(HasDynamicAxes<LengthOp>);
static_assert(!HasDynamicAxes<NoAxesOp>);
static_assert(HasDynamicKey<RowsOp>);
static_assert(HasDynamicKey<LengthOp>);
static_assert(!HasDynamicKey<NoAxesOp>);

class DynamicLengthLaunchTest : public TTNNFixtureWithDevice {
protected:
    static constexpr std::size_t kCapacity = 64;

    void SetUp() override {
        TTNNFixtureWithDevice::SetUp();
        LengthOp::outputs_created = 0;
    }

    Tensor make_input(uint32_t rows = kCapacity, uint32_t cols = 32) const {
        return create_device_tensor(
            tt::tt_metal::TensorSpec(
                ttnn::Shape{1, 1, rows, cols},
                tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), MemoryConfig{})),
            device_);
    }

    Tensor make_length_tensor(uint32_t volume) const {
        return create_device_tensor(
            tt::tt_metal::TensorSpec(
                ttnn::Shape{volume},
                tt::tt_metal::TensorLayout(
                    DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig{})),
            device_);
    }
};

TEST_F(DynamicLengthLaunchTest, CacheHitIsHeldToCapacity) {
    const auto input = make_input();
    launch<LengthOp>({.length = 16}, {.input = input});
    EXPECT_NO_THROW(launch<LengthOp>({.length = kCapacity}, {.input = input}));
    EXPECT_EQ(device_->get_program_cache().num_entries(), 1);

    const int outputs_before = LengthOp::outputs_created;
    EXPECT_THROW(launch<LengthOp>({.length = kCapacity + 1}, {.input = input}), std::exception);
    EXPECT_EQ(LengthOp::outputs_created, outputs_before);
}

TEST_F(DynamicLengthLaunchTest, PrepareIsHeldToCapacity) {
    const auto input = make_input();
    EXPECT_NO_THROW(prepare<LengthOp>({.length = kCapacity}, {.input = input}));
    const int outputs_before = LengthOp::outputs_created;
    EXPECT_THROW(prepare<LengthOp>({.length = kCapacity + 1}, {.input = input}), std::exception);
    EXPECT_EQ(LengthOp::outputs_created, outputs_before);
}

TEST_F(DynamicLengthLaunchTest, AcceptsDeviceLengthTensor) {
    const auto input = make_input();
    EXPECT_NO_THROW(launch<LengthOp>({}, {.input = input, .length_tensor = make_length_tensor(1)}));
}

TEST_F(DynamicLengthLaunchTest, RejectsMultiValuedLengthTensor) {
    const auto input = make_input();
    EXPECT_THROW(launch<LengthOp>({}, {.input = input, .length_tensor = make_length_tensor(2)}), std::exception);
    EXPECT_EQ(LengthOp::outputs_created, 0);
}

TEST_F(DynamicLengthLaunchTest, DeclaredRowsShareOneProgram) {
    launch<RowsOp>({}, {.input = make_input(32)});
    launch<RowsOp>({}, {.input = make_input(96)});
    EXPECT_EQ(device_->get_program_cache().num_entries(), 1);
    launch<RowsOp>({}, {.input = make_input(32, 64)});
    EXPECT_EQ(device_->get_program_cache().num_entries(), 2);
}

TEST_F(DynamicLengthLaunchTest, RejectsDynamicDimOnShardedTensor) {
    const MemoryConfig height_sharded{
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        tt::tt_metal::ShardSpec(CoreRangeSet(CoreRange({0, 0}, {0, 0})), {64, 32}, ShardOrientation::ROW_MAJOR)};
    const auto input = create_device_tensor(
        tt::tt_metal::TensorSpec(
            ttnn::Shape{1, 1, 64, 32},
            tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), height_sharded)),
        device_);
    EXPECT_THROW(launch<RowsOp>({}, {.input = input}), std::exception);
}

}  // namespace
}  // namespace ttnn::device_operation::test
