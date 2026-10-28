// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_program_spec_device_operation.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/config.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"

namespace ttnn::operations::generic {

namespace m2 = tt::tt_metal::experimental;

namespace {

using SpecAttributes = GenericProgramSpecDeviceOperation::operation_attributes_t;

bool nodes_contain(const m2::Nodes& nodes, const m2::NodeCoord& node) {
    return std::visit(
        ttsl::overloaded{
            [&](const m2::NodeCoord& coord) { return coord == node; },
            [&](const m2::NodeRange& range) { return range.contains(node); },
            [&](const m2::NodeRangeSet& range_set) { return range_set.contains(node); }},
        nodes);
}

// The nodes a kernel runs on: those of every work unit that lists it.
std::unordered_set<m2::NodeCoord> nodes_of_kernel(const m2::ProgramSpec& spec, const m2::KernelSpecName& kernel) {
    std::unordered_set<m2::NodeCoord> nodes;
    const auto add_range = [&nodes](const m2::NodeRange& range) {
        for (const auto& node : range) {
            nodes.insert(node);
        }
    };
    for (const auto& work_unit : spec.work_units) {
        if (std::ranges::find(work_unit.kernels, kernel) == work_unit.kernels.end()) {
            continue;
        }
        std::visit(
            ttsl::overloaded{
                [&](const m2::NodeCoord& node) { nodes.insert(node); },
                [&](const m2::NodeRange& range) { add_range(range); },
                [&](const m2::NodeRangeSet& range_set) {
                    for (const auto& range : range_set.ranges()) {
                        add_range(range);
                    }
                }},
            work_unit.target_nodes);
    }
    return nodes;
}

// A kernel's runtime vararg count on a node: num_runtime_varargs, unless the deprecated per-node counts name the
// node.
uint32_t runtime_vararg_count(const m2::KernelAdvancedOptions& options, const m2::NodeCoord& node) {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    for (const auto& [nodes, count] : options.num_runtime_varargs_per_node) {
        if (nodes_contain(nodes, node)) {
            return count;
        }
    }
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    return options.num_runtime_varargs;
}

bool has_runtime_arguments(const m2::KernelSpec& kernel) {
    const auto& schema = kernel.runtime_arg_schema;
    const auto& options = kernel.advanced_options;
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    const bool has_per_node_varargs = !options.num_runtime_varargs_per_node.empty();
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    return !schema.runtime_arg_names.empty() || !schema.common_runtime_arg_names.empty() ||
           options.num_runtime_varargs > 0 || options.num_common_runtime_varargs > 0 || has_per_node_varargs;
}

template <typename Value>
std::vector<std::string> names_of(const m2::Table<std::string, Value>& values) {
    std::vector<std::string> names;
    names.reserve(values.size());
    for (const auto& [name, value] : values) {
        names.push_back(name);
    }
    return names;
}

std::string join_names(const std::vector<std::string>& names) {
    std::string joined;
    for (const auto& name : names) {
        joined += joined.empty() ? name : ", " + name;
    }
    return joined;
}

template <typename Value>
bool has_exactly(const m2::Table<std::string, Value>& values, const m2::Group<std::string>& names) {
    return values.size() == names.size() &&
           std::ranges::all_of(names, [&values](const std::string& name) { return values.contains(name); });
}

void check_kernel_run_args(
    const m2::ProgramSpec& spec, const m2::KernelSpec& kernel, const m2::KernelRunArgs& run_args, bool check_nodes) {
    const auto& schema = kernel.runtime_arg_schema;
    TT_FATAL(
        has_exactly(run_args.runtime_arg_values, schema.runtime_arg_names),
        "generic_op: kernel '{}' declares the runtime arguments [{}], but its run args give [{}]",
        kernel.unique_id,
        join_names(schema.runtime_arg_names),
        join_names(names_of(run_args.runtime_arg_values)));
    TT_FATAL(
        has_exactly(run_args.common_runtime_arg_values, schema.common_runtime_arg_names),
        "generic_op: kernel '{}' declares the common runtime arguments [{}], but its run args give [{}]",
        kernel.unique_id,
        join_names(schema.common_runtime_arg_names),
        join_names(names_of(run_args.common_runtime_arg_values)));

    // Metal writes varargs by position on a later call without checking their count, so every call checks it.
    const auto& options = kernel.advanced_options;
    const auto& varargs = run_args.advanced_options;
    TT_FATAL(
        varargs.common_runtime_varargs.size() == options.num_common_runtime_varargs,
        "generic_op: kernel '{}' declares {} common runtime varargs, but its run args give {}",
        kernel.unique_id,
        options.num_common_runtime_varargs,
        varargs.common_runtime_varargs.size());
    for (const auto& [node, values] : varargs.runtime_varargs) {
        const uint32_t count = runtime_vararg_count(options, node);
        TT_FATAL(
            values.size() == count,
            "generic_op: kernel '{}' declares {} runtime varargs on node {}, but its run args give {}",
            kernel.unique_id,
            count,
            node.str(),
            values.size());
    }

    if (!check_nodes) {
        return;
    }
    // A later call keeps the previous value of a node it leaves out, and Metal does not report it.
    const auto nodes = nodes_of_kernel(spec, kernel.unique_id);
    for (const auto& [name, values] : run_args.runtime_arg_values) {
        const bool covers_nodes = values.size() == nodes.size() &&
                                  std::ranges::all_of(values, [&nodes](const auto& entry) {
                                      return nodes.contains(entry.first);
                                  });
        TT_FATAL(
            covers_nodes,
            "generic_op: kernel '{}' runs on {} nodes, but runtime argument '{}' does not have exactly one value for each "
            "of them",
            kernel.unique_id,
            nodes.size(),
            name);
    }
    for (const auto& node : nodes) {
        TT_FATAL(
            runtime_vararg_count(options, node) == 0 || varargs.runtime_varargs.contains(node),
            "generic_op: kernel '{}' declares runtime varargs, but its run args give none for node {}",
            kernel.unique_id,
            node.str());
    }
}

void check_tensor_args(
    const m2::ProgramSpec& spec, const m2::ProgramRunArgs& run_args, const std::vector<Tensor>& io_tensors) {
    TT_FATAL(
        run_args.tensor_args.size() == spec.tensor_parameters.size(),
        "generic_op: the ProgramSpec declares {} tensor parameters, but run_args.tensor_args gives {} tensors",
        spec.tensor_parameters.size(),
        run_args.tensor_args.size());
    std::vector<bool> claimed(io_tensors.size(), false);
    for (const auto& parameter : spec.tensor_parameters) {
        const auto tensor_arg = run_args.tensor_args.get(parameter.unique_id);
        TT_FATAL(
            tensor_arg.has_value(),
            "generic_op: tensor parameter '{}' has no tensor in run_args.tensor_args",
            parameter.unique_id);
        // A tensor argument refers to a MeshTensor, which every copy of a ttnn::Tensor shares.
        const auto* mesh_tensor = &m2::mesh_tensor_of(*tensor_arg);
        std::size_t index = 0;
        while (index < io_tensors.size() && (claimed[index] || &io_tensors[index].mesh_tensor() != mesh_tensor)) {
            ++index;
        }
        TT_FATAL(
            index < io_tensors.size(),
            "generic_op: the tensor of tensor parameter '{}' is not in io_tensors. Each tensor parameter needs its "
            "own io_tensors entry, so list a tensor once for each parameter it is bound to",
            parameter.unique_id);
        claimed[index] = true;
        const auto& tensor_spec = mesh_tensor->tensor_spec();
        TT_FATAL(
            m2::tensorspecs_match_with_relaxation(tensor_spec, parameter.spec, parameter.relaxations),
            "generic_op: the tensor of tensor parameter '{}' has the spec {}, which does not match the parameter's "
            "spec {}",
            parameter.unique_id,
            tensor_spec,
            parameter.spec);
    }
}

void check_prefetcher_pipe_args(const m2::ProgramSpec& spec, const m2::ProgramRunArgs& run_args) {
    const auto& parameters = spec.advanced_options.prefetcher_pipe_parameters;
    const auto& pipes = run_args.advanced_options.prefetcher_pipe_args;
    for (const auto& parameter : parameters) {
        TT_FATAL(
            pipes.contains(parameter.unique_id),
            "generic_op: PrefetcherPipeParameter '{}' has no pipe in run_args.advanced_options.prefetcher_pipe_args",
            parameter.unique_id);
    }
    TT_FATAL(
        pipes.size() == parameters.size(),
        "generic_op: run_args.advanced_options.prefetcher_pipe_args has a pipe for a parameter that the ProgramSpec "
        "does not declare");
}

// The checks every call runs before Metal sees the run args. Metal checks them only on the first call, and on later
// calls only with validate_program_args on, and even then it keeps the previous value of anything left out.
void check_program(
    const m2::ProgramSpec& spec,
    const m2::ProgramRunArgs& run_args,
    const std::vector<Tensor>& io_tensors,
    bool check_nodes) {
    std::unordered_set<std::string> kernels_with_run_args;
    for (const auto& kernel_run_args : run_args.kernel_run_args) {
        const auto kernel = std::ranges::find(spec.kernels, kernel_run_args.kernel, &m2::KernelSpec::unique_id);
        TT_FATAL(
            kernel != spec.kernels.end(),
            "generic_op: run_args.kernel_run_args names kernel '{}', which the ProgramSpec does not declare",
            kernel_run_args.kernel);
        TT_FATAL(
            kernels_with_run_args.insert(*kernel_run_args.kernel).second,
            "generic_op: run_args.kernel_run_args lists kernel '{}' more than once",
            kernel_run_args.kernel);
        check_kernel_run_args(spec, *kernel, kernel_run_args, check_nodes);
    }
    for (const auto& kernel : spec.kernels) {
        TT_FATAL(
            kernels_with_run_args.contains(*kernel.unique_id) || !has_runtime_arguments(kernel),
            "generic_op: kernel '{}' takes runtime arguments, but run_args.kernel_run_args has no entry for it",
            kernel.unique_id);
    }
    check_tensor_args(spec, run_args, io_tensors);
    check_prefetcher_pipe_args(spec, run_args);
}

// Only the first call needs these: the ranges and the devices holding the tensors are part of the cache key.
void check_mesh_ranges(const SpecAttributes& attributes, const tensor_args_t& tensor_args) {
    const auto& programs = attributes.programs;
    for (std::size_t i = 0; i < programs.size(); ++i) {
        for (std::size_t j = i + 1; j < programs.size(); ++j) {
            if (programs[i].range.has_value() && programs[j].range.has_value()) {
                TT_FATAL(
                    !programs[i].range->intersects(*programs[j].range),
                    "generic_op: mesh ranges {} and {} overlap",
                    *programs[i].range,
                    *programs[j].range);
            }
        }
    }
    if (ttnn::device_operation::mesh_device_operation_utils::all_tensors_have_uniform_storage(tensor_args)) {
        return;
    }
    const auto tensor_coords = ttnn::device_operation::mesh_device_operation_utils::extract_tensor_coordinates(
        tensor_args, tensor_args.output_tensor.device());
    for (const auto& program : programs) {
        if (!program.range.has_value()) {
            continue;
        }
        for (const auto& coord : *program.range) {
            TT_FATAL(
                std::ranges::find(tensor_coords, coord) != tensor_coords.end(),
                "generic_op: mesh range {} covers device {}, which does not hold every tensor in io_tensors",
                *program.range,
                coord);
        }
    }
}

// ttsl hashes an empty std::optional like a value that hashes to 0, such as DataFormat::Float32 or a config whose
// fields are all 0. So the spec hash also records which optional fields are set.
ttsl::hash::hash_t hash_optional_fields_set(const m2::ProgramSpec& spec) {
    // Tripwires: a new field in one of these types may be another std::optional to record here.
    static_assert(reflect::size<m2::DataflowBufferSpec>() == 7, "Record any new optional field below.");
    static_assert(reflect::size<m2::ScratchpadSpec>() == 5, "Record any new optional field below.");
    static_assert(reflect::size<m2::DataMovementHardwareConfig>() == 2, "Record any new optional field below.");
    static_assert(reflect::size<m2::ComputeHardwareConfig>() == 8, "Record any new optional field below.");
    ttsl::hash::hash_t hash = 0;
    for (const auto& kernel : spec.kernels) {
        std::visit(
            ttsl::overloaded{
                [&hash](const m2::DataMovementHardwareConfig& config) {
                    hash = ttsl::hash::hash_objects(hash, config.config_1xx.has_value(), config.config_2xx.has_value());
                },
                [&hash](const m2::ComputeHardwareConfig& config) {
                    hash = ttsl::hash::hash_objects(
                        hash,
                        config.config_1xx.has_value(),
                        config.config_12x.has_value(),
                        config.config_2xx.has_value());
                }},
            kernel.hw_config);
    }
    for (const auto& dfb : spec.dataflow_buffers) {
        hash = ttsl::hash::hash_objects(
            hash,
            dfb.data_format_metadata.has_value(),
            dfb.tile_format_metadata.has_value(),
            dfb.borrowed_from.has_value());
    }
    for (const auto& scratchpad : spec.scratchpads) {
        hash = ttsl::hash::hash_objects(
            hash, scratchpad.data_format_metadata.has_value(), scratchpad.tile_format_metadata.has_value());
    }
    return hash;
}

// Each kernel's vararg counts. Metal writes varargs by position into a buffer sized from these counts, so a key that
// replaces the spec hash still includes them: a program compiled with fewer varargs is never reused.
ttsl::hash::hash_t hash_vararg_counts(const m2::ProgramSpec& spec) {
    ttsl::hash::hash_t hash = spec.kernels.size();
    for (const auto& kernel : spec.kernels) {
        const auto& options = kernel.advanced_options;
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        hash = ttsl::hash::hash_objects(
            hash,
            kernel.unique_id,
            options.num_runtime_varargs,
            options.num_common_runtime_varargs,
            options.num_runtime_varargs_per_node);
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    }
    return hash;
}

}  // namespace

ttsl::hash::hash_t compute_program_spec_hash(const m2::ProgramSpec& spec) {
    // Tripwires: these three types are hashed field by field below. Every other type goes through ttsl's reflection
    // hash, which picks up a new field by itself.
    static_assert(
        reflect::size<m2::ProgramSpec>() == 8,
        "ProgramSpec gained or lost a field: hash it in compute_program_spec_hash, then update this count.");
    static_assert(
        reflect::size<m2::WorkUnitSpec>() == 3,
        "WorkUnitSpec gained or lost a field: hash it in compute_program_spec_hash, then update this count.");
    static_assert(
        reflect::size<m2::TensorParameter>() == 3,
        "TensorParameter gained or lost a field: hash it in compute_program_spec_hash, then update this count.");

    // ProgramSpec::name and WorkUnitSpec::name are left out: they are for messages only. Lists are hashed with their
    // lengths, so an element cannot move from one list to the next without changing the hash.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    ttsl::hash::hash_t hash = ttsl::hash::hash_objects_with_default_seed(
        spec.kernels, spec.dataflow_buffers, spec.semaphores, spec.scratchpads, spec.advanced_options);
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    // A tensor parameter's spec is hashed under its relaxations, so a relaxed field such as a dynamic shape does not
    // change the hash.
    hash = ttsl::hash::hash_objects(hash, spec.tensor_parameters.size());
    for (const auto& parameter : spec.tensor_parameters) {
        hash = ttsl::hash::hash_objects(
            hash,
            parameter.unique_id,
            parameter.relaxations,
            m2::hash_tensorspec_with_relaxation(parameter.spec, parameter.relaxations));
    }
    hash = ttsl::hash::hash_objects(hash, spec.work_units.size());
    for (const auto& work_unit : spec.work_units) {
        hash = ttsl::hash::hash_objects(hash, work_unit.kernels, work_unit.target_nodes);
    }
    return ttsl::hash::hash_objects(hash, hash_optional_fields_set(spec));
}

GenericProgramSpecDeviceOperation::spec_return_value_t GenericProgramSpecDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();
}

GenericProgramSpecDeviceOperation::tensor_return_value_t GenericProgramSpecDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // The caller preallocates the output.
    return tensor_args.output_tensor;
}

void GenericProgramSpecDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    check_mesh_ranges(operation_attributes, tensor_args);
    // On the first call SetProgramRunArgs checks that every node got its values.
    for (const auto& program : operation_attributes.programs) {
        check_program(*program.spec, *program.run_args, tensor_args.io_tensors, /*check_nodes=*/false);
    }
}

void GenericProgramSpecDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const bool check_nodes = ttnn::CONFIG.get<"validate_program_args">();
    for (const auto& program : operation_attributes.programs) {
        check_program(*program.spec, *program.run_args, tensor_args.io_tensors, check_nodes);
    }
}

ttsl::hash::hash_t GenericProgramSpecDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    ttsl::hash::hash_t hash = ttsl::hash::type_hash<GenericProgramSpecDeviceOperation>;
    for (const auto& program : operation_attributes.programs) {
        const ttsl::hash::hash_t program_hash =
            operation_attributes.custom_program_hash.has_value()
                ? ttsl::hash::hash_objects(*operation_attributes.custom_program_hash, hash_vararg_counts(*program.spec))
                : compute_program_spec_hash(*program.spec);
        hash = ttsl::hash::hash_objects(hash, program.range, program_hash);
    }
    return hash;
}

ttnn::device_operation::MeshWorkloadArtifacts
GenericProgramSpecDeviceOperation::ProgramFactory::create_mesh_workload_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    for (const auto& program : operation_attributes.programs) {
        if (program.range.has_value()) {
            artifacts.programs.push_back({.range = *program.range, .spec = *program.spec, .run_params = *program.run_args});
            continue;
        }
        for (const auto& range : tensor_coords.ranges()) {
            artifacts.programs.push_back({.range = range, .spec = *program.spec, .run_params = *program.run_args});
        }
    }
    return artifacts;
}

tt::tt_metal::experimental::ProgramRunArgs GenericProgramSpecDeviceOperation::ProgramFactory::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/,
    const ttnn::MeshCoordinateRange& range) {
    for (const auto& program : operation_attributes.programs) {
        if (!program.range.has_value() || *program.range == range) {
            return *program.run_args;
        }
    }
    TT_THROW("generic_op: no run args for mesh range {}", range);
}

}  // namespace ttnn::operations::generic

namespace ttnn::prim {

namespace {

using GenericProgramSpecOperation = ttnn::operations::generic::GenericProgramSpecDeviceOperation;

GenericProgramSpecOperation::tensor_args_t make_program_spec_tensor_args(const std::vector<Tensor>& io_tensors) {
    TT_FATAL(!io_tensors.empty(), "generic_op: io_tensors must hold at least the output tensor");
    for (std::size_t i = 0; i < io_tensors.size(); ++i) {
        TT_FATAL(
            io_tensors[i].storage_type() == ttnn::StorageType::DEVICE && io_tensors[i].is_allocated(),
            "generic_op: io_tensors[{}] must be a tensor allocated on a device",
            i);
    }
    // The program runs on one MeshDevice; a tensor on another would be bound by an address that means nothing there.
    for (std::size_t i = 0; i < io_tensors.size(); ++i) {
        TT_FATAL(
            io_tensors[i].device() == io_tensors.back().device(),
            "generic_op: io_tensors[{}] is on a different MeshDevice than the output tensor",
            i);
    }
    return {.io_tensors = io_tensors, .output_tensor = io_tensors.back()};
}

}  // namespace

Tensor generic_op(
    const std::vector<Tensor>& io_tensors, const GenericProgramSpecOperation::operation_attributes_t& operation_attributes) {
    return ttnn::device_operation::launch<GenericProgramSpecOperation>(
        operation_attributes, make_program_spec_tensor_args(io_tensors));
}

void prepare_generic_op(
    const std::vector<Tensor>& io_tensors, const GenericProgramSpecOperation::operation_attributes_t& operation_attributes) {
    ttnn::device_operation::prepare<GenericProgramSpecOperation>(
        operation_attributes, make_program_spec_tensor_args(io_tensors));
}

}  // namespace ttnn::prim
