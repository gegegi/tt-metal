// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn-nanobind/nanobind_fwd.hpp"

namespace ttnn::program_spec {
namespace nb = nanobind;

// Python classes for the Metal 2.0 ProgramSpec and ProgramRunArgs types (tt::tt_metal::experimental),
// re-exported under ttnn.experimental.
void py_module_types(nb::module_& mod);

// The attribute in which a Python ProgramRunArgs holds its tensor_args dict. The C++ object refers to the tensors'
// MeshTensors, so the dict keeps them alive.
inline constexpr const char* TENSOR_ARGS_ATTRIBUTE = "_tensor_args";

}  // namespace ttnn::program_spec
