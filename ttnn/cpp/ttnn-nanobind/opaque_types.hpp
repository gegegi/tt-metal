// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>

#include <tt-metalium/base_types.hpp>

// Containers that the ttnn extension binds as opaque Python types (nb::bind_vector) instead of
// converting them to Python lists. Every translation unit that converts one of these types must see
// the same declaration: a type_caster specialization that only some translation units see violates
// the one-definition rule, and unity builds can put both kinds of translation unit into one file.
NB_MAKE_OPAQUE(std::vector<tt::tt_metal::UnpackToDestMode>);
NB_MAKE_OPAQUE(std::vector<uint32_t>);
