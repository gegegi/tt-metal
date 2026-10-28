.. _generic_op_program_spec:

Running your own kernels with generic_op and a ProgramSpec
##########################################################

``ttnn.experimental.generic_op`` runs a program that you describe yourself: the kernels, the buffers between them,
the tensors they read and write, and the values of each call. It takes the program as a Metal 2.0 ``ProgramSpec``,
everything fixed about the program, and a ``ProgramRunArgs``, the values of one call. Kernels find everything they use
by name.

This form is experimental while its API settles, so everything it adds lives under ``ttnn.experimental``. The
``ProgramDescriptor`` form, ``ttnn.generic_op``, keeps working. The ``ProgramSpec`` form also runs on Quasar.

The Python classes mirror the C++ structs with the same names and fields in
``tt_metal/api/tt-metalium/experimental/metal2_host_api/``. The comments in those headers document each field.

An example
**********

This program computes ``exp(x)`` for every element of a tensor in DRAM. On each core a reader kernel copies tiles
from the input tensor into a buffer, a compute kernel applies ``exp`` to each tile, and a writer kernel copies the
results to the output tensor. The three kernels already exist in TT-NN.

.. code-block:: python

    import torch
    import ttnn

    KERNELS = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels"
    TILE_BYTES = 2 * 32 * 32  # one bfloat16 tile

    device = ttnn.open_device(device_id=0)
    num_tiles = 8
    input_tensor = ttnn.from_torch(
        torch.rand([1, num_tiles, 32, 32]).to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # generic_op does not create its output: preallocate it.
    output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, device)

    grid = device.compute_with_storage_grid_size()
    _, cores, group_1, group_2, tiles_1, tiles_2 = ttnn.split_work_to_cores(grid, num_tiles)

    E = ttnn.experimental
    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_pages", "start_id"])
    spec = E.ProgramSpec(
        name="exp",
        kernels=[
            E.KernelSpec(
                unique_id="reader",
                source=f"{KERNELS}/dataflow/reader_unary_interleaved_start_id_metal2.cpp",
                dfb_bindings=[E.producer_of("in", "in")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="src")],
                runtime_arg_schema=dm_schema,
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
            ),
            E.KernelSpec(
                unique_id="writer",
                source=f"{KERNELS}/dataflow/writer_unary_interleaved_start_id_metal2.cpp",
                dfb_bindings=[E.consumer_of("out", "out")],
                tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="dst")],
                runtime_arg_schema=dm_schema,
                hw_config=E.create_writer_datamovement_config(disable_dfb_implicit_sync_for_all=True),
            ),
            E.KernelSpec(
                unique_id="compute",
                source=f"{KERNELS}/compute/eltwise_sfpu_metal2.cpp",
                compiler_options=E.KernelSpec.CompilerOptions(
                    defines={"SFPU_OP_EXP_INCLUDE": "1", "SFPU_OP_CHAIN_0": "exp_tile_init(); exp_tile(0);"}
                ),
                dfb_bindings=[E.consumer_of("in", "in"), E.producer_of("out", "out")],
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
                hw_config=E.ComputeHardwareConfig(),
            ),
        ],
        dataflow_buffers=[
            E.DataflowBufferSpec(unique_id=name, entry_size=TILE_BYTES, num_entries=2, data_format_metadata=ttnn.bfloat16)
            for name in ("in", "out")
        ],
        tensor_parameters=[
            E.TensorParameter(unique_id="input", spec=input_tensor.spec),
            E.TensorParameter(unique_id="output", spec=output_tensor.spec),
        ],
        work_units=[E.WorkUnitSpec(name="exp", kernels=["reader", "writer", "compute"], target_nodes=cores)],
    )

    # The values of this call, by argument name and then by core. Tensors are bound by name.
    num_pages, start_id, start = {}, {}, 0
    for group, n in ((group_1, tiles_1), (group_2, tiles_2)):
        for core in ttnn.corerange_to_cores(group):
            num_pages[core], start_id[core] = n, start
            start += n
    dm_args = {"num_pages": num_pages, "start_id": start_id}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="compute", runtime_arg_values={"num_tiles": num_pages}),
        ],
        tensor_args={"input": input_tensor, "output": output_tensor},
    )

    output_tensor = E.generic_op([input_tensor, output_tensor], spec, run_args)

How a kernel finds its inputs
*****************************

Everything a kernel uses has a name, which the kernel code reaches through a generated namespace:

.. list-table::
   :header-rows: 1

   * - What a kernel needs
     - Where you set it
     - How the kernel reads it
   * - A value fixed at compile time
     - ``KernelSpec.compile_time_args``
     - ``get_arg(args::name)``
   * - A value per core
     - A name in ``RuntimeArgSchema.runtime_arg_names``, the values in ``KernelRunArgs.runtime_arg_values``
     - ``get_arg(args::name)``
   * - A value shared by all cores
     - A name in ``RuntimeArgSchema.common_runtime_arg_names``, the value in ``KernelRunArgs.common_runtime_arg_values``
     - ``get_arg(args::name)``
   * - A tensor
     - A ``TensorParameter`` and a ``TensorBinding``, the tensor in ``ProgramRunArgs.tensor_args``
     - ``TensorAccessor(tensor::name)``
   * - A buffer between two kernels
     - A ``DataflowBufferSpec``, with a producer and a consumer binding
     - ``DataflowBuffer(dfb::name)``
   * - A semaphore
     - A ``SemaphoreSpec`` and a ``SemaphoreBinding``
     - ``Semaphore(sem::name)``
   * - Private scratch memory
     - A ``ScratchpadSpec`` and a ``ScratchpadBinding``
     - ``Scratchpad<T>(scratch::name)``

A list of values whose length varies, such as a list of core coordinates, has no named form yet. Pass it as varargs:
``KernelAdvancedOptions(num_runtime_varargs=n)`` on the kernel, ``AdvancedKernelRunArgs(runtime_varargs={core: [...]})``
in its run args, and ``get_vararg(i)`` in the kernel code. Python binds only the advanced options that TT-NN's
callers use so far: the vararg counts and values, and ``DFBAdvancedOptions.alias_with``. The others are C++ only.

Tensors
*******

``io_tensors`` lists the tensors the program uses, with the output last; ``generic_op`` returns the last one. Each
tensor parameter takes its own entry, so a tensor bound to two parameters is listed twice. The list may hold only
the output, for a program with no inputs or one that works in place.

``ProgramRunArgs`` keeps its tensors alive. A tensor must be on the device and must match its parameter's
``TensorSpec``, unless the parameter's ``TensorSpecRelaxations`` allow a difference, such as a dynamic shape.

Calls and the program cache
***************************

The first call with a given program compiles its kernels and sets it up on the device. A later call with the same
program only writes the new values: argument values and tensors can change from call to call without a recompile.

The cache key is a hash of everything in the ``ProgramSpec`` that changes the compiled program, combined with the
devices it runs on. ``ProgramSpec.name`` and ``WorkUnitSpec.name`` are left out, and a tensor parameter's spec is
hashed under its relaxations. ``ttnn.experimental.compute_program_spec_hash(spec)`` returns that hash. Pass
``custom_program_hash`` to use your own key instead. The key still includes each kernel's vararg counts, so a call
never writes more varargs than a cached program has room for; keeping keys unique is otherwise your job.

``ttnn.experimental.prepare_generic_op`` takes the same arguments, compiles the program and puts it in the cache
without running it.

What every call checks
**********************

Metal checks a program fully on the first call only, and a later call keeps the previous value of anything it leaves
out. So ``generic_op`` checks every call before Metal sees it:

- each kernel gets exactly the runtime arguments it declares, and the number of varargs it declares
- each tensor parameter gets a tensor from ``io_tensors``, and that tensor matches its parameter

Whether every core got its values is checked on later calls only when ``ttnn.CONFIG.validate_program_args`` is on.
With it off, a core that a later call leaves out keeps its previous values.

DFB size overrides in ``ProgramRunArgs.dfb_run_overrides`` also carry over: an override stays in effect on later calls
until you change it.

Running on a mesh
*****************

With a ``ProgramSpec``, ``generic_op`` runs the program on every device that holds the tensors. To run different
programs on different parts of a mesh, give a program per mesh coordinate range:

.. code-block:: python

    whole_mesh = ttnn.MeshCoordinateRange(mesh_device.shape)
    E.generic_op(
        io_tensors,
        E.MeshProgramSpec(programs=[(whole_mesh, spec)]),
        E.MeshProgramRunArgs(programs=[(whole_mesh, run_args)]),
    )

The ranges must not overlap, each range must cover only devices that hold every tensor, and the run args must have
the same ranges as the spec.

Moving a kernel from the ProgramDescriptor form
***********************************************

Only the lines that fetch a kernel's inputs change; the loops that move and compute data stay as they are.

- Arguments by position become arguments by name: ``get_arg_val<uint32_t>(0)`` becomes ``get_arg(args::num_tiles)``,
  with ``num_tiles`` declared in the kernel's ``RuntimeArgSchema``.
- Circular buffers by index become dataflow buffers by name: ``tt::CBIndex::c_0`` becomes
  ``DataflowBuffer(dfb::in)``, with ``in`` declared as a ``DataflowBufferSpec`` and bound to the kernel.
- Tensor addresses passed as arguments become bound tensors: ``TensorAccessor`` built from an address and
  ``TensorAccessorArgs`` becomes ``TensorAccessor(tensor::src)``.

Some features of the ``ProgramDescriptor`` form have no ``ProgramSpec`` equivalent yet, such as global circular
buffers, global semaphores, fabric connections and per-core allocated tensors. Programs that use them stay on
``ttnn.generic_op``.

More examples
*************

``tests/ttnn/unit_tests/operations/debug/test_generic_op_program_spec.py`` builds programs with two inputs, varargs,
aliased buffers, buffers on a sharded tensor's memory, semaphores, the mesh form and, on Quasar, a compute kernel with
two threads.
