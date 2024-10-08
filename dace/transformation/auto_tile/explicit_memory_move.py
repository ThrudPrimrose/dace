# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from ast import Tuple
import copy
import dace
from dace import subsets
from dace.data import Array
from dace.properties import ListProperty, Property, SubsetProperty, make_properties
from dace.libraries.standard.nodes import CodeLibraryNode
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.symbolic import SymExpr
from dace.transformation import transformation
from dace import dtypes
from functools import reduce
from typing import Any, List, Dict
from dace import symbol


@make_properties
class MemoryMovementNode(CodeLibraryNode):
    src_subset = SubsetProperty(default=None, allow_none=True, desc="")
    src_arr_name = Property(dtype=str, default="", allow_none=False, desc="")
    src_arr = Property(dtype=dace.data.Array, default=None, allow_none=True, desc="")
    dst_arr_name = Property(dtype=str, default="", allow_none=False, desc="")
    dst_arr = Property(dtype=dace.data.Array, default=None, allow_none=True, desc="")
    num_threads = ListProperty(
        element_type=int, default=[32, 1, 1], allow_none=False, desc=""
    )
    storage = Property(
        dtype=dace.StorageType,
        default=dace.StorageType.GPU_Shared,
        allow_none=False,
        desc="",
    )
    sync = Property(dtype=bool, default=True, allow_none=False, desc="")
    tiles_evenly = Property(dtype=bool, default=False, allow_none=False, desc="")

    def __init__(
        self,
        name,
        input_names,
        output_names,
        src_subset,
        src_arr_name,
        src_arr,
        dst_arr_name,
        dst_arr,
        num_threads,
        storage,
        sync,
        tiles_evenly,
    ):
        self.src_subset = src_subset
        self.src_arr_name = src_arr_name
        self.src_arr = src_arr
        self.dst_arr_name = dst_arr_name
        self.dst_arr = dst_arr
        self.num_threads = num_threads
        self.storage = storage
        self.sync = sync
        self.tiles_evenly = tiles_evenly
        super().__init__(name=name, input_names=input_names, output_names=output_names)

    def write_inner_loop(
        self, num_threads, line_len, dim_check, dynamic_check=False, tiles_evenly=False
    ):
        code = ""
        conds = dim_check
        if num_threads > line_len:
            lines_at_a_time = num_threads // line_len
            real_lines_at_a_time = min(
                int(lines_at_a_time), int(self.dst_arr.shape[-2])
            )
            code += f"// load multiple lines at a time {real_lines_at_a_time}\n"
            if len(self.src_arr.shape) == 1:
                num_active_threads = line_len
                strides = self.src_arr.strides
                offset_expression_1d = "+".join(
                    [
                        f"({stride}*({offset}))"
                        for (offset, _, _), stride in zip(self.src_subset, strides)
                    ]
                )
                code += (
                    f"if (tid < {num_active_threads}){{\n"
                    if not dynamic_check
                    else f"if (tid < {conds[0]}){{\n"
                )
                code += f"{self.dst_arr_name}[tid] = {self.src_arr_name}[{offset_expression_1d}+tid];\n"
                code += f"}}\n"
            else:
                if not dynamic_check:
                    code += f"const int line_offset = tid % {line_len};\n"
                    code += f"const int line_num = tid / {line_len};\n"
                else:
                    code += f"const int effective_line_len = {conds[-1]};\n"
                    code += f"const int line_offset = tid % effective_line_len;\n"
                    code += f"const int line_num = tid / effective_line_len;\n"
                strides = self.src_arr.strides
                num_active_threads = real_lines_at_a_time * line_len
                offset_expression_1d = "+".join(
                    [
                        f"({stride}*({offset}))"
                        for (offset, _, _), stride in zip(self.src_subset, strides)
                    ]
                )
                if num_active_threads < num_threads:
                    code += f"if (tid < {num_active_threads}){{\n"

                var_id = 0
                for dim in conds[:-2]:
                    code += f"for (int i{var_id} = 0; i{var_id} < {dim}; i{var_id} += 1) {{\n"
                    var_id += 1

                further_access_strs_dst = []
                further_access_strs_src = []
                bound_checks = []
                for i, (dst_stride, src_stride, src_lim) in enumerate(
                    zip(
                        self.dst_arr.strides[:-2],
                        self.src_arr.strides[:-2],
                        self.src_arr.shape[:-2],
                    )
                ):
                    further_access_strs_dst.append(f"i{i} * {dst_stride}")
                    further_access_strs_src.append(f"i{i} * {src_stride}")
                further_access_str_src = " + ".join(further_access_strs_dst)
                further_access_str_dst = " + ".join(further_access_strs_src)
                if further_access_str_dst != "":
                    further_access_str_dst = " + " + further_access_str_dst
                if further_access_str_src != "":
                    further_access_str_src = " + " + further_access_str_src

                for i, (dst_stride, src_stride, src_lim) in enumerate(
                    zip(
                        self.dst_arr.strides[:-1],
                        self.src_arr.strides[:-1],
                        self.src_arr.shape[:-1],
                    )
                ):
                    bound_checks.append((f"i{i}", src_lim))

                if dynamic_check:
                    code += f"const int effectivenum_threads = {lines_at_a_time} * effective_line_len;\n"
                    code += f"if (tid < effectivenum_threads){{\n"

                if self.dst_arr.shape[-2] > real_lines_at_a_time:
                    code += f"#pragma unroll\n"
                    if not dynamic_check:
                        if conds[-2] % real_lines_at_a_time != 0:
                            remainder_iters = conds[-2] % real_lines_at_a_time
                            conds[-2] -= remainder_iters
                        else:
                            remainder_iters = 0
                    code += f"for (int i{var_id} = 0; i{var_id} < {conds[-2]}; i{var_id} += {real_lines_at_a_time}) {{\n"
                    further_access_str_src += (
                        " + " + f"((i{var_id}) * {self.src_arr.strides[-2]})"
                    )
                    further_access_str_dst += (
                        " + " + f"((i{var_id}) * {self.dst_arr.strides[-2]})"
                    )

                if dynamic_check:
                    code += f"if(line_offset < effective_line_len && line_num + {bound_checks[-1][0]} < {conds[-2]}){{\n"

                code += f"{self.dst_arr_name}[line_num*{self.dst_arr.strides[-2]} + line_offset{further_access_str_dst}] = {self.src_arr_name}[{offset_expression_1d} + line_num*{self.src_arr.strides[-2]} + line_offset{further_access_str_src}];\n"

                if self.dst_arr.shape[-2] > real_lines_at_a_time:
                    code += f"}}\n"

                if not dynamic_check:
                    if remainder_iters > 0:
                        code += f"if (tid < {remainder_iters*line_len}){{\n"
                        code += f"const int i{var_id} = {conds[-2]};\n"
                        code += f"{self.dst_arr_name}[line_num*{self.dst_arr.strides[-2]} + line_offset{further_access_str_dst}] = {self.src_arr_name}[{offset_expression_1d} + line_num*{self.src_arr.strides[-2]} + line_offset{further_access_str_src}];\n"
                        code += "}\n"

                if dynamic_check:
                    code += f"}}\n" * 2

                if num_active_threads < num_threads:
                    code += f"}}\n"
                for _ in self.dst_arr.shape[:-2]:
                    code += f"}}\n"
        else:
            code += f"// load one line at a time\n"
            if len(self.src_arr.shape) == 1:
                num_active_threads = line_len
                strides = self.src_arr.strides
                offset_expression_1d = "+".join(
                    [
                        f"({stride}*({offset}))"
                        for (offset, _, _), stride in zip(self.src_subset, strides)
                    ]
                )
                cond = conds[0]
                code += (
                    f"for (int i0 = tid; i0 < {cond}; i0 += {num_active_threads}) {{\n"
                )
                code += f"{self.dst_arr_name}[tid] = {self.src_arr_name}[{offset_expression_1d}+tid];\n"
                code += f"}}\n"
            else:
                strides = self.src_arr.strides
                num_active_threads = num_threads
                offset_expression_1d = "+".join(
                    [
                        f"({stride}*({offset}))"
                        for (offset, _, _), stride in zip(self.src_subset, strides)
                    ]
                )

                var_id = 0
                for dim in conds[:-2]:
                    code += f"for (int i{var_id} = 0; i{var_id} < {dim}; i{var_id} += 1) {{\n"
                    var_id += 1

                further_access_strs_dst = []
                further_access_strs_src = []
                for i, (dst_stride, src_stride) in enumerate(
                    zip(self.dst_arr.strides[:-2], self.src_arr.strides[:-2])
                ):
                    further_access_strs_dst.append(f"i{i} * {dst_stride}")
                    further_access_strs_src.append(f"i{i} * {src_stride}")
                further_access_str_src = " + ".join(further_access_strs_dst)
                further_access_str_dst = " + ".join(further_access_strs_src)
                if further_access_str_dst != "":
                    further_access_str_dst = " + " + further_access_str_dst
                if further_access_str_src != "":
                    further_access_str_src = " + " + further_access_str_src

                code += f"#pragma unroll\n"
                code += f"for (int i{var_id} = 0; i{var_id} < {conds[-2]}; i{var_id} += 1) {{\n"
                code += f"#pragma unroll\n"
                code += f"for (int i{var_id+1} = tid; i{var_id+1} < {conds[-1]}; i{var_id+1} += {num_active_threads}) {{\n"
                further_access_str_src += (
                    " + " + f"((i{var_id}) * {self.src_arr.strides[-2]}) + i{var_id+1}"
                )
                further_access_str_dst += (
                    f"((i{var_id}) * {self.dst_arr.strides[-2]}) + i{var_id+1}"
                )

                code += f"{self.dst_arr_name}[{further_access_str_dst}] = {self.src_arr_name}[{offset_expression_1d}{further_access_str_src}];\n"

                code += f"}}\n}}\n"
                if num_active_threads < num_threads:
                    code += f"}}\n"
                for _ in self.dst_arr.shape[:-2]:
                    code += f"}}\n"
        return code

    def generate_code(self, inputs, outputs):
        code = ""
        code += f"// {self.src_arr_name}[{','.join([str(s) for s in self.src_arr.shape])}]\n"
        code += f"// {self.dst_arr_name}[{','.join([str(s) for s in self.dst_arr.shape])}]\n"
        num_threads = reduce(lambda x, y: x * y, self.num_threads)
        tiles_evenly = self.tiles_evenly

        conds = []
        for (beg, end, step), lim in zip(self.src_subset, self.src_arr.shape):
            load_len = end + 1 - beg
            conds.append(f"{beg} <= {lim} - {load_len}")
        code += "// Inner Loop Condition: " + " && ".join(conds) + "\n"
        code += "const int tid = threadIdx.x + blockDim.x * threadIdx.y + (blockDim.x * blockDim.y) * threadIdx.z;\n"

        inner_dim_checks = [lim for lim in self.dst_arr.shape]
        outer_dim_checks = [
            f"Min({lim - beg}, {dst_lim})"
            for (beg, end, step), lim, dst_lim in zip(
                self.src_subset, self.src_arr.shape, self.dst_arr.shape
            )
        ]

        # Variant 1, load multiple lines at a time
        lb, le, ls = self.src_subset[-1]
        line_len = le + 1 - lb
        assert ls == 1
        code += f"// Num Threads: {num_threads}, Line Length (max): {line_len}\n"

        if len(conds) > 0:
            if not tiles_evenly:
                code += f"if ({' && '.join(conds)}) {{\n"
            code += self.write_inner_loop(
                num_threads, line_len, inner_dim_checks, False, tiles_evenly
            )
            if not tiles_evenly:
                code += "} else { \n"
                code += self.write_inner_loop(
                    num_threads, line_len, outer_dim_checks, True, tiles_evenly
                )
                code += "}\n"
        else:
            self.write_inner_loop(num_threads, line_len)

        if self.sync:
            code += "__syncthreads();\n"
        return code


@make_properties
class ExplicitMemoryMove(transformation.SingleStateTransformation):
    """
    For all memlets that connect the outer map to inner map,
    The src location is inferred and the memory is moved explicitly using a library not the given memory location
    """

    device_map_entry = transformation.PatternNode(nodes.MapEntry)
    thread_block_map_entry = transformation.PatternNode(nodes.MapEntry)
    map_entry = transformation.PatternNode(nodes.MapEntry)
    tiles_evenly = Property(dtype=bool, default=False, desc="No remainder loop")

    def __init__(self):
        self._added_arrays = []
        super().__init__()

    # Properties
    memory_location = Property(
        dtype=dtypes.StorageType,
        default=dtypes.StorageType.GPU_Shared,
        desc="Destination memory location",
    )

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(
                cls.device_map_entry, cls.thread_block_map_entry, cls.map_entry
            )
        ]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def infer_source(
        self, state: SDFGState, sdfg: SDFG, edge: MultiConnectorEdge[Memlet]
    ):
        u, uc, v, vc, memlet = edge
        return (memlet.data, sdfg.arrays[memlet.data].storage)

    location_to_prefix = {
        dtypes.StorageType.GPU_Global: "glb",
        dtypes.StorageType.GPU_Shared: "shr",
    }

    def filter_terms(self, expr, vars):
        if isinstance(expr, int):
            return expr

        # Base case: if the expression is a single term
        if expr.is_Atom:
            for var in vars:
                if expr.has(symbol(var)):
                    return expr
            if expr.is_number:
                return expr
            else:
                return 0

        # Recursive case: apply the function to each argument of the expression
        filtered_expr = 0
        for term in expr.args:
            filtered_term = self.filter_terms(term, vars)
            if filtered_term != 0:
                filtered_expr += filtered_term

        return filtered_expr

    def apply(self, state: SDFGState, sdfg: SDFG):
        offsets = dict()
        loc1_to_loc2_map = dict()
        num_loads = 0
        tiles_evenly = self.tiles_evenly
        for out_edge in state.out_edges(self.map_entry):
            u, uc, v, vc, memlet = out_edge
            src_arr_name, src_arrstorage_type = self.infer_source(state, sdfg, out_edge)
            if src_arrstorage_type == dtypes.StorageType.GPU_Global:
                num_loads += 1

        current_load = 0
        for out_edge in state.out_edges(self.map_entry):
            u, uc, v, vc, memlet = out_edge
            src_arr_name, src_arrstorage_type = self.infer_source(state, sdfg, out_edge)
            if src_arrstorage_type != dtypes.StorageType.GPU_Global or isinstance(
                sdfg.arrays[src_arr_name], dace.data.Scalar
            ):
                continue

            parsedstorage_type = f"{src_arrstorage_type}".split(".")[-1]
            parsed_memory_location = f"{self.memory_location}".split(".")[-1]

            # Map the subset accessed by a thread to the subset accessed by the threadblock
            subset_to_pass = []
            shape = []
            to_replace = []
            for i, (beg, end, step) in enumerate(memlet.subset):
                for sym in set.union(beg.free_symbols, end.free_symbols):
                    for param in self.thread_block_map_entry.map.params:
                        if str(sym) == param:
                            to_replace.append(i)
                            break
            for in_edge in state.in_edges(self.thread_block_map_entry):
                _, _, _, _, _memlet = in_edge
                if _memlet.data == memlet.data:
                    for j in range(len(memlet.subset)):
                        if j in to_replace:
                            subset_to_pass.append(_memlet.subset[j])
                            b, e, s = _memlet.subset[j]
                            assert s == 1
                            shape.append(e + 1 - b)
                        else:
                            subset_to_pass.append(memlet.subset[j])
                            b, e, s = memlet.subset[j]
                            assert s == 1
                            shape.append(e + 1 - b)
            # End Mapping

            mem_loc_a = parsedstorage_type
            mem_loc_b = parsed_memory_location
            lib_node_name = f"move_{memlet.data}_from_{mem_loc_a}_to_{mem_loc_b}"
            dst_arr_shape = shape
            num_threads = [
                int((e + 1 - b) / s)
                for b, e, s in self.thread_block_map_entry.map.range
            ]

            dst_arr_name = self.location_to_prefix[self.memory_location] + src_arr_name
            c = 0
            while dst_arr_name in sdfg.arrays:
                if not (dst_arr_name + str(c) in sdfg.arrays):
                    dst_arr_name = dst_arr_name + str(c)
                else:
                    c += 1
            (dst_arr_name, dst_arr) = sdfg.add_array(
                name=dst_arr_name,
                dtype=sdfg.arrays[src_arr_name].dtype,
                shape=dst_arr_shape,
                transient=True,
                storage=self.memory_location,
            )
            self._added_arrays.append(dst_arr_name)

            # raise Exception(type(subset_to_pass), ", ", type(subset_to_pass[0]))

            lib_node = MemoryMovementNode(
                name=lib_node_name,
                input_names=[vc],
                output_names=[uc],
                src_subset=dace.subsets.Range(subset_to_pass),
                src_arr_name=src_arr_name,
                src_arr=sdfg.arrays[src_arr_name],
                dst_arr_name=dst_arr_name,
                dst_arr=dst_arr,
                num_threads=num_threads,
                storage=self.memory_location,
                sync=bool(current_load == num_loads - 1),
                tiles_evenly=tiles_evenly,
            )
            current_load += 1
            state.add_node(lib_node)
            state.remove_edge(out_edge)

            # Add offsets for the rest of the accesses
            offsets[src_arr_name] = [beg for (beg, end, step) in memlet.subset]
            # Loc1 -> Loc2 mapping list
            loc1_to_loc2_map[src_arr_name] = (
                dst_arr_name,
                [beg for (beg, end, step) in memlet.subset],
            )

            dst_access_node = nodes.AccessNode(data=dst_arr_name)
            state.add_node(dst_access_node)
            # Compute thread block offset
            old_subset = memlet.subset

            # This removes any parameter that depends on the grid loop
            new_subset = []
            thread_block_offset = []
            for beg, end, step in old_subset:
                _beg = self.filter_terms(
                    SymExpr(beg), self.thread_block_map_entry.map.params
                )
                _end = self.filter_terms(
                    SymExpr(end), self.thread_block_map_entry.map.params
                )
                _step = self.filter_terms(
                    SymExpr(step), self.thread_block_map_entry.map.params
                )
                if isinstance(_beg, SymExpr) or isinstance(_beg, symbol):
                    thread_block_offset.append(
                        any(
                            [
                                _beg.has(symbol(v))
                                for v in self.thread_block_map_entry.map.params
                            ]
                        )
                    )
                else:
                    thread_block_offset.append(False)
                new_subset.append((_beg, _end, _step))

            new_range_list = new_subset
            to_dst_memlet = Memlet(
                subset=subsets.Range(new_range_list), data=dst_arr_name
            )
            to_map_memlet = Memlet(
                subset=subsets.Range(new_range_list), data=dst_arr_name
            )

            # Outer Map -> Lib Node
            state.add_edge(u, uc, lib_node, vc, memlet)

            # Lib Node -> Access Node
            state.add_edge(lib_node, uc, dst_access_node, None, to_dst_memlet)

            # Acces Node -> Inner Map
            state.add_edge(dst_access_node, None, v, vc, to_map_memlet)

            # Update any memlet that accesses any of the mapped arrays
            edges_to_check = set()

            nodeset = set([v])
            while nodeset:
                n = nodeset.pop()
                if not isinstance(n, nodes.MapExit):
                    edges_to_check = edges_to_check.union(set(state.out_edges(n)))
                    nodeset = nodeset.union(
                        set([v for u, uc, v, vc, m in state.out_edges(n)])
                    )

            for edge in edges_to_check:
                u, uc, v, vc, memlet = edge
                if memlet.data in loc1_to_loc2_map.keys():
                    dst_name, offset = loc1_to_loc2_map[memlet.data]
                    new_subset_list = [
                        (beg - offset, end - offset, step)
                        for (beg, end, step), offset in zip(memlet.subset, offset)
                    ]

                    for i, ((beg, end, step), apply_offset) in enumerate(
                        zip(new_subset_list, thread_block_offset)
                    ):
                        if apply_offset:
                            params = self.thread_block_map_entry.map.params
                            nb = (
                                beg + symbol(params[i]),
                                end + symbol(params[i]),
                                step,
                            )
                            new_subset_list[i] = nb

                    new_memlet = Memlet(
                        subset=subsets.Range(new_subset_list), data=dst_name
                    )
                    state.remove_edge(edge)
                    state.add_edge(u, uc, v, vc, new_memlet)

        self.map_entry.map.gpu_forcesyncthreads = True

    @staticmethod
    def annotates_memlets():
        return True
