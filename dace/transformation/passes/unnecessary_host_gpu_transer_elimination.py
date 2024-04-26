import dace
from dataclasses import dataclass
from typing import Any, Dict, Set, Type, Union
from dace.transformation import pass_pipeline as ppl
from dace.sdfg.analysis import cfg
from sympy import Interval, ProductSet

############
# Utility Functions / Classes that are only needed by this pass

# In a list of e.g. [1,2,3,4,5,2,4,5] prunes the repeating occurrences
# by removing the later occurrences returning, e.g., [1,2,3,4,5]
def remove_later_occurrences(lst):
    return list(set(lst))

def contains_nested_sdfg(lst):
  for node, _ in lst:
      if isinstance(node, dace.nodes.NestedSDFG):
          return True
  return False

def better_topological_sort(state : dace.sdfg.SDFGState):
  # For every node with 0 incoming degree, get the topological sort within the SDFG state
  # Do not use state.topological_sort()
  topologically_sorted_multi_nodes = dace.sdfg.utils.dfs_topological_sort(state)

  # Return them with their parent states, such that it is the same return type as
  # all_nodes_recursive()
  topologically_sorted_nodes = [(item, state) for item in topologically_sorted_multi_nodes]
  return topologically_sorted_nodes

# Returns topological sort of the whole SDFG, even if some states are not topologically sortable
# In that case, it sorts starting from every node with in_degree 0 and orders the nodes such that
# If node n2 appears after n1, then the condition is that there is no path from node n2 to n1.
# (There might also be no path from n1 to n2)
# Recursively done until there are no NestedSDFGs left
def all_nodes_topological_sort_and_recursive(sdfg):
  # Sort the nodes of every state (that can also be a nested SDFG)
  nodes = list()
  for state in list(cfg.stateorder_topological_sort(sdfg)):
    nodes += better_topological_sort(state)

  # Check if there is a Nested SDFG; while there is a nested SDFG, unpack the
  # nodes of the nested SDFG and bring it back to the list of nodes
  has_nested_sdfg = contains_nested_sdfg(nodes)
  while(has_nested_sdfg):
    idx = -1
    for _idx, (item, state) in enumerate(nodes):
      if isinstance(item, dace.nodes.NestedSDFG):
        idx = _idx
        break

    assert(idx == nodes.index((item, state)))
    assert((item, state) == nodes[idx])

    if idx != -1:
      _nodes = list()
      _state_order = list(cfg.stateorder_topological_sort(nodes[idx][0].sdfg))
      for _state in _state_order:
        _nodes += better_topological_sort(_state)
      # The nodes are basically [n1,...nM-1,nestedSDFG,nM+1,...nN]
      # and the resulting op is [n1,...,nM-1] + [<nodes result by unpacking the nestedSDFG>] + [nM+1,...,nN]
      nodes = nodes[:idx] + _nodes + nodes[idx+1:]

    has_nested_sdfg = contains_nested_sdfg(nodes)

  return nodes


# Auxiliary class to keep track of reads and writes to an Array
# It returns whether the N-dimensional data Array has been used or
# whether the data has changed.
# If the data has not been changed, copying it back from GPU to CPU is
# unnecessary
# If the data has not been used (for example, it is overwritten before
# accessed), then it is unnecessary to copy it from CPU to GPU and can
# be allocated
class DataUseTracker():
  def __init__(self, dims, name):
    self.dims = dims
    self.name = name
    self.unmarked_regions = ProductSet(*tuple(Interval(0, s - 1, False, False) for (s) in self.dims))
    self.host_data_used = False
    self.host_data_changed = False
    self.host_data_cant_be_used = False

  def reset(self):
    self.unmarked_regions = ProductSet(*tuple(Interval(0, s - 1, False, False) for (s) in self.dims))
    self.host_data_used = False
    self.host_data_changed = False
    self.host_data_cant_be_used = False

  def register_write(self, access_ranges):
    write_interval = ProductSet(*tuple(Interval(beg, end, False, False) for (beg,end) in access_ranges))

    if not write_interval.is_empty:
      self.host_data_changed = True

    if not self.host_data_used:
      self.unmarked_regions -= write_interval
      if self.unmarked_regions.is_empty:
        self.host_data_cant_be_used = True

  def register_read(self, access_ranges):
      read_interval = ProductSet(*tuple(Interval(beg, end, False, False) for (beg,end) in access_ranges))

      if not self.host_data_cant_be_used:
        # If read hits an area that has not been overwritten, then the data is used
        intersect = self.unmarked_regions.intersection(read_interval)
        if not intersect.is_empty:
          self.host_data_used = True

  def copied_host_data_used(self):
    return self.host_data_used

  def copied_host_data_changed(self):
    return self.host_data_changed

  def copied_host_data_overwritten_before_use(self):
    return (not self.host_data_used) and self.host_data_cant_be_used

  def __str__(self):
    return f"DataUseTracker(Dimensions={self.dims}, host_data_used={self.host_data_used}, host_data_changed={self.host_data_changed}, fully_overwritten_before_reads={self.host_data_cant_be_used}, unmarked_regions={self.unmarked_regions})"

  def __repr__(self):
    return self.__str__()

# Utility function that checks if the data described by the access node is already
# on the GPU
def on_gpu(sdfg: dace.SDFG | dace.SDFGState, data_name: str):
  if isinstance(sdfg, dace.SDFGState):
    _sdfg = sdfg._sdfg
  else:
    _sdfg = sdfg

  data = _sdfg.data(data_name)
  b = isinstance(data.storage, dace.dtypes.StorageType) and \
    (data.storage == dace.dtypes.StorageType.GPU_Shared or
    data.storage == dace.dtypes.StorageType.GPU_Global)
  return b

############


@dataclass
class UnnecessaryHostGPUTransferElimination(ppl.Pass):
  def __init__(self):
    super(UnnecessaryHostGPUTransferElimination, self).__init__()

  def __hash__(self):
    return hash(id(self))

  def modifies(self) -> ppl.Modifies:
    return ppl.Modifies.Edges | ppl.Modifies.Nodes | ppl.Modifies.States | ppl.Modifies.AccessNodes

  def depends_on(self) -> Set[Union[Type['Pass'], 'Pass']]:
    return {}

  def should_reapply(self, modified: ppl.Modifies) -> bool:
    return modified & ppl.Modifies.States

  def apply_pass(self, sdfg: dace.SDFG,
           pipeline_results: Dict[str, Any]) -> int:
    try:
      state_order = list(cfg.stateorder_topological_sort(sdfg))
    except KeyError:
      return None

    try:
      # Dict for all data containers
      data_container_dict = dict()
      for arrs in sdfg.arrays_recursive():
        sub_sdfg, arr_name, arr = arrs
        data_container_dict[arr_name] = (arr, sub_sdfg)

      # Order of all_edges_recursive is problematic
      gpu_to_cpu_copies_to_remove = set()
      cpu_to_gpu_copies_to_remove = set()
      cpu_to_gpu_copies = dict()
      name_map = dict()
      # Create a name map between the access nodes of a sub sdfg and parent sdfg
      for state in state_order:
        for edge, parent_sdfg in state.all_edges_recursive():
          e: dace.sdfg.graph.MultiConnectorEdge = edge
          if isinstance(e._dst, dace.sdfg.nodes.NestedSDFG):
            # The name of the array referencing to the same physical array
            # as in the parent SDFG
            dst_conn_name: str = e._dst_conn
            # Data name in the parent
            if isinstance(e._src, dace.nodes.AccessNode):
              src_name = e._src._data
              name_map[dst_conn_name] = src_name
          if isinstance(e._src, dace.sdfg.nodes.NestedSDFG):
            # The name of the array referencing to the same physical array
            # as in the parent SDFG
            src_conn_name: str = e._src_conn
            # Data name in the parent
            if isinstance(e._dst, dace.nodes.AccessNode):
              dst_name = e._dst._data
              name_map[src_conn_name] = dst_name
      # Root Level entries do not need entries in the name map

      def find_root_data_name(data_name: str):
        if not data_name in name_map:
          return data_name

        cur_name = data_name
        while(cur_name in name_map):
          cur_name = name_map[cur_name]
        return cur_name

      # Create the data name tracker for all exiting root-level arrays
      tracker_dict = dict()
      for name, (arr, sub_sdfg) in data_container_dict.items():
        original_name = find_root_data_name(name)
        # First check is to see that it is not a renamed access node
        if original_name == name:
          # Do not add CPU arrays, out of scope of this transformation
          if on_gpu(sdfg=sub_sdfg, data_name=original_name):
            tracker_dict[name] = DataUseTracker(dims=arr.shape, name=original_name)

      nodes = all_nodes_topological_sort_and_recursive(sdfg)

      # Since there can be multiple paths in a state that can execute at the same time
      # Whether the data has been used within a state must be checked that state has
      # completed its execution. Therefore on any GPU->CPU copy, the data tracker that needs
      # alongside the matching CPU->GPU copy and the analysis of whether it can be removed,
      # needs to be done after the state has ended. (The function is called after every state
      # is iterated, which is program completion)
      def check_whether_gpu_data_used():
        #print(tracker_dict)
        #print(cpu_to_gpu_copies)
        #print(data_to_reset)
        for _gpu_data in data_to_reset:
          if tracker_dict[_gpu_data].copied_host_data_overwritten_before_use() or \
            (not tracker_dict[_gpu_data].copied_host_data_used()):
            # It might be allocated on the GPU too
            if _gpu_data in cpu_to_gpu_copies:
              #print(f"Mark: {cpu_to_gpu_copies[_gpu_data]} to be removed")
              cpu_to_gpu_copies_to_remove.add(cpu_to_gpu_copies[_gpu_data])
              cpu_to_gpu_copies.pop(_gpu_data)
          tracker_dict[_gpu_data].reset()
        data_to_reset.clear()

      state_changed_this_iteration = False
      current_state = None
      data_to_reset = list()
      for (node, parent_sdfg) in nodes:
        state_changed_this_iteration = current_state and (current_state != parent_sdfg)
        current_state = parent_sdfg
        # Iterate through the CPU->GPU removal candidates at the end of the state
        if state_changed_this_iteration:
          check_whether_gpu_data_used()

        src_is_access_node = isinstance(node, dace.nodes.AccessNode)
        # Since iterating through nodes, always counting the out or in edges will be enough
        for edge in parent_sdfg.out_edges(node):
          src_node = node
          dst_node = edge._dst
          dst_is_access_node = isinstance(dst_node, dace.nodes.AccessNode)

          if src_is_access_node:
            src_node_data = src_node._data
            original_src_name = find_root_data_name(src_node_data)
          if dst_is_access_node:
            dst_node_data = dst_node._data
            original_dst_name = find_root_data_name(dst_node_data)

          # Special case is Write-Conflict-Resolution Edge
          # This is a special edge between a Tasklet and something
          # It is essentially both a read and a write
          #if isinstance(dst_node, dace.nodes.MapExit):
          if "_wcr" in edge._data.__dict__ and edge._data._wcr:
            data_name = edge._data._data
            original_name = find_root_data_name(data_name)
            arr, parent_sdfg = data_container_dict[original_name]
            if on_gpu(sdfg=parent_sdfg, data_name=original_name):
              #print("WCR EDGE: ", edge, edge.__dict__)
              access_ranges = list()
              for ar in edge._data._subset.ranges:
                access_ranges.append((ar[0], ar[1]))
              tracker_dict[original_name].register_read(access_ranges)
              tracker_dict[original_name].register_write(access_ranges)

          # When mapped from sub-SDFG changed name to root SDFG (original name) are the same
          # Then it means this results not with real(physical) memory movement
          if src_is_access_node and dst_is_access_node and \
            original_dst_name == original_src_name:
            continue

          # Copy between Host-Acc
          if src_is_access_node and dst_is_access_node:
            # GPU->CPU
            if on_gpu(sdfg=parent_sdfg, data_name=src_node_data) and \
              not on_gpu(sdfg=parent_sdfg, data_name=dst_node_data):
              if not tracker_dict[original_src_name].copied_host_data_changed():
                gpu_to_cpu_copies_to_remove.add((parent_sdfg, edge))

              data_to_reset.append(src_node_data)
            # CPU->GPU
            elif not on_gpu(sdfg=parent_sdfg, data_name=src_node_data) and \
              on_gpu(sdfg=parent_sdfg, data_name=dst_node_data):
              assert(not dst_node_data in cpu_to_gpu_copies)
              cpu_to_gpu_copies[dst_node_data] = (parent_sdfg, edge)
              if dst_node_data in tracker_dict:
                tracker_dict[dst_node_data].reset()

          if src_is_access_node or dst_is_access_node:
            access_ranges = list()
            for ar in edge._data._subset.ranges:
              access_ranges.append((ar[0], ar[1]))

          # Register read
          if src_is_access_node and not dst_is_access_node and \
              on_gpu(sdfg=parent_sdfg, data_name=src_node_data):
            tracker_dict[original_src_name].register_read(access_ranges)

          # Register write
          if not src_is_access_node and dst_is_access_node and \
              on_gpu(sdfg=parent_sdfg, data_name=dst_node_data):
            tracker_dict[original_dst_name].register_write(access_ranges)
      check_whether_gpu_data_used()
    except Exception as e:
      # An error occurred before any transformation was applied, can abort safely
      # By not removing any edges.
      return [set(), set()]

    for s in [cpu_to_gpu_copies_to_remove, gpu_to_cpu_copies_to_remove]:
      for parent_sdfg, e in s:
        # The first copy from CPU to GPU is superfluous, can be removed
        # This involves the nodes accessing the array and the edge, when accessing it for
        # first time
        parent_sdfg.remove_edge(e)
        src : dace.sdfg.nodes.AccessNode = e._src
        dst : dace.sdfg.nodes.AccessNode = e._dst
        src.remove_out_connector(e._src_conn)
        dst.remove_in_connector(e._dst_conn)

        if parent_sdfg.in_degree(src) == 0 and parent_sdfg.out_degree(src) == 0:
          parent_sdfg.remove_node(src)
        if parent_sdfg.in_degree(dst) == 0 and parent_sdfg.out_degree(dst) == 0:
          parent_sdfg.remove_node(dst)
    return [cpu_to_gpu_copies_to_remove, gpu_to_cpu_copies_to_remove]