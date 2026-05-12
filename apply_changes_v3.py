
import sys
import re

path = '/home/engine/project/core/genesis-topology/src/hnsw.rs'
with open(path, 'r') as f:
    content = f.read()

def replace_fn(name, new_body):
    global content
    m = re.search(r'fn ' + name + r'\(.*?\{', content, re.DOTALL)
    if not m:
        # Try with pub fn
        m = re.search(r'pub fn ' + name + r'\(.*?\{', content, re.DOTALL)
        if not m:
            print(f"Function {name} not found")
            return
    
    start = m.start()
    depth = 0
    end = -1
    for i in range(start, len(content)):
        if content[i] == '{': depth += 1
        elif content[i] == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    
    if end != -1:
        content = content[:start] + new_body + content[end+1:]
    else:
        print(f"Could not find end of function {name}")

search_layer_body = """    fn search_layer(
        &self,
        query: &SparseCliffordVector,
        entry_idx: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f64)> {
        SEARCH_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.out.clear();
            let search_epoch = self.next_search_epoch();
            VISITED_EPOCH.with(|visited_cell| {
                let mut visited = visited_cell.borrow_mut();
                let needed = self.nodes.len();
                if visited.len() < needed {
                    let grown_len = needed.next_power_of_two();
                    visited.resize(grown_len, 0);
                }
                if search_epoch == 1 {
                    visited.fill(0);
                }

                let limit = ef.min(MAX_FIXED_HEAP_CAPACITY);
                let mut candidates = FixedHeap::<MAX_FIXED_HEAP_CAPACITY>::new(limit);
                let mut results = FixedHeap::<MAX_FIXED_HEAP_CAPACITY>::new(limit);
                let query_f32 = Self::dense_to_query_f32(query);
                let approx_threshold_sq = self.adaptive_precision_threshold_sq_f32();
                let slab_blocks = self.layer0_soa.blocks.len();
                let nodes_len = self.nodes.len();
                let audit = self.should_audit_escape();

                let d0 = self.distance_to_node_sq(query, entry_idx, layer) as f32;
                visited[entry_idx] = search_epoch;
                candidates.push_or_replace(d0, entry_idx as u32);
                results.push_or_replace(d0, entry_idx as u32);

                while let Some((c_dist, c_idx_u32)) = candidates.pop_best() {
                    if results.len() >= limit && c_dist > results.worst() {
                        break;
                    }

                    let c_idx = c_idx_u32 as usize;
                    if layer > self.nodes[c_idx].max_layer {
                        continue;
                    }
                    if layer == 0 && !self.layer0_soa.blocks.is_empty() {
                        // HOT PATH: O(groups), called per beam expansion at layer 0.
                        // Layer-0 block projection is materialized at insertion/removal time.
                        for group in self.node_layer0_groups(c_idx) {
                            let block = group.block as usize;
                            if block >= slab_blocks {
                                continue;
                            }
                            let base = block * SLAB_LANES;
                            let block_ptr = self.layer0_soa.blocks[block].lanes.as_ptr();
                            debug_assert_eq!(block_ptr as usize % 64, 0, "slab alignment");
                            // SAFETY: `block < slab_blocks`, slab is persistently materialized and
                            // 64-byte aligned; query buffer has fixed 16-lane shape.
                            let batch =
                                slab_distance(block_ptr, &query_f32, approx_threshold_sq);
                            let mut m = group.lane_mask;
                            while m != 0 {
                                let lane_u8 = m.trailing_zeros() as u8;
                                let lane = usize::from(lane_u8);
                                let original_bit = 1_u8 << lane_u8;
                                let nb_idx = base + lane;
                                if nb_idx >= nodes_len {
                                    m &= m - 1;
                                    continue;
                                }
                                if visited[nb_idx] == search_epoch {
                                    m &= m - 1;
                                    continue;
                                }
                                visited[nb_idx] = search_epoch;
                                let d = batch.distances[lane];
                                let escape = (batch.escape_mask & original_bit) != 0;
                                let recall_drop = escape
                                    & self.compute_recall_drop(
                                        audit, d, &query_f32, block, lane, block_ptr,
                                    );
                                self.record_escape_result(escape, recall_drop);
                                if results.push_or_replace(d, nb_idx as u32) {
                                    candidates.push_or_replace(d, nb_idx as u32);
                                }
                                m &= m - 1;
                            }
                        }
                    } else {
                        for nb_idx_u32 in self.node_neighbors_iter(c_idx, layer) {
                            let nb_idx = nb_idx_u32 as usize;
                            if visited[nb_idx] == search_epoch {
                                continue;
                            }
                            visited[nb_idx] = search_epoch;
                            let d = self.distance_to_node_sq(query, nb_idx, layer) as f32;
                            if results.push_or_replace(d, nb_idx as u32) {
                                candidates.push_or_replace(d, nb_idx as u32);
                            }
                        }
                    }
                }

                scratch.out.reserve(results.len());
                for &(dist_sq, idx) in results.as_slice() {
                    scratch.out.push((idx as usize, f64::from(dist_sq)));
                }
                std::mem::take(&mut scratch.out)
            })
        })
    }"""

search_layer_work_count_body = """    fn search_layer_with_work_count(
        &self,
        query: &SparseCliffordVector,
        entry_idx: usize,
        ef: usize,
        layer: usize,
        work_count: &mut usize,
    ) -> Vec<(usize, f64)> {
        SEARCH_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.out.clear();
            let search_epoch = self.next_search_epoch();
            VISITED_EPOCH.with(|visited_cell| {
                let mut visited = visited_cell.borrow_mut();
                let needed = self.nodes.len();
                if visited.len() < needed {
                    let grown_len = needed.next_power_of_two();
                    visited.resize(grown_len, 0);
                }
                if search_epoch == 1 {
                    visited.fill(0);
                }

                let limit = ef.min(MAX_FIXED_HEAP_CAPACITY);
                let mut candidates = FixedHeap::<MAX_FIXED_HEAP_CAPACITY>::new(limit);
                let mut results = FixedHeap::<MAX_FIXED_HEAP_CAPACITY>::new(limit);
                let query_f32 = Self::dense_to_query_f32(query);
                let approx_threshold_sq = self.adaptive_precision_threshold_sq_f32();
                let slab_blocks = self.layer0_soa.blocks.len();
                let nodes_len = self.nodes.len();
                let audit = self.should_audit_escape();

                let d0 = self.distance_to_node_sq(query, entry_idx, layer) as f32;
                *work_count += 1;
                visited[entry_idx] = search_epoch;
                candidates.push_or_replace(d0, entry_idx as u32);
                results.push_or_replace(d0, entry_idx as u32);

                while let Some((c_dist, c_idx_u32)) = candidates.pop_best() {
                    if results.len() >= limit && c_dist > results.worst() {
                        break;
                    }

                    let c_idx = c_idx_u32 as usize;
                    if layer > self.nodes[c_idx].max_layer {
                        continue;
                    }
                    if layer == 0 && !self.layer0_soa.blocks.is_empty() {
                        for group in self.node_layer0_groups(c_idx) {
                            let block = group.block as usize;
                            if block >= slab_blocks {
                                continue;
                            }
                            let base = block * SLAB_LANES;
                            let block_ptr = self.layer0_soa.blocks[block].lanes.as_ptr();
                            debug_assert_eq!(block_ptr as usize % 64, 0, "slab alignment");
                            // SAFETY: `block < slab_blocks`, slab is persistently materialized and
                            // 64-byte aligned; query buffer has fixed 16-lane shape.
                            let batch =
                                slab_distance(block_ptr, &query_f32, approx_threshold_sq);
                            let mut m = group.lane_mask;
                            while m != 0 {
                                let lane_u8 = m.trailing_zeros() as u8;
                                let lane = usize::from(lane_u8);
                                let original_bit = 1_u8 << lane_u8;
                                let nb_idx = base + lane;
                                if nb_idx >= nodes_len {
                                    m &= m - 1;
                                    continue;
                                }
                                if visited[nb_idx] == search_epoch {
                                    m &= m - 1;
                                    continue;
                                }
                                visited[nb_idx] = search_epoch;
                                let d = batch.distances[lane];
                                let escape = (batch.escape_mask & original_bit) != 0;
                                let recall_drop = escape
                                    & self.compute_recall_drop(
                                        audit, d, &query_f32, block, lane, block_ptr,
                                    );
                                self.record_escape_result(escape, recall_drop);
                                *work_count += 1;
                                if results.push_or_replace(d, nb_idx as u32) {
                                    candidates.push_or_replace(d, nb_idx as u32);
                                }
                                m &= m - 1;
                            }
                        }
                    } else {
                        for nb_idx_u32 in self.node_neighbors_iter(c_idx, layer) {
                            let nb_idx = nb_idx_u32 as usize;
                            if visited[nb_idx] == search_epoch {
                                continue;
                            }
                            visited[nb_idx] = search_epoch;
                            let d = self.distance_to_node_sq(query, nb_idx, layer) as f32;
                            *work_count += 1;
                            if results.push_or_replace(d, nb_idx as u32) {
                                candidates.push_or_replace(d, nb_idx as u32);
                            }
                        }
                    }
                }

                scratch.out.reserve(results.len());
                for &(dist_sq, idx) in results.as_slice() {
                    scratch.out.push((idx as usize, f64::from(dist_sq)));
                }
                std::mem::take(&mut scratch.out)
            })
        })
    }"""

replace_fn('search_layer', search_layer_body)
replace_fn('search_layer_with_work_count', search_layer_work_count_body)

with open(path, 'w') as f:
    f.write(content)
