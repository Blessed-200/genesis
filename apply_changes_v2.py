
import sys
import re

path = '/home/engine/project/core/genesis-topology/src/hnsw.rs'
with open(path, 'r') as f:
    content = f.read()

# 1. HnswLayer0Slab definition
content = content.replace(
    'struct HnswLayer0Slab {\n    blocks: Vec<SlabBlock>,',
    'struct HnswLayer0Slab {\n    blocks: Arc<Vec<Arc<SlabBlock>>>,'
)

# 2. HnswGraph::new
content = re.sub(
    r'blocks: Vec::with_capacity\(INITIAL_SLAB_BLOCK_CAPACITY\),',
    r'blocks: Arc::new(Vec::with_capacity(INITIAL_SLAB_BLOCK_CAPACITY)),',
    content
)

# 3. write_node_to_slab
new_write_node = """    fn write_node_to_slab(&mut self, dense_idx: usize, vec: &SparseCliffordVector) {
        let slab_idx = dense_idx;
        let block = slab_idx / SLAB_LANES;
        let lane = slab_idx % SLAB_LANES;
        if block >= self.layer0_soa.blocks.len() {
            Arc::make_mut(&mut self.layer0_soa.blocks).push(Arc::new(SlabBlock::zeroed()));
        }
        let blocks = Arc::make_mut(&mut self.layer0_soa.blocks);
        let block_ref = Arc::make_mut(&mut blocks[block]);
        let mut d = 0;
        while d < SLAB_DIM {
            block_ref.lanes[d * SLAB_LANES + lane] = vec.coeffs[d] as f32;
            d += 1;
        }
        let slab_idx_u32 = u32::try_from(slab_idx).unwrap_or(u32::MAX - 1);
        debug_assert_ne!(slab_idx_u32, u32::MAX);
        Self::cow_vec_mut(&mut self.layer0_soa.node_to_slab).push(slab_idx_u32);
    }"""
content = re.sub(r'fn write_node_to_slab\(&mut self, dense_idx: usize, vec: &SparseCliffordVector\) \{.*?\}', new_write_node, content, flags=re.DOTALL)

# 4. slab_distance functions
content = content.replace(
    'fn slab_distance_scalar(\n    slab_ptr: *const f32,\n    block: usize,',
    'fn slab_distance_scalar(\n    block_ptr: *const f32,'
)
# Fix scalar body
content = content.replace('let block_base = block * BLOCK_STRIDE;', '// block_ptr is direct')
content = content.replace('slab_ptr.add(block_base + lane)', 'block_ptr.add(lane)')
content = content.replace('slab_ptr.add(offset)', 'block_ptr.add(offset)')
content = content.replace('block_base + ', '')

content = content.replace(
    'unsafe fn slab_distance_avx2(\n    slab_ptr: *const f32,\n    block: usize,',
    'unsafe fn slab_distance_avx2(\n    block_ptr: *const f32,'
)
content = content.replace('let block_base = unsafe { slab_ptr.add(block * BLOCK_STRIDE) };', 'let block_base = block_ptr;')

content = content.replace(
    'fn slab_distance(\n    slab_ptr: *const f32,\n    block: usize,',
    'fn slab_distance(\n    block_ptr: *const f32,'
)
content = content.replace('slab_distance_avx2(slab_ptr, block,', 'slab_distance_avx2(block_ptr,')
content = content.replace('slab_distance_scalar(slab_ptr, block,', 'slab_distance_scalar(block_ptr,')

# 5. distance_to_layer0_node_sq
new_dist_l0 = """    #[inline]
    fn distance_to_layer0_node_sq(&self, query_f32: &[f32; SLAB_DIM], idx: usize) -> f64 {
        let slab_idx = self.layer0_soa.node_to_slab[idx] as usize;
        let block = slab_idx / SLAB_LANES;
        let lane = slab_idx % SLAB_LANES;
        if self.layer0_soa.blocks.is_empty() {
            return f64::INFINITY;
        }
        let block_ptr = self.layer0_soa.blocks[block].lanes.as_ptr();
        let node_mask = self.nodes[idx].vec.active_mask;
        let base_threshold_sq = self.adaptive_precision_threshold_sq_f32();
        let approx_threshold_sq =
            if base_threshold_sq > 0.0 && (node_mask & GRADE3_GRADE4_MASK) == 0 {
                f32::INFINITY
            } else {
                base_threshold_sq
            };
        let batch = slab_distance(block_ptr, query_f32, approx_threshold_sq);
        let d = batch.distances[lane];
        let approx = f64::from(d);
        let escape = (batch.escape_mask & (1u8 << lane)) != 0;
        let audit = self.should_audit_escape();
        let recall_drop =
            escape & self.compute_recall_drop(audit, d, query_f32, block, lane, block_ptr);
        self.record_escape_result(escape, recall_drop);
        approx
    }"""
content = re.sub(r'#\[inline\]\n\s+fn distance_to_layer0_node_sq\(&self, query_f32: &\[f32; SLAB_DIM\], idx: usize\) -> f64 \{.*?\}', new_dist_l0, content, flags=re.DOTALL)

# 6. compute_recall_drop and layer0_exact_distance_sq
content = content.replace('self.compute_recall_drop(audit, d, query_f32, block, lane, slab_ptr)', 'self.compute_recall_drop(audit, d, query_f32, block, lane, block_ptr)')
content = content.replace(
    'fn compute_recall_drop(\n        &self,\n        audit: bool,\n        d: f32,\n        query_f32: &[f32; SLAB_DIM],\n        block: usize,\n        lane: usize,\n        slab_ptr: *const f32,',
    'fn compute_recall_drop(\n        &self,\n        audit: bool,\n        d: f32,\n        query_f32: &[f32; SLAB_DIM],\n        block: usize,\n        lane: usize,\n        block_ptr: *const f32,'
)
content = content.replace('self.layer0_exact_distance_sq(query_f32, block, lane, slab_ptr)', 'self.layer0_exact_distance_sq(query_f32, block, lane, block_ptr)')
content = content.replace(
    'fn layer0_exact_distance_sq(\n        &self,\n        query_f32: &[f32; SLAB_DIM],\n        block: usize,\n        lane: usize,\n        slab_ptr: *const f32,',
    'fn layer0_exact_distance_sq(\n        &self,\n        query_f32: &[f32; SLAB_DIM],\n        block: usize,\n        lane: usize,\n        block_ptr: *const f32,'
)
# Fix layer0_exact_distance_sq body
content = content.replace('let block_base = block * BLOCK_STRIDE;', 'let block_base = 0;')
content = content.replace('slab_ptr.add(offset)', 'block_ptr.add(offset)')

# 7. search_layer
search_layer_call_pattern = r'let batch =\s+slab_distance\(slab_ptr, block, &query_f32, approx_threshold_sq\);'
content = re.sub(
    search_layer_call_pattern,
    r'let block_ptr = self.layer0_soa.blocks[block].lanes.as_ptr();\n                            let batch = slab_distance(block_ptr, &query_f32, approx_threshold_sq);',
    content
)
# Also fix recall_drop call in search_layer
content = content.replace(
    'audit, d, &query_f32, block, lane, slab_ptr,',
    'audit, d, &query_f32, block, lane, block_ptr,'
)

# 8. remove_node
# Re-read content to make sure we are working on current state
new_remove_slab_logic = """        let blocks = Arc::make_mut(&mut self.layer0_soa.blocks);
        if let Some(block_arc) = blocks.get_mut(block) {
            let block_ref = Arc::make_mut(block_arc);
            block_ref.lanes[lane] = f32::NAN;
        }"""
# Use a more specific pattern for remove_node
content = re.sub(
    r'if let Some\(block_ref\) = self\.layer0_soa\.blocks\.get_mut\(block\) \{\s+block_ref\.lanes\[lane\] = f32::NAN;\s+\}',
    new_remove_slab_logic,
    content,
    flags=re.MULTILINE
)

# 9. layer0_soa() method
content = content.replace(
    '.flat_map(|block| block.lanes)',
    '.flat_map(|block| block.lanes.iter().copied())'
)

# 10. remove layer0_slab_ptr
content = re.sub(r'fn layer0_slab_ptr\(&self\) -> \*const f32 \{.*?\}', '', content, flags=re.DOTALL)

# 11. insert_batch
content = content.replace(
    'self.layer0_soa\n                .blocks\n                .reserve',
    'Arc::make_mut(&mut self.layer0_soa.blocks)\n                .reserve'
)

# 12. Fix tests
content = content.replace(
    'let slab_ptr = graph.layer0_slab_ptr();\n        let query_f32 = HnswGraph::dense_to_query_f32(&query);\n        let distances = slab_distance_scalar(slab_ptr, 0, &query_f32, 0.0);',
    'let block_ptr = graph.layer0_soa.blocks[0].lanes.as_ptr();\n        let query_f32 = HnswGraph::dense_to_query_f32(&query);\n        let distances = slab_distance_scalar(block_ptr, &query_f32, 0.0);'
)

content = content.replace(
    'let slab_ptr = graph.layer0_slab_ptr();\n        let query_f32 = HnswGraph::dense_to_query_f32(&query);\n        let threshold = (BASE_APPROX_PRECISION_THRESHOLD as f32).powi(2);\n        let distances = slab_distance_scalar(slab_ptr, 0, &query_f32, threshold);',
    'let block_ptr = graph.layer0_soa.blocks[0].lanes.as_ptr();\n        let query_f32 = HnswGraph::dense_to_query_f32(&query);\n        let threshold = (BASE_APPROX_PRECISION_THRESHOLD as f32).powi(2);\n        let distances = slab_distance_scalar(block_ptr, &query_f32, threshold);'
)

with open(path, 'w') as f:
    f.write(content)
