
import sys

path = '/home/engine/project/core/genesis-topology/src/hnsw.rs'
with open(path, 'r') as f:
    lines = f.readlines()

def find_line(pattern, start=0):
    for i in range(start, len(lines)):
        if pattern in lines[i]:
            return i
    return -1

# Fix HnswGraph::new (line 1303 approx)
idx = find_line('blocks: Vec::with_capacity(INITIAL_SLAB_BLOCK_CAPACITY),')
if idx != -1:
    lines[idx] = lines[idx].replace('Vec::with_capacity', 'Arc::new(Vec::with_capacity').replace('),', ')),')

# Fix HnswGraph::clone (line 1190 approx)
# lines[1189] = '                blocks: self.layer0_soa.blocks.clone(),\n'
# This is fine as is, but let's be sure.

# Fix apply_delta (line 1367 approx)
# lines[1366] = '                blocks: self.layer0_soa.blocks.clone(),\n'
# This is also fine.

# Fix write_node_to_slab
start = find_line('fn write_node_to_slab(&mut self')
end = find_line('    }', start)
if start != -1 and end != -1:
    new_impl = [
        '    fn write_node_to_slab(&mut self, dense_idx: usize, vec: &SparseCliffordVector) {\n',
        '        let slab_idx = dense_idx;\n',
        '        let block = slab_idx / SLAB_LANES;\n',
        '        let lane = slab_idx % SLAB_LANES;\n',
        '        if block >= self.layer0_soa.blocks.len() {\n',
        '            Arc::make_mut(&mut self.layer0_soa.blocks).push(Arc::new(SlabBlock::zeroed()));\n',
        '        }\n',
        '        let blocks = Arc::make_mut(&mut self.layer0_soa.blocks);\n',
        '        let block_ref = Arc::make_mut(&mut blocks[block]);\n',
        '        let mut d = 0;\n',
        '        while d < SLAB_DIM {\n',
        '            block_ref.lanes[d * SLAB_LANES + lane] = vec.coeffs[d] as f32;\n',
        '            d += 1;\n',
        '        }\n',
        '        let slab_idx_u32 = u32::try_from(slab_idx).unwrap_or(u32::MAX - 1);\n',
        '        debug_assert_ne!(slab_idx_u32, u32::MAX);\n',
        '        Self::cow_vec_mut(&mut self.layer0_soa.node_to_slab).push(slab_idx_u32);\n',
        '    }\n'
    ]
    lines[start:end+1] = new_impl

# Fix remove_node (line 2815 approx)
idx = find_line('if let Some(block_ref) = self.layer0_soa.blocks.get_mut(block) {')
if idx != -1:
    lines[idx:idx+3] = [
        '        let blocks = Arc::make_mut(&mut self.layer0_soa.blocks);\n',
        '        if let Some(block_arc) = blocks.get_mut(block) {\n',
        '            let block_ref = Arc::make_mut(block_arc);\n',
        '            block_ref.lanes[lane] = f32::NAN;\n',
        '        }\n'
    ]

# Fix insert_batch (line 3322 approx)
idx = find_line('if needed_blocks > self.layer0_soa.blocks.len() {')
if idx != -1:
    lines[idx:idx+5] = [
        '        if needed_blocks > self.layer0_soa.blocks.len() {\n',
        '            Arc::make_mut(&mut self.layer0_soa.blocks)\n',
        '                .reserve(needed_blocks - self.layer0_soa.blocks.len());\n',
        '        }\n'
    ]

# Fix layer0_soa() (line 2705 approx)
idx = find_line('slab: self')
if idx != -1 and '.blocks' in lines[idx+2]:
    lines[idx+1:idx+6] = [
        '            slab: self\n',
        '                .layer0_soa\n',
        '                .blocks\n',
        '                .iter()\n',
        '                .flat_map(|block| block.lanes.iter().copied())\n',
        '                .collect(),\n'
    ]

# Fix slab_distance_scalar signature and implementation
start = find_line('fn slab_distance_scalar(')
end = find_line('    }', start + 10) # Find the end of the function
if start != -1:
    lines[start] = 'fn slab_distance_scalar(\n'
    lines[start+1] = '    block_ptr: *const f32,\n'
    lines[start+2] = '    query_f32: &[f32; SLAB_DIM],\n'
    lines[start+3] = '    approx_threshold_sq: f32,\n'
    lines[start+4] = ') -> SlabDistanceBatch {\n'
    
    # Remove block_base
    idx_base = find_line('let block_base = block * BLOCK_STRIDE;', start)
    if idx_base != -1:
        lines[idx_base] = '    // block_ptr points directly to the start of the block\n'
    
    # Replace uses of block_base
    for i in range(start, end + 1):
        lines[i] = lines[i].replace('slab_ptr.add(block_base + lane)', 'block_ptr.add(lane)')
        lines[i] = lines[i].replace('slab_ptr.add(offset)', 'block_ptr.add(offset)')
        lines[i] = lines[i].replace('block_base + ', '')

# Fix slab_distance_avx2 signature and implementation
start = find_line('unsafe fn slab_distance_avx2(')
if start != -1:
    lines[start] = 'unsafe fn slab_distance_avx2(\n'
    lines[start+1] = '    block_ptr: *const f32,\n'
    lines[start+2] = '    query_f32: &[f32; SLAB_DIM],\n'
    lines[start+3] = '    approx_threshold_sq: f32,\n'
    lines[start+4] = ') -> SlabDistanceBatch {\n'
    
    # Remove block_base calc
    idx_base = find_line('let block_base = unsafe { slab_ptr.add(block * BLOCK_STRIDE) };', start)
    if idx_base != -1:
        lines[idx_base] = '    let block_base = block_ptr;\n'
    # The rest of the function uses block_base correctly.

# Fix slab_distance signature
start = find_line('fn slab_distance(')
if start != -1:
    lines[start+1] = '    block_ptr: *const f32,\n'
    lines[start+2] = '    query_f32: &[f32; SLAB_DIM],\n'
    lines[start+3] = '    approx_threshold_sq: f32,\n'
    # Delete line with block
    del lines[start+4]
    
    # Fix calls in the body
    idx_call = find_line('return unsafe { slab_distance_avx2(slab_ptr, block, query_f32, approx_threshold_sq) };', start)
    if idx_call != -1:
        lines[idx_call] = lines[idx_call].replace('slab_ptr, block', 'block_ptr')
    idx_call = find_line('slab_distance_scalar(slab_ptr, block, query_f32, approx_threshold_sq)', start)
    if idx_call != -1:
        lines[idx_call] = lines[idx_call].replace('slab_ptr, block', 'block_ptr')

# Fix distance_to_layer0_node_sq calls
start = find_line('fn distance_to_layer0_node_sq(&self')
if start != -1:
    idx_ptr = find_line('let slab_ptr = self.layer0_slab_ptr();', start)
    if idx_ptr != -1:
        lines[idx_ptr] = '        // block accessed via self.layer0_soa.blocks[block]\n'
    
    idx_call = find_line('let batch = slab_distance(slab_ptr, block, query_f32, approx_threshold_sq);', start)
    if idx_call != -1:
        lines[idx_call] = '        let block_ptr = self.layer0_soa.blocks[block].lanes.as_ptr();\n'
        lines[idx_call+1] = '        let batch = slab_distance(block_ptr, query_f32, approx_threshold_sq);\n'

# Fix search_layer calls
idx_call = find_line('let batch =', find_line('search_layer'))
while idx_call != -1:
    if 'slab_distance(slab_ptr, block, &query_f32, approx_threshold_sq)' in lines[idx_call]:
        # We need to find block. It's group.block
        lines[idx_call-1] = '                            let block_idx = group.block as usize;\n'
        lines[idx_call] = '                            let block_ptr = self.layer0_soa.blocks[block_idx].lanes.as_ptr();\n'
        lines[idx_call+1] = '                            let batch =\n'
        lines[idx_call+2] = '                                slab_distance(block_ptr, &query_f32, approx_threshold_sq);\n'
        # Reset search from current point to find next occurrence (if any)
    idx_call = find_line('let batch =', idx_call + 1)

# Fix test slab_distance_matches_scalar_all_counts
idx_ptr = find_line('let slab_ptr = graph.layer0_slab_ptr();')
if idx_ptr != -1:
    lines[idx_ptr] = '        let block_ptr = graph.layer0_soa.blocks[0].lanes.as_ptr();\n'
    idx_call = find_line('let distances = slab_distance_scalar(slab_ptr, 0, &query_f32, 0.0);', idx_ptr)
    if idx_call != -1:
        lines[idx_call] = '        let distances = slab_distance_scalar(block_ptr, &query_f32, 0.0);\n'

# Fix test slab_distance_uses_scalar_projection_when_high_grade_is_negligible
idx_ptr = find_line('let slab_ptr = graph.layer0_slab_ptr();', idx_ptr + 1)
if idx_ptr != -1:
    lines[idx_ptr] = '        let block_ptr = graph.layer0_soa.blocks[0].lanes.as_ptr();\n'
    idx_call = find_line('let distances = slab_distance_scalar(slab_ptr, 0, &query_f32, threshold);', idx_ptr)
    if idx_call != -1:
        lines[idx_call] = '        let distances = slab_distance_scalar(block_ptr, &query_f32, threshold);\n'

# Remove layer0_slab_ptr
idx_fn = find_line('fn layer0_slab_ptr(&self)')
if idx_fn != -1:
    del lines[idx_fn:idx_fn+7]

# Fix layer0_exact_distance_sq signature and implementation
start = find_line('fn layer0_exact_distance_sq(')
if start != -1:
    lines[start+4] = '        block_ptr: *const f32,\n'
    idx_base = find_line('let block_base = block * BLOCK_STRIDE;', start)
    if idx_base != -1:
        lines[idx_base] = '        let block_base = 0;\n' # Simplest way to fix offsets
    for i in range(start, start + 20):
        lines[i] = lines[i].replace('slab_ptr.add', 'block_ptr.add')

# Fix caller of layer0_exact_distance_sq in compute_recall_drop
idx_call = find_line('let exact = self.layer0_exact_distance_sq(query_f32, block, lane, slab_ptr);')
if idx_call != -1:
    lines[idx_call] = '        let block_ptr = self.layer0_soa.blocks[block].lanes.as_ptr();\n'
    lines[idx_call] = lines[idx_call] + '        let exact = self.layer0_exact_distance_sq(query_f32, block, lane, block_ptr);\n'

# Fix search_layer recall_drop call
idx_call = find_line('audit, d, &query_f32, block, lane, slab_ptr,')
if idx_call != -1:
    lines[idx_call] = '                                        audit, d, &query_f32, block, lane, block_ptr,\n'

with open(path, 'w') as f:

    f.writelines(lines)
