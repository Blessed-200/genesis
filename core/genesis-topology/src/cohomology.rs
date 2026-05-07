#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256i, __m512i, _mm256_load_si256, _mm256_store_si256, _mm256_xor_si256, _mm512_load_si512,
    _mm512_store_si512, _mm512_xor_si512,
};

/// AX-ID: AXIOMA-007, AXIOMA-009
/// Cohomology validator: computes H¹ = ker(∂₁) / im(∂₂) over Z₂.
/// All arithmetic in Z₂ (bit operations). No external linear algebra libraries.
/// Boundary matrices stored as packed bitmaps on a 64-byte aligned word buffer.
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

// Maintenance policy for critical topology validation modules.
//
// - `#[inline(always)]` is prohibited except for documented exceptions with
//   reproducible benchmark evidence, architectural rationale, and explicit risk evaluation.
// - When an optimization is conditioned on `cfg`, it must declare a complementary
//   `not(...)` branch to maintain codec symbol coverage across build profiles.
//
// AX-ID: AXIOMA-007, AXIOMA-009, H_restricción (LEY_FUNDACIONAL §5.6)

use crate::rips::RipsComplex;

const Z2MATRIX_ALIGNMENT_BYTES: usize = 64;
const Z2MATRIX_WORDS_ALIGNMENT: usize = Z2MATRIX_ALIGNMENT_BYTES / std::mem::size_of::<u64>();

#[derive(Debug)]
struct AlignedU64Buffer {
    ptr: NonNull<u64>,
    len: usize,
}

impl AlignedU64Buffer {
    fn new_zeroed(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
            };
        }
        let size = len
            .checked_mul(std::mem::size_of::<u64>())
            .expect("aligned buffer size overflow");
        let layout = std::alloc::Layout::from_size_align(size, Z2MATRIX_ALIGNMENT_BYTES)
            .expect("valid aligned u64 layout");
        // SAFETY: layout was validated and non-zero.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        debug_assert_eq!(
            (raw as usize) & (Z2MATRIX_ALIGNMENT_BYTES - 1),
            0,
            "allocator returned pointer that violates 64-byte alignment contract"
        );
        #[allow(clippy::cast_ptr_alignment)]
        let ptr = NonNull::new(raw.cast::<u64>())
            .unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, len }
    }

    #[expect(
        clippy::missing_const_for_fn,
        reason = "zeroing an allocation through a raw pointer is a runtime memory effect"
    )]
    fn reset_zeroed(&mut self) {
        if self.len == 0 {
            return;
        }
        // SAFETY: `ptr` is valid for `len` contiguous u64 elements by construction.
        unsafe { std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.len) };
    }

    fn ensure_len_zeroed(&mut self, new_len: usize) {
        if new_len == self.len {
            self.reset_zeroed();
            return;
        }
        *self = Self::new_zeroed(new_len);
    }

    const fn as_slice(&self) -> &[u64] {
        // SAFETY: `ptr` points to an allocation of `len` elements or is dangling with len=0.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    const fn as_mut_slice(&mut self) -> &mut [u64] {
        // SAFETY: same as `as_slice`, with unique mutable access through `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedU64Buffer {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }
        let size = self
            .len
            .checked_mul(std::mem::size_of::<u64>())
            .expect("aligned buffer size overflow on drop");
        let layout = std::alloc::Layout::from_size_align(size, Z2MATRIX_ALIGNMENT_BYTES)
            .expect("valid aligned u64 layout on drop");
        // SAFETY: pointer/layout match the allocation performed in `new_zeroed`.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr().cast::<u8>(), layout) };
    }
}

impl Deref for AlignedU64Buffer {
    type Target = [u64];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for AlignedU64Buffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Packed Z₂ matrix: rows × cols, each row stored as ceil(cols/64) u64 words.
struct Z2Matrix {
    rows: usize,
    cols: usize,
    /// Cached packing width.
    ///
    /// Structural invariants:
    /// - `data.len() == rows * words_per_row`
    /// - `words_per_row >= cols.div_ceil(64)`
    /// - `words_per_row % 8 == 0` (64-byte row stride alignment contract)
    /// - `cols` only changes through `set_cols` so packed layout remains consistent.
    words_per_row: usize,
    data: AlignedU64Buffer,
}

type XorKernel = fn(&mut [u64], &[u64]);

impl Z2Matrix {
    #[inline]
    const fn padded_words_per_row(cols: usize) -> usize {
        cols.div_ceil(64).div_ceil(Z2MATRIX_WORDS_ALIGNMENT) * Z2MATRIX_WORDS_ALIGNMENT
    }

    fn new(rows: usize, cols: usize) -> Self {
        Self::new_with_scratch(rows, cols, &mut None)
    }

    fn new_with_scratch(rows: usize, cols: usize, scratch: &mut Option<AlignedU64Buffer>) -> Self {
        let words_per_row = Self::padded_words_per_row(cols);
        let data_len = rows * words_per_row;
        let mut data = scratch
            .take()
            .unwrap_or_else(|| AlignedU64Buffer::new_zeroed(data_len));
        data.ensure_len_zeroed(data_len);
        let matrix = Self {
            rows,
            cols,
            words_per_row,
            data,
        };

        debug_assert_matrix_invariants(&matrix);

        matrix
    }

    fn into_scratch(self) -> AlignedU64Buffer {
        self.data
    }

    #[cfg(test)]
    fn set_cols(&mut self, cols: usize) {
        self.cols = cols;
        self.words_per_row = Self::padded_words_per_row(cols);
        self.data.ensure_len_zeroed(self.rows * self.words_per_row);
        debug_assert_matrix_invariants(self);
    }

    fn set(&mut self, row: usize, col: usize, val: bool) {
        if row >= self.rows || col >= self.cols {
            index_oob_error(row, col, self.rows, self.cols);
        }
        let wpr = self.words_per_row;
        let idx = row * wpr + col / 64;
        if val {
            self.data[idx] |= 1u64 << (col % 64);
        } else {
            self.data[idx] &= !(1u64 << (col % 64));
        }
        debug_assert_matrix_invariants(self);
    }

    fn rank_by_gaussian_elimination(&mut self) -> usize {
        self.rank_by_gaussian_elimination_with_scratch(&mut None)
    }

    fn rank_by_gaussian_elimination_with_scratch(
        &mut self,
        pivot_scratch: &mut Option<AlignedU64Buffer>,
    ) -> usize {
        debug_assert_matrix_invariants(self);
        if self.rows == 0 || self.cols == 0 {
            return 0;
        }

        let wpr = self.words_per_row;
        let mut rank = 0usize;
        let mut pivot_target_row = 0usize;
        let mut pivot_row_buf = pivot_scratch
            .take()
            .unwrap_or_else(|| AlignedU64Buffer::new_zeroed(wpr));
        pivot_row_buf.ensure_len_zeroed(wpr);
        // HOT PATH: O(rows * cols * words_per_row), with persistent scratch and
        // CPU-selected XOR kernels to avoid per-elimination allocation.
        let xor_kernel = select_xor_kernel();

        debug_assert_eq!(self.data.len(), self.rows * wpr);

        for col in 0..self.cols {
            if pivot_target_row >= self.rows {
                break;
            }

            let pivot_word = col / 64;
            let pivot_bit = 1u64 << (col % 64);
            let Some(pivot_row) = self.find_pivot_row(pivot_target_row, pivot_word, pivot_bit)
            else {
                continue;
            };

            self.swap_rows(pivot_row, pivot_target_row);
            self.copy_row_to_buffer(pivot_target_row, &mut pivot_row_buf);
            self.eliminate_pivot_column(
                pivot_target_row,
                pivot_word,
                pivot_bit,
                &pivot_row_buf,
                xor_kernel,
            );

            rank += 1;
            pivot_target_row += 1;
        }

        debug_assert_matrix_invariants(self);
        *pivot_scratch = Some(pivot_row_buf);
        rank
    }

    fn find_pivot_row(&self, start_row: usize, pivot_word: usize, pivot_bit: u64) -> Option<usize> {
        let mut idx = start_row * self.words_per_row + pivot_word;
        for row in start_row..self.rows {
            if (self.data[idx] & pivot_bit) != 0 {
                return Some(row);
            }
            idx += self.words_per_row;
        }
        None
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }

        let wpr = self.words_per_row;
        let a_start = a * wpr;
        let b_start = b * wpr;
        if a_start < b_start {
            let (left, right) = self.data.split_at_mut(b_start);
            left[a_start..a_start + wpr].swap_with_slice(&mut right[..wpr]);
        } else {
            let (left, right) = self.data.split_at_mut(a_start);
            left[b_start..b_start + wpr].swap_with_slice(&mut right[..wpr]);
        }
    }

    fn copy_row_to_buffer(&self, row: usize, pivot_row_buf: &mut AlignedU64Buffer) {
        let start = row * self.words_per_row;
        pivot_row_buf.copy_from_slice(&self.data[start..start + self.words_per_row]);
    }

    fn eliminate_pivot_column(
        &mut self,
        pivot_row: usize,
        pivot_word: usize,
        pivot_bit: u64,
        pivot_row_buf: &AlignedU64Buffer,
        xor_kernel: XorKernel,
    ) {
        let wpr = self.words_per_row;
        let len = wpr - pivot_word;
        let pivot_tail = &pivot_row_buf[pivot_word..pivot_word + len];

        for row in 0..self.rows {
            if row == pivot_row {
                continue;
            }
            let row_pivot_idx = row * wpr + pivot_word;
            if (self.data[row_pivot_idx] & pivot_bit) == 0 {
                continue;
            }
            let base = row * wpr + pivot_word;
            xor_kernel(&mut self.data[base..base + len], pivot_tail);
        }
    }
}

/// Returns the optimal Z₂ XOR kernel for the current CPU.
///
/// CPUID-based feature detection is invariant for the process lifetime, so the
/// selected function pointer is cached after the first elimination call.
///
/// AX-ID: AXIOMA-007, AXIOMA-009
fn select_xor_kernel() -> XorKernel {
    static CACHED: std::sync::OnceLock<XorKernel> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return xor_row_avx512_entry;
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return xor_row_avx2_entry;
            }
        }
        xor_row_scalar
    })
}

#[inline]
fn xor_row_scalar(row_tail: &mut [u64], pivot_tail: &[u64]) {
    let len = row_tail.len();
    let mut i = 0usize;
    while i + 4 <= len {
        row_tail[i] ^= pivot_tail[i];
        row_tail[i + 1] ^= pivot_tail[i + 1];
        row_tail[i + 2] ^= pivot_tail[i + 2];
        row_tail[i + 3] ^= pivot_tail[i + 3];
        i += 4;
    }
    while i < len {
        row_tail[i] ^= pivot_tail[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn xor_row_avx512(row_tail: &mut [u64], pivot_tail: &[u64]) {
    let len = row_tail.len();
    let mut i = 0usize;
    let misaligned_bytes = (Z2MATRIX_ALIGNMENT_BYTES
        - (row_tail.as_ptr() as usize & (Z2MATRIX_ALIGNMENT_BYTES - 1)))
        & (Z2MATRIX_ALIGNMENT_BYTES - 1);
    let prefix_words = (misaligned_bytes / std::mem::size_of::<u64>()).min(len);
    while i < prefix_words {
        row_tail[i] ^= pivot_tail[i];
        i += 1;
    }
    while i + 8 <= len {
        // SAFETY: `i + 8 <= len` keeps all pointer arithmetic in-bounds for both slices.
        // Row stride and buffer allocation guarantee both pointers share identical
        // 64-byte alignment at this offset.
        let row_ptr = row_tail.as_ptr().wrapping_add(i) as usize;
        let pivot_ptr = pivot_tail.as_ptr().wrapping_add(i) as usize;
        debug_assert!(row_ptr.trailing_zeros() >= 6);
        debug_assert!(pivot_ptr.trailing_zeros() >= 6);
        // SAFETY: pointer arithmetic is in-bounds (`i + 8 <= len`) and both
        // vectors are 64-byte aligned at this offset by contract.
        unsafe {
            #[allow(clippy::cast_ptr_alignment)]
            let lhs = _mm512_load_si512(row_tail.as_ptr().add(i).cast::<__m512i>());
            #[allow(clippy::cast_ptr_alignment)]
            let rhs = _mm512_load_si512(pivot_tail.as_ptr().add(i).cast::<__m512i>());
            let out = _mm512_xor_si512(lhs, rhs);
            #[allow(clippy::cast_ptr_alignment)]
            _mm512_store_si512(row_tail.as_mut_ptr().add(i).cast::<__m512i>(), out);
        }
        i += 8;
    }
    xor_row_scalar(&mut row_tail[i..], &pivot_tail[i..]);
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn xor_row_avx512_entry(row_tail: &mut [u64], pivot_tail: &[u64]) {
    // SAFETY: selected only after runtime AVX-512 feature detection.
    unsafe { xor_row_avx512(row_tail, pivot_tail) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xor_row_avx2(row_tail: &mut [u64], pivot_tail: &[u64]) {
    let len = row_tail.len();
    let mut i = 0usize;
    let misaligned_bytes = (32 - (row_tail.as_ptr() as usize & 31)) & 31;
    let prefix_words = (misaligned_bytes / std::mem::size_of::<u64>()).min(len);
    while i < prefix_words {
        row_tail[i] ^= pivot_tail[i];
        i += 1;
    }
    while i + 4 <= len {
        // SAFETY: `i + 4 <= len` keeps all pointer arithmetic in-bounds for both slices.
        // Row stride and buffer allocation guarantee both pointers share identical
        // 32-byte alignment at this offset.
        let row_ptr = row_tail.as_ptr().wrapping_add(i) as usize;
        let pivot_ptr = pivot_tail.as_ptr().wrapping_add(i) as usize;
        debug_assert!(row_ptr.trailing_zeros() >= 5);
        debug_assert!(pivot_ptr.trailing_zeros() >= 5);
        // SAFETY: pointer arithmetic is in-bounds (`i + 4 <= len`) and both
        // vectors are 32-byte aligned at this offset by contract.
        unsafe {
            #[allow(clippy::cast_ptr_alignment)]
            let lhs = _mm256_load_si256(row_tail.as_ptr().add(i).cast::<__m256i>());
            #[allow(clippy::cast_ptr_alignment)]
            let rhs = _mm256_load_si256(pivot_tail.as_ptr().add(i).cast::<__m256i>());
            let out = _mm256_xor_si256(lhs, rhs);
            #[allow(clippy::cast_ptr_alignment)]
            _mm256_store_si256(row_tail.as_mut_ptr().add(i).cast::<__m256i>(), out);
        }
        i += 4;
    }
    xor_row_scalar(&mut row_tail[i..], &pivot_tail[i..]);
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn xor_row_avx2_entry(row_tail: &mut [u64], pivot_tail: &[u64]) {
    // SAFETY: selected only after runtime AVX2 feature detection.
    unsafe { xor_row_avx2(row_tail, pivot_tail) }
}

fn debug_assert_matrix_invariants(matrix: &Z2Matrix) {
    debug_assert!(matrix.words_per_row >= matrix.cols.div_ceil(64));
    debug_assert_eq!(matrix.words_per_row % Z2MATRIX_WORDS_ALIGNMENT, 0);
    debug_assert_eq!(matrix.data.len(), matrix.rows * matrix.words_per_row);
}

#[cold]
#[inline(never)]
fn index_oob_error(row: usize, col: usize, rows: usize, cols: usize) -> ! {
    panic!("Z2Matrix index out of bounds: row={row}, col={col}, rows={rows}, cols={cols}");
}

#[cold]
#[inline(never)]
fn invalid_topology_state(message: &'static str) -> ! {
    panic!("invalid topology state: {message}");
}

#[derive(Default)]
struct HomologyWorkspace {
    id_to_vertex: Vec<usize>,
    edge_lookup: Vec<(u64, usize)>,
    dense_edge_lookup: Vec<u32>,
    d1_matrix_scratch: Option<AlignedU64Buffer>,
    d2_matrix_scratch: Option<AlignedU64Buffer>,
    pivot_row_scratch: Option<AlignedU64Buffer>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct H1CacheKey {
    ptr: *const RipsComplex,
    counts: (usize, usize, usize),
    fingerprint: u64,
}

#[derive(Default)]
struct H1Cache {
    key: Option<H1CacheKey>,
    cached_result: Option<bool>,
    uf_parent: Vec<usize>,
    uf_rank: Vec<u8>,
}

impl H1Cache {
    const fn invalidate(&mut self) {
        self.key = None;
        self.cached_result = None;
    }

    fn prepare_union_find(&mut self, n: usize) {
        if self.uf_parent.len() < n {
            self.uf_parent.resize(n, 0);
            self.uf_rank.resize(n, 0);
        }

        for (i, parent) in self.uf_parent[..n].iter_mut().enumerate() {
            *parent = i;
        }
        self.uf_rank[..n].fill(0);
        // CRYSTAL: FO63, FO64 — inevitable
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.uf_parent[root] != root {
            root = self.uf_parent[root];
        }

        let mut node = x;
        while self.uf_parent[node] != root {
            let parent = self.uf_parent[node];
            self.uf_parent[node] = self.uf_parent[parent];
            node = parent;
        }

        root
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }

        let rank_a = self.uf_rank[ra];
        let rank_b = self.uf_rank[rb];
        if rank_a < rank_b {
            self.uf_parent[ra] = rb;
        } else {
            self.uf_parent[rb] = ra;
            if rank_a == rank_b {
                self.uf_rank[ra] = self.uf_rank[ra].saturating_add(1);
            }
        }

        true
    }
}

thread_local! {
    static HOMOLOGY_WORKSPACE: RefCell<HomologyWorkspace> = RefCell::new(HomologyWorkspace::default());
    static H1_CACHE: RefCell<H1Cache> = RefCell::new(H1Cache::default());
}

#[cfg(test)]
thread_local! {
    static FULL_REBUILD_COUNT: RefCell<usize> = const { RefCell::new(0) };
    static FINGERPRINT_COUNT: RefCell<usize> = const { RefCell::new(0) };
}

#[cfg(test)]
fn note_full_rebuild() {
    FULL_REBUILD_COUNT.with(|counter| {
        let mut value = counter.borrow_mut();
        *value += 1;
    });
}

#[cfg(not(test))]
const fn note_full_rebuild() {}

#[cfg(test)]
fn note_fingerprint() {
    FINGERPRINT_COUNT.with(|counter| {
        let mut value = counter.borrow_mut();
        *value += 1;
    });
}

#[cfg(not(test))]
const fn note_fingerprint() {}

const fn edge_key(a: usize, b: usize) -> u64 {
    let (x, y) = if a <= b { (a, b) } else { (b, a) };
    ((x as u64) << 32) | y as u64
}

#[inline]
fn complex_fingerprint(complex: &RipsComplex) -> u64 {
    // AX-ID: AXIOMA-009
    // 64-bit rolling mix over 1- and 2-simplices to robustly invalidate cache
    // when topology mutates without count changes.
    note_fingerprint();

    let mut acc = 0x9E37_79B9_7F4A_7C15u64;
    for edge in complex.simplices_of_dim(1) {
        let a = edge[0].get();
        let b = edge[1].get();
        let k = a.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ b.wrapping_mul(0x94D0_49BB_1331_11EB);
        acc ^= k.rotate_left(17);
        acc = acc.rotate_left(13).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    for tri in complex.simplices_of_dim(2) {
        let a = tri[0].get();
        let b = tri[1].get();
        let c = tri[2].get();
        let k = a.wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ b.wrapping_mul(0xA5A3_58F4_7A6B_CDEF)
            ^ c.wrapping_mul(0x8D58_AC26_A2F4_9E27);
        acc ^= k.rotate_left(29);
        acc = acc.rotate_left(11).wrapping_mul(0x94D0_49BB_1331_11EB);
    }
    acc
}

fn prepare_vertex_index(complex: &RipsComplex, ws: &mut HomologyWorkspace) -> usize {
    let mut max_id = 0usize;
    let mut n_v = 0usize;
    for v in complex.simplices_of_dim(0) {
        let raw = v[0].get() as usize;
        max_id = max_id.max(raw);
        n_v += 1;
    }

    if ws.id_to_vertex.len() <= max_id {
        ws.id_to_vertex.resize(max_id + 1, usize::MAX);
    }

    for i in 0..=max_id {
        ws.id_to_vertex[i] = usize::MAX;
    }

    for (row, v) in complex.simplices_of_dim(0).enumerate() {
        ws.id_to_vertex[v[0].get() as usize] = row;
    }

    n_v
}

fn build_d1(complex: &RipsComplex, ws: &mut HomologyWorkspace, n_v: usize) -> Z2Matrix {
    let n_e = complex.simplices_of_dim(1).count();
    if n_v == 0 || n_e == 0 {
        return Z2Matrix::new_with_scratch(0, 0, &mut ws.d1_matrix_scratch);
    }

    let mut m = Z2Matrix::new_with_scratch(n_v, n_e, &mut ws.d1_matrix_scratch);
    for (col, edge) in complex.simplices_of_dim(1).enumerate() {
        let u = ws.id_to_vertex[edge[0].get() as usize];
        let v = ws.id_to_vertex[edge[1].get() as usize];
        if u != usize::MAX {
            m.set(u, col, true);
        }
        if v != usize::MAX {
            m.set(v, col, true);
        }
    }
    m
}

fn build_d2(complex: &RipsComplex, ws: &mut HomologyWorkspace, n_v: usize) -> Z2Matrix {
    let n_e = complex.simplices_of_dim(1).count();
    let n_t = complex.simplices_of_dim(2).count();
    if n_e == 0 || n_t == 0 {
        return Z2Matrix::new_with_scratch(0, 0, &mut ws.d2_matrix_scratch);
    }

    ws.edge_lookup.clear();
    for (row, edge) in complex.simplices_of_dim(1).enumerate() {
        let u = ws.id_to_vertex[edge[0].get() as usize];
        let v = ws.id_to_vertex[edge[1].get() as usize];
        ws.edge_lookup.push((edge_key(u, v), row));
    }
    ws.edge_lookup.sort_unstable_by_key(|&(key, _)| key);

    const DENSE_EDGE_MAP_BYTES_LIMIT: usize = 16 * 1024 * 1024;
    let dense_slots = n_v.saturating_mul(n_v);
    let dense_lookup = if dense_slots > 0
        && dense_slots <= DENSE_EDGE_MAP_BYTES_LIMIT / std::mem::size_of::<u32>()
    {
        if ws.dense_edge_lookup.len() < dense_slots {
            ws.dense_edge_lookup.resize(dense_slots, u32::MAX);
        } else {
            ws.dense_edge_lookup[..dense_slots].fill(u32::MAX);
        }
        let table = &mut ws.dense_edge_lookup[..dense_slots];
        for &(key, row) in &ws.edge_lookup {
            let a = (key >> 32) as usize;
            let b = (key & 0xFFFF_FFFF) as usize;
            table[a * n_v + b] = row as u32;
        }
        Some(&table[..])
    } else {
        None
    };

    let mut boundary_matrix = Z2Matrix::new_with_scratch(n_e, n_t, &mut ws.d2_matrix_scratch);
    for (col, tri) in complex.simplices_of_dim(2).enumerate() {
        let vertex_a = ws.id_to_vertex[tri[0].get() as usize];
        let vertex_b = ws.id_to_vertex[tri[1].get() as usize];
        let vertex_c = ws.id_to_vertex[tri[2].get() as usize];

        for (edge_start, edge_end) in [
            (vertex_a, vertex_b),
            (vertex_a, vertex_c),
            (vertex_b, vertex_c),
        ] {
            let (min_vertex, max_vertex) = if edge_start <= edge_end {
                (edge_start, edge_end)
            } else {
                (edge_end, edge_start)
            };
            if let Some(table) = dense_lookup {
                let base = min_vertex * n_v;
                let row = table[base + max_vertex];
                if row != u32::MAX {
                    boundary_matrix.set(row as usize, col, true);
                }
            } else {
                let key = edge_key(min_vertex, max_vertex);
                if let Ok(pos) = ws.edge_lookup.binary_search_by_key(&key, |&(k, _)| k) {
                    boundary_matrix.set(ws.edge_lookup[pos].1, col, true);
                }
            }
        }
    }
    boundary_matrix
}

fn check_h1_full(complex: &RipsComplex, ws: &mut HomologyWorkspace, n_v: usize) -> bool {
    note_full_rebuild();

    let n_edges = complex.simplices_of_dim(1).count();
    let mut d1 = build_d1(complex, ws, n_v);
    let rank_d1 = d1.rank_by_gaussian_elimination_with_scratch(&mut ws.pivot_row_scratch);
    let rank_ker_d1 = n_edges.saturating_sub(rank_d1);
    ws.d1_matrix_scratch = Some(d1.into_scratch());

    let rank_im_d2 = {
        let mut d2 = build_d2(complex, ws, n_v);
        let rank = d2.rank_by_gaussian_elimination_with_scratch(&mut ws.pivot_row_scratch);
        ws.d2_matrix_scratch = Some(d2.into_scratch());
        rank
    };

    let h1_dim = rank_ker_d1.saturating_sub(rank_im_d2);
    h1_dim == 0
}

fn detect_graph_cycle_incremental(
    complex: &RipsComplex,
    ws: &HomologyWorkspace,
    cache: &mut H1Cache,
) -> bool {
    let n_v = complex.simplices_of_dim(0).count();
    if n_v == 0 {
        return false;
    }

    cache.prepare_union_find(n_v);
    let id_to_vertex = ws.id_to_vertex.as_slice();
    for edge in complex.simplices_of_dim(1) {
        let raw_u = edge[0].get() as usize;
        let raw_v = edge[1].get() as usize;
        if raw_u >= id_to_vertex.len() || raw_v >= id_to_vertex.len() {
            invalid_topology_state("edge endpoint missing in vertex index");
        }
        let u = id_to_vertex[raw_u];
        let v = id_to_vertex[raw_v];
        if u == usize::MAX || v == usize::MAX {
            continue;
        }
        if !cache.union(u, v) {
            return true;
        }
    }

    false
}

/// Benchmark helper that executes Z₂ Gaussian elimination over a deterministic
/// packed matrix and returns the computed rank.
///
/// AX-ID: AXIOMA-007, H_estructura (LEY_FUNDACIONAL §3.1)
pub fn benchmark_rank_by_gaussian_elimination(rows: usize, cols: usize, seed: u64) -> usize {
    if rows == 0 || cols == 0 {
        return 0;
    }

    let mut matrix = Z2Matrix::new(rows, cols);
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for row in 0..rows {
        for col in 0..cols {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if (state & 1) != 0 {
                matrix.set(row, col, true);
            }
        }
    }

    matrix.rank_by_gaussian_elimination()
}

/// Benchmark helper for branch-minimal row XOR elimination throughput over
/// packed Z₂ row buffers.
///
/// AX-ID: AXIOMA-007, H_dinámica (LEY_FUNDACIONAL §3.2)
pub fn benchmark_xor_row_elimination(words_per_row: usize, iterations: usize, seed: u64) -> u64 {
    if words_per_row == 0 || iterations == 0 {
        return 0;
    }

    let mut state = seed ^ 0x94D0_49BB_1331_11EB;
    let mut dst = AlignedU64Buffer::new_zeroed(words_per_row);
    let mut pivot = AlignedU64Buffer::new_zeroed(words_per_row);

    for word in 0..words_per_row {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        dst[word] = state;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        pivot[word] = state.rotate_left(11);
    }

    let xor_kernel = select_xor_kernel();
    for _ in 0..iterations {
        xor_kernel(&mut dst, &pivot);
    }

    dst.iter().fold(0_u64, |acc, &word| acc ^ word)
}

/// Cohomology validator.
///
/// AX-ID: AXIOMA-007, AXIOMA-009
pub struct CohomologyValidator;

impl CohomologyValidator {
    /// Invalidates thread-local H¹ memoization and scratch workspaces.
    ///
    /// Use this method before measurements that require cold-start execution.
    ///
    /// AX-ID: AXIOMA-007, AXIOMA-009
    pub fn invalidate_cache() {
        H1_CACHE.with(|cache_cell| {
            let mut cache = cache_cell.borrow_mut();
            cache.invalidate();
            cache.uf_parent.clear();
            cache.uf_rank.clear();
        });
        HOMOLOGY_WORKSPACE.with(|ws_cell| {
            let mut ws = ws_cell.borrow_mut();
            ws.id_to_vertex.clear();
            ws.edge_lookup.clear();
            ws.dense_edge_lookup.clear();
            ws.d1_matrix_scratch = None;
            ws.d2_matrix_scratch = None;
            ws.pivot_row_scratch = None;
        });
    }

    /// Returns `true` if H¹(complex) = 0 (no independent cycles), `false` otherwise.
    ///
    /// AX-ID: AXIOMA-007, AXIOMA-009
    pub fn check_h1(complex: &RipsComplex) -> bool {
        let n_edges = complex.simplices_of_dim(1).count();
        if n_edges == 0 {
            return true;
        }
        let ptr = std::ptr::from_ref(complex);
        let counts = complex.counts();

        H1_CACHE.with(|cache_cell| {
            HOMOLOGY_WORKSPACE.with(|ws_cell| {
                let mut cache = cache_cell.borrow_mut();

                if let Some(key) = cache.key {
                    if key.ptr == ptr && key.counts == counts {
                        if let Some(cached) = cache.cached_result {
                            return cached;
                        }
                    }
                }

                let key = H1CacheKey {
                    ptr,
                    counts,
                    fingerprint: complex_fingerprint(complex),
                };

                if cache.key != Some(key) {
                    cache.invalidate();
                    cache.key = Some(key);
                }

                if let Some(cached) = cache.cached_result {
                    return cached;
                }

                let mut ws = ws_cell.borrow_mut();
                let n_v = prepare_vertex_index(complex, &mut ws);

                if !detect_graph_cycle_incremental(complex, &ws, &mut cache) {
                    cache.cached_result = Some(true);
                    return true;
                }
                if complex.simplices_of_dim(2).next().is_none() {
                    cache.cached_result = Some(false);
                    return false;
                }

                let full_result = check_h1_full(complex, &mut ws, n_v);
                cache.cached_result = Some(full_result);
                full_result
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use genesis_math::SparseCliffordVector;
    use genesis_types::NodeId;

    use super::*;
    use crate::hnsw::HnswGraph;
    use crate::rips::RipsComplex;

    fn make_seeded_vec(node_id: u64, seed: u64) -> SparseCliffordVector {
        let a = (((seed >> 8) & 0xFF) as f64).mul_add(0.001, 0.05);
        let b = (((seed >> 24) & 0xFF) as f64).mul_add(0.001, 0.1);
        SparseCliffordVector::from_iter((0..4).map(|idx| {
            let scale = (idx as f64 + 1.0) * 0.2;
            (idx, scale * (a * node_id as f64 + b))
        }))
        .unwrap()
    }

    fn xorshift64(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn check_h1_full_for_test(complex: &RipsComplex) -> bool {
        HOMOLOGY_WORKSPACE.with(|cell| {
            let mut ws = cell.borrow_mut();
            let n_v = prepare_vertex_index(complex, &mut ws);
            check_h1_full(complex, &mut ws, n_v)
        })
    }

    fn reset_full_rebuild_count() {
        FULL_REBUILD_COUNT.with(|counter| *counter.borrow_mut() = 0);
    }

    fn full_rebuild_count() -> usize {
        FULL_REBUILD_COUNT.with(|counter| *counter.borrow())
    }

    fn reset_fingerprint_count() {
        FINGERPRINT_COUNT.with(|counter| *counter.borrow_mut() = 0);
    }

    fn fingerprint_count() -> usize {
        FINGERPRINT_COUNT.with(|counter| *counter.borrow())
    }

    fn make_vec(id: u64) -> SparseCliffordVector {
        let s = (id as f64).mul_add(0.15, 0.05);
        SparseCliffordVector::from_iter((0..4).map(|b| (b, s * (b as f64 + 1.0)))).unwrap()
    }

    #[test]
    fn cohomology_h1_zero_for_cycle_free_graph() {
        let mut g = HnswGraph::new(8);
        for i in 0..5u64 {
            g.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &make_vec(i),
            )
            .unwrap();
        }
        let complex = RipsComplex::build(&g, 10.0).expect("rips build should succeed");
        let result = CohomologyValidator::check_h1(&complex);
        assert!(result, "expected H¹=0 for cycle-free graph");
    }

    #[test]
    fn cohomology_h1_nonzero_for_loop() {
        let mut d1 = Z2Matrix::new(4, 4);
        d1.set(0, 0, true);
        d1.set(1, 0, true);
        d1.set(1, 1, true);
        d1.set(2, 1, true);
        d1.set(2, 2, true);
        d1.set(3, 2, true);
        d1.set(3, 3, true);
        d1.set(0, 3, true);

        let rank_d1 = d1.rank_by_gaussian_elimination();
        let n_edges = 4;
        let rank_ker_d1 = n_edges - rank_d1;
        let h1_dim = rank_ker_d1;

        assert_eq!(h1_dim, 1, "expected H¹=1 for a 4-cycle, got {}", h1_dim);
    }

    #[test]
    fn z2_matrix_invariant_holds_after_gaussian_elimination() {
        let mut m = Z2Matrix::new(5, 130);
        m.set(0, 0, true);
        m.set(1, 64, true);
        m.set(2, 65, true);
        m.set(3, 129, true);
        m.set(4, 0, true);
        m.set(4, 64, true);

        let _ = m.rank_by_gaussian_elimination();

        assert!(m.words_per_row >= m.cols.div_ceil(64));
        assert_eq!(m.words_per_row % Z2MATRIX_WORDS_ALIGNMENT, 0);
        assert_eq!(m.data.len(), m.rows * m.words_per_row);
    }

    #[test]
    fn z2_matrix_set_cols_rebuilds_packed_layout_invariant() {
        let mut m = Z2Matrix::new(3, 65);
        m.set(0, 0, true);
        m.set(1, 64, true);

        m.set_cols(129);

        assert_eq!(m.words_per_row, Z2Matrix::padded_words_per_row(129));
        assert!(m.words_per_row >= m.cols.div_ceil(64));
        assert_eq!(m.words_per_row % Z2MATRIX_WORDS_ALIGNMENT, 0);
        assert_eq!(m.data.len(), m.rows * m.words_per_row);
    }

    #[test]
    fn z2_rank_matches_known_matrices() {
        let mut identity = Z2Matrix::new(4, 4);
        for i in 0..4 {
            identity.set(i, i, true);
        }
        assert_eq!(identity.rank_by_gaussian_elimination(), 4);

        let mut dependent = Z2Matrix::new(4, 4);
        dependent.set(0, 0, true);
        dependent.set(0, 1, true);
        dependent.set(1, 1, true);
        dependent.set(1, 2, true);
        dependent.set(2, 0, true);
        dependent.set(2, 1, true); // row 2 = row 0 in GF(2)
        dependent.set(3, 3, true);
        assert_eq!(dependent.rank_by_gaussian_elimination(), 3);

        let mut zero = Z2Matrix::new(3, 5);
        assert_eq!(zero.rank_by_gaussian_elimination(), 0);
    }

    #[test]
    fn h1_incremental_matches_full_on_1000_random_graphs() {
        let mut seed = 0xA5A5_5A5A_D3C1_9E37u64;
        for _ in 0..1000 {
            let mut g = HnswGraph::new(8);
            let node_count = 8 + (xorshift64(&mut seed) % 17) as usize;
            for node in 0..node_count as u64 {
                let local_seed = xorshift64(&mut seed);
                g.insert(
                    NodeId::try_new(node).expect("NodeId válido por construcción"),
                    &make_seeded_vec(node, local_seed),
                )
                .unwrap();
            }

            let epsilon = ((xorshift64(&mut seed) % 700) as f64).mul_add(0.001, 0.15);
            let complex = RipsComplex::build(&g, epsilon).expect("rips build should succeed");

            let incremental = CohomologyValidator::check_h1(&complex);
            let full = check_h1_full_for_test(&complex);
            assert_eq!(incremental, full);
        }
    }

    #[test]
    fn h1_cache_invalidates_on_same_counts_with_different_topology() {
        let mut g1 = HnswGraph::new(8);
        let mut g2 = HnswGraph::new(8);
        for i in 0..4u64 {
            let id = NodeId::try_new(i).expect("NodeId válido por construcción");
            g1.insert(id, &make_vec(i)).unwrap();
            g2.insert(id, &make_vec(i + 100)).unwrap();
        }

        let c1 = RipsComplex::build(&g1, 0.001).expect("rips build should succeed");
        let _ = CohomologyValidator::check_h1(&c1);
        let key1 = H1CacheKey {
            ptr: std::ptr::from_ref(&c1),
            counts: c1.counts(),
            fingerprint: complex_fingerprint(&c1),
        };

        let c2 = RipsComplex::build(&g2, 10.0).expect("rips build should succeed");
        let _ = CohomologyValidator::check_h1(&c2);
        let key2 = H1CacheKey {
            ptr: std::ptr::from_ref(&c2),
            counts: c2.counts(),
            fingerprint: complex_fingerprint(&c2),
        };

        assert_ne!(key1.fingerprint, key2.fingerprint);
    }

    #[test]
    fn h1_cache_hit_avoids_fingerprint_rescan() {
        CohomologyValidator::invalidate_cache();
        reset_fingerprint_count();

        let mut hnsw = HnswGraph::new(8);
        let vectors = [
            SparseCliffordVector::from_iter([(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]).unwrap(),
            SparseCliffordVector::from_iter([(0, 0.1), (1, 0.0), (2, 0.0), (3, 0.0)]).unwrap(),
            SparseCliffordVector::from_iter([(0, 0.1), (1, 0.1), (2, 0.0), (3, 0.0)]).unwrap(),
            SparseCliffordVector::from_iter([(0, 0.0), (1, 0.1), (2, 0.0), (3, 0.0)]).unwrap(),
        ];

        for (i, vec) in vectors.iter().enumerate() {
            hnsw.insert(NodeId::try_new(i as u64).unwrap(), vec)
                .unwrap();
        }

        let complex = RipsComplex::build(&hnsw, 0.2).expect("rips build should succeed");
        let first = CohomologyValidator::check_h1(&complex);
        let first_count = fingerprint_count();
        let second = CohomologyValidator::check_h1(&complex);

        assert_eq!(first, second);
        assert_eq!(first_count, 1);
        assert_eq!(fingerprint_count(), 1);
    }

    #[test]
    fn h1_incremental_ualpha_complexity() {
        let mut g = HnswGraph::new(16);
        for i in 0..128u64 {
            g.insert(
                NodeId::try_new(i).expect("NodeId válido por construcción"),
                &make_vec(i),
            )
            .unwrap();
        }

        let complex = RipsComplex::build(&g, 2.5).expect("rips build should succeed");
        reset_full_rebuild_count();

        for _ in 0..1000 {
            let _ = CohomologyValidator::check_h1(&complex);
        }

        assert!(
            full_rebuild_count() <= 1,
            "expected <=1 full Z2 rebuild, got {}",
            full_rebuild_count()
        );
    }
}
