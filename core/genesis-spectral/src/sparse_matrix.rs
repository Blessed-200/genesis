#![allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct SparseMatrix16 {
    entries: Vec<(u8, u8, f64)>,
    row_ptr: [u32; 17],
}

impl SparseMatrix16 {
    pub(crate) fn from_dense(dense: &[[f64; 16]; 16]) -> Self {
        let mut entries = Vec::new();
        let mut row_ptr = [0_u32; 17];
        for (r, row) in dense.iter().enumerate() {
            row_ptr[r] = entries.len() as u32;
            for (c, value) in row.iter().enumerate() {
                if value.abs() > 1e-14 {
                    entries.push((r as u8, c as u8, *value));
                }
            }
        }
        row_ptr[16] = entries.len() as u32;
        Self { entries, row_ptr }
    }

    pub(crate) fn matvec(&self, v: &[f64; 16]) -> [f64; 16] {
        let mut out = [0.0; 16];
        for (row, out_cell) in out.iter_mut().enumerate() {
            let start = self.row_ptr[row] as usize;
            let end = self.row_ptr[row + 1] as usize;
            let mut acc = 0.0;
            for &(_, c, value) in &self.entries[start..end] {
                acc += value * v[c as usize];
            }
            *out_cell = acc;
        }
        out
    }

    pub(crate) fn trace(&self) -> f64 {
        self.entries
            .iter()
            .filter(|&&(r, c, _)| r == c)
            .map(|&(_, _, v)| v)
            .sum()
    }

    pub(crate) fn frobenius_norm_sq(&self) -> f64 {
        self.entries.iter().map(|&(_, _, v)| v * v).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::SparseMatrix16;

    #[test]
    fn sparse_matrix_matvec_identity() {
        let mut id = [[0.0; 16]; 16];
        for (i, row) in id.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let m = SparseMatrix16::from_dense(&id);
        let v = std::array::from_fn(|i| i as f64 * 0.5);
        assert_eq!(m.matvec(&v), v);
    }
}
