#![allow(dead_code)]
#[derive(Clone, Debug)]
pub struct SparseMatrix16 {
    entries: Vec<(u8, u8, f64)>,
    row_ptr: [u32; 17],
}

impl SparseMatrix16 {
    pub(crate) fn from_dense(dense: &[[f64; 16]; 16]) -> Self {
        let mut entries = Vec::new();
        let mut row_ptr = [0_u32; 17];
        for (r, row) in dense.iter().enumerate() {
            row_ptr[r] = u32::try_from(entries.len()).unwrap_or(u32::MAX);
            for (c, value) in row.iter().enumerate() {
                if value.abs() > 1e-14 {
                    entries.push((
                        u8::try_from(r).unwrap_or(u8::MAX),
                        u8::try_from(c).unwrap_or(u8::MAX),
                        *value,
                    ));
                }
            }
        }
        row_ptr[16] = u32::try_from(entries.len()).unwrap_or(u32::MAX);
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
        let v = std::array::from_fn(|i| f64::from(u32::try_from(i).unwrap_or(0)) * 0.5);
        let out = m.matvec(&v);
        for i in 0..16 {
            assert!((out[i] - v[i]).abs() < 1e-12);
        }
    }
}
