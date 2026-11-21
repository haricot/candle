//! Architecture-neutral MMQ geometry interpreter.
//!
//! The producer owns tile/shared-memory numbers. This module only checks
//! bounded geometry and selects the first feasible tile minimizing grid Y.
//! No CUDA toolkit, SM61 special cases, tensor types or kernel symbols here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmqTile {
    pub width: u32,
    pub shared_bytes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmqCodec {
    pub name: &'static str,
    pub tiles: &'static [MmqTile],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmqRecipe {
    pub row_tile: u32,
    pub quant_padding: u32,
    pub warp: u32,
    pub max_shared_bytes: u32,
    pub quant_block: (u32, u32, u32),
    pub matrix_block: (u32, u32, u32),
    pub codecs: &'static [MmqCodec],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmqGeometry {
    pub matrix_key: String,
    pub quant_grid: (u32, u32, u32),
    pub quant_block: (u32, u32, u32),
    pub matrix_grid: (u32, u32, u32),
    pub matrix_block: (u32, u32, u32),
    pub matrix_shared: u32,
}

impl MmqRecipe {
    /// Validates shape-dependent inputs and emits only bounded kernel geometry.
    /// The producer must validate the codec/tile table against its ABI manifest
    /// and the actual exported kernel entry names at build time.
    pub fn select(
        &self, codec: &str, rows: u32, batch: u32,
        padded: u32, warp_size: i32, shared_limit: i32,
    ) -> Result<MmqGeometry, &'static str> {
        if rows == 0 || batch == 0 || padded == 0
            || self.quant_padding == 0 || self.row_tile == 0
            || self.warp == 0 || warp_size <= 0 || warp_size as u32 != self.warp
            || shared_limit <= 0 || self.max_shared_bytes == 0
            || self.quant_block.0 == 0 || self.matrix_block.0 == 0
            || self.quant_block.1 == 0 || self.quant_block.2 == 0
            || self.matrix_block.1 == 0 || self.matrix_block.2 == 0
        {
            return Err("invalid MMQ geometry or device capability");
        }
        let pad_blocks = padded.div_ceil(self.quant_padding);
        if pad_blocks > 65535 || batch > 65535 {
            return Err("MMQ grid exceeds native ABI");
        }
        let candidate = self.codecs.iter().find(|c| c.name == codec)
            .ok_or("MMQ codec not declared by package")?;
        let limit = (shared_limit as u32).min(self.max_shared_bytes);
        let mut best = None;
        let mut best_y = u32::MAX;
        for tile in candidate.tiles {
            if tile.width == 0 || tile.shared_bytes == 0
                || tile.shared_bytes > limit {
                continue;
            }
            let count = batch.div_ceil(tile.width);
            if count < best_y {
                best = Some(*tile);
                best_y = count;
            }
            if best_y == 1 {
                break;
            }
        }
        let tile = best.ok_or("no MMQ tile fits device shared memory")?;
        let grid_x = rows.div_ceil(self.row_tile);
        if grid_x > 65535 || best_y > 65535 {
            return Err("MMQ matrix grid exceeds CUDA limit");
        }
        Ok(MmqGeometry {
            matrix_key: format!("{}:{}", tile.width, u8::from(rows % self.row_tile != 0)),
            quant_grid: (batch, pad_blocks, 1),
            quant_block: self.quant_block,
            matrix_grid: (grid_x, best_y, 1),
            matrix_block: self.matrix_block,
            matrix_shared: tile.shared_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TILES: [MmqTile; 3] = [
        MmqTile { width: 8, shared_bytes: 512 },
        MmqTile { width: 16, shared_bytes: 1024 },
        MmqTile { width: 32, shared_bytes: 2048 },
    ];
    const CODECS: [MmqCodec; 1] = [MmqCodec { name: "example", tiles: &TILES }];
    const RECIPE: MmqRecipe = MmqRecipe {
        row_tile: 64, quant_padding: 512, warp: 32, max_shared_bytes: 2048,
        quant_block: (128, 1, 1), matrix_block: (32, 8, 1),
        codecs: &CODECS,
    };

    #[test]
    fn selects_first_minimum_grid_candidate_without_architecture_policy() {
        assert_eq!(RECIPE.select("example", 65, 9, 1024, 32, 65536).unwrap(),
            MmqGeometry {
                matrix_key: "16:1".into(),
                quant_grid: (9, 2, 1), quant_block: (128, 1, 1),
                matrix_grid: (2, 1, 1), matrix_block: (32, 8, 1),
                matrix_shared: 1024,
            });
        assert_eq!(RECIPE.select("example", 64, 33, 1024, 32, 1024)
            .unwrap().matrix_key, "16:0");
        assert_eq!(RECIPE.select("example", 64, 1, 1024, 32, 65536)
            .unwrap().matrix_key, "8:0");
    }

    #[test]
    fn rejects_incompatible_dimensions_and_missing_recipes() {
        assert!(RECIPE.select("other", 64, 9, 1024, 32, 65536).is_err());
        assert!(RECIPE.select("example", 0, 9, 1024, 32, 65536).is_err());
        assert!(RECIPE.select("example", 64, 9, 1024, 16, 65536).is_err());
        assert!(RECIPE.select("example", 64, 9, 1024, 32, 400).is_err());
        assert!(RECIPE.select("example", 64, 9, 65536 * 512, 32, 65536).is_err());
    }
}
