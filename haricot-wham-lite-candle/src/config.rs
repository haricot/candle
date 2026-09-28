#[derive(Clone, Debug)]
pub struct WhamLiteConfig {
    pub n_joints: usize,
    pub input_dim: usize,
    pub embed_dim: usize,
    pub context_dim: usize,
    pub n_layers: usize,
    pub pose_joints: usize,
    pub contact_dim: usize,
}

impl Default for WhamLiteConfig {
    fn default() -> Self {
        let n_joints = 17;
        let embed_dim = 512;
        Self {
            n_joints,
            input_dim: n_joints * 2 + 3,
            embed_dim,
            context_dim: embed_dim + n_joints * 3,
            n_layers: 3,
            pose_joints: 24,
            contact_dim: 4,
        }
    }
}

impl WhamLiteConfig {
    pub fn pose_dim(&self) -> usize {
        self.pose_joints * 6
    }

    pub fn kp3d_dim(&self) -> usize {
        self.n_joints * 3
    }

    pub fn init_kp_dim(&self) -> usize {
        self.kp3d_dim() + self.input_dim
    }

    pub fn main_pose_dim(&self) -> usize {
        // WHAM uses 20 MAIN_JOINTS for its SMPL recurrent initializer.
        20 * 6
    }
}
