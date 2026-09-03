pub const NUM_KEYPOINTS: usize = 26;

pub const NAMES: [&str; NUM_KEYPOINTS] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "head",
    "neck",
    "hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
];

pub const SKELETON: [(usize, usize); 27] = [
    (15, 13), (13, 11), (11, 19),
    (16, 14), (14, 12), (12, 19),
    (17, 18), (18, 19),
    (18, 5), (5, 7), (7, 9),
    (18, 6), (6, 8), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6),
    (15, 20), (15, 22), (15, 24),
    (16, 21), (16, 23), (16, 25),
];

pub const LEFT_RIGHT_SWAP: [(usize, usize); 10] = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (20, 21), (22, 23),
];
