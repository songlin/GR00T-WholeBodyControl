#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201
"""Generate a constant "stand still" G1 reference motion.

Holds the G1 default standing pose (the IsaacLab ``init_state`` in
``gear_sonic/envs/manager_env/robots/g1.py``) for a fixed number of frames and writes it in
both formats used by this repo:

  1. Deployment CSV folder (docs/source/references/motion_reference.md):
     joint_pos/joint_vel (IsaacLab order), body_pos/body_quat (wxyz) and body velocities for
     the 14 tracked bodies, metadata.txt, info.txt.
  2. Training motion_lib PKL ({name: entry} with root_trans_offset, pose_aa, dof, root_rot,
     smpl_joints, fps), built with the same ``convert_sequence`` used for Bones-SEED data.
  3. Matching SMPL PKL (pose_aa, transl, smpl_joints, fps) for the SMPL encoder: the stillest
     frame of a real human idle clip, re-headed to face the same way as the robot clip.

Body poses come from MuJoCo forward kinematics of the training MJCF. The pelvis height is
chosen so the lowest foot contact sphere touches z = 0.

SMPL conventions (see TrackingCommand.smpl_root_quat_w*): the policy sees the SMPL root as
``q_eff = Rx(90deg) * q_pose_aa * conj(q_base)`` with q_base = [0.5, 0.5, 0.5, 0.5], and SMPL
joints are stored z-up in the world heading. Rotating ``q_eff`` and the joints by the same yaw
leaves the root-relative joint observation unchanged and only changes the heading.

Usage (from the repo root):
    python gear_sonic/data_process/make_stance_motion.py
    python gear_sonic/data_process/make_stance_motion.py --frames 1000 --yaw 90
"""

import argparse
import os

import joblib
import mujoco
import numpy as np
from scipy.spatial import transform

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import convert_sequence
from gear_sonic.envs.env_utils.joint_utils import G1_ISAACLab_ORDER

MJCF_PATH = "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"

# G1 default standing pose (matches G1_CYLINDER_MODEL_12_DEX_CFG.init_state); other joints are 0.
STANCE_JOINT_POS = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
}

# Tracked bodies, in the column-group order of the example deployment clips. Their IsaacLab
# body indexes (metadata.txt) are listed alongside.
TRACKED_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
TRACKED_BODY_ISAACLAB_INDEXES = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]

FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]

SMPL_BASE_ROT = transform.Rotation.from_quat([0.5, 0.5, 0.5, 0.5])
SMPL_Y_TO_Z_UP = transform.Rotation.from_rotvec([np.pi / 2, 0.0, 0.0])


def compute_stance_frame(yaw_deg: float, height: float | None):
    """Run FK for the stance pose; return IsaacLab joint pos and tracked body pos/quat (wxyz)."""
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    joint_pos_il = np.array([STANCE_JOINT_POS.get(n, 0.0) for n in G1_ISAACLab_ORDER])
    for name, value in zip(G1_ISAACLab_ORDER, joint_pos_il):  # noqa: B905
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value

    root_quat_xyzw = transform.Rotation.from_euler("z", yaw_deg, degrees=True).as_quat()
    data.qpos[3:7] = root_quat_xyzw[[3, 0, 1, 2]]  # MuJoCo free joint quat is wxyz

    if height is None:
        # Place the pelvis so the lowest foot collision sphere rests on the ground.
        data.qpos[:3] = [0.0, 0.0, 1.0]
        mujoco.mj_forward(model, data)
        foot_ids = {model.body(n).id for n in FOOT_BODIES}
        lowest = min(
            data.geom_xpos[g][2] - model.geom_size[g][0]
            for g in range(model.ngeom)
            if model.geom_bodyid[g] in foot_ids
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
            and model.geom_contype[g] > 0
        )
        height = 1.0 - lowest
    data.qpos[:3] = [0.0, 0.0, height]
    mujoco.mj_forward(model, data)

    body_pos = np.array([data.xpos[model.body(n).id] for n in TRACKED_BODIES])
    body_quat = np.array([data.xquat[model.body(n).id] for n in TRACKED_BODIES])  # wxyz
    return joint_pos_il.astype(np.float32), body_pos.astype(np.float32), body_quat.astype(
        np.float32
    ), height


def make_smpl_stance(source_path, frame, num_frames, yaw_deg):
    """Build a static SMPL clip from one frame of a y-up SMPL idle clip, facing ``yaw_deg``.

    If ``frame`` is None, pick the frame with the lowest mean joint speed over a +-0.5 s window
    (plus a small penalty on root tilt).
    """
    src = joblib.load(source_path)
    fps = float(src["fps"])
    joints = src["smpl_joints"]
    q_eff = SMPL_Y_TO_Z_UP * transform.Rotation.from_rotvec(src["pose_aa"][:, :3]) * SMPL_BASE_ROT.inv()

    if frame is None:
        speed = np.linalg.norm(np.diff(joints, axis=0), axis=-1).mean(-1) * fps
        half = int(fps * 0.5)
        score = np.convolve(speed, np.ones(2 * half + 1) / (2 * half + 1), mode="same")
        tilt = np.degrees(np.arccos(np.clip(q_eff.apply([0.0, 0.0, 1.0])[:, 2], -1.0, 1.0)))
        valid = slice(half, len(speed) - half)
        frame = int(np.argmin(score[valid] + 0.01 * tilt[valid])) + half

    # Re-head: rotate root orientation and joints about z so the heading equals yaw_deg.
    src_yaw = q_eff[frame].as_euler("ZYX")[0]
    re_head = transform.Rotation.from_euler("z", np.radians(yaw_deg) - src_yaw)
    q_eff_new = re_head * q_eff[frame]
    q_pose_new = SMPL_Y_TO_Z_UP.inv() * q_eff_new * SMPL_BASE_ROT

    pose_aa = src["pose_aa"][frame].astype(np.float32).copy()  # (72,)
    pose_aa[:3] = q_pose_new.as_rotvec()
    smpl_joints = re_head.apply(joints[frame]).astype(np.float32)  # (24, 3), z-up
    transl = np.array([0.0, src["transl"][frame][1], 0.0], dtype=np.float32)  # y-up; keep height

    smpl = {
        "pose_aa": np.tile(pose_aa, (num_frames, 1)),
        "transl": np.tile(transl, (num_frames, 1)),
        "smpl_joints": np.tile(smpl_joints, (num_frames, 1, 1)),
        "fps": fps,
    }
    return smpl, frame


def write_csv(path, array_2d, prefixes):
    """Write a (T, D) array with a header row; ``prefixes`` gives one column name per entry."""
    np.savetxt(path, array_2d, delimiter=",", header=",".join(prefixes), comments="", fmt="%.6f")


def write_deploy_folder(out_dir, name, joint_pos, body_pos, body_quat):
    """Write the C++ deployment CSV folder for one motion."""
    os.makedirs(out_dir, exist_ok=True)
    T, num_joints = joint_pos.shape
    num_bodies = body_pos.shape[1]
    pos_cols = [f"body_{b}_{a}" for b in range(num_bodies) for a in "xyz"]
    quat_cols = [f"body_{b}_{a}" for b in range(num_bodies) for a in "wxyz"]
    zeros_vel = np.zeros((T, num_bodies * 3), dtype=np.float32)

    write_csv(f"{out_dir}/joint_pos.csv", joint_pos, [f"joint_{i}" for i in range(num_joints)])
    write_csv(
        f"{out_dir}/joint_vel.csv",
        np.zeros_like(joint_pos),
        [f"joint_vel_{i}" for i in range(num_joints)],
    )
    write_csv(f"{out_dir}/body_pos.csv", body_pos.reshape(T, -1), pos_cols)
    write_csv(f"{out_dir}/body_quat.csv", body_quat.reshape(T, -1), quat_cols)
    write_csv(f"{out_dir}/body_lin_vel.csv", zeros_vel, pos_cols)
    write_csv(f"{out_dir}/body_ang_vel.csv", zeros_vel, pos_cols)

    indexes = " ".join(f"{i:2d}" for i in TRACKED_BODY_ISAACLAB_INDEXES)
    with open(f"{out_dir}/metadata.txt", "w") as f:
        f.write(f"Metadata for: {name}\n{'=' * 30}\n\n")
        f.write(f"Body part indexes:\n[{indexes}]\n\nTotal timesteps: {T}\n")

    with open(f"{out_dir}/info.txt", "w") as f:
        f.write(f"Motion Information: {name}\n{'=' * 50}\n\n")
        f.write("Constant G1 default standing pose (generated by make_stance_motion.py).\n\n")
        for key, arr in [
            ("joint_pos", joint_pos),
            ("body_pos_w", body_pos),
            ("body_quat_w", body_quat),
        ]:
            f.write(f"{key}:\n  Shape: {arr.shape}\n  Dtype: {arr.dtype}\n")
            f.write(f"  Range: [{arr.min():.3f}, {arr.max():.3f}]\n\n")
        f.write(f"Tracked bodies: {TRACKED_BODIES}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default="stand_still")
    parser.add_argument("--frames", type=int, default=500, help="Number of frames at --fps")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--yaw", type=float, default=0.0, help="Root heading in degrees")
    parser.add_argument(
        "--height", type=float, default=None, help="Pelvis height (default: feet on ground)"
    )
    parser.add_argument("--deploy-out", default="gear_sonic_deploy/reference/stance")
    parser.add_argument("--pkl-out", default="sample_data/stance/robot_filtered")
    parser.add_argument(
        "--smpl-source",
        default="data/smpl_filtered/idle_001__A419.pkl",
        help="Human SMPL idle clip to take the standing frame from ('' to skip SMPL output)",
    )
    parser.add_argument(
        "--smpl-frame", type=int, default=None, help="Source frame (default: stillest frame)"
    )
    parser.add_argument("--smpl-out", default="sample_data/stance/smpl_filtered")
    args = parser.parse_args()

    joint_pos, body_pos, body_quat, height = compute_stance_frame(args.yaw, args.height)
    T = args.frames
    joint_pos_t = np.tile(joint_pos, (T, 1))
    body_pos_t = np.tile(body_pos, (T, 1, 1))
    body_quat_t = np.tile(body_quat, (T, 1, 1))

    deploy_dir = os.path.join(args.deploy_out, args.name)
    write_deploy_folder(deploy_dir, args.name, joint_pos_t, body_pos_t, body_quat_t)
    print(f"Wrote deployment CSVs: {deploy_dir}")

    entry = convert_sequence(
        {"joint_pos": joint_pos_t, "body_pos_w": body_pos_t, "body_quat_w": body_quat_t},
        args.fps,
    )
    os.makedirs(args.pkl_out, exist_ok=True)
    pkl_path = os.path.join(args.pkl_out, f"{args.name}.pkl")
    joblib.dump({args.name: entry}, pkl_path, compress=True)
    print(f"Wrote training PKL:   {pkl_path}")
    print(f"Frames: {T} @ {args.fps} Hz ({T / args.fps:.1f} s), pelvis height {height:.4f} m")

    if args.smpl_source:
        smpl_fps = float(joblib.load(args.smpl_source)["fps"])
        num_smpl_frames = round(T * smpl_fps / args.fps)  # same duration as the robot clip
        smpl, frame = make_smpl_stance(args.smpl_source, args.smpl_frame, num_smpl_frames, args.yaw)
        os.makedirs(args.smpl_out, exist_ok=True)
        # Directory-mode SMPL files are flat dicts named after the robot motion key.
        smpl_path = os.path.join(args.smpl_out, f"{args.name}.pkl")
        joblib.dump(smpl, smpl_path, compress=True)
        print(f"Wrote SMPL PKL:       {smpl_path} (from {args.smpl_source} frame {frame})")


if __name__ == "__main__":
    main()
