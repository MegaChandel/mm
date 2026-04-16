# DEEP ANALYSIS & COMPLETE AGENT INSTRUCTIONS
# Biped Platform Jump — Assignment 3 (SAC)

---

## PART 0: CRITICAL ANALYSIS BEFORE TOUCHING ANY CODE

### 0.1 — File/Directory Structure (CREATE EXACTLY THIS)
```
project_root/
├── main.py          ← implement all TODOs
├── utils.py         ← implement all TODOs
├── requirements.txt ← provided, do not change
├── assest/          ← NOTE: intentional typo "assest" not "assets" — DO NOT rename
│   ├── biped_.urdf
│   └── stair.urdf
```

### 0.2 — README LIES (DO NOT FOLLOW THESE PARTS)
The README has CLI examples that reference flags `--algo` and `--task` that DO NOT EXIST in `main.py`'s `parse_args()`.  
**Ignore all README CLI examples that use `--algo` or `--task`.**  
The actual valid flags are: `--mode`, `--timesteps`, `--model_path`, `--episodes`, `--render`.

Also the README references a `TASK_ENV` dict — that does NOT exist in main.py either. Only `ALGO_MAP` exists.

### 0.3 — Robot URDF Analysis (Critical for correct implementation)
From `assest/biped_.urdf`:
- **6 revolute joints** (3 per leg): left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle
- All joints rotate around the **x-axis** (sagittal plane motion)
- Joint limits and effort values:
  | Joint | Lower | Upper | Effort (N·m) | Max Vel |
  |-------|-------|-------|-------------|---------|
  | left_hip | 0.0 | 0.87 | 15 | 2 |
  | left_knee | -1.0 | 0.0 | 15 | 2 |
  | left_ankle | -0.2 | 0.78 | 7.5 | 1 |
  | right_hip | 0.0 | 0.87 | 15 | 2 |
  | right_knee | -1.0 | 0.0 | 15 | 2 |
  | right_ankle | -0.2 | 0.78 | 7.5 | 1 |
- **Total robot mass: 2.05 kg** (0.6 pelvis + 0.3+0.3+0.125 × 2 legs) — matches `ROBOT_MASS_KG`
- Robot's standing height from ground to pelvis ≈ 0.81 m (this is `GROUND_Z`)

### 0.4 — Environment Physics (Critical)
From `BipedJumpEnv.__init__()`:
- **Platform**: box halfExtents [0.6, 0.6, 0.5], center at [0, 0, 0.5] → **top surface at z=1.0m**
- **Robot spawn**: pelvis at [0, 0, **1.81**] = PLATFORM_H(1.0) + GROUND_Z(0.81)
- **Ground plane**: `plane_id` at z=0
- **Timestep**: 1/50 s = 0.02 s
- **Max steps**: 500 per episode
- **n_actuated**: 6 joints
- **obs_dim**: 6×2 + 3 + 3 + 3 + 2 + 1 + 1 = **25 total**

### 0.5 — Observation Vector Breakdown (MUST MATCH EXACTLY)
`obs_dim = n_actuated * 2 + 3 + 3 + 3 + 2 + 1 + 1`  
Indices for 6 joints (obs_dim = 25):
```
[0:6]   joint positions (radians) — 6 values
[6:12]  joint velocities (rad/s) — 6 values
[12:15] base position [x, y, z] — 3 values
[15:18] base orientation as Euler [roll, pitch, yaw] — 3 values
[18:21] base linear velocity [vx, vy, vz] — 3 values
[21:23] foot contacts [left_ground, right_ground] — 2 binary floats
[23]    height above ground = pos[2] - GROUND_Z — 1 value
[24]    has_landed flag = float(self.has_landed) — 1 value
```
**This MUST produce exactly 25 float32 values or gymnasium will crash.**

### 0.6 — Landing Logic (Critical for reward/termination)
- "Landed" = robot feet contact `plane_id` (ground plane, z=0) — NOT the platform
- `has_landed` starts False, becomes True the FIRST time a foot touches `plane_id` while `pos[2] < PLATFORM_H + 0.5`
- `land_stable_steps` counts consecutive steps with at least one foot on ground after landing
- **Fall condition**: `abs(roll) > 1.2` OR `abs(pitch) > 1.2` (≈69°)

---

## PART 1: COMPLETE utils.py IMPLEMENTATION

Replace the entire `utils.py` file with the following. KEEP all the provided code that already works (RewardPlotCallback, __init__ structure, etc.) and ONLY fill in the TODOs:

### 1.1 — Top constants (fill in the None values):
```python
TOTAL_TIMESTEPS    = 1_000_000
EVAL_FREQ          = 10_000
MAX_EPISODE_STEPS  = 500
EVAL_EPISODES      = 10
ROBOT_MASS_KG      = 2.05

SAC_CONFIG = dict(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,
    buffer_size   = 1_000_000,
    batch_size    = 256,
    tau           = 0.005,
    gamma         = 0.99,
    ent_coef      = "auto",
    verbose       = 1,
)
```

### 1.2 — `reset()` implementation:
```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    # Reset robot base pose and velocity
    p.resetBasePositionAndOrientation(
        self.robot_id,
        [0, 0, self.SPAWN_Z],
        [0, 0, 0, 1],          # identity quaternion = upright
        physicsClientId=self.physics_client
    )
    p.resetBaseVelocity(
        self.robot_id,
        linearVelocity=[0, 0, 0],
        angularVelocity=[0, 0, 0],
        physicsClientId=self.physics_client
    )

    # Reset all joints to zero position and velocity
    for idx in self.joint_indices:
        p.resetJointState(
            self.robot_id, idx,
            targetValue=0.0,
            targetVelocity=0.0,
            physicsClientId=self.physics_client
        )

    # Reset counters and state flags
    self.step_counter      = 0
    self.land_stable_steps = 0
    self.has_landed        = False
    self.prev_z            = self.SPAWN_Z
    self._initial_pos      = [0.0, 0.0, self.SPAWN_Z]

    # Warm-up: let robot settle on platform (prevents initial physics explosion)
    for _ in range(5):
        p.stepSimulation(physicsClientId=self.physics_client)

    return self._get_obs(), {}
```

### 1.3 — `_get_obs()` implementation:
```python
def _get_obs(self):
    # Joint positions and velocities
    joint_states = [
        p.getJointState(self.robot_id, idx, physicsClientId=self.physics_client)
        for idx in self.joint_indices
    ]
    joint_pos = np.array([s[0] for s in joint_states], dtype=np.float32)
    joint_vel = np.array([s[1] for s in joint_states], dtype=np.float32)

    # Base position and orientation
    pos, orn = p.getBasePositionAndOrientation(
        self.robot_id, physicsClientId=self.physics_client
    )
    euler = p.getEulerFromQuaternion(orn)   # returns (roll, pitch, yaw)

    # Base linear velocity
    lin_vel, _ = p.getBaseVelocity(
        self.robot_id, physicsClientId=self.physics_client
    )

    # Foot contacts with ground plane only (plane_id, NOT platform)
    left_contact = float(
        len(p.getContactPoints(
            self.robot_id, self.plane_id,
            self.left_foot_link, -1,
            physicsClientId=self.physics_client
        )) > 0
    )
    right_contact = float(
        len(p.getContactPoints(
            self.robot_id, self.plane_id,
            self.right_foot_link, -1,
            physicsClientId=self.physics_client
        )) > 0
    )

    # Height above standing ground level (0 when standing on ground, ~1.0 on platform)
    height_above_ground = float(pos[2]) - self.GROUND_Z

    # Has-landed flag
    landed_flag = float(self.has_landed)

    obs = np.concatenate([
        joint_pos,                                          # 6
        joint_vel,                                          # 6
        np.array(pos,     dtype=np.float32),                # 3
        np.array(euler,   dtype=np.float32),                # 3
        np.array(lin_vel, dtype=np.float32),                # 3
        np.array([left_contact, right_contact], dtype=np.float32),  # 2
        np.array([height_above_ground], dtype=np.float32),  # 1
        np.array([landed_flag],          dtype=np.float32), # 1
    ], dtype=np.float32)

    # Safety check — REMOVE after confirming correct (should always be 25)
    assert obs.shape[0] == self.observation_space.shape[0], \
        f"obs shape mismatch: {obs.shape[0]} vs {self.observation_space.shape[0]}"

    return obs
```

### 1.4 — `_compute_reward()` implementation:
```python
def _compute_reward(self, pos, orn, lin_vel, landed_now):
    euler    = p.getEulerFromQuaternion(orn)
    roll, pitch, yaw = euler

    # ── 1. Upright penalty (active throughout entire episode) ──────────
    # Penalizes tilting in roll and pitch. Keeps robot upright during flight.
    upright_penalty = -(roll ** 2 + pitch ** 2) * 0.5

    # ── 2. Z-progress reward (reward for descending off platform) ───────
    # Encourages the robot to actually leave the platform and fall.
    # Only during flight (not after landing, to avoid jitter reward).
    if not self.has_landed:
        z_progress = (self.prev_z - pos[2]) * 3.0   # positive when descending
    else:
        z_progress = 0.0

    # ── 3. Flight bonus (small per-step reward for being off platform) ──
    # Rewards robot for clearing the platform edge.
    if pos[2] < self.PLATFORM_H + 0.15:
        flight_bonus = 0.3
    else:
        # Still on platform: reward horizontal speed toward edge
        horiz_speed = float(np.sqrt(lin_vel[0]**2 + lin_vel[1]**2))
        flight_bonus = horiz_speed * 0.4

    # ── 4. Landing reward (one-time, large bonus) ───────────────────────
    # Given when feet first touch the ground (plane_id), scaled by uprightness.
    landing_reward = 0.0
    if landed_now:
        upright_factor = max(0.0, 1.0 - abs(roll) - abs(pitch))
        landing_reward = 100.0 * upright_factor + 50.0    # 50 base + 100 if perfectly upright

    # ── Small stability bonus after landing ────────────────────────────
    # Rewards staying upright after touching down
    stability_bonus = 0.0
    if self.has_landed and not landed_now:
        stability_bonus = max(0.0, 1.0 - abs(roll) - abs(pitch)) * 0.5

    return float(upright_penalty + z_progress + flight_bonus + landing_reward + stability_bonus)
```

### 1.5 — `get_joint_indices()` implementation:
```python
def get_joint_indices(self):
    return self.joint_indices
```

### 1.6 — `robot_initial_position()` implementation:
```python
def robot_initial_position(self):
    # Returns the spawn position stored at the last reset()
    return list(getattr(self, '_initial_pos', [0.0, 0.0, self.SPAWN_Z]))
```

### 1.7 — `robot_current_position()` implementation:
```python
def robot_current_position(self):
    pos, _ = p.getBasePositionAndOrientation(
        self.robot_id, physicsClientId=self.physics_client
    )
    return list(pos)
```

### 1.8 — `step()` implementation:
```python
def step(self, action):
    action = np.clip(action, -1.0, 1.0)

    # ── Apply position control ──────────────────────────────────────────
    # Map action [-1, 1] → target joint angle [lower_limit, upper_limit]
    # Use URDF effort limits as max force.
    effort_limits = [15.0, 15.0, 7.5, 15.0, 15.0, 7.5]  # hip, knee, ankle × 2

    for i, (idx, (lo, hi)) in enumerate(zip(self.joint_indices, self.joint_limits)):
        target_angle = lo + (float(action[i]) + 1.0) / 2.0 * (hi - lo)
        p.setJointMotorControl2(
            self.robot_id, idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_angle,
            force=effort_limits[i],
            physicsClientId=self.physics_client
        )

    p.stepSimulation(physicsClientId=self.physics_client)

    if self.render_mode:
        time.sleep(self.timestep)

    self.step_counter += 1

    # ── Read state ──────────────────────────────────────────────────────
    pos, orn = p.getBasePositionAndOrientation(
        self.robot_id, physicsClientId=self.physics_client
    )
    lin_vel, _ = p.getBaseVelocity(
        self.robot_id, physicsClientId=self.physics_client
    )

    # ── Foot contact with ground plane ─────────────────────────────────
    left_gnd = len(p.getContactPoints(
        self.robot_id, self.plane_id,
        self.left_foot_link, -1,
        physicsClientId=self.physics_client
    )) > 0
    right_gnd = len(p.getContactPoints(
        self.robot_id, self.plane_id,
        self.right_foot_link, -1,
        physicsClientId=self.physics_client
    )) > 0
    any_foot_on_ground = left_gnd or right_gnd

    # ── Landing detection ───────────────────────────────────────────────
    # "Landed" = first time a foot touches ground plane while below platform
    landed_now = False
    if (not self.has_landed
            and pos[2] < self.PLATFORM_H + 0.5
            and any_foot_on_ground):
        self.has_landed        = True
        landed_now             = True
        self.land_stable_steps = 0

    if self.has_landed and any_foot_on_ground:
        self.land_stable_steps += 1

    # ── Reward ──────────────────────────────────────────────────────────
    reward = self._compute_reward(pos, orn, lin_vel, landed_now)
    self.prev_z = pos[2]

    # ── Termination conditions ──────────────────────────────────────────
    euler    = p.getEulerFromQuaternion(orn)
    roll, pitch = euler[0], euler[1]
    terminated = False

    # Robot fell over
    if abs(roll) > 1.2 or abs(pitch) > 1.2:
        terminated = True
        reward    -= 20.0   # fall penalty

    # Robot somehow went underground
    elif pos[2] < 0.05:
        terminated = True

    # Stable landing achieved (feet on ground for 20+ consecutive steps)
    elif self.land_stable_steps >= 20:
        terminated = True

    truncated = (self.step_counter >= self.max_steps)

    obs = self._get_obs()
    return obs, float(reward), terminated, truncated, {}
```

---

## PART 2: COMPLETE main.py IMPLEMENTATION

### 2.1 — `ALGO_MAP` (fill in the TODO at the top of main.py):
```python
ALGO_MAP = {
    "sac": (SAC, SAC_CONFIG),
}
```

### 2.2 — `train()` complete implementation:
```python
def train(timesteps: int, render: bool = False):
    # ── Create output directories ────────────────────────────────────────
    os.makedirs("models/sac_best", exist_ok=True)
    os.makedirs("logs/sac_goal",   exist_ok=True)
    os.makedirs("logs/sac_eval",   exist_ok=True)

    # ── Create environments ──────────────────────────────────────────────
    env      = Monitor(BipedJumpEnv(render=render),  "logs/sac_monitor.csv")
    eval_env = Monitor(BipedJumpEnv(render=False))

    # ── Instantiate SAC model ────────────────────────────────────────────
    model = SAC(
        env=env,
        tensorboard_log="logs/sac_goal/",
        **SAC_CONFIG
    )

    # ── Callbacks ────────────────────────────────────────────────────────
    reward_cb = RewardPlotCallback()
    eval_cb   = EvalCallback(
        eval_env,
        best_model_save_path="models/sac_best/",
        log_path="logs/sac_eval/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # ── Training ─────────────────────────────────────────────────────────
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[reward_cb, eval_cb],
        )
    except KeyboardInterrupt:
        print("\n[train] Keyboard interrupt — saving crash checkpoint...")
        model.save("models/sac_biped_crashsave")
        print("[train] Crash save written to models/sac_biped_crashsave.zip")

    # ── Save final model and plot ─────────────────────────────────────────
    model.save("models/sac_biped_goal")
    print("[train] Final model saved to models/sac_biped_goal.zip")

    reward_cb.plot_rewards("reward_curve_sac.png")

    env.close()
    eval_env.close()
```

### 2.3 — `test()` complete implementation:
```python
def test(model_path: str, episodes: int, render: bool):
    DT = 1.0 / 50.0   # must match BipedJumpEnv.timestep

    # ── Load environment and model ────────────────────────────────────────
    env   = BipedJumpEnv(render=render)
    model = SAC.load(model_path, env=env)

    joint_indices  = env.get_joint_indices()
    total_energy   = 0.0
    total_distance = 0.0
    total_reward   = 0.0
    fall_count     = 0

    for ep in range(episodes):
        obs, _     = env.reset()
        init_pos   = env.robot_initial_position()
        ep_reward  = 0.0
        ep_energy  = 0.0
        ep_steps   = 0
        fell       = False
        done       = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_steps  += 1
            done       = terminated or truncated

            # Energy calculation: sum |torque × velocity| × dt
            for idx in joint_indices:
                js = p.getJointState(
                    env.robot_id, idx,
                    physicsClientId=env.physics_client
                )
                torque = js[3]   # applied joint torque
                vel    = js[1]   # joint velocity
                ep_energy += abs(torque * vel) * DT

            # Track fall: large roll or pitch
            pos_b, orn_b = p.getBasePositionAndOrientation(
                env.robot_id, physicsClientId=env.physics_client
            )
            euler_b = p.getEulerFromQuaternion(orn_b)
            if abs(euler_b[0]) > 1.2 or abs(euler_b[1]) > 1.2:
                fell = True

        curr_pos = env.robot_current_position()
        dist = float(np.linalg.norm(
            np.array(curr_pos[:2]) - np.array(init_pos[:2])   # horizontal distance only
        ))

        total_energy   += ep_energy
        total_distance += dist
        total_reward   += ep_reward
        if fell:
            fall_count += 1

        print(
            f"Episode {ep+1:>3}/{episodes}  |  "
            f"steps={ep_steps:>4}  |  "
            f"reward={ep_reward:>8.2f}  |  "
            f"energy={ep_energy:>7.2f} J  |  "
            f"dist={dist:>5.2f} m  |  "
            f"{'FELL' if fell else 'OK'}"
        )

    # ── Summary ───────────────────────────────────────────────────────────
    avg_reward   = total_reward   / episodes
    fall_rate    = 100.0 * fall_count / episodes
    avg_distance = total_distance / episodes
    avg_energy   = total_energy   / episodes
    CoT          = total_energy   / (ROBOT_MASS_KG * 9.81 * total_distance + 1e-8)

    print("\n" + "="*55)
    print("  EVALUATION SUMMARY")
    print("="*55)
    print(f"  Avg Reward       : {avg_reward:>10.2f}")
    print(f"  Fall Rate        : {fall_rate:>9.1f} %")
    print(f"  Avg Distance     : {avg_distance:>9.2f} m")
    print(f"  Avg Energy       : {avg_energy:>9.2f} J")
    print(f"  Cost of Transport: {CoT:>10.4f}")
    print("="*55)

    env.close()
```

### 2.4 — `main()` complete implementation:
```python
def main():
    args = parse_args()

    if args.mode == "view":
        view()

    elif args.mode == "train":
        ts = args.timesteps if args.timesteps is not None else TOTAL_TIMESTEPS
        train(ts, args.render)

    elif args.mode == "test":
        model_path = args.model_path if args.model_path is not None \
                     else "models/sac_best/best_model"
        test(model_path, args.episodes, args.render)
```

---

## PART 3: PUTTING IT ALL TOGETHER — EXACT FILE CONTENTS

### utils.py — Final combined file:

Write the file **exactly** as shown below. Do NOT change anything that was already provided (imports, RewardPlotCallback, __init__ structure). Only fill in the TODO sections:

```python
"""
utils.py — Shared utilities for Assignment 3: Biped 1 m Platform Jump.
"""

# ===========================================================================
# Hyperparameters
# ===========================================================================

TOTAL_TIMESTEPS   = 1_000_000
EVAL_FREQ         = 10_000
MAX_EPISODE_STEPS = 500

SAC_CONFIG = dict(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,
    buffer_size   = 1_000_000,
    batch_size    = 256,
    tau           = 0.005,
    gamma         = 0.99,
    ent_coef      = "auto",
    verbose       = 1,
)

EVAL_EPISODES = 10
ROBOT_MASS_KG = 2.05


# ===========================================================================
# RewardPlotCallback  — DO NOT MODIFY (already provided)
# ===========================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class RewardPlotCallback(BaseCallback):
    """Records episode rewards during training and saves a plot at the end."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._current_episode_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        done   = self.locals.get("dones",   [False])[0]
        self._current_episode_reward += reward
        if done:
            self.episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0
        return True

    def plot_rewards(self, save_path="reward_curve_sac.png"):
        if not self.episode_rewards:
            print("No episode rewards recorded yet.")
            return
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, alpha=0.6, label="Episode Reward")
        window = 20
        if len(self.episode_rewards) >= window:
            rolling = [
                sum(self.episode_rewards[max(0, i - window):i]) / min(i, window)
                for i in range(1, len(self.episode_rewards) + 1)
            ]
            plt.plot(rolling, color="red", linewidth=2, label=f"{window}-ep Rolling Avg")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("SAC Training Reward Curve — Biped 1 m Jump")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Reward plot saved to {save_path}")


# ===========================================================================
# BipedJumpEnv
# ===========================================================================

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

_ASSEST_DIR = os.path.join(os.path.dirname(__file__), "assest")


class BipedJumpEnv(gym.Env):
    """
    Task: biped spawns on a 1 m platform and must jump off, landing upright.
    """

    PLATFORM_H = 1.0
    SPAWN_Z    = 1.0 + 0.81   # = 1.81
    GROUND_Z   = 0.81

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render
        cid = p.connect(p.GUI if render else p.DIRECT)
        self.physics_client = cid

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=cid)
        self.timestep = 1.0 / 50.0
        p.setTimeStep(self.timestep, physicsClientId=cid)

        self.max_steps         = 500
        self.step_counter      = 0
        self.land_stable_steps = 0

        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, physicsClientId=cid)

        # 1 m platform
        plat_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.5],
                                          physicsClientId=cid)
        plat_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.5],
                                       rgbaColor=[0.55, 0.27, 0.07, 1],
                                       physicsClientId=cid)
        self.platform_id = p.createMultiBody(0, plat_col, plat_vis,
                                              [0, 0, 0.5], physicsClientId=cid)

        # Robot
        urdf_path = os.path.join(_ASSEST_DIR, "biped_.urdf")
        self.robot_id = p.loadURDF(urdf_path, [0, 0, self.SPAWN_Z],
                                    useFixedBase=False, physicsClientId=cid)
        p.changeDynamics(self.robot_id, -1,
                         linearDamping=0.5, angularDamping=0.5,
                         physicsClientId=cid)

        # Joint discovery
        self.joint_indices   = []
        self.joint_limits    = []
        self.left_foot_link  = 2
        self.right_foot_link = 5

        for i in range(p.getNumJoints(self.robot_id, physicsClientId=cid)):
            ji = p.getJointInfo(self.robot_id, i, physicsClientId=cid)
            if ji[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_limits.append((ji[8], ji[9]))
            if b"left_foot"  in ji[12]: self.left_foot_link  = i
            if b"right_foot" in ji[12]: self.right_foot_link = i

        p.changeDynamics(self.robot_id, self.left_foot_link,
                         lateralFriction=2.0, physicsClientId=cid)
        p.changeDynamics(self.robot_id, self.right_foot_link,
                         lateralFriction=2.0, physicsClientId=cid)

        self.n_actuated = len(self.joint_indices)

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0,
                                       shape=(self.n_actuated,), dtype=np.float32)
        obs_dim  = self.n_actuated * 2 + 3 + 3 + 3 + 2 + 1 + 1
        obs_high = np.full(obs_dim, np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.prev_z      = self.SPAWN_Z
        self.has_landed  = False
        self._initial_pos = [0.0, 0.0, self.SPAWN_Z]
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetBasePositionAndOrientation(
            self.robot_id,
            [0, 0, self.SPAWN_Z],
            [0, 0, 0, 1],
            physicsClientId=self.physics_client
        )
        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=self.physics_client
        )

        for idx in self.joint_indices:
            p.resetJointState(
                self.robot_id, idx,
                targetValue=0.0,
                targetVelocity=0.0,
                physicsClientId=self.physics_client
            )

        self.step_counter      = 0
        self.land_stable_steps = 0
        self.has_landed        = False
        self.prev_z            = self.SPAWN_Z
        self._initial_pos      = [0.0, 0.0, self.SPAWN_Z]

        for _ in range(5):
            p.stepSimulation(physicsClientId=self.physics_client)

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _get_obs(self):
        joint_states = [
            p.getJointState(self.robot_id, idx, physicsClientId=self.physics_client)
            for idx in self.joint_indices
        ]
        joint_pos = np.array([s[0] for s in joint_states], dtype=np.float32)
        joint_vel = np.array([s[1] for s in joint_states], dtype=np.float32)

        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        euler = p.getEulerFromQuaternion(orn)

        lin_vel, _ = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.physics_client
        )

        left_contact = float(
            len(p.getContactPoints(
                self.robot_id, self.plane_id,
                self.left_foot_link, -1,
                physicsClientId=self.physics_client
            )) > 0
        )
        right_contact = float(
            len(p.getContactPoints(
                self.robot_id, self.plane_id,
                self.right_foot_link, -1,
                physicsClientId=self.physics_client
            )) > 0
        )

        height_above_ground = float(pos[2]) - self.GROUND_Z
        landed_flag         = float(self.has_landed)

        obs = np.concatenate([
            joint_pos,
            joint_vel,
            np.array(pos,     dtype=np.float32),
            np.array(euler,   dtype=np.float32),
            np.array(lin_vel, dtype=np.float32),
            np.array([left_contact, right_contact], dtype=np.float32),
            np.array([height_above_ground],          dtype=np.float32),
            np.array([landed_flag],                  dtype=np.float32),
        ], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------
    def _compute_reward(self, pos, orn, lin_vel, landed_now):
        euler        = p.getEulerFromQuaternion(orn)
        roll, pitch, _ = euler

        upright_penalty = -(roll ** 2 + pitch ** 2) * 0.5

        if not self.has_landed:
            z_progress = (self.prev_z - pos[2]) * 3.0
        else:
            z_progress = 0.0

        if pos[2] < self.PLATFORM_H + 0.15:
            flight_bonus = 0.3
        else:
            horiz_speed  = float(np.sqrt(lin_vel[0]**2 + lin_vel[1]**2))
            flight_bonus = horiz_speed * 0.4

        landing_reward = 0.0
        if landed_now:
            upright_factor = max(0.0, 1.0 - abs(roll) - abs(pitch))
            landing_reward = 100.0 * upright_factor + 50.0

        stability_bonus = 0.0
        if self.has_landed and not landed_now:
            stability_bonus = max(0.0, 1.0 - abs(roll) - abs(pitch)) * 0.5

        return float(upright_penalty + z_progress + flight_bonus
                     + landing_reward + stability_bonus)

    # ------------------------------------------------------------------
    def get_joint_indices(self):
        return self.joint_indices

    def robot_initial_position(self):
        return list(getattr(self, '_initial_pos', [0.0, 0.0, self.SPAWN_Z]))

    def robot_current_position(self):
        pos, _ = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        return list(pos)

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        effort_limits = [15.0, 15.0, 7.5, 15.0, 15.0, 7.5]

        for i, (idx, (lo, hi)) in enumerate(zip(self.joint_indices, self.joint_limits)):
            target_angle = lo + (float(action[i]) + 1.0) / 2.0 * (hi - lo)
            p.setJointMotorControl2(
                self.robot_id, idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_angle,
                force=effort_limits[i],
                physicsClientId=self.physics_client
            )

        p.stepSimulation(physicsClientId=self.physics_client)
        if self.render_mode:
            time.sleep(self.timestep)

        self.step_counter += 1

        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        lin_vel, _ = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.physics_client
        )

        left_gnd = len(p.getContactPoints(
            self.robot_id, self.plane_id,
            self.left_foot_link, -1,
            physicsClientId=self.physics_client
        )) > 0
        right_gnd = len(p.getContactPoints(
            self.robot_id, self.plane_id,
            self.right_foot_link, -1,
            physicsClientId=self.physics_client
        )) > 0
        any_foot_on_ground = left_gnd or right_gnd

        landed_now = False
        if (not self.has_landed
                and pos[2] < self.PLATFORM_H + 0.5
                and any_foot_on_ground):
            self.has_landed        = True
            landed_now             = True
            self.land_stable_steps = 0

        if self.has_landed and any_foot_on_ground:
            self.land_stable_steps += 1

        reward  = self._compute_reward(pos, orn, lin_vel, landed_now)
        self.prev_z = pos[2]

        euler   = p.getEulerFromQuaternion(orn)
        roll, pitch = euler[0], euler[1]

        terminated = False
        if abs(roll) > 1.2 or abs(pitch) > 1.2:
            terminated = True
            reward    -= 20.0
        elif pos[2] < 0.05:
            terminated = True
        elif self.land_stable_steps >= 20:
            terminated = True

        truncated = (self.step_counter >= self.max_steps)

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, {}

    # ------------------------------------------------------------------
    def close(self):
        p.disconnect(self.physics_client)
```

---

### main.py — Final combined file:

```python
"""
main.py — Assignment 3: Biped RL (1 m Platform Jump with SAC)

Usage:
    python main.py --mode view
    python main.py --mode train
    python main.py --mode train --timesteps 500000
    python main.py --mode test
    python main.py --mode test --render --episodes 5
    python main.py --mode test --model_path "models/sac_best/best_model"
"""

import argparse
import os
import time

import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from utils import (
    BipedJumpEnv, RewardPlotCallback,
    TOTAL_TIMESTEPS, EVAL_FREQ,
    SAC_CONFIG,
    EVAL_EPISODES, ROBOT_MASS_KG,
)

# ── Algorithm registry ────────────────────────────────────────────────────────
ALGO_MAP = {
    "sac": (SAC, SAC_CONFIG),
}


# ── Environment Preview ────────────────────────────────────────────────────────
def view():
    """Spawns the biped + stair in GUI mode. Press Ctrl+C to quit."""
    import pybullet_data

    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8, physicsClientId=cid)

    p.loadURDF("plane.urdf", physicsClientId=cid)

    assest = os.path.join(os.path.dirname(__file__), "assest")
    p.loadURDF(os.path.join(assest, "biped_.urdf"), [0, 0, 0.81],
               useFixedBase=False, physicsClientId=cid)
    p.loadURDF(os.path.join(assest, "stair.urdf"),  [0, 2, 0],
               p.getQuaternionFromEuler([0, 0, -3.1416]),
               useFixedBase=True, physicsClientId=cid)

    print("[view] Biped + stair spawned. Press Ctrl+C to quit.")
    try:
        while True:
            p.stepSimulation(physicsClientId=cid)
            time.sleep(1 / 240)
    except KeyboardInterrupt:
        pass
    p.disconnect(cid)


# ── Training ──────────────────────────────────────────────────────────────────
def train(timesteps: int, render: bool = False):
    os.makedirs("models/sac_best", exist_ok=True)
    os.makedirs("logs/sac_goal",   exist_ok=True)
    os.makedirs("logs/sac_eval",   exist_ok=True)

    env      = Monitor(BipedJumpEnv(render=render), "logs/sac_monitor.csv")
    eval_env = Monitor(BipedJumpEnv(render=False))

    model = SAC(
        env=env,
        tensorboard_log="logs/sac_goal/",
        **SAC_CONFIG
    )

    reward_cb = RewardPlotCallback()
    eval_cb   = EvalCallback(
        eval_env,
        best_model_save_path="models/sac_best/",
        log_path="logs/sac_eval/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[reward_cb, eval_cb],
        )
    except KeyboardInterrupt:
        print("\n[train] Interrupt — saving crash checkpoint...")
        model.save("models/sac_biped_crashsave")
        print("[train] Saved to models/sac_biped_crashsave.zip")

    model.save("models/sac_biped_goal")
    print("[train] Final model saved to models/sac_biped_goal.zip")
    reward_cb.plot_rewards("reward_curve_sac.png")
    env.close()
    eval_env.close()


# ── Evaluation ────────────────────────────────────────────────────────────────
def test(model_path: str, episodes: int, render: bool):
    DT = 1.0 / 50.0

    env   = BipedJumpEnv(render=render)
    model = SAC.load(model_path, env=env)

    joint_indices  = env.get_joint_indices()
    total_energy   = 0.0
    total_distance = 0.0
    total_reward   = 0.0
    fall_count     = 0

    for ep in range(episodes):
        obs, _    = env.reset()
        init_pos  = env.robot_initial_position()
        ep_reward = 0.0
        ep_energy = 0.0
        ep_steps  = 0
        fell      = False
        done      = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_steps  += 1
            done       = terminated or truncated

            for idx in joint_indices:
                js     = p.getJointState(env.robot_id, idx,
                                          physicsClientId=env.physics_client)
                torque = js[3]
                vel    = js[1]
                ep_energy += abs(torque * vel) * DT

            pos_b, orn_b = p.getBasePositionAndOrientation(
                env.robot_id, physicsClientId=env.physics_client
            )
            euler_b = p.getEulerFromQuaternion(orn_b)
            if abs(euler_b[0]) > 1.2 or abs(euler_b[1]) > 1.2:
                fell = True

        curr_pos = env.robot_current_position()
        dist = float(np.linalg.norm(
            np.array(curr_pos[:2]) - np.array(init_pos[:2])
        ))

        total_energy   += ep_energy
        total_distance += dist
        total_reward   += ep_reward
        if fell:
            fall_count += 1

        print(
            f"Episode {ep+1:>3}/{episodes}  |  "
            f"steps={ep_steps:>4}  |  "
            f"reward={ep_reward:>8.2f}  |  "
            f"energy={ep_energy:>7.2f} J  |  "
            f"dist={dist:>5.2f} m  |  "
            f"{'FELL' if fell else 'OK'}"
        )

    avg_reward   = total_reward   / episodes
    fall_rate    = 100.0 * fall_count / episodes
    avg_distance = total_distance / episodes
    avg_energy   = total_energy   / episodes
    CoT          = total_energy   / (ROBOT_MASS_KG * 9.81 * total_distance + 1e-8)

    print("\n" + "="*55)
    print("  EVALUATION SUMMARY")
    print("="*55)
    print(f"  Avg Reward       : {avg_reward:>10.2f}")
    print(f"  Fall Rate        : {fall_rate:>9.1f} %")
    print(f"  Avg Distance     : {avg_distance:>9.2f} m")
    print(f"  Avg Energy       : {avg_energy:>9.2f} J")
    print(f"  Cost of Transport: {CoT:>10.4f}")
    print("="*55)

    env.close()


# ── CLI entry-point ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Assignment 3 — Biped 1 m Platform Jump (SAC)"
    )
    parser.add_argument("--mode",       choices=["view", "train", "test"], required=True)
    parser.add_argument("--timesteps",  type=int,  default=None)
    parser.add_argument("--model_path", type=str,  default=None)
    parser.add_argument("--episodes",   type=int,  default=EVAL_EPISODES)
    parser.add_argument("--render",     action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "view":
        view()

    elif args.mode == "train":
        ts = args.timesteps if args.timesteps is not None else TOTAL_TIMESTEPS
        train(ts, args.render)

    elif args.mode == "test":
        model_path = (args.model_path if args.model_path is not None
                      else "models/sac_best/best_model")
        test(model_path, args.episodes, args.render)


if __name__ == "__main__":
    main()
```

---

## PART 4: KNOWN GOTCHAS — DO NOT FALL INTO THESE TRAPS

### Gotcha 1: The folder is named "assest" (NOT "assets")
Both `main.py` and `utils.py` use `assest`. Keep it exactly as is.

### Gotcha 2: The README CLI flags do NOT exist
- `--algo` → does NOT exist in parse_args()
- `--task` → does NOT exist in parse_args()
- `TASK_ENV` dict → does NOT exist in main.py
Do not add them. The code works without them.

### Gotcha 3: obs_dim MUST be exactly n_actuated*2 + 13
With 6 joints: obs_dim = 25. The concatenation in `_get_obs()` must produce exactly 25 floats. If you're off by even 1, gymnasium throws a shape mismatch error on the first `env.step()`.

### Gotcha 4: Contact detection uses plane_id NOT platform_id
- `self.plane_id` = ground (z=0)  
- `self.platform_id` = the 1m box  
Foot contacts for landing detection must check against `self.plane_id`. Checking against `platform_id` means the robot thinks it "landed" when it's still on the starting platform.

### Gotcha 5: Do NOT call p.disconnect in reset() or step()
`close()` handles disconnection. Calling it elsewhere causes crashes.

### Gotcha 6: EvalCallback requires `log_path` to be a directory that exists
`os.makedirs("logs/sac_eval", exist_ok=True)` MUST be called before EvalCallback is created.

### Gotcha 7: SAC.load() with env= parameter
When loading for test, always pass `env=env` to `SAC.load()` to ensure observation/action space compatibility.

### Gotcha 8: Monitor wraps the environment — test() should NOT use Monitor
`train()` wraps envs with Monitor. `test()` uses a raw `BipedJumpEnv` directly. This is correct.

---

## PART 5: TRAINING COMMANDS

```bash
# Install dependencies
pip install -r requirements.txt

# Verify environment loads (should open GUI window)
python main.py --mode view

# Train (default 1M steps — takes ~2-4 hours on CPU)
python main.py --mode train

# Quick smoke test (200k steps, ~30 min)
python main.py --mode train --timesteps 200000

# Evaluate best checkpoint
python main.py --mode test --episodes 10

# Evaluate with visualization
python main.py --mode test --render --episodes 3
```

---

## PART 6: HYPERPARAMETER TUNING (Task 2)

Three configurations to try:

**Config A (provided default — baseline):**
```python
learning_rate = 3e-4, batch_size = 256, gamma = 0.99, ent_coef = "auto"
```

**Config B (more exploration, larger buffer):**
```python
learning_rate = 1e-3, batch_size = 512, gamma = 0.995, ent_coef = 0.1
```

**Config C (conservative, slower but more stable):**
```python
learning_rate = 1e-4, batch_size = 128, gamma = 0.98, ent_coef = "auto"
```

Change only `SAC_CONFIG` in `utils.py` between runs. Save each reward curve with a different filename.

---

## PART 7: FINAL VERIFICATION CHECKLIST

Before running, verify:
- [ ] Folder named exactly `assest/` (not `assets/`)
- [ ] `assest/biped_.urdf` and `assest/stair.urdf` exist
- [ ] `utils.py` has no `None` values in SAC_CONFIG or top constants
- [ ] `python main.py --mode view` opens GUI without crash
- [ ] `python main.py --mode train --timesteps 1000` runs for 1000 steps without error
- [ ] `obs.shape == (25,)` (verify by printing in _get_obs during smoke test)
- [ ] After training, `models/sac_best/best_model.zip` exists before running test
