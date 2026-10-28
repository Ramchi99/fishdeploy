#!/usr/bin/env python3
"""
00_coordinate_frames.py
-----------------------
Presentation-quality coordinate frame reference for the field-test pipeline.

Five frames:
  ① Camera   (OpenCV)    – overhead camera above the pool
  ② AprilTag (Body)      – raw tag pose from dt_apriltags
  ③ World                – derived from floor tag + gravity; origin at ID0
  ④ MuJoCo               – field2sim simulation (X=swim, Y=lat, Z=vertical)
  ⑤ IMU      (BNO055)    – inertial sensor on the fish body (notes only)

Saved next to this script:
  00_coordinate_frames.pdf   (embed in slides / paper)
  00_coordinate_frames.png   (raster preview)

Run from anywhere:
  python3 field_test/pipeline/00_coordinate_frames.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, Circle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D                     # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# ── Output ──────────────────────────────────────────────────────────────────
OUT = os.path.dirname(os.path.abspath(__file__))

# ── Presentation style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9, "font.family": "sans-serif",
    "pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": False,
})
# Standard axis colours:  X=red  Y=green  Z=blue  (RGB = XYZ — universal convention)
CX, CY, CZ = '#D62728', '#2CA02C', '#1F77B4'
_AKW = dict(arrow_length_ratio=0.20, linewidth=2.5, alpha=0.92)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def draw_3d_frame(ax, origin, R, xlbl, ylbl, zlbl, L=0.72):
    """Draw RGB X/Y/Z arrows and labels.  R: 3×3 column-stacked axes."""
    o = np.asarray(origin, float)
    for col, lbl, axis in zip([CX, CY, CZ], [xlbl, ylbl, zlbl], R.T):
        v = axis / np.linalg.norm(axis) * L
        ax.quiver(*o, *v, color=col, **_AKW)
        ax.text(*(o + v * 1.44), lbl, color=col, fontsize=8, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.13', fc='white', alpha=0.78, ec='none'))


def style_3d(ax, title, elev=22, azim=-55):
    ax.set_title(title, fontsize=9, fontweight='bold', pad=5, linespacing=1.4)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#cccccc')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(True, alpha=0.10)
    ax.set_box_aspect([1, 1, 1])
    lim = 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=elev, azim=azim)


def add_face(ax, verts, fc='lightblue', ec='gray', alpha=0.13, lw=0.8):
    ax.add_collection3d(Poly3DCollection([verts], alpha=alpha,
                                          facecolor=fc, edgecolor=ec, linewidth=lw))


def arrow2d(ax, ox, oy, dx, dy, color, lw=2.5, lbl=None, lbl_offset=1.22, fs=8.5):
    ax.annotate('', xy=(ox + dx, oy + dy), xytext=(ox, oy),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=16))
    if lbl:
        ax.text(ox + dx * lbl_offset, oy + dy * lbl_offset, lbl,
                color=color, fontsize=fs, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.82, ec='none'))


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 12))
fig.patch.set_facecolor('#F8F9FA')

gs = gridspec.GridSpec(2, 4, figure=fig,
                       left=0.02, right=0.98, top=0.93, bottom=0.04,
                       wspace=0.06, hspace=0.38)

ax_cam  = fig.add_subplot(gs[0, 0], projection='3d')
ax_tag  = fig.add_subplot(gs[0, 1], projection='3d')
ax_wld  = fig.add_subplot(gs[0, 2], projection='3d')
ax_mu   = fig.add_subplot(gs[0, 3], projection='3d')
ax_pool = fig.add_subplot(gs[1, :3])   # 2-D pool overview
ax_tx   = fig.add_subplot(gs[1, 3])    # transform chain

fig.text(0.5, 0.975,
         'Coordinate Frame Reference  —  Field-Test Pipeline',
         ha='center', va='top', fontsize=15, fontweight='bold', color='#1a1a1a')


# ═══════════════════════════════════════════════════════════════════════════
# ① CAMERA FRAME  (OpenCV / dt_apriltags)
# ═══════════════════════════════════════════════════════════════════════════
# Camera hangs overhead pointing DOWN at the pool.
#   X  →  image right
#   Y  →  image down    (⚠ NOT up — OpenCV convention)
#   Z  →  optical axis into scene  (toward pool floor)
# The camera is mounted in landscape; a +90° CCW rotation (rot_around_z) is
# applied to the gravity vector in 02_process_swimming.py to compensate.

draw_3d_frame(ax_cam, [0, 0, 0], np.eye(3),
              'X\n(image right)', 'Y\n(image down)', 'Z\n(optical axis\n→ pool floor)')

sensor = [[-0.46, -0.29, -0.42], [0.46, -0.29, -0.42],
          [0.46,   0.29, -0.42], [-0.46, 0.29, -0.42]]
add_face(ax_cam, sensor, fc='#888888', ec='#888888', alpha=0.14)
ax_cam.text(0, 0, -0.60, 'sensor plane', fontsize=7, ha='center', color='#888888')

floor = [[-0.9, -0.6, 0.92], [0.9, -0.6, 0.92],
         [0.9,   0.6, 0.92], [-0.9, 0.6, 0.92]]
add_face(ax_cam, floor, fc='#4488CC', ec='#4488CC', alpha=0.10)
ax_cam.text(0, 0, 1.06, 'pool surface', fontsize=7, ha='center', color='#4488CC')

style_3d(ax_cam, '① Camera Frame\n(OpenCV convention)', elev=18, azim=-52)


# ═══════════════════════════════════════════════════════════════════════════
# ② APRILTAG BODY FRAME
# ═══════════════════════════════════════════════════════════════════════════
# dt_apriltags returns pose_R and pose_t in the *camera* frame.
# The tag's own body axes (used to read off orientation):
#   Tag X  →  right side of printed square
#   Tag Y  →  top  of printed square  ← KEY: used as "swim-forward / North"
#             when constructing the world frame from the floor tag
#   Tag Z  →  normal out of page (toward camera, right-hand rule)

draw_3d_frame(ax_tag, [0, 0, 0], np.eye(3),
              'Tag X\n(right side)', 'Tag Y\n(top = swim fwd)', 'Tag Z\n(toward cam)')

sq = [[-0.56, -0.56, 0], [0.56, -0.56, 0], [0.56, 0.56, 0], [-0.56, 0.56, 0]]
add_face(ax_tag, sq, fc='#FFCC44', ec='#333333', alpha=0.32, lw=1.5)
ax_tag.text(0, 0, -0.10, '⊞ tag', fontsize=12, ha='center', va='top',
            color='#555555', alpha=0.45)
ax_tag.text(0.10, 1.10, 0.20,
            '← "North" used\n   to build world\n   frame from ID 0',
            fontsize=7, color=CY, style='italic')

style_3d(ax_tag, '② AprilTag Body Frame\n(fish tag ID 1  |  floor tag ID 0)',
         elev=32, azim=-42)


# ═══════════════════════════════════════════════════════════════════════════
# ③ WORLD FRAME  (02_process_swimming.py → get_rotation_matrix)
# ═══════════════════════════════════════════════════════════════════════════
# Built from the *floor* tag (ID0) average pose + IMU gravity vector:
#   Y_world  =  gravity direction  (DOWN  — IMU reports g, not reaction force)
#   X_world  =  cross(Tag-Y_floor, Y_world)  →  lateral LEFT
#   Z_world  =  cross(X_world, Y_world)      →  swim FORWARD
#   Origin   =  average floor-tag translation in camera frame
#
# Right-handed, Y-DOWN  (inherits the OpenCV/camera Y-down lineage).

draw_3d_frame(ax_wld, [0, 0, 0], np.eye(3),
              'X\n(lateral\nLEFT)',
              'Y\n(DOWN\n= gravity)',
              'Z\n(swim\nFORWARD)')

# gravity dashed arrow
ax_wld.quiver(0, 0, 0, 0, 0.55, 0, color='#999999', linewidth=1.5,
              linestyle=':', alpha=0.65, arrow_length_ratio=0.22)
ax_wld.text(0.12, 0.70, 0.0, 'g↓', fontsize=9, color='#999999', style='italic')

# floor tag at origin
ax_wld.scatter([0], [0], [0], color='#333333', s=75, zorder=5)
ax_wld.text(0.12, 0.02, 0.12,
            'floor tag\n(ID 0)\nworld origin', fontsize=7, color='#333333')

# pool floor plane at Y=0 (horizontal surface, X-Z plane)
surf = [[-0.9, -0.03, -0.9], [0.9, -0.03, -0.9],
        [0.9,  -0.03,  0.9], [-0.9, -0.03,  0.9]]
add_face(ax_wld, surf, fc='#AACCFF', ec='#AACCFF', alpha=0.12)
ax_wld.text(0, -0.20, 0.95, 'pool floor (XZ plane)', fontsize=7,
            ha='center', color='#5577AA')

style_3d(ax_wld, '③ World Frame\n(Y-down, origin at floor tag)', elev=18, azim=-55)


# ═══════════════════════════════════════════════════════════════════════════
# ④ MUJOCO FRAME  (03_prep_field2sim.py → field2sim simulation)
# ═══════════════════════════════════════════════════════════════════════════
# Axis relabelling from World (03_prep_field2sim.py lines 85–91):
#   MuJoCo X  =  World Z   (swim FORWARD)
#   MuJoCo Y  =  World X   (lateral  — same sign, lateral LEFT)
#   MuJoCo Z  =  World Y   (vertical — World Y=DOWN so Z≈down;
#                            fish is constrained to Z=0, so sign rarely matters)
#
# Yaw convention: positive = CCW in XY-plane viewed from above (+Z).
#   Yaw = 0  →  fish points along +X (swim forward)
#   +Yaw     →  fish turns LEFT (toward +Y)

draw_3d_frame(ax_mu, [0, 0, 0], np.eye(3),
              'X\n(swim\nFORWARD)',
              'Y\n(lateral\nLEFT)',
              'Z\n(vertical\n= World Y)')

# fish body (cylinder sketch)
theta = np.linspace(0, 2 * np.pi, 40)
r = 0.10
for xi in [0.0, 0.62]:
    ax_mu.plot(np.full(40, xi), r * np.cos(theta), r * np.sin(theta),
               color='#7B68EE', alpha=0.25, linewidth=0.8)
ax_mu.plot([0, 0.62], [0, 0], [0, 0], color='#7B68EE', linewidth=2.8, alpha=0.55)
ax_mu.text(0.74, 0, 0, 'fish', fontsize=8, color='#7B68EE', va='center')

# +Yaw arc (CCW from +X toward +Y in XY plane at Z=0)
t_arc = np.linspace(0, np.pi / 2.5, 50)
arc_r = 0.48
ax_mu.plot(arc_r * np.cos(t_arc), arc_r * np.sin(t_arc), np.zeros(50),
           color='purple', linewidth=2.2, alpha=0.80)
# arrowhead at end of arc
i = -1
ax_mu.quiver(arc_r * np.cos(t_arc[i]), arc_r * np.sin(t_arc[i]), 0,
             -np.sin(t_arc[i]) * 0.001, np.cos(t_arc[i]) * 0.001, 0,
             color='purple', arrow_length_ratio=800, linewidth=2.2)
ax_mu.text(0.18, 0.54, 0.10, '+Yaw\n(CCW)', fontsize=8, color='purple',
           fontweight='bold')

style_3d(ax_mu, '④ MuJoCo Frame\n(field2sim simulation, X=swim Y=lat)',
         elev=28, azim=-52)


# ═══════════════════════════════════════════════════════════════════════════
# ⑤ POOL TOP-DOWN VIEW  (2-D schematic)
# ═══════════════════════════════════════════════════════════════════════════
# Orientation matches the trajectory plot in 04_plot_field_test.py:
#   plot-X  =  World Z  =  MuJoCo X   (swim forward → right)
#   plot-Y  =  −World X =  MuJoCo Y   (positive = fish goes UP in pool)

ax = ax_pool
ax.set_facecolor('#ECF4FF')
ax.set_aspect('equal')

pool_w, pool_h = 3.2, 1.5
# pool boundary
pool_rect = FancyBboxPatch((-0.18, -pool_h / 2 - 0.18),
                            pool_w + 0.36, pool_h + 0.36,
                            boxstyle='round,pad=0.0', linewidth=2.2,
                            edgecolor='#2255AA', facecolor='#C8E0FF', alpha=0.35)
ax.add_patch(pool_rect)
ax.text(pool_w / 2, pool_h / 2 + 0.30,
        'Pool  —  top-down view  (camera mounted overhead, looking down)',
        ha='center', fontsize=9, color='#2255AA', style='italic')

# ── Floor tag at origin ────────────────────────────────────────────────────
ts = 0.10
floor_sq = FancyBboxPatch((-ts, -ts), 2 * ts, 2 * ts,
                           boxstyle='square,pad=0', lw=1.8,
                           edgecolor='black', facecolor='#FFCC44')
ax.add_patch(floor_sq)
ax.text(0, -0.28, 'Floor Tag\n(ID 0)\nworld origin', ha='center',
        fontsize=7.5, color='#333333')

# World frame at floor tag
arrow2d(ax, 0, 0,  0.60, 0,    CZ, lw=2.8, lbl='World +Z\n(MuJoCo +X)\nswim FORWARD')
arrow2d(ax, 0, 0,  0,    0.50, CX, lw=2.8, lbl='World +X\n(MuJoCo +Y)\nlateral LEFT')
# World +Y = DOWN (into page) shown as circled cross
ax.plot(-0.42, -0.30, 'o', ms=13, color=CY, alpha=0.55, mew=0)
ax.plot(-0.42, -0.30, 'x', ms=9,  color=CY, alpha=0.95, mew=2.5)
ax.text(-0.42, -0.47, 'World +Y\n(↓ into page\n= gravity)',
        ha='center', fontsize=7.5, color=CY)

# ── Fish at a representative mid-pool position with yaw 25° ───────────────
fish_cx, fish_cy = 2.0, 0.32
yaw_deg = 25.0
yaw_rad = np.radians(yaw_deg)

fish_body = Ellipse((fish_cx, fish_cy), width=0.40, height=0.15,
                     angle=yaw_deg, fc='#7B68EE', ec='#333388', alpha=0.75, lw=1.6)
ax.add_patch(fish_body)
ax.text(fish_cx, fish_cy + 0.24, 'Fish  (ID 1)', ha='center',
        fontsize=8.5, color='#333388', fontweight='bold')

# swim direction arrow
sw = 0.42
arrow2d(ax, fish_cx, fish_cy,
        sw * np.cos(yaw_rad), sw * np.sin(yaw_rad),
        '#7B68EE', lw=2.2, lbl='swim dir', lbl_offset=1.3, fs=7.5)

# positive yaw arc (CCW from +Z/swim-axis)
arc_r2 = 0.30
t_a = np.linspace(0, yaw_rad, 50)
ax.plot(fish_cx + arc_r2 * np.cos(t_a),
        fish_cy + arc_r2 * np.sin(t_a), color='purple', lw=2.2, alpha=0.85)
ax.annotate('', xy=(fish_cx + arc_r2 * np.cos(t_a[-1]),
                    fish_cy + arc_r2 * np.sin(t_a[-1])),
            xytext=(fish_cx + arc_r2 * np.cos(t_a[-2]),
                    fish_cy + arc_r2 * np.sin(t_a[-2])),
            arrowprops=dict(arrowstyle='->', color='purple', lw=1.8, mutation_scale=12))
ax.text(fish_cx + arc_r2 * 1.28, fish_cy + arc_r2 * 0.45,
        f'+Yaw = {yaw_deg:.0f}°\n(CCW, same in\nWorld & MuJoCo)',
        fontsize=7.5, color='purple', fontweight='bold')

# AprilTag mini-square on fish
tag_fx = fish_cx + 0.02 * np.sin(yaw_rad)
tag_fy = fish_cy - 0.02 * np.cos(yaw_rad)
tag_sq = FancyBboxPatch((tag_fx - 0.048, tag_fy - 0.048), 0.096, 0.096,
                         boxstyle='square,pad=0', lw=1.0,
                         edgecolor='black', facecolor='#FFCC44', alpha=0.90)
ax.add_patch(tag_sq)

# Tag Y on fish (top of tag = swim fwd)
ty_len = 0.26
arrow2d(ax, tag_fx, tag_fy,
        ty_len * np.cos(yaw_rad), ty_len * np.sin(yaw_rad),
        CY, lw=1.8, lbl='Tag Y\n(top=fwd)', lbl_offset=1.55, fs=7)

# Tag X on fish (right side of tag = lateral right relative to fish)
tx_len = 0.18
arrow2d(ax, tag_fx, tag_fy,
        tx_len * np.cos(yaw_rad - np.pi / 2),
        tx_len * np.sin(yaw_rad - np.pi / 2),
        CX, lw=1.5, lbl='Tag X', lbl_offset=1.7, fs=7)

# ── Camera icon ────────────────────────────────────────────────────────────
cx, cy = 2.85, -0.72
ax.add_patch(FancyBboxPatch((cx - 0.20, cy - 0.11), 0.40, 0.22,
                             boxstyle='round,pad=0.03', lw=1.5,
                             edgecolor='#444', facecolor='#DDDDDD'))
ax.add_patch(Circle((cx, cy), 0.068, fc='#888', ec='#444', lw=1.2))
ax.text(cx, cy - 0.26, 'Camera\n(overhead)', ha='center', fontsize=7.5, color='#444444')
ax.plot([cx, 0], [cy, 0], '--', color='#BBBBBB', lw=1.1, alpha=0.55)

# ── IMU notes ──────────────────────────────────────────────────────────────
ax.text(0.85, -0.72,
        'IMU (BNO055) on fish:\n'
        '  GyrZ  = yaw rate  (rad/s)\n'
        '  RawY  = fused heading (°)\n'
        '           ⚠ negated in code:\n'
        '           BNO055 RawY↓ for CCW,\n'
        '           MuJoCo needs +ψ for CCW\n'
        '  AccXYZ → avg = gravity vector\n'
        '           → defines World Y (down)',
        fontsize=7.5, va='top', color='#333333',
        bbox=dict(boxstyle='round,pad=0.4', fc='#FFFDE8', ec='#CCAA33', lw=1.2))

ax.set_xlim(-0.42, pool_w + 0.32)
ax.set_ylim(-pool_h / 2 - 0.60, pool_h / 2 + 0.55)
ax.set_xlabel('World Z  /  MuJoCo X  (swim axis, m)', fontsize=9)
ax.set_ylabel('−World X  /  MuJoCo Y  (lateral, m)', fontsize=9)
ax.set_title('⑤ Pool Top-Down View  (axes match the trajectory plot in 04_plot_field_test.py)',
             fontsize=9, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.28)


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORMATION CHAIN (text panel)
# ═══════════════════════════════════════════════════════════════════════════
ax_tx.axis('off')
summary = (
    'Transformation Chain\n'
    '─────────────────────────────────────\n\n'
    '(1)Camera -> (3)World\n'
    '  R = get_rotation_matrix(R_ID0, avg_g).T\n'
    '  p_world = R @ (t_fish - t_floor)\n'
    '  (floor tag pose + gravity vector)\n\n'
    '(3)World -> (4)MuJoCo\n'
    '  X_mu =  Z_world  (swim forward)\n'
    '  Y_mu = -X_world  (lateral; negated vs World X)\n'
    '  Z_mu =  Y_world  (vertical; locked\n'
    '                   to 0 in sim)\n\n'
    '─────────────────────────────────────\n'
    'Yaw  (Vision / AprilTag):\n'
    '  fwd = R_fish_world[:,1]  (Tag Y)\n'
    '  psi = arctan2(fwd[0], fwd[2])\n'
    '  zero: fish along +Z_world/+X_mu\n\n'
    'Yaw  (IMU / BNO055):\n'
    '  psi_imu = -(RawY - RawY0)  [rad]\n'
    '  negate: RawY down for CCW, but\n'
    '  MuJoCo needs +psi for CCW\n\n'
    'Motor  (encoder -> radians):\n'
    '  alpha = (ticks-center)*2pi/4096\n\n'
    '─────────────────────────────────────\n'
    'Colour key:   '
)
ax_tx.text(0.04, 0.97, summary, transform=ax_tx.transAxes,
           fontsize=7.8, va='top', ha='left', family='monospace',
           bbox=dict(boxstyle='round,pad=0.55', fc='#F5F5F5', ec='#CCCCCC', lw=1.2))

# colour legend dots
for i, (col, lbl) in enumerate([(CX, 'X'), (CY, 'Y'), (CZ, 'Z')]):
    ax_tx.text(0.48 + i * 0.175, 0.032, f'■ {lbl}', transform=ax_tx.transAxes,
               fontsize=9, color=col, fontweight='bold', va='bottom')


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
for ext in ('pdf', 'png'):
    path = os.path.join(OUT, f'00_coordinate_frames.{ext}')
    fig.savefig(path, dpi=200 if ext == 'png' else None, bbox_inches='tight')
    print(f'Saved: {path}')

plt.show()
