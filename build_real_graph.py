"""
build_real_graph.py — 从 YJK 计算结果构建完整的 structure_graph.pt

从以下文件提取真实数据:
  1. STRUCTURE.DAT → 图结构 (节点、边、截面属性)
  2. Chi-Chi *.DAT → 地震波加速度时程 (X/Y/Z, 6017 步, dt=0.005s)
  3. PostEP/PLSTIC/*.DISP → 楼层响应时程 (位移/速度/加速度)

输出格式与 GNN-LSTM 的 GraphDataset 完全兼容:
  ground_motion: [20000] float64   (200Hz, 100s)
  Acceleration/Velocity/Displacement: [2000, max_story] float64  (20Hz, 100s)
"""

import sys
import os
import struct
import argparse
import numpy as np
import torch
from torch_geometric.data import Data

sys.stdout.reconfigure(encoding='utf-8')

# ─── 参数 ───
MAX_STORY = 8
SAMPLE_RATE = 0.005       # 200 Hz
DURATION = 100.0           # 目标总时长 100s
SRC_LEN = int(DURATION / SAMPLE_RATE)  # 20000
TRG_LEN = SRC_LEN // 10                # 2000 (输出 20Hz)


def load_ground_motion(dat_path):
    """从 YJK 的 .DAT 文件提取地震波加速度时程。

    Returns:
        gm_x: [N] float64, X 方向加速度 (cm/s²)
        gm_y: [N] float64, Y 方向加速度 (cm/s²)
        dt: float, 时间步长 (s)
    """
    with open(dat_path, 'rb') as f:
        raw = f.read()

    # 头部: int32(N_steps), float32(总能量?), float32(阻尼比), float32(X峰值), float32(Y峰值)
    n_steps = struct.unpack('<i', raw[:4])[0]
    header = np.frombuffer(raw[4:20], dtype=np.float32)
    print(f"  .DAT header: N={n_steps}, peaks: X={header[2]:.2f}, Y={header[3]:.2f} cm/s²")

    # 数据: 从 byte 4 + 4*4=20 开始, 每步 3 个 float32 (x, y, z=0)
    # 但实际数据从 float32[4] 开始, 格式: (z, x, y) 或 (x, y, z)
    arr32 = np.frombuffer(raw[4:], dtype=np.float32)
    # 验证: 4 header float32 + n_steps * 3 data float32 = 4 + 18051 = 18055
    # arr32 有 18052 个, 所以 header=5 个 float32?
    # 实际测试: 从 arr32[4] 开始, 每3个为一组, 共6017组, 峰值匹配
    data = arr32[4:4 + n_steps * 3]
    triplets = data.reshape(n_steps, 3)

    gm_x = triplets[:, 0].astype(np.float64)  # X 方向
    gm_y = triplets[:, 1].astype(np.float64)   # Y 方向

    return gm_x, gm_y, SAMPLE_RATE


def load_floor_response(disp_path, postdb_path, earthquake_angle=90):
    """从 DISP 文件提取楼层响应时程。

    DISP 文件格式: float32, 无 header, shape = [6017, 2215]
    每步 2215 个 float32:
      - 246 节点 × 9 值 = 2214 (dx,dy,dz, vx,vy,vz, ax,ay,az)
      - 1 个 flag (999999)

    Args:
        disp_path: DISP 文件路径
        postdb_path: postdb.db 路径 (含节点-楼层映射)
        earthquake_angle: 地震方向角 (度, 0=X, 90=Y)

    Returns:
        disp: [N, n_floors] 楼层位移 (mm)
        vel:  [N, n_floors] 楼层速度 (mm/s)
        acc:  [N, n_floors] 楼层加速度 (mm/s²)
    """
    import sqlite3

    # 读取节点-楼层映射
    conn = sqlite3.connect(postdb_path)
    cur = conn.cursor()
    cur.execute('SELECT NodeLable, FlrNo FROM tblNodeInfo ORDER BY NodeLable')
    node_floors = {r[0]: r[1] for r in cur.fetchall()}
    cur.execute('SELECT COUNT(DISTINCT FlrNo) FROM tblNodeInfo')
    n_floors = cur.fetchone()[0]  # 含 FlrNo=0 (base)
    conn.close()

    print(f"  节点数: {len(node_floors)}, 楼层数: {n_floors} (含基底)")

    # 读取 DISP
    raw = open(disp_path, 'rb').read()
    arr32 = np.frombuffer(raw, dtype=np.float32)
    n_steps = 6017
    cols_per_step = len(arr32) // n_steps
    mat = arr32.reshape(n_steps, cols_per_step)
    print(f"  DISP: {mat.shape}, 每步 {cols_per_step} 列")

    # 确定位移/速度/加速度列索引
    # 地震方向 90° → Y方向; 0° → X方向
    if earthquake_angle == 90:
        dy_col, vy_col, ay_col = 1, 4, 7  # Y方向在9值组内的索引
        dir_name = "Y"
    else:
        dy_col, vy_col, ay_col = 0, 3, 6  # X方向
        dir_name = "X"
    print(f"  方向: {dir_name}, 位移列={dy_col}, 速度列={vy_col}, 加速度列={ay_col}")

    # 按楼层聚合
    disp = np.zeros((n_steps, n_floors), dtype=np.float64)
    vel = np.zeros((n_steps, n_floors), dtype=np.float64)
    acc = np.zeros((n_steps, n_floors), dtype=np.float64)
    counts = np.zeros(n_floors, dtype=np.float64)

    for nid, flr in node_floors.items():
        base = (nid - 1) * 9
        d_col = base + dy_col
        v_col = base + vy_col
        a_col = base + ay_col
        if d_col < cols_per_step and v_col < cols_per_step and a_col < cols_per_step:
            disp[:, flr] += mat[:, d_col].astype(np.float64)
            vel[:, flr] += mat[:, v_col].astype(np.float64)
            acc[:, flr] += mat[:, a_col].astype(np.float64)
            counts[flr] += 1

    for flr in range(n_floors):
        if counts[flr] > 0:
            disp[:, flr] /= counts[flr]
            vel[:, flr] /= counts[flr]
            acc[:, flr] /= counts[flr]

    return disp, vel, acc


def resample_to_target(data, src_dt, trg_dt, trg_len):
    """将时程数据重采样/对齐到目标长度。

    Args:
        data: [N] 或 [N, C] 原始数据
        src_dt: 原始时间步长
        trg_dt: 目标时间步长
        trg_len: 目标长度

    Returns:
        resampled: [trg_len] 或 [trg_len, C]
    """
    n_src = len(data)
    src_time = np.arange(n_src) * src_dt
    trg_time = np.arange(trg_len) * trg_dt

    if data.ndim == 1:
        resampled = np.interp(trg_time, src_time, data, left=0, right=0)
    else:
        resampled = np.zeros((trg_len, data.shape[1]), dtype=data.dtype)
        for c in range(data.shape[1]):
            resampled[:, c] = np.interp(trg_time, src_time, data[:, c], left=0, right=0)

    return resampled


def pad_to_max_story(data, max_story):
    """将楼层数据填充到 max_story 列 (不足部分补零)."""
    n_floors = data.shape[1]
    if n_floors >= max_story:
        return data[:, :max_story]
    padded = np.zeros((data.shape[0], max_story), dtype=data.dtype)
    padded[:, :n_floors] = data
    return padded


def convert_units(disp_mm, vel_mms, acc_mmss2):
    """单位转换: YJK 内部使用 kN-mm-s, GNN-LSTM 也使用 kN-mm-s.

    检查是否需要转换:
    - YJK DISP 输出单位: mm (位移), mm/s (速度), mm/s² (加速度)
    - GNN-LSTM 期望单位: 同上
    所以通常不需要转换。
    """
    # 检查量级是否合理
    max_d = np.max(np.abs(disp_mm))
    max_v = np.max(np.abs(vel_mms))
    max_a = np.max(np.abs(acc_mmss2))
    print(f"  位移最大值: {max_d:.4f} mm")
    print(f"  速度最大值: {max_v:.4f} mm/s")
    print(f"  加速度最大值: {max_a:.2f} mm/s² = {max_a/9800:.4f} g")

    return disp_mm, vel_mms, acc_mmss2


def build_graph(
    structure_graph_path,  # inp_to_graph.py 生成的图结构
    dat_path,              # 地震波 .DAT
    disp_path,             # 楼层响应 .DISP
    postdb_path,           # 模型信息 postdb.db
    output_path,           # 输出 structure_graph.pt
    earthquake_angle=90,   # 地震方向角 (0=X, 90=Y)
    gm_name="Unknown",
    n_real_floors=None,    # 实际楼层数 (None=自动检测)
):
    """构建完整的 structure_graph.pt"""
    print("=" * 70)
    print("构建真实 structure_graph.pt")
    print("=" * 70)

    # ─── Step 1: 加载图结构 ───
    print("\n[Step 1] 加载图结构")
    graph = torch.load(structure_graph_path, weights_only=False)
    print(f"  节点: {graph.x.shape}, 边: {graph.edge_index.shape}")
    print(f"  边特征: {graph.edge_attr.shape}")

    # ─── Step 1b: 修复 story (楼层数) ───
    # dataset.py 用 graph.story = int(graph.x[0, 1]) - 1 确定楼层
    # 原始 x[:,1] 是 Y 方向网格线数 (对5层结构=4), 导致只画3层
    # 修复: 将 x[:,1] 设为 n_real_floors + 1, 使得 int(x[0,1])-1 = n_real_floors
    # 同时修复 col 0 (X网格线数) 和 col 2 (Z网格线数) 以保持一致性
    if n_real_floors is not None:
        # Z方向网格线数 = n_real_floors + 1 (含基底)
        graph.x[:, 2] = float(n_real_floors + 1)
        # Y方向网格线数设为 n_real_floors + 1 (使得 story = n_real_floors)
        graph.x[:, 1] = float(n_real_floors + 1)
        print(f"  修复 story: x[:,1] = {n_real_floors + 1}, x[:,2] = {n_real_floors + 1}")
        print(f"  → int(x[0,1])-1 = {n_real_floors} 层")
    else:
        print(f"  x[0,1] = {graph.x[0,1].item()}, int(x[0,1])-1 = {int(graph.x[0,1].item()) - 1}")

    # ─── Step 1c: 填充节点特征的占位数据 ───
    # 对于 YJK 导出的结构，以下特征可能为零（无模态/质量数据）：
    #   col 6: 第一周期 T1 (s)
    #   col 7: DOF 标记 (0=固定, 1=自由)
    #   col 8: 节点质量 (ton = kN·s²/mm → 约 10³ kg)
    #   col 9-11: X/Y/Z 惯性矩
    #   col 12-14: X/Y/Z 振型
    # 这些为零会导致归一化除以零 → NaN，必须填充非零占位值
    n_nodes = graph.x.shape[0]

    # col 6: 第一周期 T1 (钢筋混凝土框架结构典型值 0.4~1.0s)
    if torch.max(torch.abs(graph.x[:, 6])) == 0:
        # 5层框架结构 T1 ≈ 0.08*N ≈ 0.4s
        T1_placeholder = 0.4363  # s
        graph.x[:, 6] = T1_placeholder
        print(f"  填充 T1 = {T1_placeholder} s (占位)")

    # col 7: DOF 标记
    if torch.max(torch.abs(graph.x[:, 7])) == 0:
        graph.x[:, 7] = 1.0  # 全部设为自由节点
        print(f"  填充 DOF = 1.0 (全部自由)")

    # col 8: 节点质量 (单位: ton = kN·s²/mm)
    if torch.max(torch.abs(graph.x[:, 8])) == 0:
        # 典型楼层质量: 5层框架约 500 ton 总重 → 每节点约 2~5 ton
        graph.x[:, 8] = 1.0
        print(f"  填充 mass = 1.0 (占位)")

    # col 9-11: 惯性矩 (Ixx, Iyy, Izz)
    if torch.max(torch.abs(graph.x[:, 9:12])) == 0:
        graph.x[:, 9:12] = 1.0
        print(f"  填充 XYZ_inertia = 1.0 (占位)")

    # col 12-14: 振型分量 (X, Y, Z)
    if torch.max(torch.abs(graph.x[:, 12:15])) == 0:
        # 使用线性振型: 按节点高度比例分配 (0→1)
        z_coords = graph.x[:, 5]  # Z方向网格索引
        if torch.max(torch.abs(z_coords)) > 0:
            z_norm = z_coords / torch.max(z_coords)
        else:
            z_norm = torch.ones(n_nodes)
        # Y方向地震占优 → Y振型分量最大
        graph.x[:, 12] = 0.3 * z_norm   # X振型 (小)
        graph.x[:, 13] = z_norm          # Y振型 (主方向)
        graph.x[:, 14] = 0.1 * z_norm   # Z振型 (小)
        print(f"  填充 XYZ_mode_shape: 线性Y方向占优振型")

    # ─── Step 2: 加载地震波 ───
    print("\n[Step 2] 加载地震波")
    gm_x, gm_y, gm_dt = load_ground_motion(dat_path)

    if earthquake_angle == 90:
        gm_raw = gm_y
    else:
        gm_raw = gm_x

    print(f"  原始: {len(gm_raw)} 点, dt={gm_dt}s, 峰值={np.max(np.abs(gm_raw)):.2f} cm/s²")

    # 转换为 mm/s² (GNN-LSTM 使用 kN-mm-s)
    # 1 cm/s² = 10 mm/s²
    gm_raw_mmss2 = gm_raw * 10.0
    print(f"  转换后: 峰值={np.max(np.abs(gm_raw_mmss2)):.2f} mm/s²")

    # 重采样到 20000 点 (100s, 0.005s)
    gm_resampled = resample_to_target(gm_raw_mmss2, gm_dt, SAMPLE_RATE, SRC_LEN)
    print(f"  重采样: {len(gm_resampled)} 点, 峰值={np.max(np.abs(gm_resampled)):.2f} mm/s²")

    # 计算实际数据持续时间 (原始数据长度)
    actual_duration = len(gm_raw) * gm_dt  # 秒
    actual_src_len = len(gm_raw)            # 原始步数
    actual_trg_len = int(actual_duration / (SAMPLE_RATE * 10))  # 对应的输出步数
    print(f"  实际持续时间: {actual_duration:.1f}s ({actual_src_len} 步), 对应输出 {actual_trg_len} 步")

    graph.ground_motion = torch.tensor(gm_resampled, dtype=torch.float64)
    graph.ground_motion_name = gm_name

    # ─── Step 3: 加载楼层响应 ───
    print("\n[Step 3] 加载楼层响应")
    disp_raw, vel_raw, acc_raw = load_floor_response(disp_path, postdb_path, earthquake_angle)

    print(f"  原始: {disp_raw.shape[0]} 步 × {disp_raw.shape[1]} 层")

    # 单位检查/转换
    disp_raw, vel_raw, acc_raw = convert_units(disp_raw, vel_raw, acc_raw)

    # 对齐到目标格式:
    # - 输入: 200Hz (0.005s), 6017 点 → 重采样到 20000 点
    # - 输出: 20Hz (0.05s), 先从原始 200Hz 降采样到 20Hz
    #   - 原始输出频率 = 200Hz, 目标输出频率 = 20Hz (每10个取1个)
    #   - 但原始只有 6017 点 ≈ 30s, 目标 2000 点 = 100s
    #   - 所以需要: 先插值到 20000 点 (100s), 然后降采样到 2000 点 (20Hz)

    # 方案A: 直接从 6017@200Hz 重采样到 2000@20Hz
    # 输出步长 = 10 * 输入步长 = 0.05s
    trg_dt_output = SAMPLE_RATE * 10  # 0.05s

    disp_resampled = resample_to_target(disp_raw, SAMPLE_RATE, trg_dt_output, TRG_LEN)
    vel_resampled = resample_to_target(vel_raw, SAMPLE_RATE, trg_dt_output, TRG_LEN)
    acc_resampled = resample_to_target(acc_raw, SAMPLE_RATE, trg_dt_output, TRG_LEN)

    print(f"  重采样: {TRG_LEN} 步 × {disp_resampled.shape[1]} 层")

    # 填充到 max_story=8 列
    disp_final = pad_to_max_story(disp_resampled, MAX_STORY)
    vel_final = pad_to_max_story(vel_resampled, MAX_STORY)
    acc_final = pad_to_max_story(acc_resampled, MAX_STORY)

    print(f"  填充后: {disp_final.shape}")

    graph.Displacement = torch.tensor(disp_final, dtype=torch.float32)
    graph.Velocity = torch.tensor(vel_final, dtype=torch.float32)
    graph.Acceleration = torch.tensor(acc_final, dtype=torch.float32)

    # ─── Step 4: 设置元数据 ───
    print("\n[Step 4] 设置元数据")
    graph.time_steps = SRC_LEN  # 20000 (int, 对应100s)
    graph.sample_rate = SAMPLE_RATE  # 0.005 (float)
    graph.response_type = "Displacement"  # 默认
    graph.actual_duration = actual_duration  # 实际数据持续时间 (s)
    graph.actual_trg_len = actual_trg_len   # 实际输出步数

    # ─── Step 5: 保存 ───
    print("\n[Step 5] 保存")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save(graph, output_path)

    # ─── 验证 ───
    print("\n[验证] 重新加载检查")
    verify = torch.load(output_path, weights_only=False)
    print(f"  x: {verify.x.shape}")
    print(f"  edge_index: {verify.edge_index.shape}")
    print(f"  edge_attr: {verify.edge_attr.shape}")
    print(f"  ground_motion: {verify.ground_motion.shape}, dtype={verify.ground_motion.dtype}")
    print(f"  Displacement: {verify.Displacement.shape}, dtype={verify.Displacement.dtype}")
    print(f"  Velocity: {verify.Velocity.shape}, dtype={verify.Velocity.dtype}")
    print(f"  Acceleration: {verify.Acceleration.shape}, dtype={verify.Acceleration.dtype}")
    print(f"  time_steps: {verify.time_steps}")
    print(f"  sample_rate: {verify.sample_rate}")
    print(f"  ground_motion_name: {verify.ground_motion_name}")

    # 值域检查
    print(f"\n  ground_motion 峰值: {torch.max(torch.abs(verify.ground_motion)):.2f} mm/s²")
    for resp in ["Displacement", "Velocity", "Acceleration"]:
        r = getattr(verify, resp)
        print(f"  {resp} 峰值: {torch.max(torch.abs(r)):.4f} (非零层: {torch.any(r != 0, dim=0).sum().item()})")

    print("\n" + "=" * 70)
    print(f"完成! 已保存到: {output_path}")
    print("=" * 70)

    return graph


def main():
    parser = argparse.ArgumentParser(
        description="从 YJK 计算结果构建 structure_graph.pt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程
  python build_real_graph.py \\
      --structure_graph ./output/structure_graph.pt \\
      --dat ./earthquake.DAT \\
      --disp ./PLSTIC/earthquake.DISP \\
      --postdb ./PostEP/postdb.db \\
      --output ./result/structure_graph.pt \\
      --n_floors 5
        """,
    )
    parser.add_argument("--structure_graph", type=str, required=True,
                        help="inp_to_graph.py 生成的图结构 .pt 文件")
    parser.add_argument("--dat", type=str, required=True,
                        help="地震波 .DAT 文件")
    parser.add_argument("--disp", type=str, required=True,
                        help="DISP 响应文件 (PostEP/PLSTIC/*.DISP)")
    parser.add_argument("--postdb", type=str, required=True,
                        help="模型信息数据库 (PostEP/postdb.db)")
    parser.add_argument("--output", type=str,
                        default="./output/structure_graph.pt",
                        help="输出路径 (default: ./output/structure_graph.pt)")
    parser.add_argument("--angle", type=float, default=90,
                        help="地震方向角: 0=X方向, 90=Y方向 (default: 90)")
    parser.add_argument("--gm_name", type=str, default="Unknown",
                        help="地震波名称 (default: Unknown)")
    parser.add_argument("--n_floors", type=int, default=None,
                        help="实际楼层数，用于修正 story 字段 (default: 自动检测)")
    args = parser.parse_args()

    build_graph(
        structure_graph_path=args.structure_graph,
        dat_path=args.dat,
        disp_path=args.disp,
        postdb_path=args.postdb,
        output_path=args.output,
        earthquake_angle=args.angle,
        gm_name=args.gm_name,
        n_real_floors=args.n_floors,
    )


if __name__ == "__main__":
    main()
