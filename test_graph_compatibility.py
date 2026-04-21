"""
test_graph_compatibility.py — 最小兼容性测试

验证 inp_to_graph.py 生成的 structure_graph.pt 能否被 GNN-LSTM 模型正确加载。

测试内容:
  1. 加载图数据，检查字段完整性和形状
  2. 填充模拟的 ground_motion 和 response 数据（测试格式兼容）
  3. 构建模型并执行前向传播（验证维度匹配）
  4. 模拟 GraphDataset 的加载和归一化流程
"""

import sys
import os
import torch
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# ─── 配置 ───
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--graph", type=str, default="./output/structure_graph.pt",
                     help="structure_graph.pt 文件路径")
_args, _ = _parser.parse_known_args()
GRAPH_PATH = _args.graph
MAX_STORY = 8           # 与 train_GAT_LSTM_arg.py 默认值一致
COMPRESSION_RATE = 40   # 与默认值一致
SAMPLE_RATE = 0.005     # 200 Hz
DURATION = 100.0        # 秒
SRC_LEN = int(DURATION / SAMPLE_RATE)  # 20000
TRG_LEN = SRC_LEN // 10               # 2000

print("=" * 70)
print("GNN-LSTM 兼容性测试")
print("=" * 70)


# ─── Step 1: 加载原始图 ───
print("\n[Step 1] 加载图数据")
data = torch.load(GRAPH_PATH, weights_only=False)
print(f"  原始字段: {list(data.keys())}")
print(f"  x: {data.x.shape}, dtype={data.x.dtype}")
print(f"  edge_index: {data.edge_index.shape}, dtype={data.edge_index.dtype}")
print(f"  edge_attr: {data.edge_attr.shape}, dtype={data.edge_attr.dtype}")
print(f"  time_steps: {data.time_steps}, type={type(data.time_steps)}")
print(f"  sample_rate: {data.sample_rate}, type={type(data.sample_rate)}")

# ─── Step 2: 修复格式 + 填充模拟数据 ───
print("\n[Step 2] 修复格式并填充模拟数据")

# 2a. time_steps 必须是 int
num_real_stories = 5  # 该结构实际5层
data.time_steps = SRC_LEN  # 20000 (int)
print(f"  time_steps → {data.time_steps} (int)")

# 2b. sample_rate 必须是 float
data.sample_rate = SAMPLE_RATE
print(f"  sample_rate → {data.sample_rate}")

# 2c. ground_motion: 模拟地震动加速度时程 [SRC_LEN]
# 使用类似真实地震动的随机信号（低通滤波后的噪声）
np.random.seed(42)
dt = SAMPLE_RATE
t = np.arange(SRC_LEN) * dt
# 生成包络调制的噪声，模拟地震动
envelope = np.exp(-((t - 15) ** 2) / 200) * (t < 40).astype(float)
noise = np.random.randn(SRC_LEN)
# 简单低通: 累积和 + 差分
gm = np.cumsum(noise) * 0.01
gm = gm - np.mean(gm)
gm = gm * envelope * 3000  # 缩放到 mm/s² 量级
data.ground_motion = torch.tensor(gm, dtype=torch.float64)
print(f"  ground_motion: {data.ground_motion.shape}, dtype={data.ground_motion.dtype}")
print(f"  ground_motion range: [{gm.min():.1f}, {gm.max():.1f}]")

# 2d. response 数据: [TRG_LEN, MAX_STORY]
# 模拟各楼层位移响应（放大版的低频信号）
for resp_type in ["Acceleration", "Velocity", "Displacement"]:
    resp = np.zeros((TRG_LEN, MAX_STORY), dtype=np.float64)
    t_resp = np.arange(TRG_LEN) * (dt * 10)  # 0.05s 间隔
    for floor in range(num_real_stories):
        # 每层不同频率和幅值
        amp = 5.0 * (floor + 1)  # 层越高幅值越大
        phase = floor * 0.3
        freq = 1.0 + floor * 0.2  # Hz
        env = np.exp(-((t_resp - 20) ** 2) / 300) * (t_resp < 60)
        resp[:, floor] = amp * np.sin(2 * np.pi * freq * t_resp + phase) * env
    setattr(data, resp_type, torch.tensor(resp, dtype=torch.float64))
    print(f"  {resp_type}: {resp.shape}, range=[{resp.min():.2f}, {resp.max():.2f}]")

data.ground_motion_name = "Simulated_GM"

# ─── Step 3: 模拟 GraphDataset 加载流程 ───
print("\n[Step 3] 模拟 GraphDataset 加载流程")

# 3a. 检查 sample_rate
assert data.sample_rate == SAMPLE_RATE, f"sample_rate 不匹配: {data.sample_rate} != {SAMPLE_RATE}"
print(f"  ✓ sample_rate = {data.sample_rate}")

# 3b. 楼层数: story = int(graph.x[0, 1]) - 1
story_from_graph = int(data.x[0, 1]) - 1
print(f"  x[0,1] = {data.x[0, 1]} → story = {story_from_graph}")
print(f"  (注意: x[0,1] 是 Y 方向网格线数 = {int(data.x[0, 1])}，不是楼层数)")
print(f"  实际楼层数 = {num_real_stories}")

# 3c. 设定 response_type 并选择 y
response_type = "Displacement"
data.response_type = response_type
data.story = num_real_stories  # 使用实际楼层数
data.y = data[response_type]
print(f"  ✓ response_type = {response_type}")
print(f"  ✓ y shape = {data.y.shape}")

# 3d. 删除其他 response
for rt in ["Acceleration", "Velocity", "Moment_Z", "Shear_Y"]:
    if rt in data.keys():
        del data[rt]
print(f"  ✓ 清理后字段: {list(data.keys())}")

# 3e. unsqueeze ground_motion → [1, SRC_LEN]
data.ground_motion = torch.unsqueeze(data.ground_motion, dim=0).float()
data.ground_motion_name = "Simulated_GM"
print(f"  ✓ ground_motion unsqueezed: {data.ground_motion.shape}")

# 3f. y unsqueeze → [1, TRG_LEN, MAX_STORY]
data.y = torch.unsqueeze(data.y, dim=0)
print(f"  ✓ y unsqueezed: {data.y.shape}")

# ─── Step 4: 归一化（模拟 get_normalized_item_dict + normalize） ───
print("\n[Step 4] 归一化")
from collections import OrderedDict

normalized_item_dict = OrderedDict()
x_norm = OrderedDict()
x_norm["XYZ_gridline_num"] = torch.max(torch.abs(data.x[:, :3]))
x_norm["XYZ_grid_index"] = torch.max(torch.abs(data.x[:, 3:6]))
x_norm["period"] = torch.max(torch.abs(data.x[:, 6])) if torch.max(torch.abs(data.x[:, 6])) > 0 else 1.0
x_norm["DOF"] = torch.max(torch.abs(data.x[:, 7])) if torch.max(torch.abs(data.x[:, 7])) > 0 else 1.0
x_norm["mass"] = torch.max(torch.abs(data.x[:, 8])) if torch.max(torch.abs(data.x[:, 8])) > 0 else 1.0
x_norm["XYZ_inertia"] = torch.max(torch.abs(data.x[:, 9:12])) if torch.max(torch.abs(data.x[:, 9:12])) > 0 else 1.0
x_norm["XYZ_mode_shape"] = torch.max(torch.abs(data.x[:, 12:15])) if torch.max(torch.abs(data.x[:, 12:15])) > 0 else 1.0

edge_norm = OrderedDict()
edge_norm["S_y"] = torch.max(torch.abs(data.edge_attr[:, 2]))
edge_norm["S_z"] = torch.max(torch.abs(data.edge_attr[:, 3]))
edge_norm["area"] = torch.max(torch.abs(data.edge_attr[:, 4]))
edge_norm["element_length"] = torch.max(torch.abs(data.edge_attr[:, 5]))

normalized_item_dict["x"] = x_norm
normalized_item_dict["ground_motion"] = torch.max(torch.abs(data.ground_motion))
normalized_item_dict["y"] = torch.max(torch.abs(data.y))
normalized_item_dict["edge_attr"] = edge_norm
normalized_item_dict["response_type"] = response_type

print(f"  归一化字典:")
for k, v in normalized_item_dict.items():
    if isinstance(v, dict):
        for kk, vv in v.items():
            print(f"    {k}/{kk} = {vv:.4e}")
    elif isinstance(v, str):
        print(f"    {k} = {v}")
    else:
        print(f"    {k} = {v:.4e}")

# 执行归一化
data.x[:, :3] = data.x[:, :3] / x_norm["XYZ_gridline_num"]
data.x[:, 3:6] = data.x[:, 3:6] / x_norm["XYZ_grid_index"]
if x_norm["period"] > 0:
    data.x[:, 6] = data.x[:, 6] / x_norm["period"]
if x_norm["DOF"] > 0:
    data.x[:, 7] = data.x[:, 7] / x_norm["DOF"]
if x_norm["mass"] > 0:
    data.x[:, 8] = data.x[:, 8] / x_norm["mass"]
if x_norm["XYZ_inertia"] > 0:
    data.x[:, 9:12] = data.x[:, 9:12] / x_norm["XYZ_inertia"]
if x_norm["XYZ_mode_shape"] > 0:
    data.x[:, 12:15] = data.x[:, 12:15] / x_norm["XYZ_mode_shape"]
data.edge_attr[:, 2] = data.edge_attr[:, 2] / edge_norm["S_y"]
data.edge_attr[:, 3] = data.edge_attr[:, 3] / edge_norm["S_z"]
data.edge_attr[:, 4] = data.edge_attr[:, 4] / edge_norm["area"]
data.edge_attr[:, 5] = data.edge_attr[:, 5] / edge_norm["element_length"]
data.ground_motion = data.ground_motion / normalized_item_dict["ground_motion"]
data.y = data.y / normalized_item_dict["y"]
print(f"  ✓ 归一化完成")

# ─── Step 5: 构建 DataLoader + 模型前向传播 ───
print("\n[Step 5] 模型前向传播测试")

from torch_geometric.loader import DataLoader as GraphDataLoader
from Models.GAT_LSTM import GAT_LSTM, LSTM, GAT_Encoder

# 添加 batch 索引（DataLoader 会自动加，这里手动模拟）
data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)

# 创建 DataLoader
loader = GraphDataLoader([data], batch_size=1, shuffle=False)
batch_data = next(iter(loader))

print(f"  batch_data.x: {batch_data.x.shape}")
print(f"  batch_data.edge_index: {batch_data.edge_index.shape}")
print(f"  batch_data.edge_attr: {batch_data.edge_attr.shape}")
print(f"  batch_data.batch: {batch_data.batch.shape}, unique={batch_data.batch.unique().tolist()}")
print(f"  batch_data.ground_motion: {batch_data.ground_motion.shape}")
print(f"  batch_data.time_steps: {batch_data.time_steps}")
print(f"  batch_data.y: {batch_data.y.shape}")

# 构建模型
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"  设备: {device}")

input_dim = data.x.shape[1]  # 15
gat_encoder = GAT_Encoder(
    input_dim=input_dim,
    hid_dim=128,
    edge_dim=6,
    gnn_embed_dim=128,
    dropout=0.2,
)
lstm = LSTM(
    gnn_embed_dim=128,
    hid_dim=128,
    n_layers=2,
    dropout=0.2,
    pack_mode=True,
    compression_rate=COMPRESSION_RATE,
    max_story=MAX_STORY,
)
model = GAT_LSTM(gat_encoder, lstm, device).to(device)

# 前向传播
model.eval()
with torch.no_grad():
    try:
        output, edge_idx_ret, attn_weights = model(batch_data)
        print(f"\n  ✓ 前向传播成功!")
        print(f"  output shape: {output.shape}")
        print(f"  期望 shape: [1, {TRG_LEN}, {MAX_STORY}]")
        assert output.shape == torch.Size([1, TRG_LEN, MAX_STORY]), \
            f"输出形状不匹配: {output.shape} != [1, {TRG_LEN}, {MAX_STORY}]"
        print(f"  ✓ 输出形状匹配!")
        print(f"  output range: [{output.min():.6f}, {output.max():.6f}]")
        print(f"  edge_idx_ret: {edge_idx_ret.shape}")
        print(f"  attn_weights: {len(attn_weights)} entries")

        # 计算 loss
        y = batch_data.y.to(device)
        loss = torch.nn.MSELoss()(output, y)
        print(f"  MSE Loss: {loss.item():.8f}")

    except Exception as e:
        print(f"\n  ✗ 前向传播失败!")
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成!")
print("=" * 70)
