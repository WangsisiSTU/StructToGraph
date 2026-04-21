# StructToGraph

将结构分析模型（Abaqus INP、YJK INP、YJK STRUCTURE.DAT）转换为 PyTorch Geometric 图数据（`structure_graph.pt`），用于 GNN-LSTM 等图神经网络的结构动力响应预测。

## 功能

- **多格式支持**：Abaqus INP、YJK（盈建科）INP、YJK STRUCTURE.DAT
- **自动拓扑提取**：从节点坐标推导正交网格系统，提取梁柱拓扑关系
- **壳单元聚合**：自动将剪力墙（壳单元）聚合为等效虚拟边
- **完整数据集成**：支持加载地震动时程、结构响应时程、模态分析结果
- **兼容 GNN-LSTM**：输出格式与 [GNN-LSTM Fusion Model](https://github.com/kuopochih/GNN-LSTM-based-Fusion-Model-for-Structural-Dynamic-Responses-Prediction) 完全兼容

## 安装

```bash
pip install -r requirements.txt
```

依赖：
- Python >= 3.9
- PyTorch >= 1.10
- PyTorch Geometric >= 2.0
- NumPy >= 1.21

## 脚本说明

### 1. `inp_to_graph.py` — 通用结构图转换

从结构模型文件提取拓扑和截面属性，生成 `structure_graph.pt`。

```bash
# 仅解析 INP，生成结构图
python inp_to_graph.py --inp model.inp --output_dir ./output

# 完整流程：INP + 模态数据 + 地震动 + 响应数据
python inp_to_graph.py \
    --inp model.inp \
    --modal modal_data.json \
    --ground_motion gm_record.txt \
    --response_dir ./responses/ \
    --output_dir ./output

# YJK STRUCTURE.DAT 格式
python inp_to_graph.py \
    --structure_dat STRUCTURE.DAT \
    --fea_dat fea.dat \
    --tag_map M0.ElemTagID.txt \
    --output_dir ./output
```

**主要参数：**

| 参数 | 说明 |
|------|------|
| `--inp` | Abaqus INP 或 YJK INP 文件路径 |
| `--structure_dat` | YJK STRUCTURE.DAT 文件路径 |
| `--fea_dat` | YJK fea.dat（可选） |
| `--tag_map` | M0.ElemTagID.txt（可选） |
| `--modal` | 模态数据 JSON（可选） |
| `--ground_motion` | 地震动时程文件（可选） |
| `--response_dir` | 响应数据目录（可选） |
| `--max_story` | 最大楼层数（default: 8） |
| `--unit_system` | 单位系统：`auto` / `kN_mm_s` / `N_m_s` / `kN_m_s` |
| `--output_dir` | 输出目录 |

### 2. `build_real_graph.py` — YJK 计算结果转换

将 YJK 弹塑性时程分析的计算结果（STRUCTURE.DAT + 地震波 + 楼层响应）组装为完整的 `structure_graph.pt`。

```bash
python build_real_graph.py \
    --structure_graph ./output/structure_graph.pt \
    --dat ./earthquake.DAT \
    --disp ./PostEP/PLSTIC/earthquake.DISP \
    --postdb ./PostEP/postdb.db \
    --output ./result/structure_graph.pt \
    --n_floors 5 \
    --gm_name ChiChi_CHY028 \
    --angle 90
```

**主要参数：**

| 参数 | 说明 |
|------|------|
| `--structure_graph` | `inp_to_graph.py` 生成的图结构 `.pt` 文件 |
| `--dat` | YJK 地震波 `.DAT` 文件 |
| `--disp` | 楼层响应 `.DISP` 文件 |
| `--postdb` | `PostEP/postdb.db` 模型信息数据库 |
| `--output` | 输出路径 |
| `--n_floors` | 实际楼层数（用于修正 story 字段） |
| `--angle` | 地震方向角：0=X 方向，90=Y 方向 |
| `--gm_name` | 地震波名称 |

### 3. `extract_modal.py` — Abaqus ODB 模态提取

从 Abaqus ODB 文件提取模态分析结果（周期、振型），供 `inp_to_graph.py` 使用。

```bash
# 需要在 Abaqus Python 环境中运行
abaqus python extract_modal.py --odb job.odb --output modal_data.json
```

### 4. `test_graph_compatibility.py` — 兼容性验证

验证生成的 `structure_graph.pt` 能否被 GNN-LSTM 模型正确加载。

```bash
python test_graph_compatibility.py --graph ./output/structure_graph.pt
```

## 输出格式

输出的 `structure_graph.pt` 是一个 PyG `Data` 对象，包含以下字段：

### 节点特征 `x: [N, 15]`

| 列 | 内容 | 说明 |
|----|------|------|
| 0 | X 方向网格线总数 | |
| 1 | Y 方向网格线总数 | `int(x[0,1]) - 1` = 楼层数 |
| 2 | Z 方向网格线总数 | |
| 3 | 节点 X 网格索引 | 0-based |
| 4 | 节点 Y 网格索引 | |
| 5 | 节点 Z 网格索引 | |
| 6 | 第一模态周期 T1 | (s) |
| 7 | DOF 标记 | 0=固定, 1=自由 |
| 8 | 节点等效质量 | (ton) |
| 9 | Ixx 惯性矩 | |
| 10 | Iyy 惯性矩 | |
| 11 | Izz 惯性矩 | |
| 12 | 第一振型分量 dx | |
| 13 | 第一振型分量 dy | |
| 14 | 第一振型分量 dz | |

### 边 `edge_index: [2, 2E]` + `edge_attr: [2E, 6]`

每条结构边生成双向边。边特征：

| 列 | 内容 | 说明 |
|----|------|------|
| 0 | 梁标记 | 水平构件 → 1 |
| 1 | 柱/墙标记 | 竖向构件 → 1 |
| 2 | Sy | 截面模量 |
| 3 | Sz | 截面模量 |
| 4 | 截面面积 A | |
| 5 | 构件长度 L | |

### 时程数据（可选）

| 字段 | 形状 | 说明 |
|------|------|------|
| `ground_motion` | `[20000]` | 地震动加速度 (mm/s²), 200Hz, 100s |
| `Displacement` | `[2000, max_story]` | 位移响应 (mm), 20Hz, 100s |
| `Velocity` | `[2000, max_story]` | 速度响应 (mm/s), 20Hz, 100s |
| `Acceleration` | `[2000, max_story]` | 加速度响应 (mm/s²), 20Hz, 100s |

### 元数据

| 字段 | 类型 | 说明 |
|------|------|------|
| `time_steps` | int | 原始时间步数 |
| `sample_rate` | float | 采样率 (s) |
| `ground_motion_name` | str | 地震动名称 |

## 典型工作流

```
                    ┌─────────────────────┐
                    │   结构模型文件        │
                    │  (INP / DAT / ODB)   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌────────────┐  ┌──────────────┐  ┌────────────┐
     │inp_to_graph│  │extract_modal │  │ YJK 计算    │
     │   .py      │  │    .py       │  │ 结果文件    │
     └─────┬──────┘  └──────┬───────┘  └─────┬──────┘
           │                │                │
           │    modal_data.json              │
           │                │                │
           ▼                ▼                │
     structure_graph.pt (结构图)             │
           │                                 │
           └────────────┬────────────────────┘
                        ▼
              ┌──────────────────┐
              │build_real_graph  │
              │      .py         │
              └────────┬─────────┘
                       ▼
              structure_graph.pt (完整图+响应)
                       │
                       ▼
              ┌──────────────────┐
              │test_graph_       │
              │compatibility.py  │
              └────────┬─────────┘
                       ▼
                  验证通过 → 用于 GNN-LSTM 训练
```

## 单位系统

默认使用 **kN-mm-s** 单位制，与 GNN-LSTM 项目一致。

| 物理量 | 单位 |
|--------|------|
| 力 | kN |
| 长度 | mm |
| 时间 | s |
| 加速度 | mm/s² |
| 位移 | mm |



## License

MIT License
