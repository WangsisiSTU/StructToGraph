"""
extract_modal.py — 从 Abaqus ODB 文件提取模态分析结果

此脚本需要在 Abaqus Python 环境中运行:
    abaqus python extract_modal.py --odb job.odb --output modal_data.json

或在 Abaqus CAE 中通过 File > Run Script 执行。

输出 JSON 格式:
{
    "T1": 0.944,               // 第一模态周期 (s)
    "frequencies": [1.059, ...], // 所有提取的频率 (Hz)
    "mode_shapes": {
        "1":  [dx, dy, dz],    // 节点1的第一阶振型分量
        "2":  [dx, dy, dz],
        ...
    }
}
"""

import argparse
import json
import sys
import os


def extract_modal(odb_path: str, output_path: str, num_modes: int = 1):
    """
    从 ODB 中提取模态数据。

    Args:
        odb_path: ODB 文件路径
        output_path: 输出 JSON 路径
        num_modes: 提取的模态阶数（默认只取第1阶）
    """
    try:
        from odbAccess import openOdb
    except ImportError:
        print("错误: 需要 Abaqus Python 环境来读取 ODB 文件。")
        print("请使用以下命令运行:")
        print(f"  abaqus python {sys.argv[0]} --odb {odb_path} --output {output_path}")
        sys.exit(1)

    print(f"打开 ODB 文件: {odb_path}")
    odb = openOdb(odb_path)

    # 查找频率分析步
    freq_step = None
    for step_name, step in odb.steps.items():
        # FREQUENCY 或频率分析
        procedure = getattr(step, 'procedure', '')
        if 'FREQUENCY' in procedure.upper() or 'frequency' in step_name.lower():
            freq_step = step
            freq_step_name = step_name
            break

    if freq_step is None:
        # 尝试任何步骤的第一个帧
        for step_name, step in odb.steps.items():
            if len(step.frames) > 0:
                freq_step = step
                freq_step_name = step_name
                break

    if freq_step is None:
        print("错误: 在 ODB 中未找到频率分析步。")
        odb.close()
        sys.exit(1)

    print(f"使用分析步: {freq_step_name}")

    # 提取频率
    frequencies = []
    # 尝试从历史输出获取频率
    try:
        for region_name, region in freq_step.historyRegions.items():
            if 'EIGFREQ' in region.historyOutputs:
                freq_data = region.historyOutputs['EIGFREQ'].data
                for t, f in freq_data:
                    frequencies.append(f)
                break
    except Exception:
        pass

    # 如果没有历史输出，从帧标签解析
    if not frequencies:
        for frame in freq_step.frames:
            try:
                freq_label = frame.frequency
                if freq_label and freq_label > 0:
                    frequencies.append(freq_label)
            except AttributeError:
                pass

    # 如果仍然没有，从帧描述解析
    if not frequencies:
        for frame in freq_step.frames:
            desc = frame.description if hasattr(frame, 'description') else ''
            # 尝试从描述中提取频率值
            import re
            match = re.search(r'(\d+\.?\d*)', desc)
            if match:
                frequencies.append(float(match.group(1)))

    # 提取第一阶周期
    T1 = 0.0
    if frequencies and frequencies[0] > 0:
        T1 = 1.0 / frequencies[0]

    print(f"提取到 {len(frequencies)} 个频率")
    if T1 > 0:
        print(f"第一模态周期 T1 = {T1:.6f} s")

    # 提取第一阶振型
    mode_shapes = {}
    if len(freq_step.frames) > 0:
        frame = freq_step.frames[0]  # 第一阶

        if 'U' in frame.fieldOutputs:
            displacement = frame.fieldOutputs['U']

            for block in displacement.bulkDataBlocks:
                node_labels = block.nodeLabels
                data = block.data

                for i in range(len(node_labels)):
                    nid = int(node_labels[i])
                    if data.shape[1] >= 3:
                        dx = float(data[i, 0])
                        dy = float(data[i, 1])
                        dz = float(data[i, 2])
                    elif data.shape[1] >= 1:
                        dx = float(data[i, 0])
                        dy = 0.0
                        dz = 0.0
                    else:
                        continue

                    mode_shapes[str(nid)] = [dx, dy, dz]

            # 如果没有 bulkDataBlocks，尝试其他方式
            if not mode_shapes:
                try:
                    for value in displacement.values:
                        nid = value.nodeLabel
                        data = value.data
                        mode_shapes[str(nid)] = [float(data[0]), float(data[1]), float(data[2])]
                except Exception:
                    pass

    print(f"提取到 {len(mode_shapes)} 个节点的振型数据")

    # 振型归一化（最大绝对值归一到 1）
    if mode_shapes:
        max_val = max(
            max(abs(v[0]), abs(v[1]), abs(v[2]))
            for v in mode_shapes.values()
        )
        if max_val > 0:
            mode_shapes = {
                k: [v[0]/max_val, v[1]/max_val, v[2]/max_val]
                for k, v in mode_shapes.items()
            }

    # 组装结果
    result = {
        "T1": T1,
        "frequencies": frequencies[:10],  # 最多保存前10阶
        "mode_shapes": mode_shapes
    }

    # 保存
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n模态数据已保存到: {output_path}")
    print(f"  T1 = {T1:.6f} s")
    print(f"  振型节点数 = {len(mode_shapes)}")

    odb.close()


def main():
    parser = argparse.ArgumentParser(description="从 Abaqus ODB 提取模态分析结果")
    parser.add_argument("--odb", type=str, required=True, help="ODB 文件路径")
    parser.add_argument("--output", type=str, default="modal_data.json", help="输出 JSON 路径")
    parser.add_argument("--num_modes", type=int, default=1, help="提取的模态阶数")
    args = parser.parse_args()

    extract_modal(args.odb, args.output, args.num_modes)


if __name__ == "__main__":
    main()
