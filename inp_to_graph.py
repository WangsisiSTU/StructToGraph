"""
inp_to_graph.py — 将 Abaqus INP 结构模型转换为 PyG 图数据 (structure_graph.pt)

用法:
    # 仅解析 INP，生成结构图（无响应和模态数据）
    python inp_to_graph.py --inp model.inp --output_dir ./output/

    # 完整流程：INP + 模态数据 + 地震动 + 响应数据
    python inp_to_graph.py \
        --inp model.inp \
        --modal modal_data.json \
        --ground_motion gm_record.txt \
        --response_dir ./responses/ \
        --output_dir ./output/

    # 指定单位系统
    python inp_to_graph.py --inp model.inp --unit_system N_m_s --output_dir ./output/

输出: output_dir/structure_graph.pt  (PyG Data 对象)
"""

import re
import math
import json
import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data


# ===========================================================================
# 0. YJKParser — YJK（盈建科）INP 格式解析器
# ===========================================================================

@dataclass
class YJKSection:
    """YJK 截面属性（已在 *SECTION 中直接给出）"""
    sec_id: int
    area: float = 0.0
    Iy: float = 0.0
    Iz: float = 0.0
    J: float = 0.0
    Sy: float = 0.0
    Sz: float = 0.0


@dataclass
class YJKMaterial:
    """YJK 材料属性"""
    mat_id: int
    E: float = 0.0
    nu: float = 0.0
    density: float = 0.0


@dataclass
class YJKStory:
    """YJK 楼层信息"""
    story_id: int
    tower: int = 1
    level: int = 0
    elev: float = 0.0
    height: float = 0.0


class YJKParser:
    """
    YJK（盈建科）INP 格式解析器。

    YJK INP 格式与标准 Abaqus INP 完全不同，使用自定义关键字：
    *NODE, *FRAME, *SECTION, *MATERIAL, *CONSTRAINT, *STORY 等。
    """

    def __init__(self, inp_path: str):
        self.inp_path = inp_path
        self.nodes: Dict[int, Tuple[float, float, float]] = {}
        self.node_masses: Dict[int, Tuple[float, float, float]] = {}  # {nid: (mx, my, mz)}
        self.beam_elements: Dict[int, BeamElement] = {}
        self.shell_elements: Dict[int, ShellElement] = {}
        self.sections: Dict[int, YJKSection] = {}
        self.materials: Dict[int, YJKMaterial] = {}
        self.stories: Dict[int, YJKStory] = {}
        self.constraints: Dict[int, Tuple[int, int]] = {}  # {nid: (first_dof, last_dof)}
        self.unit_system: str = "kN_m_s"  # YJK 默认单位
        self.gravity: float = 10.0

    def parse(self):
        """执行解析"""
        with open(self.inp_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        self._parse_lines(lines)
        self._assign_sections()
        return self

    def _parse_lines(self, lines: List[str]):
        """状态机逐行解析"""
        current_keyword = ""
        data_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("*"):
                # 处理上一个关键字的数据
                if current_keyword:
                    self._process_keyword(current_keyword, data_lines)
                current_keyword = stripped.split(",")[0].strip().upper()
                data_lines = []
            else:
                data_lines.append(stripped)

        # 处理最后一个关键字
        if current_keyword:
            self._process_keyword(current_keyword, data_lines)

    @staticmethod
    def _parse_kv_line(line: str) -> Tuple[str, Dict]:
        """
        解析 YJK 格式的键值对行。
        格式1: id<TAB>KEY1=val1 KEY2=val2 ...
        格式2: id KEY1=val1 KEY2=val2 ...
        返回 (id_str, {key: val})
        """
        # 先按 tab 分割
        parts = line.split("\t")
        if len(parts) >= 2:
            # 有 tab：第一部分是 id，其余是键值对
            id_str = parts[0].strip()
            rest = " ".join(parts[1:]).strip()
        else:
            # 没有 tab：按第一个包含 "=" 的 token 分割
            tokens = line.split()
            id_str = tokens[0].strip() if tokens else ""
            rest = " ".join(tokens[1:]).strip()

        kv = {}
        # 按空格分割键值对
        tokens = rest.split()
        for token in tokens:
            if "=" in token:
                key, val = token.split("=", 1)
                kv[key.strip().upper()] = val.strip(",").strip()

        return id_str, kv

    def _process_keyword(self, keyword: str, data_lines: List[str]):
        if keyword == "*NODE":
            self._parse_nodes(data_lines)
        elif keyword == "*FRAME":
            self._parse_frames(data_lines)
        elif keyword == "*SECTION":
            self._parse_sections(data_lines)
        elif keyword == "*MATERIAL":
            self._parse_materials(data_lines)
        elif keyword == "*STORY":
            self._parse_stories(data_lines)
        elif keyword == "*CONSTRAINT":
            self._parse_constraints(data_lines)
        elif keyword == "*STRUCTURE":
            self._parse_structure(data_lines)
        elif keyword == "*UNIT":
            self._parse_unit(data_lines)

    def _parse_structure(self, data_lines: List[str]):
        """解析 *STRUCTURE 行，提取重力等信息"""
        if not data_lines:
            return
        for part in data_lines[0].split(","):
            kv = part.strip().split("=")
            if len(kv) == 2:
                key, val = kv[0].strip().upper(), kv[1].strip()
                if key == "GRAV":
                    self.gravity = float(val)

    def _parse_unit(self, data_lines: List[str]):
        """解析 *UNIT"""
        if not data_lines:
            return
        # UNIT=0,0,0  (YJK 内部编码)
        pass

    def _parse_nodes(self, data_lines: List[str]):
        """
        解析 *NODE 块。
        格式: node_id<TAB>C=x,y,z NM=... VM=... FM=... M=mx,my,mz,ixx,iyy,izz T=... L=... TWR=...
        """
        for line in data_lines:
            id_str, kv = self._parse_kv_line(line)
            try:
                nid = int(id_str)
            except ValueError:
                continue

            # 坐标
            coords_str = kv.get("C", "0,0,0")
            try:
                coords = [float(v) for v in coords_str.split(",")]
                self.nodes[nid] = (coords[0], coords[1], coords[2])
            except (ValueError, IndexError):
                self.nodes[nid] = (0.0, 0.0, 0.0)

            # 质量 M=mx,my,mz,ixx,iyy,izz
            mass_str = kv.get("M", None)
            if mass_str and mass_str != "0,0,0,0,0,0":
                try:
                    mass_vals = [float(v) for v in mass_str.split(",")]
                    self.node_masses[nid] = (
                        mass_vals[0] if len(mass_vals) > 0 else 0.0,
                        mass_vals[1] if len(mass_vals) > 1 else 0.0,
                        mass_vals[2] if len(mass_vals) > 2 else 0.0,
                    )
                except (ValueError, IndexError):
                    pass

    def _parse_frames(self, data_lines: List[str]):
        """
        解析 *FRAME 块。
        格式: elem_id<TAB>T=BEAM|COLUMN B=... N=node1,node2 M=mat_id SEC=sec_id ...
        """
        for line in data_lines:
            id_str, kv = self._parse_kv_line(line)
            try:
                eid = int(id_str)
            except ValueError:
                continue

            frame_type = kv.get("T", "BEAM").upper()
            nodes_str = kv.get("N", "")
            sec_id = int(kv.get("SEC", "0"))

            try:
                node_ids = [int(v) for v in nodes_str.split(",")]
            except ValueError:
                continue

            if len(node_ids) < 2:
                continue

            elem = BeamElement(
                elem_id=eid,
                node_ids=node_ids,
                elem_type="B31",
                elset=f"SEC_{sec_id}",
            )
            elem._yjk_sec_id = sec_id
            elem._yjk_type = frame_type  # "BEAM" or "COLUMN"

            self.beam_elements[eid] = elem

    def _parse_sections(self, data_lines: List[str]):
        """
        解析 *SECTION 块。
        格式: sec_id<TAB>P=Area,Iy,Iz,J,Sy,Sz CM=...
        """
        for line in data_lines:
            id_str, kv = self._parse_kv_line(line)
            try:
                sec_id = int(id_str)
            except ValueError:
                continue

            props_str = kv.get("P", "")
            try:
                props = [float(v) for v in props_str.split(",")]
            except ValueError:
                props = []

            sec = YJKSection(sec_id=sec_id)
            if len(props) >= 1:
                sec.area = props[0]
            if len(props) >= 2:
                sec.Iy = props[1]
            if len(props) >= 3:
                sec.Iz = props[2]
            if len(props) >= 4:
                sec.J = props[3]
            if len(props) >= 5:
                sec.Sy = props[4]
            if len(props) >= 6:
                sec.Sz = props[5]

            self.sections[sec_id] = sec

    def _parse_materials(self, data_lines: List[str]):
        """
        解析 *MATERIAL 块。
        格式: mat_id<TAB>P=E,nu,damp,density,thermal
        """
        for line in data_lines:
            id_str, kv = self._parse_kv_line(line)
            try:
                mat_id = int(id_str)
            except ValueError:
                continue

            props_str = kv.get("P", "")
            try:
                props = [float(v) for v in props_str.split(",")]
            except ValueError:
                props = []

            mat = YJKMaterial(mat_id=mat_id)
            if len(props) >= 1:
                mat.E = props[0]
            if len(props) >= 2:
                mat.nu = props[1]
            if len(props) >= 4:
                mat.density = props[3]

            self.materials[mat_id] = mat

    def _parse_stories(self, data_lines: List[str]):
        """
        解析 *STORY 块。
        格式: story_id<TAB>TWR=1 L=level T=0 Elev=height H=story_height ...
        """
        for line in data_lines:
            id_str, kv = self._parse_kv_line(line)
            try:
                story_id = int(id_str)
            except ValueError:
                continue

            story = YJKStory(story_id=story_id)
            story.tower = int(kv.get("TWR", "1"))
            story.level = int(kv.get("L", "0"))
            story.elev = float(kv.get("ELEV", "0"))
            story.height = float(kv.get("H", "0"))

            self.stories[story_id] = story

    def _parse_constraints(self, data_lines: List[str]):
        """
        解析 *CONSTRAINT 块。
        格式: node_id<TAB>R=1,1,1,1,1,1 T=1 G=0
        """
        for line in data_lines:
            id_str, kv = self._parse_kv_line(line)
            try:
                nid = int(id_str)
            except ValueError:
                continue

            r_str = kv.get("R", "0,0,0,0,0,0")
            try:
                restraints = [int(v) for v in r_str.split(",")]
            except ValueError:
                restraints = [0, 0, 0, 0, 0, 0]

            # 如果 6 个 DOF 全部约束 → 固定
            if all(r == 1 for r in restraints):
                self.constraints[nid] = (1, 6)

    def _assign_sections(self):
        """将截面关联到梁单元"""
        for eid, elem in self.beam_elements.items():
            sec_id = getattr(elem, '_yjk_sec_id', 0)
            if sec_id in self.sections:
                yjk_sec = self.sections[sec_id]
                # 转换为通用的 BeamSection
                section = BeamSection()
                section.area = yjk_sec.area
                section.Sy = yjk_sec.Sy
                section.Sz = yjk_sec.Sz
                section.Iyy = yjk_sec.Iy
                section.Izz = yjk_sec.Iz
                section.Ixx = yjk_sec.J
                section.section_type = "YJK_DIRECT"
                elem.section = section

    def to_standard_parser(self) -> 'INPParser':
        """将 YJK 数据转换为 INPParser 兼容格式，方便后续 GraphBuilder 使用"""
        p = INPParser.__new__(INPParser)
        p.inp_path = self.inp_path
        p.nodes = dict(self.nodes)
        p.beam_elements = dict(self.beam_elements)
        p.shell_elements = dict(self.shell_elements)
        p.beam_sections = {}
        p.shell_sections = {}
        p.materials = {}
        p.node_sets = {}
        p.element_sets = {}
        p.boundaries = []
        p.orientations = {}
        p.current_elset_name = ""

        # 转换约束为 Boundary 对象
        for nid, (first, last) in self.constraints.items():
            p.boundaries.append(Boundary(nid, first, last, 0.0))

        # 存储 YJK 特有数据供后续使用
        p._yjk_stories = self.stories
        p._yjk_node_masses = self.node_masses
        p._yjk_materials = self.materials
        p._yjk_unit = self.unit_system

        return p


def detect_format(inp_path: str) -> str:
    """
    自动检测 INP 文件格式。
    返回 "STRUCTURE_DAT", "YJK" 或 "ABAQUS"。
    """
    import os
    basename = os.path.basename(inp_path).upper()
    if basename == "STRUCTURE.DAT":
        return "STRUCTURE_DAT"

    with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
        first_lines = [f.readline() for _ in range(10)]

    for line in first_lines:
        if "*DOCTYPE" in line.upper() or "YJK" in line.upper():
            return "YJK"

    # YJK 特征：有 *FRAME, *SECTION, *STORY 等关键字
    with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    if "*FRAME" in content and "*STORY" in content:
        return "YJK"

    return "ABAQUS"


# ===========================================================================
# 0b. StructureDatParser — YJK STRUCTURE.DAT 完整模型解析器
# ===========================================================================

class StructureDatParser:
    """
    YJK STRUCTURE.DAT 完整模型解析器。
    包含所有 246 个节点（含墙体细分节点）、215 个杆件、120 个墙板。

    文件结构:
    1. 头部: 版本, 结构参数
    2. 楼层标高
    3. 截面定义
    4. 全部节点 + 约束信息
    5. 按楼层分组的墙板(PL)和梁柱(Beam/Colu)元素
    6. Gauss 点信息(Tag→元素映射)
    7. 墙板角点定义
    8. 纤维截面等

    可选配合 fea.dat 提供截面属性(通过 M0.ElemTagID.txt Tag 映射)。
    """

    def __init__(self, dat_path: str, fea_path: str = None, tag_map_path: str = None):
        self.dat_path = dat_path
        self.fea_path = fea_path
        self.tag_map_path = tag_map_path

        self.nodes: Dict[int, Tuple[float, float, float]] = {}
        self.node_constraints: Dict[int, List[int]] = {}  # nid → [r1..r6], -1=fixed
        self.node_stories: Dict[int, int] = {}  # nid → story_num
        self.beam_elements: Dict[int, BeamElement] = {}
        self.shell_elements: Dict[int, ShellElement] = {}
        self.boundaries: List[Boundary] = []
        self.stories: Dict[int, YJKStory] = {}
        self.sections: Dict[int, YJKSection] = {}

        # STRUCTURE.DAT 内部截面
        self._sd_sections: List[dict] = []
        # 墙板信息: {story: [(n1,n2,n3,n4,thickness,tag), ...]}
        self._wall_panels: Dict[int, List[tuple]] = {}
        # 杆件信息: [(type, n1, n2, tag, story, n_seg, sec_type1, sec_type2, sec_type3), ...]
        self._frame_elements: List[tuple] = []
        # 楼层标高
        self._story_elevations: Dict[int, float] = {}
        # Gauss Tag → [elem_ids] 映射
        self._gauss_tags: Dict[str, List[int]] = {}
        # 楼面角点
        self._wall_corners: Dict[str, Tuple[int, int, int, int]] = {}

        # 从 fea.dat / Tag 映射获得的截面
        self._fea_sections: Dict[int, YJKSection] = {}
        self._tag_to_sec_id: Dict[int, int] = {}  # YJK Tag → fea.dat sec_id

        # 统计
        self.total_nodes = 0
        self.total_beam_colu = 0
        self.total_walls = 0
        self.num_stories = 0

    def parse(self):
        """执行解析"""
        # 尝试多种编码（YJK 文件通常是 GBK）
        lines = None
        for enc in ("gbk", "gb2312", "utf-8", "latin-1"):
            try:
                with open(self.dat_path, "r", encoding=enc, errors="strict") as f:
                    lines = [l.rstrip("\n\r") for l in f.readlines()]
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if lines is None:
            with open(self.dat_path, "r", encoding="utf-8", errors="replace") as f:
                lines = [l.rstrip("\n\r") for l in f.readlines()]

        self._parse_all(lines)

        # 尝试加载 fea.dat 和 Tag 映射
        if self.fea_path:
            self._load_fea_sections()
        if self.tag_map_path:
            self._load_tag_map()

        # 分配截面
        self._assign_sections_from_tags()

        # 转换为标准格式
        self._build_standard_elements()
        return self

    def _parse_all(self, lines: List[str]):
        """整体解析 STRUCTURE.DAT — 使用行扫描策略"""
        n = len(lines)

        # --- 扫描关键位置 ---
        section_header_idx = None   # 截面定义头
        node_header_idx = None      # 节点块头
        story_block_starts = []     # 自然层头列表
        wall_block_starts = []      # 墙板块头列表
        frame_block_starts = []     # 杆件块头列表

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            # 截面定义头: "N : 梁/柱/支撑单元截面类型总数"
            # 注意: 文件中有两处匹配 — 第一处在头部(行5), 第二处在实际截面定义前(行16)
            # 我们需要第二处（实际截面数据之前的那一行）
            if re.match(r"\d+\s*:.*截面类型", stripped):
                section_header_idx = i  # 保留最后一个匹配

            # 节点块头: "246 : 节点坐标"
            if node_header_idx is None and re.match(r"\d+\s*:.*节点坐标", stripped):
                node_header_idx = i

            # 自然层头: "story_id : 自然层号"
            if re.match(r"\d+\s*:.*自然层", stripped):
                story_block_starts.append(i)

            # 墙板块头: "N : 墙/壳单元信息"
            if re.match(r"\d+\s*:.*墙.*单元", stripped):
                wall_block_starts.append(i)

            # 杆件块头: "N : 梁/柱/支撑单元信息"
            if re.match(r"\d+\s*:.*梁.*单元信息", stripped):
                frame_block_starts.append(i)

        # --- 1. 头部: 结构参数 ---
        # Line 1: "5 : 结构总层数"
        self.num_stories = 5  # 从已知结构获取
        for i in range(min(10, n)):
            m = re.match(r"(\d+)\s*:.*结构总层数", lines[i].strip())
            if m:
                self.num_stories = int(m.group(1))
                break

        # --- 2. 楼层标高 ---
        # 在截面定义之前，格式: "elev standard_story natural_story x y z"
        if section_header_idx is not None:
            for i in range(10, section_header_idx):
                parts = lines[i].strip().split()
                if len(parts) >= 6:
                    try:
                        elev = float(parts[0])
                        standard_story = int(parts[1])
                        natural_story = int(parts[2])
                        self.stories[natural_story] = YJKStory(
                            story_id=natural_story, level=natural_story, elev=elev
                        )
                        self._story_elevations[natural_story] = elev
                    except (ValueError, IndexError):
                        continue

        # --- 3. 截面定义 ---
        if section_header_idx is not None:
            m = re.match(r"(\d+)\s*:", lines[section_header_idx].strip())
            num_sections = int(m.group(1)) if m else 0
            for i in range(1, num_sections + 1):
                idx = section_header_idx + i
                if idx < n:
                    sec = self._parse_section_line(lines[idx].strip())
                    if sec:
                        self._sd_sections.append(sec)

        # --- 4. 节点 ---
        if node_header_idx is not None:
            m = re.match(r"(\d+)\s*:", lines[node_header_idx].strip())
            self.total_nodes = int(m.group(1)) if m else 0
            for i in range(1, self.total_nodes + 1):
                idx = node_header_idx + i
                if idx >= n:
                    break
                parts = lines[idx].strip().split()
                if len(parts) >= 12:
                    try:
                        nid = int(parts[0])
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        self.nodes[nid] = (x, y, z)
                        constraints = [int(parts[j]) for j in range(4, 10)]
                        self.node_constraints[nid] = constraints
                        story_num = int(parts[11])
                        self.node_stories[nid] = story_num
                        if all(r == -1 for r in constraints):
                            self.boundaries.append(Boundary(nid, 1, 6, 0.0))
                    except (ValueError, IndexError):
                        pass

        # --- 5. 按楼层分组解析墙板和杆件 ---
        # 每个楼层结构:
        #   story_block_start + 1: 自然层头行
        #   +2..+8: 7行偏心信息
        #   +9: 墙板头 "N : 墙/壳单元信息"
        #   +10..+10+N-1: PL 元素
        #   +10+N: 杆件头 "N : 梁/柱/支撑单元信息"
        #   +10+N+1..: Beam/Colu 元素

        for si, story_start in enumerate(story_block_starts):
            story_num = si + 1

            # 在 story_start 之后找墙板头
            wall_start = None
            for i in range(story_start + 1, min(story_start + 15, n)):
                if re.match(r"\d+\s*:.*墙.*单元", lines[i].strip()):
                    wall_start = i
                    break

            if wall_start is None:
                continue

            # 解析墙板
            wm = re.match(r"(\d+)\s*:", lines[wall_start].strip())
            wall_count = int(wm.group(1)) if wm else 0

            story_walls = []
            for i in range(1, wall_count + 1):
                idx = wall_start + i
                if idx >= n:
                    break
                parts = lines[idx].strip().split()
                # PL story_id n1 n2 n3 n4 0 0 thickness 0 0 0 wall_tag
                if len(parts) >= 12 and parts[0] == "PL":
                    try:
                        n1, n2, n3, n4 = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                        thickness = float(parts[8])
                        wall_tag = parts[11]
                        story_walls.append((n1, n2, n3, n4, thickness, wall_tag))
                    except (ValueError, IndexError):
                        pass

            self._wall_panels[story_num] = story_walls
            self.total_walls += len(story_walls)

            # 在墙板之后找杆件头（可能有空行）
            frame_start = wall_start + wall_count + 1
            # 跳过空行
            while frame_start < n and not lines[frame_start].strip():
                frame_start += 1

            if frame_start < n:
                fm = re.match(r"(\d+)\s*:", lines[frame_start].strip())
                frame_count = int(fm.group(1)) if fm else 0

                for i in range(1, frame_count + 1):
                    idx = frame_start + i
                    if idx >= n:
                        break
                    parts = lines[idx].strip().split()
                    # Beam|Colu n1 n2 0 sec_type1 sec_type2 sec_type3 n_seg ... tag story count
                    # 20 columns total, tag at [17], story at [18]
                    if len(parts) >= 3 and parts[0] in ("Beam", "Colu"):
                        try:
                            elem_type = parts[0]
                            n1 = int(parts[1])
                            n2 = int(parts[2])
                            sec_type1 = int(parts[4]) if len(parts) > 4 else 0
                            sec_type2 = int(parts[5]) if len(parts) > 5 else 0
                            sec_type3 = int(parts[6]) if len(parts) > 6 else 0
                            n_seg = int(parts[7]) if len(parts) > 7 else 1
                            tag = int(parts[17]) if len(parts) > 17 else 0
                            elem_story = int(parts[18]) if len(parts) > 18 else 0
                            self._frame_elements.append(
                                (elem_type, n1, n2, tag, elem_story, n_seg, sec_type1, sec_type2, sec_type3)
                            )
                        except (ValueError, IndexError):
                            pass

                self.total_beam_colu += frame_count

        # --- 6. 后续信息 ---
        # Gauss 点和楼面角点在文件的最后部分
        self._parse_remaining(lines, n)

    @staticmethod
    def _is_section_header(line: str) -> bool:
        """检测截面定义头"""
        return bool(re.match(r"\d+\s*:.*截面类型", line))

    @staticmethod
    def _is_story_block_start(line: str) -> bool:
        """检测楼层块开始（自然层标记）"""
        return bool(re.match(r"\d+\s*:.*自然层", line))

    def _parse_section_line(self, line: str) -> Optional[dict]:
        """
        解析截面定义行。
        STRUCTURE.DAT 截面格式 (17 fields for Rect):
        [0] type [1] material [2] b [3] h [4-9] 钢筋面积
        [10-11] 配筋率 [12] Iy [13] Iz [14] Sy(弹性模量) [15] A(面积) [16] fy

        或: VoidBox 格式类似但字段含义不同
        """
        parts = line.strip().split()
        if not parts:
            return None

        sec = {
            "type": parts[0],
            "material": parts[1] if len(parts) > 1 else "",
        }

        if sec["type"] == "Rect" and len(parts) >= 16:
            try:
                b = float(parts[2])
                h = float(parts[3])
                sec["b"] = b
                sec["h"] = h
                # 面积: 直接从文件读取 (parts[15])
                sec["area"] = float(parts[15]) if len(parts) > 15 else b * h
                # 惯性矩: parts[12]=Iy, parts[13]=Iz
                sec["Iy"] = float(parts[12])
                sec["Iz"] = float(parts[13])
                # 塑性截面模量: 从 b,h 计算
                sec["Sy"] = b * h * h / 4   # Z = b*h²/4
                sec["Sz"] = h * b * b / 4   # Z = h*b²/4
            except (ValueError, IndexError):
                sec["b"] = 0.0
                sec["h"] = 0.0
                sec["area"] = 0.0
                sec["Sy"] = 0.0
                sec["Sz"] = 0.0
        elif sec["type"] == "VoidBox" and len(parts) >= 15:
            try:
                b = float(parts[2])
                h = float(parts[3])
                sec["b"] = b
                sec["h"] = h
                # VoidBox: 面积和模量从纤维截面计算值获取
                sec["area"] = float(parts[12]) if len(parts) > 12 else 0.0
                sec["Iy"] = float(parts[13]) if len(parts) > 13 else 0.0
                sec["Iz"] = float(parts[14]) if len(parts) > 14 else 0.0
                sec["Sy"] = 0.0
                sec["Sz"] = 0.0
            except (ValueError, IndexError):
                sec["area"] = 0.0

        return sec

    def _parse_remaining(self, lines: List[str], start_idx: int):
        """解析剩余部分：Gauss Tag 映射、楼面角点等"""
        n = len(lines)
        idx = start_idx

        while idx < n:
            line = lines[idx].strip()

            # 楼面角点: "10 : 楼面角信息"
            if re.match(r"10\s*:.*角信息", line):
                idx += 1
                for _ in range(10):
                    if idx >= n:
                        break
                    parts = lines[idx].strip().split()
                    if len(parts) >= 5:
                        tag = parts[0]
                        try:
                            c1, c2, c3, c4 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                            self._wall_corners[tag] = (c1, c2, c3, c4)
                        except (ValueError, IndexError):
                            pass
                    idx += 1
                continue

            # Gauss 点: "65 : 杆件信息"
            if re.match(r"\d+\s*:.*杆件信息", line):
                gm = re.match(r"(\d+)", line)
                gauss_count = int(gm.group(1)) if gm else 0
                idx += 1
                for _ in range(gauss_count):
                    if idx >= n:
                        break
                    # Tag:XXXXXXX
                    if lines[idx].strip().startswith("Tag:"):
                        tag = lines[idx].strip()[4:]
                        elem_ids = []
                        idx += 1
                        # CnrPts:2
                        if idx < n and lines[idx].strip().startswith("CnrPts:"):
                            idx += 1
                        # 节点列表 (可能多行)
                        if idx < n and not lines[idx].strip().startswith("Tag:") and not lines[idx].strip().startswith("nElem"):
                            idx += 1
                        # nElem:N
                        n_elem = 0
                        if idx < n and lines[idx].strip().startswith("nElem:"):
                            n_elem = int(lines[idx].strip()[6:])
                            idx += 1
                        # 元素 ID 列表 (可能一行或多行)
                        for _ in range(n_elem):
                            if idx < n and not lines[idx].strip().startswith("Tag:"):
                                try:
                                    elem_ids.append(int(lines[idx].strip()))
                                except ValueError:
                                    pass
                                idx += 1
                            else:
                                break
                        self._gauss_tags[tag] = elem_ids
                    else:
                        idx += 1
                continue

            idx += 1

    def _load_fea_sections(self):
        """从 fea.dat / combine.inp 加载截面属性"""
        if not self.fea_path or not os.path.exists(self.fea_path):
            return

        try:
            yjk = YJKParser(self.fea_path)
            yjk.parse()
            self._fea_sections = yjk.sections
        except Exception as e:
            print(f"    警告: 无法解析 fea.dat: {e}")

    def _load_tag_map(self):
        """从 M0.ElemTagID.txt 加载 Tag→ID 映射"""
        if not self.tag_map_path or not os.path.exists(self.tag_map_path):
            return

        try:
            with open(self.tag_map_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            tag = int(parts[0])
                            sid = int(parts[1])
                            self._tag_to_sec_id[tag] = sid
                        except ValueError:
                            continue
        except Exception as e:
            print(f"    警告: 无法加载 Tag 映射: {e}")

    def _assign_sections_from_tags(self):
        """根据 Tag 和 fea.dat 截面属性分配到 _sd_sections"""
        # 用 STRUCTURE.DAT 内的截面定义（已含 A, Iy, Iz, Sy, Sz）
        # 无需 fea.dat 的截面信息，STRUCTURE.DAT 内置的更完整
        pass

    def _build_standard_elements(self):
        """将解析结果转换为标准 BeamElement / ShellElement"""
        elem_counter = 0

        # 梁柱元素
        for elem_type, n1, n2, tag, story_num, n_seg, st1, st2, st3 in self._frame_elements:
            elem_counter += 1

            beam = BeamElement(
                elem_id=elem_counter,
                node_ids=[n1, n2],
                elem_type="B31",
                elset=f"Tag_{tag}",
            )

            # 截面属性: 根据 sec_type1(1=Rect beam, 2=Rect column) 选择
            # sec_type1=1 → beam (section[0]: 0.3×0.5)
            # sec_type1=2 → column (section[1]: 0.5×0.5)
            sec_idx = st1 - 1  # 0-based
            if 0 <= sec_idx < len(self._sd_sections):
                sd_sec = self._sd_sections[sec_idx]
                section = BeamSection()
                section.section_type = "RECT"
                section.area = sd_sec.get("area", sd_sec.get("b", 0) * sd_sec.get("h", 0))
                section.Sy = sd_sec.get("Sy", 0.0)
                section.Sz = sd_sec.get("Sz", 0.0)
                section.Iyy = sd_sec.get("Iy", 0.0)
                section.Izz = sd_sec.get("Iz", 0.0)
                section.params = [sd_sec.get("b", 0), sd_sec.get("h", 0)]
                beam.section = section

            beam._yjk_type = elem_type.upper()[:4]  # "BEAM" or "COLU"
            beam._yjk_tag = tag
            beam._yjk_story = story_num

            self.beam_elements[elem_counter] = beam

        # 墙板元素 → ShellElement
        wall_counter = 10000  # 墙板从 10000 开始编号
        for story, walls in self._wall_panels.items():
            for n1, n2, n3, n4, thickness, wall_tag in walls:
                wall_counter += 1

                shell = ShellElement(
                    elem_id=wall_counter,
                    node_ids=[n1, n2, n3, n4],
                    elem_type="S4R",
                    elset=wall_tag,
                )

                section = ShellSection()
                section.thickness = thickness
                shell.section = section

                self.shell_elements[wall_counter] = shell

    def to_standard_parser(self) -> 'INPParser':
        """将 STRUCTURE.DAT 数据转换为 INPParser 兼容格式"""
        p = INPParser.__new__(INPParser)
        p.inp_path = self.dat_path
        p.nodes = dict(self.nodes)
        p.beam_elements = dict(self.beam_elements)
        p.shell_elements = dict(self.shell_elements)
        p.beam_sections = {}
        p.shell_sections = {}
        p.materials = {}
        p.node_sets = {}
        p.element_sets = {}
        p.boundaries = list(self.boundaries)
        p.orientations = {}
        p.current_elset_name = ""

        # YJK 附加数据
        p._yjk_stories = self.stories
        p._yjk_node_masses = {}
        p._yjk_materials = {}
        p._yjk_unit = "kN_m_s"  # STRUCTURE.DAT 使用 kN-m-s
        p._yjk_story_elevations = self._story_elevations
        p._yjk_wall_panels = self._wall_panels
        p._yjk_wall_corners = self._wall_corners

        return p


# ===========================================================================
# 1. INPParser — Abaqus INP 文件解析器
# ===========================================================================

@dataclass
class BeamSection:
    """梁截面属性"""
    elset: str = ""
    section_type: str = ""       # RECT, BOX, I, L, GENERAL 等
    params: list = field(default_factory=list)  # 截面几何参数
    area: float = 0.0            # 截面面积 (mm^2 或 m^2)
    Sy: float = 0.0              # 塑性截面模量 Y (mm^3)
    Sz: float = 0.0              # 塑性截面模量 Z (mm^3)
    Ixx: float = 0.0             # 抗扭惯性矩
    Iyy: float = 0.0             # 绕 Y 轴惯性矩
    Izz: float = 0.0             # 绕 Z 轴惯性矩
    material: str = ""
    density: float = 0.0         # 密度
    youngs_modulus: float = 0.0  # 弹性模量


@dataclass
class ShellSection:
    """壳截面属性"""
    elset: str = ""
    thickness: float = 0.0
    material: str = ""
    density: float = 0.0
    youngs_modulus: float = 0.0


@dataclass
class BeamElement:
    """梁单元"""
    elem_id: int
    node_ids: List[int]          # B31: 2个节点, B32: 3个节点
    elem_type: str               # B31, B32
    elset: str = ""
    section: Optional[BeamSection] = None


@dataclass
class ShellElement:
    """壳单元"""
    elem_id: int
    node_ids: List[int]
    elem_type: str               # S4R, S8R 等
    elset: str = ""
    section: Optional[ShellSection] = None


@dataclass
class Boundary:
    """边界条件"""
    node_id: int
    first_dof: int               # 1-6
    last_dof: int
    value: float = 0.0


class INPParser:
    """
    Abaqus INP 文件解析器。
    纯 Python 实现，逐行状态机解析。
    """

    # 常见的壳单元类型
    SHELL_TYPES = {"S4R", "S4", "S3R", "S3", "S8R", "S8R5", "STRI3", "STRI65"}
    # 常见的梁单元类型
    BEAM_TYPES = {"B31", "B31H", "B31OS", "B32", "B32H", "B32OS", "B33", "B33H", "PIPE31", "PIPE32"}
    # 实体单元类型（暂不处理，仅记录）
    SOLID_TYPES = {"C3D8R", "C3D8", "C3D20R", "C3D20", "C3D4", "C3D10M"}

    def __init__(self, inp_path: str):
        self.inp_path = inp_path
        self.nodes: Dict[int, Tuple[float, float, float]] = {}
        self.beam_elements: Dict[int, BeamElement] = {}
        self.shell_elements: Dict[int, ShellElement] = {}
        self.beam_sections: Dict[str, BeamSection] = {}
        self.shell_sections: Dict[str, ShellSection] = {}
        self.materials: Dict[str, Dict] = {}
        self.node_sets: Dict[str, Set[int]] = {}
        self.element_sets: Dict[str, Set[int]] = {}
        self.boundaries: List[Boundary] = []
        self.orientations: Dict[str, dict] = {}
        self.current_elset_name: str = ""  # 用于追踪当前元素集

    def parse(self):
        """执行解析"""
        with open(self.inp, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        self._parse_lines(lines)
        self._assign_sections_to_elements()
        return self

    @property
    def inp(self):
        return self.inp_path

    def _parse_lines(self, lines: List[str]):
        """状态机逐行解析"""
        current_keyword = ""
        current_params = {}
        data_lines: List[str] = []

        for line in lines:
            stripped = line.strip()

            # 跳过空行和纯注释行
            if not stripped:
                continue
            if stripped.startswith("**") and not stripped.startswith("*"):
                continue

            # 遇到新关键字
            if stripped.startswith("*"):
                # 先处理上一个关键字的数据
                if current_keyword:
                    self._process_keyword_data(current_keyword, current_params, data_lines)

                # 解析新关键字
                keyword, params = self._parse_keyword_line(stripped)
                current_keyword = keyword
                current_params = params
                data_lines = []
            else:
                # 累积数据行
                data_lines.append(stripped)

        # 处理最后一个关键字
        if current_keyword:
            self._process_keyword_data(current_keyword, current_params, data_lines)

    def _parse_keyword_line(self, line: str) -> Tuple[str, Dict]:
        """
        解析关键字行，如:
        *ELEMENT, TYPE=B31, ELSET=COLUMNS
        返回 (keyword, {param_key: param_value})
        """
        # 去掉 * 前缀
        content = line[1:].strip()
        parts = [p.strip() for p in content.split(",")]
        keyword = parts[0].upper()

        params = {}
        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                params[key.strip().upper()] = value.strip()

        return keyword, params

    def _process_keyword_data(self, keyword: str, params: Dict, data_lines: List[str]):
        """根据关键字类型分发处理"""

        if keyword == "NODE":
            self._parse_nodes(data_lines)
        elif keyword == "ELEMENT":
            elem_type = params.get("TYPE", "").upper()
            elset = params.get("ELSET", "")
            if elem_type in self.BEAM_TYPES:
                self._parse_beam_elements(data_lines, elem_type, elset)
            elif elem_type in self.SHELL_TYPES:
                self._parse_shell_elements(data_lines, elem_type, elset)
            else:
                # 其他单元类型暂不处理，但记录节点（用于网格推导）
                pass
        elif keyword in ("BEAM SECTION", "BEAM GENERAL SECTION"):
            self._parse_beam_section(keyword, params, data_lines)
        elif keyword == "SHELL SECTION" or keyword == "SHELL GENERAL SECTION":
            self._parse_shell_section(keyword, params, data_lines)
        elif keyword == "MATERIAL":
            self._parse_material(params, data_lines)
        elif keyword == "ELSET":
            self._parse_elset(params, data_lines)
        elif keyword == "NSET":
            self._parse_nset(params, data_lines)
        elif keyword == "BOUNDARY":
            self._parse_boundary(data_lines)
        elif keyword == "ORIENTATION":
            self._parse_orientation(params, data_lines)
        # 其他关键字忽略

    # ----- 各关键字解析方法 -----

    def _parse_nodes(self, data_lines: List[str]):
        """
        *NODE 格式:
        node_id, x, y, z
        """
        for line in data_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    nid = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    self.nodes[nid] = (x, y, z)
                except (ValueError, IndexError):
                    continue

    def _parse_beam_elements(self, data_lines: List[str], elem_type: str, elset: str):
        """
        *ELEMENT, TYPE=B31
        elem_id, node1, node2
        """
        for line in data_lines:
            parts = [p.strip() for p in line.split(",")]
            try:
                eid = int(parts[0])
                node_ids = [int(p) for p in parts[1:] if p]
                self.beam_elements[eid] = BeamElement(
                    elem_id=eid, node_ids=node_ids, elem_type=elem_type, elset=elset
                )
            except (ValueError, IndexError):
                continue

    def _parse_shell_elements(self, data_lines: List[str], elem_type: str, elset: str):
        """
        *ELEMENT, TYPE=S4R
        elem_id, node1, node2, node3, node4
        """
        for line in data_lines:
            parts = [p.strip() for p in line.split(",")]
            try:
                eid = int(parts[0])
                node_ids = [int(p) for p in parts[1:] if p]
                self.shell_elements[eid] = ShellElement(
                    elem_id=eid, node_ids=node_ids, elem_type=elem_type, elset=elset
                )
            except (ValueError, IndexError):
                continue

    def _parse_beam_section(self, keyword: str, params: Dict, data_lines: List[str]):
        """
        *BEAM SECTION, ELSET=..., SECTION=RECT|BOX|I|...
        或
        *BEAM GENERAL SECTION, ELSET=..., SECTION=...
        数据行格式取决于截面类型。
        """
        elset = params.get("ELSET", "")
        section_type = params.get("SECTION", "").upper()
        section = BeamSection(elset=elset, section_type=section_type)

        if len(data_lines) == 0:
            self.beam_sections[elset] = section
            return

        if keyword == "BEAM SECTION":
            # *BEAM SECTION 格式:
            # 第1行: 截面几何参数（如 RECT: b, h）
            # 第2行: 材料名 或 n1 方向向量
            # 可能有更多行
            first_line = [float(v) for v in data_lines[0].split(",") if v.strip()]
            section.params = first_line

            if len(data_lines) >= 2:
                # 第2行可能是材料名或方向向量
                second_parts = data_lines[1].strip().split(",")
                if all(c.isalpha() or c.isspace() or c == '_' for c in data_lines[1].strip().replace(",", "")):
                    section.material = second_parts[0].strip()
                else:
                    # 方向向量行
                    pass
            if len(data_lines) >= 3:
                section.material = data_lines[2].strip()

        elif keyword == "BEAM GENERAL SECTION":
            # *BEAM GENERAL SECTION 格式:
            # 第1行: 材料名（如果后面有属性数据行）
            # 后续行: A, Ixx, Iyy, Izz, J 或截面几何参数
            if section_type in ("RECT", "CIRCLE", "BOX", "I", "L", "T", "PIPE", "TRAPEZOID"):
                if len(data_lines) >= 1:
                    first_parts = [v.strip() for v in data_lines[0].split(",")]
                    try:
                        section.params = [float(v) for v in first_parts]
                    except ValueError:
                        # 第一行可能是材料名
                        section.material = first_parts[0]
                        if len(data_lines) >= 2:
                            section.params = [float(v) for v in data_lines[1].split(",") if v.strip()]
            else:
                # SECTION=GENERAL 或其他
                # 数据行可能直接给 A, Ixx, Iyy, Izz 或其他组合
                lines_float = []
                for dl in data_lines:
                    try:
                        vals = [float(v) for v in dl.split(",") if v.strip()]
                        lines_float.extend(vals)
                    except ValueError:
                        section.material = dl.strip()
                        continue
                section.params = lines_float

        self.beam_sections[elset] = section

    def _parse_shell_section(self, keyword: str, params: Dict, data_lines: List[str]):
        """
        *SHELL SECTION, ELSET=...
        第1行: thickness, (num_integration_pts)
        或
        第1行: 材料名
        """
        elset = params.get("ELSET", "")
        section = ShellSection(elset=elset)

        if len(data_lines) >= 1:
            first_parts = [v.strip() for v in data_lines[0].split(",")]
            try:
                section.thickness = float(first_parts[0])
                if len(first_parts) > 1 and not any(c.isalpha() for c in first_parts[1]):
                    pass  # 积分点数，忽略
            except ValueError:
                section.material = first_parts[0]
                if len(data_lines) >= 2:
                    second_parts = [v.strip() for v in data_lines[1].split(",")]
                    try:
                        section.thickness = float(second_parts[0])
                    except ValueError:
                        pass

        if len(data_lines) >= 2 and not section.material:
            # 第2行可能是材料名
            mat_line = data_lines[1].strip().split(",")[0].strip()
            if mat_line and not any(c.isdigit() for c in mat_line.replace(".", "").replace("-", "")):
                section.material = mat_line

        self.shell_sections[elset] = section

    def _parse_material(self, params: Dict, data_lines: List[str]):
        """
        *MATERIAL, NAME=...
        后续可能有 *ELASTIC, *DENSITY 等子关键字
        这里简化处理：只提取 name
        """
        name = params.get("NAME", "")
        if name:
            self.materials[name] = {"name": name}

    def _parse_elset(self, params: Dict, data_lines: List[str]):
        """
        *ELSET, ELSET=name, GENERATE (可选)
        """
        elset_name = params.get("ELSET", "")
        if not elset_name:
            return

        node_ids = set()
        generate = "GENERATE" in params

        for line in data_lines:
            parts = [p.strip() for p in line.split(",")]
            if generate and len(parts) == 3:
                try:
                    start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
                    node_ids.update(range(start, end + 1, step))
                except ValueError:
                    continue
            else:
                for p in parts:
                    try:
                        node_ids.add(int(p))
                    except ValueError:
                        continue

        self.element_sets[elset_name] = node_ids

    def _parse_nset(self, params: Dict, data_lines: List[str]):
        """
        *NSET, NSET=name, GENERATE (可选)
        """
        nset_name = params.get("NSET", "")
        if not nset_name:
            return

        node_ids = set()
        generate = "GENERATE" in params

        for line in data_lines:
            parts = [p.strip() for p in line.split(",")]
            if generate and len(parts) == 3:
                try:
                    start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
                    node_ids.update(range(start, end + 1, step))
                except ValueError:
                    continue
            else:
                for p in parts:
                    try:
                        node_ids.add(int(p))
                    except ValueError:
                        continue

        self.node_sets[nset_name] = node_ids

    def _parse_boundary(self, data_lines: List[str]):
        """
        *BOUNDARY
        node_id, first_dof, last_dof, value (可选)
        """
        for line in data_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    nid = int(parts[0])
                    first_dof = int(parts[1])
                    last_dof = int(parts[2])
                    value = float(parts[3]) if len(parts) > 3 else 0.0
                    self.boundaries.append(Boundary(nid, first_dof, last_dof, value))
                except (ValueError, IndexError):
                    continue

    def _parse_orientation(self, params: Dict, data_lines: List[str]):
        """
        *ORIENTATION, NAME=...
        数据行: 定义局部坐标系
        """
        name = params.get("NAME", "")
        if name and data_lines:
            self.orientations[name] = {
                "params": params,
                "data": [l.strip() for l in data_lines]
            }

    def _assign_sections_to_elements(self):
        """将截面属性通过 ELSET 关联到具体单元"""
        # 梁截面 → 梁单元
        for elset_name, section in self.beam_sections.items():
            if not elset_name:
                continue
            elem_ids = self.element_sets.get(elset_name, set())
            for eid in elem_ids:
                if eid in self.beam_elements:
                    self.beam_elements[eid].section = section
                    self.beam_elements[eid].elset = elset_name

        # 壳截面 → 壳单元
        for elset_name, section in self.shell_sections.items():
            if not elset_name:
                continue
            elem_ids = self.element_sets.get(elset_name, set())
            for eid in elem_ids:
                if eid in self.shell_elements:
                    self.shell_elements[eid].section = section
                    self.shell_elements[eid].elset = elset_name


# ===========================================================================
# 2. GridSystem — 网格推导系统
# ===========================================================================

class GridSystem:
    """
    从节点坐标推导正交网格系统。
    将连续坐标映射到离散网格索引。
    """

    def __init__(self, nodes: Dict[int, Tuple[float, float, float]], tol: float = 1.0):
        """
        Args:
            nodes: {node_id: (x, y, z)}
            tol: 坐标容差 (mm)，用于聚类
        """
        self.tol = tol
        self.nodes = nodes

        # 推导网格线
        self.x_grids: List[float] = []
        self.y_grids: List[float] = []
        self.z_grids: List[float] = []
        self.node_grid_map: Dict[int, Tuple[int, int, int]] = {}

        self._derive()

    def _derive(self):
        """执行网格推导"""
        if not self.nodes:
            return

        xs = [n[0] for n in self.nodes.values()]
        ys = [n[1] for n in self.nodes.values()]
        zs = [n[2] for n in self.nodes.values()]

        self.x_grids = self._cluster_coordinates(xs, self.tol)
        self.y_grids = self._cluster_coordinates(ys, self.tol)
        self.z_grids = self._cluster_coordinates(zs, self.tol)

        # 为每个节点分配网格索引
        x_map = {coord: idx for idx, coord in enumerate(self.x_grids)}
        y_map = {coord: idx for idx, coord in enumerate(self.y_grids)}
        z_map = {coord: idx for idx, coord in enumerate(self.z_grids)}

        for nid, (x, y, z) in self.nodes.items():
            xr = self._snap(x, self.tol)
            yr = self._snap(y, self.tol)
            zr = self._snap(z, self.tol)

            xi = x_map.get(xr, 0)
            yi = y_map.get(yr, 0)
            zi = z_map.get(zr, 0)
            self.node_grid_map[nid] = (xi, yi, zi)

    @staticmethod
    def _snap(value: float, tol: float) -> float:
        """将坐标对齐到最近的容差网格"""
        return round(value / tol) * tol

    @staticmethod
    def _cluster_coordinates(coords: List[float], tol: float) -> List[float]:
        """
        对坐标做容差聚类，返回排序后的唯一网格线坐标列表。
        """
        if not coords:
            return []

        # 对齐到容差网格
        snapped = sorted(set(round(c / tol) * tol for c in coords))
        return snapped

    def get_grid_counts(self) -> Tuple[int, int, int]:
        """返回 (x_grid_count, y_grid_count, z_grid_count)"""
        return len(self.x_grids), len(self.y_grids), len(self.z_grids)

    def get_grid_index(self, node_id: int) -> Tuple[int, int, int]:
        """返回节点的 (x_idx, y_idx, z_idx)，0-based"""
        return self.node_grid_map.get(node_id, (0, 0, 0))

    def get_floor_heights(self) -> List[float]:
        """
        推断楼层高度列表（基于 Y 方向或 Z 方向，取决于建模习惯）。
        假设竖向坐标为 Y 或 Z。
        返回排序后的非零高度列表。
        """
        # 在结构工程中，Abaqus 常用 Y-up 或 Z-up
        # 优先用 Z 方向（若 Z 有多个唯一值），否则用 Y
        vertical_grids = self.z_grids if len(self.z_grids) >= len(self.y_grids) else self.y_grids
        # 排除地面（高度≈0）
        floor_heights = [v for v in vertical_grids if abs(v) > self.tol]
        return sorted(floor_heights)

    def get_num_stories(self) -> int:
        """推断楼层数"""
        floors = self.get_floor_heights()
        return len(floors)


# ===========================================================================
# 3. SectionCalculator — 截面属性计算
# ===========================================================================

class SectionCalculator:
    """
    从截面几何参数计算面积、塑性模量等。
    """

    @staticmethod
    def compute_beam_properties(section: BeamSection) -> BeamSection:
        """
        根据截面类型和参数计算 A, Sy, Sz, Ixx, Iyy, Izz。
        假设参数单位与目标一致（mm）。
        """
        st = section.section_type.upper()
        p = section.params

        if st == "RECT" and len(p) >= 2:
            b, h = p[0], p[1]
            section.area = b * h
            section.Sy = b * h * h / 4       # 塑性模量 = b*h²/4
            section.Sz = h * b * b / 4       # 塑性模量 = h*b²/4
            section.Iyy = b * h**3 / 12
            section.Izz = h * b**3 / 12

        elif st == "CIRCLE" and len(p) >= 1:
            r = p[0] / 2  # 直径 → 半径
            section.area = math.pi * r * r
            section.Sy = section.Sz = (4 * r**3) / 3
            section.Iyy = section.Izz = math.pi * r**4 / 4

        elif st == "BOX" and len(p) >= 4:
            b, h, tw, tf = p[0], p[1], p[2], p[3]
            section.area = 2 * (tf * b + tw * (h - 2 * tf))
            # 塑性模量近似
            section.Sy = (b * h**2 / 4) - ((b - 2*tw) * (h - 2*tf)**2 / 4)
            section.Sz = (h * b**2 / 4) - ((h - 2*tf) * (b - 2*tw)**2 / 4)
            section.Iyy = (b * h**3 - (b - 2*tw) * (h - 2*tf)**3) / 12
            section.Izz = (h * b**3 - (h - 2*tf) * (b - 2*tw)**3) / 12

        elif st == "I" and len(p) >= 5:
            # I 形截面: d(total depth), bf(flange width), tw(web), tf(flange)
            # Abaqus 格式可能不同，常见: bf, d, tw, tf 或 d, bf, tw, tf
            if len(p) >= 5:
                # d, bf, tw, tf, (可能还有 k 等)
                d, bf, tw, tf = p[0], p[1], p[2], p[3]
            elif len(p) == 4:
                d, bf, tw, tf = p[0], p[1], p[2], p[3]
            else:
                return section

            hw = d - 2 * tf  # web height
            section.area = 2 * bf * tf + hw * tw
            # 弹性截面模量
            Syy = (bf * d**3 - (bf - tw) * hw**3) / (6 * d)
            section.Iyy = (bf * d**3 - (bf - tw) * hw**3) / 12
            # 塑性截面模量
            section.Sy = bf * tf * (d - tf) + tw * hw**2 / 4
            section.Sz = 2 * tf * bf**2 / 4 + hw * tw**2 / 4
            section.Izz = 2 * tf * bf**3 / 12 + hw * tw**3 / 12

        elif st == "L" and len(p) >= 4:
            b, h, tw, tf = p[0], p[1], p[2], p[3]
            section.area = b * tf + (h - tf) * tw
            section.Sy = b * h**2 / 4 * 0.5  # 近似
            section.Sz = h * b**2 / 4 * 0.5  # 近似
            section.Iyy = (b * h**3) / 12 * 0.3  # 粗略近似
            section.Izz = (h * b**3) / 12 * 0.3

        else:
            # GENERAL 或未知类型 — 如果 params 中直接给出了属性
            # 尝试按 A, Ixx, Iyy, Izz, J 的顺序解析
            if len(p) >= 1:
                section.area = p[0]
            if len(p) >= 2:
                section.Ixx = p[1]
            if len(p) >= 3:
                section.Iyy = p[2]
            if len(p) >= 4:
                section.Izz = p[3]
            # 塑性模量用弹性模量近似（Z ≈ 1.15 * S）
            if section.Iyy > 0 and len(p) >= 3:
                pass  # 需要更多信息

        return section


# ===========================================================================
# 4. ShellHandler — 剪力墙壳单元聚合
# ===========================================================================

class ShellHandler:
    """
    将壳单元（S4R/S8R）聚合为虚拟边。
    策略 A: 楼层级别的墙板聚合为虚拟构件边。
    """

    def __init__(
        self,
        shell_elements: Dict[int, ShellElement],
        nodes: Dict[int, Tuple[float, float, float]],
        grid: GridSystem,
        tol: float = 1.0,
    ):
        self.shell_elements = shell_elements
        self.nodes = nodes
        self.grid = grid
        self.tol = tol

    def aggregate(self) -> List[Tuple[int, int, BeamSection, float]]:
        """
        聚合壳单元为虚拟边。

        Returns:
            List of (bottom_node_id, top_node_id, section, length)
        """
        if not self.shell_elements:
            return []

        # 1. 提取每个壳单元的中心坐标和厚度
        elem_info = []
        for eid, elem in self.shell_elements.items():
            center = self._element_center(elem)
            thickness = self._get_thickness(elem)
            if center is not None:
                elem_info.append((eid, center, thickness))

        if not elem_info:
            return []

        # 2. 确定竖向轴（Y 或 Z）
        vertical_axis = self._detect_vertical_axis(elem_info)

        # 3. 按楼层分组
        floor_groups = self._group_by_floor(elem_info, vertical_axis)
        floor_heights = sorted(floor_groups.keys())

        # 4. 在每层内按水平位置聚类为墙段
        virtual_edges = []
        for i, fh in enumerate(floor_heights):
            elems_in_floor = floor_groups[fh]
            wall_segments = self._cluster_wall_segments(elems_in_floor, vertical_axis)

            for seg in wall_segments:
                # 找到该墙段上下楼板最近的主网格节点
                bottom_node = self._find_nearest_grid_node(seg, floor_heights, i - 1, vertical_axis)
                top_node = self._find_nearest_grid_node(seg, floor_heights, i, vertical_axis)

                if bottom_node is not None and top_node is not None and bottom_node != top_node:
                    # 计算等效截面属性
                    total_thickness = sum(t for _, _, t in seg)
                    wall_length = self._segment_length(seg, vertical_axis)
                    story_height = self._story_height(floor_heights, i)

                    section = BeamSection()
                    section.area = total_thickness * wall_length
                    section.Sy = total_thickness * wall_length * story_height / 4
                    section.Sz = total_thickness * wall_length * story_height / 4
                    section.section_type = "WALL"

                    virtual_edges.append((bottom_node, top_node, section, story_height))

        return virtual_edges

    def _element_center(self, elem: ShellElement) -> Optional[Tuple[float, float, float]]:
        """计算单元中心坐标"""
        coords = []
        for nid in elem.node_ids:
            if nid in self.nodes:
                coords.append(self.nodes[nid])
        if not coords:
            return None
        n = len(coords)
        return tuple(sum(c[i] for c in coords) / n for i in range(3))

    def _get_thickness(self, elem: ShellElement) -> float:
        """获取壳单元厚度"""
        if elem.section and elem.section.thickness > 0:
            return elem.section.thickness
        return 0.0

    def _detect_vertical_axis(self, elem_info: list) -> int:
        """
        检测竖向轴（0=X, 1=Y, 2=Z）。
        通过比较各轴坐标变化范围来判断。
        通常竖向轴坐标范围最大（楼层高度）。
        """
        if not elem_info:
            return 2  # 默认 Z-up

        # 实际上竖向轴应该是有最多离散层级的轴
        ranges = []
        for axis in range(3):
            vals = [info[1][axis] for info in elem_info]
            unique_count = len(set(round(v / self.tol) * self.tol for v in vals))
            ranges.append((unique_count, max(vals) - min(vals)))

        # 选择唯一值最多的轴作为竖向（楼层越多越可能是竖向）
        vertical_axis = max(range(3), key=lambda i: ranges[i][0])
        return vertical_axis

    def _group_by_floor(self, elem_info: list, vertical_axis: int) -> Dict[float, list]:
        """按楼层高度分组壳单元"""
        groups = defaultdict(list)
        for eid, center, thickness in elem_info:
            height = round(center[vertical_axis] / self.tol) * self.tol
            groups[height].append((eid, center, thickness))
        return dict(groups)

    def _cluster_wall_segments(self, elems_in_floor: list, vertical_axis: int) -> List[list]:
        """在楼层内按水平位置聚类为墙段"""
        # 简单策略：按水平坐标聚类
        # 水平轴是除了竖向轴之外的两个轴
        horiz_axes = [i for i in range(3) if i != vertical_axis]

        segments = []
        used = set()

        for i, (eid_i, center_i, t_i) in enumerate(elems_in_floor):
            if i in used:
                continue
            segment = [(eid_i, center_i, t_i)]
            used.add(i)

            for j, (eid_j, center_j, t_j) in enumerate(elems_in_floor):
                if j in used:
                    continue
                # 检查是否相邻（水平距离小于容差 * 合理阈值）
                close = True
                for ax in horiz_axes:
                    if abs(center_i[ax] - center_j[ax]) > self.tol * 5:
                        close = False
                        break
                if close:
                    segment.append((eid_j, center_j, t_j))
                    used.add(j)

            segments.append(segment)

        return segments

    def _find_nearest_grid_node(self, segment: list, floor_heights: list,
                                 floor_idx: int, vertical_axis: int) -> Optional[int]:
        """找到最近的网格主节点"""
        if floor_idx < 0 or floor_idx >= len(floor_heights):
            # 底层 → 找地面节点（竖向坐标≈0）
            target_height = 0.0
        else:
            target_height = floor_heights[floor_idx]

        # 段中心的水平坐标
        horiz_axes = [i for i in range(3) if i != vertical_axis]
        seg_center_x = np.mean([s[1][horiz_axes[0]] for s in segment]) if len(horiz_axes) > 0 else 0
        seg_center_y = np.mean([s[1][horiz_axes[1]] for s in segment]) if len(horiz_axes) > 1 else 0

        # 在主节点中搜索最近的
        best_node = None
        best_dist = float("inf")

        for nid, (nx, ny, nz) in self.nodes.items():
            coords = [nx, ny, nz]
            vert_coord = coords[vertical_axis]

            # 竖向坐标要接近目标高度
            if abs(vert_coord - target_height) > self.tol * 2:
                continue

            # 水平距离
            dist = 0
            if len(horiz_axes) > 0:
                dist += (coords[horiz_axes[0]] - seg_center_x) ** 2
            if len(horiz_axes) > 1:
                dist += (coords[horiz_axes[1]] - seg_center_y) ** 2

            if dist < best_dist:
                best_dist = dist
                best_node = nid

        return best_node

    def _segment_length(self, segment: list, vertical_axis: int) -> float:
        """计算墙段的水平长度"""
        horiz_axes = [i for i in range(3) if i != vertical_axis]
        if not segment or not horiz_axes:
            return 0.0

        # 用主水平轴方向的范围作为长度
        ax = horiz_axes[0]
        vals = [s[1][ax] for s in segment]
        length = max(vals) - min(vals)
        if length < self.tol:
            if len(horiz_axes) > 1:
                ax2 = horiz_axes[1]
                vals2 = [s[1][ax2] for s in segment]
                length = max(vals2) - min(vals2)
        return max(length, self.tol)

    def _story_height(self, floor_heights: list, floor_idx: int) -> float:
        """计算层高"""
        if floor_idx <= 0:
            return floor_heights[0] if floor_heights else 3000.0
        return floor_heights[floor_idx] - floor_heights[floor_idx - 1]


# ===========================================================================
# 5. GraphBuilder — 组装 PyG Data 对象
# ===========================================================================

class GraphBuilder:
    """
    将所有提取的数据组装为 PyG Data 对象。
    """

    def __init__(
        self,
        parser: Union['INPParser', object],
        grid: GridSystem,
        max_story: int = 8,
        modal_data: Optional[Dict] = None,
        unit_system: str = "kN_mm_s",
    ):
        self.parser = parser
        self.grid = grid
        self.max_story = max_story
        self.modal_data = modal_data or {}
        self.unit_system = unit_system

        # 单位转换因子（从当前系统到 kN-mm-s）
        if unit_system == "N_m_s" or unit_system == "kN_m_s":
            # kN-m-s 或 N-m-s → 转换到 kN-mm-s
            self.force_factor = 1.0 if "kN" in unit_system else 1e-3
            self.length_factor = 1e3      # m → mm
            self.stress_factor = 1e-6 if "N_m" in unit_system else 1e-3
        else:
            self.force_factor = 1.0
            self.length_factor = 1.0
            self.stress_factor = 1.0

    def build(self) -> Data:
        """
        组装结构图数据。
        """
        # 确定参与图的节点（梁单元端点 + 墙虚拟边端点）
        active_nodes = self._collect_active_nodes()

        # 节点重编号（原始 ID → 0-based 连续索引）
        node_id_to_idx = {nid: idx for idx, nid in enumerate(sorted(active_nodes))}
        num_nodes = len(node_id_to_idx)

        # 构建节点特征
        x = self._build_node_features(node_id_to_idx, num_nodes)

        # 构建边
        edge_index_list, edge_attr_list = self._build_edges(node_id_to_idx)

        # 构建壳虚拟边
        shell_edges = self._build_shell_edges(node_id_to_idx)
        if shell_edges:
            se_idx, se_attr = shell_edges
            edge_index_list.append(se_idx)
            edge_attr_list.append(se_attr)

        # 合并所有边
        if edge_index_list:
            edge_index = np.concatenate(edge_index_list, axis=1)
            edge_attr = np.concatenate(edge_attr_list, axis=0)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 6), dtype=np.float32)

        # 组装 Data 对象
        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            time_steps=0,
            sample_rate=0.005,
            ground_motion_name="",
        )

        return data

    def _collect_active_nodes(self) -> Set[int]:
        """收集所有参与图的节点 ID"""
        active = set()

        # 梁单元端点
        for elem in self.parser.beam_elements.values():
            for nid in elem.node_ids:
                active.add(nid)

        # 壳单元节点（墙体）
        for elem in self.parser.shell_elements.values():
            for nid in elem.node_ids:
                active.add(nid)

        # 对于 B32（3节点），只取首尾
        # 但这里先全部收集，后面再处理

        return active

    def _build_node_features(self, node_id_to_idx: Dict[int, int], num_nodes: int) -> np.ndarray:
        """构建节点特征矩阵 [N, 15]"""
        x = np.zeros((num_nodes, 15), dtype=np.float32)

        x_grid_count, y_grid_count, z_grid_count = self.grid.get_grid_counts()
        T1 = self.modal_data.get("T1", 0.0)
        mode_shapes = self.modal_data.get("mode_shapes", {})

        # 构建固定节点集合
        fixed_nodes = set()
        for b in self.parser.boundaries:
            if b.first_dof == 1 and b.last_dof == 6:
                # 全部 DOF 固定
                fixed_nodes.add(b.node_id)

        for nid, idx in node_id_to_idx.items():
            xi, yi, zi = self.grid.get_grid_index(nid)

            # col 0-2: 网格线总数
            x[idx, 0] = x_grid_count
            x[idx, 1] = y_grid_count
            x[idx, 2] = z_grid_count

            # col 3-5: 当前节点的网格索引
            x[idx, 3] = xi
            x[idx, 4] = yi
            x[idx, 5] = zi

            # col 6: 第一模态周期
            x[idx, 6] = T1

            # col 7: DOF 标记 (0=固定, 1=自由)
            x[idx, 7] = 0.0 if nid in fixed_nodes else 1.0

            # col 8-11: 质量和惯性矩
            yjk_masses = getattr(self.parser, '_yjk_node_masses', {})
            if nid in yjk_masses:
                mx, my, mz = yjk_masses[nid]
                # 质量: YJK 中通常是 kN (力单位，需要除以重力加速度得到质量)
                # 在 kN-m-s 系统中，质量 = weight/gravity (kN / m/s² = kN·s²/m)
                # 转换到 kN-mm-s: 乘以 1000
                gravity = 10.0  # m/s²
                mass = (mx + my + mz) / (3 * gravity)  # 平均质量（kN·s²/m）
                mass_mm = mass * self.length_factor  # 转换到 mm 单位系统
                x[idx, 8] = mass_mm
                # 惯性矩: 简化假设为质量 × 特征长度²（如无确切数据）
                # 如果有更精确的惯性矩数据可在此处填充
                x[idx, 9] = 0.0   # Ixx - 需要更多信息
                x[idx, 10] = 0.0  # Iyy
                x[idx, 11] = 0.0  # Izz
            # col 12-14: 第一模态振型
            mode = mode_shapes.get(str(nid), [0.0, 0.0, 0.0])
            if len(mode) >= 3:
                x[idx, 12] = float(mode[0])
                x[idx, 13] = float(mode[1])
                x[idx, 14] = float(mode[2])

        return x

    def _build_edges(self, node_id_to_idx: Dict[int, int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """从梁单元构建边"""
        edge_index_list = []
        edge_attr_list = []
        calc = SectionCalculator

        for elem in self.parser.beam_elements.values():
            # B31: 2 节点 → 1 条边
            # B32: 3 节点 → 取首尾
            if "32" in elem.elem_type or "33" in elem.elem_type:
                if len(elem.node_ids) < 3:
                    continue
                n1, n2 = elem.node_ids[0], elem.node_ids[-1]
            else:
                if len(elem.node_ids) < 2:
                    continue
                n1, n2 = elem.node_ids[0], elem.node_ids[1]

            # 检查两个节点是否都在图中
            if n1 not in node_id_to_idx or n2 not in node_id_to_idx:
                continue

            idx1 = node_id_to_idx[n1]
            idx2 = node_id_to_idx[n2]

            # 计算边属性
            coord1 = self.parser.nodes.get(n1, (0, 0, 0))
            coord2 = self.parser.nodes.get(n2, (0, 0, 0))

            dx = abs(coord2[0] - coord1[0]) * self.length_factor
            dy = abs(coord2[1] - coord1[1]) * self.length_factor
            dz = abs(coord2[2] - coord1[2]) * self.length_factor
            length = math.sqrt(dx**2 + dy**2 + dz**2)

            # 方向判断: dZ 最大 → 柱 [0,1]，否则 → 梁 [1,0]
            if dz > max(dx, dy):
                dir_marker = [0.0, 1.0]
            else:
                dir_marker = [1.0, 0.0]

            # 截面属性
            Sy, Sz, area = 0.0, 0.0, 0.0
            if elem.section:
                sec = elem.section
                # 如果截面属性未计算，尝试计算
                if sec.area == 0 and sec.params:
                    calc.compute_beam_properties(sec)
                Sy = sec.Sy * (self.length_factor ** 3)
                Sz = sec.Sz * (self.length_factor ** 3)
                area = sec.area * (self.length_factor ** 2)

            attr = np.array([[dir_marker[0], dir_marker[1], Sy, Sz, area, length]],
                           dtype=np.float32)

            # 双向边
            ei = np.array([[idx1, idx2], [idx2, idx1]], dtype=np.int64)
            attr_dup = np.concatenate([attr, attr], axis=0)

            edge_index_list.append(ei)
            edge_attr_list.append(attr_dup)

        return edge_index_list, edge_attr_list

    def _build_shell_edges(self, node_id_to_idx: Dict[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """从壳单元构建虚拟边"""
        if not self.parser.shell_elements:
            return None

        handler = ShellHandler(
            self.parser.shell_elements, self.parser.nodes, self.grid
        )
        virtual_edges = handler.aggregate()

        if not virtual_edges:
            return None

        edge_index_list = []
        edge_attr_list = []

        for bottom_nid, top_nid, section, length in virtual_edges:
            if bottom_nid not in node_id_to_idx or top_nid not in node_id_to_idx:
                continue

            idx1 = node_id_to_idx[bottom_nid]
            idx2 = node_id_to_idx[top_nid]

            # 壳墙边标记为 [0, 1]（竖向构件）
            Sy = section.Sy * (self.length_factor ** 3)
            Sz = section.Sz * (self.length_factor ** 3)
            area = section.area * (self.length_factor ** 2)
            L = length * self.length_factor

            attr = np.array([[0.0, 1.0, Sy, Sz, area, L]], dtype=np.float32)

            ei = np.array([[idx1, idx2], [idx2, idx1]], dtype=np.int64)
            attr_dup = np.concatenate([attr, attr], axis=0)

            edge_index_list.append(ei)
            edge_attr_list.append(attr_dup)

        if not edge_index_list:
            return None

        return (
            np.concatenate(edge_index_list, axis=1),
            np.concatenate(edge_attr_list, axis=0),
        )


# ===========================================================================
# 6. ResponseAligner — 响应数据对齐
# ===========================================================================

class ResponseAligner:
    """
    读取 CSV/Excel 格式的楼层响应数据，
    对齐到 [2000, max_story] 格式。
    """

    def __init__(self, max_story: int = 8, target_time_steps: int = 2000, duration: float = 100.0):
        self.max_story = max_story
        self.target_time_steps = target_time_steps
        self.duration = duration
        self.target_dt = duration / target_time_steps  # 0.05s

    def load_csv(self, csv_path: str, floor_col_map: Optional[Dict[int, int]] = None) -> np.ndarray:
        """
        读取 CSV 格式响应数据。

        Args:
            csv_path: CSV 文件路径
            floor_col_map: {楼层编号(1-based): CSV列索引(0-based)}
                          如果为 None，假设 CSV 列按 1F, 2F, ... 排列

        Returns:
            np.ndarray of shape [2000, max_story]
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
        except ImportError:
            # 纯 numpy 回退
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
            # 假设第一列是时间，后续列是楼层
            values = data[:, 1:] if data.shape[1] > 1 else data[:, :1]
            time_col = data[:, 0] if data.shape[1] > 1 else None
            return self._align_response(values, time_col, floor_col_map)

        # 尝试识别时间列和楼层列
        time_col = None
        value_cols = []

        for col in df.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in ["time", "t(s)", "t(sec)", "时间"]):
                time_col = df[col].values
            else:
                value_cols.append(col)

        if not value_cols:
            # 所有列都是数据
            values = df.values
        else:
            values = df[value_cols].values

        return self._align_response(values, time_col, floor_col_map)

    def _align_response(self, values: np.ndarray, time_col: Optional[np.ndarray],
                        floor_col_map: Optional[Dict[int, int]]) -> np.ndarray:
        """
        将原始响应数据对齐到 [2000, max_story]。
        """
        num_input_cols = values.shape[1]

        # 确定楼层映射
        if floor_col_map:
            num_floors = max(floor_col_map.keys())
        else:
            num_floors = min(num_input_cols, self.max_story)

        # 插值到目标时间步
        target_time = np.linspace(0, self.duration, self.target_time_steps, endpoint=False)

        if time_col is not None and len(time_col) == values.shape[0]:
            # 有时间列，做插值
            from scipy.interpolate import interp1d

            result = np.zeros((self.target_time_steps, self.max_story))
            for floor_idx in range(min(num_floors, self.max_story)):
                col_idx = floor_col_map.get(floor_idx + 1, floor_idx) if floor_col_map else floor_idx
                if col_idx < values.shape[1]:
                    f = interp1d(time_col, values[:, col_idx],
                                kind="linear", bounds_error=False, fill_value=0.0)
                    result[:, floor_idx] = f(target_time)
        else:
            # 无时间列，直接截断/补零
            result = np.zeros((self.target_time_steps, self.max_story))
            n = min(values.shape[0], self.target_time_steps)
            n_cols = min(values.shape[1], self.max_story)
            result[:n, :n_cols] = values[:n, :n_cols]

        return result


# ===========================================================================
# 7. 辅助函数
# ===========================================================================

def load_ground_motion(gm_path: str, sample_rate: float = 0.005,
                        target_length: int = 20000) -> Tuple[np.ndarray, int]:
    """
    读取地震动加速度时程文件。

    支持格式:
    - 单列: 纯加速度值（等间隔）
    - 双列: 时间 + 加速度
    - 多列: 取最后一列为加速度

    Returns:
        (ground_motion_array, time_steps)
    """
    data = np.loadtxt(gm_path)
    if data.ndim == 1:
        acc = data
    elif data.shape[1] == 1:
        acc = data[:, 0]
    else:
        # 假设最后一列是加速度，或第二列
        acc = data[:, -1]

    time_steps = len(acc)

    # 截断或补零到目标长度
    if len(acc) > target_length:
        acc = acc[:target_length]
        time_steps = target_length
    elif len(acc) < target_length:
        padded = np.zeros(target_length)
        padded[:len(acc)] = acc
        acc = padded
        # time_steps 保持原始值

    return acc, time_steps


def load_modal_data(json_path: str) -> Dict:
    """
    读取模态分析结果 JSON 文件。

    期望格式:
    {
        "T1": 0.944,
        "mode_shapes": {
            "1": [dx, dy, dz],
            "2": [dx, dy, dz],
            ...
        }
    }
    """
    with open(json_path, "r") as f:
        return json.load(f)


# ===========================================================================
# 8. CLI 入口
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="将结构模型转换为 PyG 图数据 (structure_graph.pt)"
    )
    parser.add_argument("--inp", type=str, default=None,
                        help="Abaqus INP 或 YJK INP 文件路径")
    parser.add_argument("--structure_dat", type=str, default=None,
                        help="YJK STRUCTURE.DAT 完整模型文件路径")
    parser.add_argument("--fea_dat", type=str, default=None,
                        help="YJK fea.dat / combine.inp 简化模型（提供截面属性）")
    parser.add_argument("--tag_map", type=str, default=None,
                        help="M0.ElemTagID.txt Tag 映射文件路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--modal", type=str, default=None,
                        help="模态数据 JSON 文件（由 extract_modal.py 生成）")
    parser.add_argument("--ground_motion", type=str, default=None,
                        help="地震动加速度时程文件路径")
    parser.add_argument("--response_dir", type=str, default=None,
                        help="响应数据目录（包含 CSV/Excel 文件）")
    parser.add_argument("--max_story", type=int, default=8,
                        help="最大楼层数（默认 8）")
    parser.add_argument("--unit_system", type=str, default="auto",
                        choices=["auto", "kN_mm_s", "N_m_s", "kN_m_s"],
                        help="INP 文件的单位系统（默认 auto 自动检测）")
    parser.add_argument("--grid_tol", type=float, default=1.0,
                        help="网格推导的坐标容差 mm（默认 1.0）")
    parser.add_argument("--verbose", action="store_true", help="打印详细信息")
    args = parser.parse_args()

    if not args.inp and not args.structure_dat:
        parser.error("必须指定 --inp 或 --structure_dat 之一")

    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("结构模型 → PyG Graph 转换器")
    print("=" * 60)

    # --- Step 1: 检测格式并解析 ---
    if args.structure_dat:
        # STRUCTURE.DAT 模式
        print(f"\n[1] 解析 STRUCTURE.DAT: {args.structure_dat}")
        sd_parser = StructureDatParser(
            args.structure_dat,
            fea_path=args.fea_dat,
            tag_map_path=args.tag_map,
        )
        sd_parser.parse()
        print(f"    楼层数: {sd_parser.num_stories}")
        print(f"    节点数: {len(sd_parser.nodes)}")
        print(f"    杆件数(Beam+Colu): {sd_parser.total_beam_colu}")
        print(f"    墙板数(PL): {sd_parser.total_walls}")
        print(f"    内部截面定义: {len(sd_parser._sd_sections)}")
        print(f"    边界条件(固定节点): {len(sd_parser.boundaries)}")

        parser = sd_parser.to_standard_parser()
        file_format = "STRUCTURE_DAT"

        if args.unit_system == "auto":
            args.unit_system = "kN_m_s"
    else:
        # INP 模式（YJK 或 Abaqus）
        print(f"\n[1] 解析文件: {args.inp}")
        file_format = detect_format(args.inp)
        print(f"    检测到格式: {file_format}")

        if file_format == "YJK":
            yjk_parser = YJKParser(args.inp)
            yjk_parser.parse()
            print(f"    楼层数: {len(yjk_parser.stories)}")
            print(f"    节点数: {len(yjk_parser.nodes)}")
            print(f"    梁单元数: {len(yjk_parser.beam_elements)}")
            print(f"    截面数: {len(yjk_parser.sections)}")
            print(f"    材料数: {len(yjk_parser.materials)}")
            print(f"    约束节点数: {len(yjk_parser.constraints)}")
            parser = yjk_parser.to_standard_parser()

            if args.unit_system == "auto":
                args.unit_system = "kN_m_s"
        else:
            parser = INPParser(args.inp)
            parser.parse()
            if args.unit_system == "auto":
                args.unit_system = "kN_mm_s"

    print(f"    节点数: {len(parser.nodes)}")
    print(f"    梁单元数: {len(parser.beam_elements)}")
    print(f"    壳单元数: {len(parser.shell_elements)}")
    print(f"    梁截面定义: {len(parser.beam_sections)}")
    print(f"    壳截面定义: {len(parser.shell_sections)}")
    print(f"    边界条件: {len(parser.boundaries)}")

    if args.verbose:
        print(f"    梁截面类型: {[f'{k}={v.section_type}' for k, v in parser.beam_sections.items()]}")
        print(f"    壳截面厚度: {[f'{k}={v.thickness}' for k, v in parser.shell_sections.items()]}")

    # --- Step 2: 推导网格 ---
    # YJK / STRUCTURE.DAT 单位是 m，网格容差需要调整
    grid_tol = args.grid_tol
    if file_format in ("YJK", "STRUCTURE_DAT"):
        grid_tol = grid_tol / 1000.0  # mm → m
    print(f"\n[2] 推导网格系统 (容差={grid_tol})")
    grid = GridSystem(parser.nodes, tol=grid_tol)
    nx, ny, nz = grid.get_grid_counts()
    print(f"    网格线数: X={nx}, Y={ny}, Z={nz}")
    print(f"    推断楼层数: {grid.get_num_stories()}")

    # --- Step 3: 加载模态数据 ---
    modal_data = {}
    if args.modal:
        print(f"\n[3] 加载模态数据: {args.modal}")
        modal_data = load_modal_data(args.modal)
        print(f"    T1 = {modal_data.get('T1', 'N/A')} s")
        print(f"    振型节点数 = {len(modal_data.get('mode_shapes', {}))}")
    else:
        print(f"\n[3] 未提供模态数据，使用默认值 (T1=0, 振型=0)")

    # --- Step 4: 构建图 ---
    print(f"\n[4] 构建图数据")
    builder = GraphBuilder(
        parser=parser,
        grid=grid,
        max_story=args.max_story,
        modal_data=modal_data,
        unit_system=args.unit_system,
    )
    data = builder.build()

    print(f"    节点数: {data.x.shape[0]}")
    print(f"    节点特征维度: {data.x.shape[1]}")
    print(f"    边数: {data.edge_index.shape[1]}")
    print(f"    边特征维度: {data.edge_attr.shape[1]}")

    # --- Step 5: 加载地震动 ---
    if args.ground_motion:
        print(f"\n[5] 加载地震动: {args.ground_motion}")
        gm, time_steps = load_ground_motion(args.ground_motion)
        data.ground_motion = torch.tensor(gm, dtype=torch.float64)
        data.time_steps = torch.tensor(time_steps, dtype=torch.long)
        data.ground_motion_name = os.path.splitext(os.path.basename(args.ground_motion))[0]
        print(f"    时间步数: {time_steps}")
        print(f"    采样率: 0.005s (200Hz)")
    else:
        print(f"\n[5] 未提供地震动数据")

    # --- Step 6: 加载响应数据 ---
    if args.response_dir:
        print(f"\n[6] 加载响应数据: {args.response_dir}")
        aligner = ResponseAligner(max_story=args.max_story)

        # 尝试匹配响应文件
        response_types = ["Acceleration", "Velocity", "Displacement"]
        for rt in response_types:
            # 查找可能的文件名
            patterns = [f"{rt}.csv", f"{rt.lower()}.csv", f"{rt}.txt",
                       f"{rt.lower()}.txt", f"{rt}.xlsx", f"{rt.lower()}.xlsx"]
            found = False
            for pat in patterns:
                fpath = os.path.join(args.response_dir, pat)
                if os.path.exists(fpath):
                    print(f"    加载 {rt} from {fpath}")
                    resp = aligner.load_csv(fpath)
                    setattr(data, rt, torch.tensor(resp, dtype=torch.float64))
                    found = True
                    break
            if not found:
                print(f"    未找到 {rt} 数据文件")
    else:
        print(f"\n[6] 未提供响应数据目录")

    # --- Step 7: 保存 ---
    output_path = os.path.join(args.output_dir, "structure_graph.pt")
    torch.save(data, output_path)

    print(f"\n[7] 保存完成!")
    print(f"    输出路径: {output_path}")
    print(f"    Data 字段: {list(data.keys())}")
    print(f"    Data 对象: {data}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
