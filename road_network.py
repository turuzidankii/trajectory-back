import pandas as pd
import csv
import networkx as nx
from shapely.geometry import Point, LineString, box
from shapely.strtree import STRtree
from leuvenmapmatching.map.inmem import InMemMap
import logging
import os
import pickle  # 👈 引入 pickle

# 配置路径
DATA_DIR = "../data"
LOCAL_ROAD_FILE = os.path.join(DATA_DIR, "25M3_rbeijing_gcj02.csv") # 确保这个文件名对
# 🔥 缓存文件路径 (自动生成)
CACHE_FILE = os.path.join(DATA_DIR, "road_network_cache.pkl")
CACHE_VERSION = 2

class RoadNetwork:
    def __init__(self):
        self.gdf = None
        self.sindex = None
        self.geometries = []
        self.indices = []
        self.is_loaded = False
        self.hmm_map = None 
        self.graph = None
        self.edge_geom_map = {}

    @staticmethod
    def _parse_linestring_wkt(wkt: str):
        wkt = str(wkt).strip().strip('"')
        if not wkt.upper().startswith("LINESTRING"):
            return []
        start = wkt.find("(")
        end = wkt.rfind(")")
        if start == -1 or end == -1:
            return []
        coords_part = wkt[start + 1: end]
        coords = []
        for pair in coords_part.split(","):
            parts = pair.strip().split()
            if len(parts) < 2:
                continue
            lon = float(parts[0])
            lat = float(parts[1])
            coords.append((lat, lon))
        return coords

    @staticmethod
    def _read_header(path: str):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header_parts = []
            for line in f:
                if line.lstrip().startswith('"LINESTRING') or line.lstrip().startswith("LINESTRING"):
                    break
                header_parts.append(line.rstrip("\n"))
            header_line = "".join(part.strip() for part in header_parts)
            return next(csv.reader([header_line]))

    @staticmethod
    def _read_header_and_first_data_line(path: str):
        f = open(path, "r", encoding="utf-8", errors="ignore")
        header_parts = []
        first_data_line = ""
        for line in f:
            if line.lstrip().startswith('"LINESTRING') or line.lstrip().startswith("LINESTRING"):
                first_data_line = line.rstrip("\n")
                break
            header_parts.append(line.rstrip("\n"))
        header_line = "".join(part.strip() for part in header_parts)
        header = next(csv.reader([header_line]))
        return header, first_data_line, f

    def _iter_rows(self, path: str):
        header, first_data_line, f = self._read_header_and_first_data_line(path)
        ncols = len(header)

        def parse_line(line: str):
            return next(csv.reader([line], skipinitialspace=True))

        buffer = first_data_line
        for line in f:
            if not buffer:
                buffer = line.rstrip("\n")
                continue
            row = parse_line(buffer)
            if len(row) < ncols:
                buffer += line.rstrip("\n")
                continue
            yield row
            buffer = line.rstrip("\n")

        if buffer:
            try:
                row = parse_line(buffer)
                if len(row) == ncols:
                    yield row
            finally:
                f.close()
        else:
            f.close()

    def _ensure_hmm_index(self):
        """为 leuvenmapmatching 的 InMemMap 构建空间索引（如可用）。"""
        if self.hmm_map is None:
            return
        try:
            if hasattr(self.hmm_map, "build_spatial_index"):
                self.hmm_map.build_spatial_index()
                logging.info("✅ HMM 空间索引已构建")
                return
            if hasattr(self.hmm_map, "index"):
                # 某些版本可能提供 index 属性或方法
                idx = self.hmm_map.index
                if callable(idx):
                    idx()
                    logging.info("✅ HMM 索引已构建 (index())")
                else:
                    logging.info("✅ HMM 索引已存在")
        except Exception as e:
            logging.warning(f"⚠️ HMM 索引构建失败: {e}")

    def _normalize_node_id(self, node_id):
        """统一节点ID为字符串，避免 123 与 123.0 不匹配。"""
        s = str(node_id)
        if s.endswith('.0'):
            s = s[:-2]
        return s

    def normalize_node_id(self, node_id):
        """对外暴露的节点ID归一化方法。"""
        return self._normalize_node_id(node_id)

    def has_node(self, node_id):
        """判断节点是否存在于拓扑图中。"""
        if self.graph is None:
            return False
        key = self._normalize_node_id(node_id)
        return key in self.graph

    def load_local_file(self):
        """
        智能加载：优先读取 .pkl 缓存，如果没有则解析 CSV 并生成缓存
        """
        # 1. 尝试读取缓存 (极速模式)
        if os.path.exists(CACHE_FILE):
            try:
                logging.info(f"🚀 发现缓存文件，正在快速恢复路网: {CACHE_FILE}")
                with open(CACHE_FILE, 'rb') as f:
                    cache_data = pickle.load(f)

                if cache_data.get('cache_version') != CACHE_VERSION:
                    raise ValueError("cache_version_mismatch")
                
                # 恢复数据
                self.gdf = cache_data['gdf']
                self.hmm_map = cache_data['hmm_map']
                self.geometries = list(self.gdf['shapely_geom']) # 确保转回 list
                self.indices = list(self.gdf.index)
                # 统一节点ID类型，避免匹配几何失败
                if 'SnodeID' in self.gdf.columns and 'EnodeID' in self.gdf.columns:
                    self.gdf['SnodeID'] = self.gdf['SnodeID'].astype(str).map(self._normalize_node_id)
                    self.gdf['EnodeID'] = self.gdf['EnodeID'].astype(str).map(self._normalize_node_id)
                
                # ⚠️ STRtree 通常不能直接 pickle (包含 C 指针)，需要重新构建
                # 但构建树比解析 CSV 快得多，瞬间就能完成
                logging.info("⚡ 正在重建空间索引...")
                self.sindex = STRtree(self.geometries)
                
                self.is_loaded = True
                # 重建拓扑图与边几何索引
                self._build_graph_and_edge_map()
                # 重新构建 HMM 空间索引
                self._ensure_hmm_index()
                logging.info(f"✅ 缓存加载成功! (HMM节点数: {self.hmm_map.size()})")
                return True, "缓存加载成功"
            except Exception as e:
                logging.warning(f"⚠️ 缓存文件损坏或版本不兼容，将重新解析 CSV: {e}")
                # 如果读取缓存失败，就继续往下走，重新解析 CSV

        # 2. 解析 CSV (慢速模式 - 仅第一次)
        if not os.path.exists(LOCAL_ROAD_FILE):
            logging.error(f"路网文件未找到: {LOCAL_ROAD_FILE}")
            return False, "文件不存在"
        
        try:
            logging.info("🐢 未找到缓存，正在从 CSV 解析路网 (耗时操作)...")
            
            # --- 解析逻辑：模仿 match_and_plot 的 CSV 读取方式 ---
            header = self._read_header(LOCAL_ROAD_FILE)
            geom_idx = header.index("geometry") if "geometry" in header else None
            records = []
            valid_geoms = []
            valid_indices = []

            self.hmm_map = InMemMap("beijing_net", use_latlon=True)
            coord_index = {}

            for idx, row in enumerate(self._iter_rows(LOCAL_ROAD_FILE)):
                record = {header[i]: row[i] if i < len(row) else "" for i in range(len(header))}
                records.append(record)

                try:
                    geom_str = record.get("geometry", "") if geom_idx is not None else ""
                    coords = self._parse_linestring_wkt(geom_str)
                    if not coords:
                        continue

                    geom = LineString([(lon, lat) for lat, lon in coords])
                    valid_geoms.append(geom)
                    valid_indices.append(idx)

                    direction = record.get('Direction', 1)
                    try:
                        direction = int(float(direction))
                    except Exception:
                        direction = 1

                    prev_node_id = None
                    for lat, lon in coords:
                        key = (round(lat, 6), round(lon, 6))
                        node_id = coord_index.get(key)
                        if node_id is None:
                            node_id = f"n{len(coord_index)}"
                            coord_index[key] = node_id
                            self.hmm_map.add_node(node_id, (lat, lon))

                        if prev_node_id is not None:
                            if direction == 1:
                                self.hmm_map.add_edge(prev_node_id, node_id)
                                self.hmm_map.add_edge(node_id, prev_node_id)
                            elif direction == 2:
                                self.hmm_map.add_edge(prev_node_id, node_id)
                            elif direction == 3:
                                self.hmm_map.add_edge(node_id, prev_node_id)
                            else:
                                self.hmm_map.add_edge(prev_node_id, node_id)
                                self.hmm_map.add_edge(node_id, prev_node_id)

                        prev_node_id = node_id
                except Exception:
                    continue

            df = pd.DataFrame.from_records(records)

            # 保存到 self
            df['shapely_geom'] = [None] * len(df)
            for i, list_idx in enumerate(valid_indices):
                df.at[list_idx, 'shapely_geom'] = valid_geoms[i]

            self.gdf = df.dropna(subset=['shapely_geom'])
            # 统一节点ID类型，避免匹配几何失败
            if 'SnodeID' in self.gdf.columns and 'EnodeID' in self.gdf.columns:
                self.gdf['SnodeID'] = self.gdf['SnodeID'].astype(str).map(self._normalize_node_id)
                self.gdf['EnodeID'] = self.gdf['EnodeID'].astype(str).map(self._normalize_node_id)
            self.geometries = valid_geoms
            self.indices = valid_indices
            self.sindex = STRtree(self.geometries)
            self.is_loaded = True
            # 构建拓扑图与边几何索引
            self._build_graph_and_edge_map()
            # 构建 HMM 空间索引
            self._ensure_hmm_index()
            
            # 3. 🔥 生成缓存文件 🔥
            logging.info(f"💾 正在生成缓存文件 (下次启动将秒开)...")
            try:
                cache_data = {
                    'cache_version': CACHE_VERSION,
                    'gdf': self.gdf,
                    'hmm_map': self.hmm_map
                }
                with open(CACHE_FILE, 'wb') as f:
                    pickle.dump(cache_data, f)
                logging.info(f"✅ 缓存保存完毕: {CACHE_FILE}")
            except Exception as e:
                logging.error(f"❌ 缓存保存失败: {e}")

            return True, f"加载成功 (已生成缓存)"
            
        except Exception as e:
            logging.error(f"加载异常: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)

    def _build_graph_and_edge_map(self):
        """基于 gdf 构建有向图和边->几何映射，便于路径补全。"""
        if self.gdf is None or self.gdf.empty:
            self.graph = None
            self.edge_geom_map = {}
            return

        self.graph = nx.DiGraph()
        self.edge_geom_map = {}

        def _line_length_m(line):
            # 基于经纬度的近似长度 (米)
            coords = list(line.coords)
            if len(coords) < 2:
                return 0.0
            total = 0.0
            for i in range(len(coords) - 1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i + 1]
                # 简单近似：经纬度转米
                dx = (lon2 - lon1) * 111000 * 0.76
                dy = (lat2 - lat1) * 111000
                total += (dx * dx + dy * dy) ** 0.5
            return total

        for _, row in self.gdf.iterrows():
            geom = row.get('shapely_geom', None)
            if geom is None:
                continue
            s_node = self._normalize_node_id(row.get('SnodeID', ''))
            e_node = self._normalize_node_id(row.get('EnodeID', ''))
            if not s_node or not e_node:
                continue

            length_m = _line_length_m(geom)
            direction = row.get('Direction', 1)
            try:
                direction = int(float(direction))
            except Exception:
                direction = 1

            def _add_edge(u, v):
                self.graph.add_edge(u, v, length=length_m)
                key = (u, v)
                if key not in self.edge_geom_map:
                    self.edge_geom_map[key] = []
                self.edge_geom_map[key].append((length_m, geom))

            if direction == 1:
                _add_edge(s_node, e_node)
                _add_edge(e_node, s_node)
            elif direction == 2:
                _add_edge(s_node, e_node)
            elif direction == 3:
                _add_edge(e_node, s_node)
            else:
                _add_edge(s_node, e_node)
                _add_edge(e_node, s_node)

    def get_shortest_path_edges(self, u, v):
        """返回最短路径的边序列 [(n1,n2), ...]，失败则空列表。"""
        try:
            if self.graph is None:
                return []
            u_key = self._normalize_node_id(u)
            v_key = self._normalize_node_id(v)
            if u_key not in self.graph or v_key not in self.graph:
                return []
            path_nodes = nx.shortest_path(self.graph, source=u_key, target=v_key, weight='length')
            if not path_nodes or len(path_nodes) < 2:
                return []
            return list(zip(path_nodes[:-1], path_nodes[1:]))
        except Exception:
            return []

    def get_candidates(self, lat, lon, radius=50):
        if not self.is_loaded: return []
        buffer_deg = radius / 111000.0
        query_box = box(lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg)
        candidate_indices = self.sindex.query(query_box)
        candidates = []
        point = Point(lon, lat)
        for idx in candidate_indices:
            geom = self.geometries[idx]
            dist_deg = point.distance(geom)
            dist_m = dist_deg * 111000 
            if dist_m <= radius:
                row_idx = self.indices[idx] 
                row = self.gdf.loc[row_idx]
                proj_dist = geom.project(point) 
                proj_point = geom.interpolate(proj_dist)
                candidates.append({
                    "edge_id": row.get('ID', idx),
                    "dist_m": dist_m,
                    "proj_point": proj_point,
                    "geometry": geom # 供手写HMM使用
                })
        return candidates

    def query_roads_in_bounds(self, min_lat, min_lon, max_lat, max_lon, buffer=0.01):
        if not self.is_loaded: return []
        search_box = box(min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)
        indices = self.sindex.query(search_box)
        segments = []
        for idx in indices:
            geom = self.geometries[idx]
            line = [[p[1], p[0]] for p in geom.coords]
            segments.append(line)
        return segments

    def get_edge_geometry(self, u, v):
        try:
            u_key = self._normalize_node_id(u)
            v_key = self._normalize_node_id(v)
            # 先从边映射中取几何
            key = (u_key, v_key)
            if key in self.edge_geom_map and self.edge_geom_map[key]:
                # 选择最短的一条作为代表
                return sorted(self.edge_geom_map[key], key=lambda x: x[0])[0][1]
            rows = self.gdf[(self.gdf['SnodeID'] == u_key) & (self.gdf['EnodeID'] == v_key)]
            if not rows.empty: return rows.iloc[0]['shapely_geom']
            
            rows = self.gdf[(self.gdf['SnodeID'] == v_key) & (self.gdf['EnodeID'] == u_key)]
            if not rows.empty: return rows.iloc[0]['shapely_geom']
            return None
        except:
            return None

road_network_service = RoadNetwork()