import pandas as pd
import numpy as np
import time
import os
from road_network import road_network_service
from ivmm import SimpleIVMM

# 引入 LeuvenMapMatching
try:
    from leuvenmapmatching.matcher.distance import DistanceMatcher
    from leuvenmapmatching.map.inmem import InMemMap
    HAS_LEUVEN = True
except ImportError:
    HAS_LEUVEN = False
    print("⚠️ 警告: 未检测到 leuvenmapmatching 库，HMM 功能不可用。请运行 pip install leuvenmapmatching scipy")

class TrajectoryProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df = self.df.sort_values('timestamp')

    def preprocess_pipeline(self, config: dict):
        """
        数据清洗管道：停留点聚类 -> 卡尔曼滤波
        """
        print(f"\n>>> [预处理] 开始执行...")
        start_time = time.time()
        df = self.df.copy()
        
        if 'lat' not in df.columns or 'lon' not in df.columns:
            return df

        # 1. 停留点聚类
        if config.get('remove_stop_points', False):
            print(f"    -> 执行: 停留点聚类")
            radius = float(config.get('stop_radius', 5.0))
            duration = float(config.get('stop_duration', 30.0))

            df['prev_lat'] = df['lat'].shift(1)
            df['prev_lon'] = df['lon'].shift(1)
            lat_diff = (df['lat'] - df['prev_lat']) * 111000
            lon_diff = (df['lon'] - df['prev_lon']) * 111000 * 0.76
            df['dist_diff'] = np.sqrt(lat_diff**2 + lon_diff**2).fillna(9999)

            if 'timestamp' in df.columns:
                df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(9999)
                df['is_moving'] = (df['dist_diff'] > radius) | (df['time_diff'] > duration)
            else:
                df['is_moving'] = df['dist_diff'] > radius

            df['group_id'] = df['is_moving'].cumsum()
            
            agg_rules = {'lat': 'mean', 'lon': 'mean'}
            for col in ['timestamp', 'road', 'status', 'id']:
                if col in df.columns: agg_rules[col] = 'first'
            df = df.groupby('group_id', as_index=False).agg(agg_rules)

        # 2. 卡尔曼滤波
        if config.get('enable_kalman', True):
            print(f"    -> 执行: 卡尔曼滤波")
            try:
                user_R = float(config.get('kalman_R', 0.01))
                user_Q = float(config.get('kalman_Q', 1000.0))
                
                dim_x, dim_z, dt = 4, 2, 1.0
                F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
                H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
                P = np.eye(dim_x) * 1000.0
                R = np.eye(dim_z) * user_R
                Q = np.eye(dim_x) * user_Q
                
                if len(df) > 0:
                    x = np.array([df.iloc[0]['lat'], df.iloc[0]['lon'], 0., 0.])
                    smoothed = []
                    for _, row in df.iterrows():
                        z = np.array([row['lat'], row['lon']])
                        # Predict
                        x = F @ x
                        P = F @ P @ F.T + Q
                        # Update
                        S = H @ P @ H.T + R
                        K = P @ H.T @ np.linalg.inv(S)
                        y = z - H @ x
                        x = x + K @ y
                        P = (np.eye(dim_x) - K @ H) @ P
                        smoothed.append({'lat': x[0], 'lon': x[1]})
                    
                    res_df = pd.DataFrame(smoothed)
                    df['lat'] = res_df['lat']
                    df['lon'] = res_df['lon']
            except Exception as e:
                print(f"❌ 卡尔曼滤波出错: {e}")

        print(f">>> [预处理] 结束，耗时 {time.time() - start_time:.2f}s")
        return df

    def map_match(self, df_input, algorithm='Simple', config=None):
        """路径匹配入口"""
        print(f"\n>>> [路径匹配] 算法: {algorithm}")
        start_time = time.time()
        config = config or {}
        
        if not road_network_service.is_loaded:
            return df_input.copy(), "❌ 路网未加载"
        
        if algorithm == 'HMM':
            res, msg = self._match_hmm_leuven(df_input, config)
        elif algorithm == 'IVMM':
            res, msg = self._match_ivmm(df_input, config)
        else:
            res, msg = self._match_simple(df_input)
            
        print(f">>> [路径匹配] 结束，耗时 {time.time() - start_time:.2f}s")
        return res, msg

    def _match_hmm_leuven(self, df, config=None):
        """
        使用 LeuvenMapMatching 进行 HMM 匹配 (最终优化版)
        """
        if not HAS_LEUVEN:
            return df, "❌ 未安装 leuvenmapmatching"

        config = config or {}
        debug_match = bool(config.get('debug_match', False))

        if not road_network_service.is_loaded or road_network_service.hmm_map is None:
            return df, "❌ 路网尚未初始化"

        print(f"    -> [Leuven] 开始匹配...")
        try:
            # 1. 直接获取预构建好的地图 (省时!)
            map_con = road_network_service.hmm_map

            # 2. 准备轨迹 (确保按时间排序)
            if 'timestamp' in df.columns:
                 df = df.sort_values('timestamp')
            path = list(zip(df['lat'], df['lon']))
            timestamps = df['timestamp'].tolist() if 'timestamp' in df.columns else [None] * len(path)
            # for i, ((lat, lon), ts) in enumerate(zip(path, timestamps)):
            #     print(f"    -> [Leuven] path[{i}]: lat={lat}, lon={lon}, ts={ts}")

            # 诊断：轨迹范围与候选路段覆盖情况
            if len(path) > 0:
                lats = [p[0] for p in path]
                lons = [p[1] for p in path]
                print(f"    -> [Diag] 轨迹范围 lat[{min(lats):.6f}, {max(lats):.6f}], lon[{min(lons):.6f}, {max(lons):.6f}]")
                sample_idxs = sorted(set([0, len(path) // 2, len(path) - 1]))
                for idx in sample_idxs:
                    lat, lon = path[idx]
                    cands_50 = road_network_service.get_candidates(lat, lon, radius=50)
                    cands_200 = road_network_service.get_candidates(lat, lon, radius=200)
                    print(f"    -> [Diag] sample[{idx}] candidates r=50: {len(cands_50)}, r=200: {len(cands_200)}")

            # 3. 初始化匹配器
            # 这里的参数根据你的数据稀疏度调整
            matcher = DistanceMatcher(
                map_con,
                max_dist=500,          # 最大候选距离（米）
                obs_noise=30,         # 观测噪声（米）
                obs_noise_ne=50,      # 观测噪声（非欧式）
                max_lattice_width=50,  # 限制搜索宽度
            )
            
            print(f"    -> [Leuven] 执行 Viterbi 算法...")
            states, _ = matcher.match(path)
            print(f"    -> [Leuven] states: {states}")

            # 可视化输出（LeuvenMapMatching 自带）
            if config.get('hmm_viz', False) or config.get('hmm_visualize', False):
                try:
                    from leuvenmapmatching import visualization as mmviz

                    base_dir = os.path.dirname(__file__)
                    out_path = config.get('hmm_viz_filename') or os.path.join(base_dir, 'data', 'hmm_match.png')
                    if not os.path.isabs(out_path):
                        out_path = os.path.join(base_dir, out_path)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    mmviz.plot_map(
                        map_con,
                        matcher=matcher,
                        show_labels=bool(config.get('hmm_viz_show_labels', False)),
                        show_matching=True,
                        show_graph=bool(config.get('hmm_viz_show_graph', True)),
                        use_osm=bool(config.get('hmm_viz_use_osm', False)),
                        zoom_path=bool(config.get('hmm_viz_zoom_path', True)),
                        filename=out_path,
                    )
                    print(f"✅ [Leuven] 可视化已保存: {out_path}")
                except ImportError:
                    print("⚠️ [Leuven] 可视化依赖未安装，请运行 pip install smopy")
                except Exception as viz_e:
                    print(f"⚠️ [Leuven] 可视化失败: {viz_e}")
            
            if not states:
                print("⚠️ [Leuven] 未找到匹配路径")
                # 自动降级到 Simple 匹配
                return self._match_simple(df)
            
            # 4. 解析结果（模仿 match_and_plot 的写法）
            matched_points = []
            print(f"    -> [Leuven] 解析匹配路径...")
            last_point = None

            for item in states:
                try:
                    if isinstance(item, tuple) and len(item) == 2:
                        u, v = item
                        cu = map_con.node_coordinates(u)
                        cv = map_con.node_coordinates(v)
                        if cu is not None:
                            pt = (cu[0], cu[1])
                            if pt != last_point:
                                matched_points.append({'lat': cu[0], 'lon': cu[1]})
                                last_point = pt
                        if cv is not None:
                            pt = (cv[0], cv[1])
                            if pt != last_point:
                                matched_points.append({'lat': cv[0], 'lon': cv[1]})
                                last_point = pt
                    else:
                        coord = map_con.node_coordinates(item)
                        if coord is not None:
                            pt = (coord[0], coord[1])
                            if pt != last_point:
                                matched_points.append({'lat': coord[0], 'lon': coord[1]})
                                last_point = pt
                except Exception as parse_e:
                    if debug_match:
                        print(f"    -> [Diag] 解析路径失败: {parse_e}")

            print(f"✅ [Leuven] 成功生成 {len(matched_points)} 个匹配点")

            if len(matched_points) < 2:
                if debug_match:
                    print("    -> [Diag] HMM 输出过短，回退到 Simple(半径200m)")
                return self._match_simple(df, radius=200)

            return pd.DataFrame(matched_points), "✅ HMM 匹配成功"

        except Exception as e:
            print(f"❌ [Leuven] 异常: {e}")
            import traceback
            traceback.print_exc()
            return self._match_simple(df)

    def _match_simple(self, df, radius=50):
        """最近邻吸附"""
        print(f"    -> [Simple] 执行最近邻搜索...")
        matched = []
        for _, row in df.iterrows():
            cands = road_network_service.get_candidates(row['lat'], row['lon'], radius=radius)
            if cands:
                best = min(cands, key=lambda x: x['dist_m'])
                p = best['proj_point']
                matched.append({'lat': p.y, 'lon': p.x})
            else:
                matched.append({'lat': row['lat'], 'lon': row['lon']})
        return pd.DataFrame(matched), "Simple 匹配完成"

    def _match_ivmm(self, df, config=None):
        """IVMM 匹配 (调用 ivmm.py)"""
        config = config or {}
        print(f"    -> [IVMM] 开始匹配...")

        try:
            # 从配置读取参数
            search_radius = float(config.get('ivmm_search_radius', 100))
            w_dist = float(config.get('ivmm_w_dist', 0.6))
            w_heading = float(config.get('ivmm_w_heading', 0.4))

            matcher = SimpleIVMM(
                search_radius=search_radius,
                w_dist=w_dist,
                w_heading=w_heading
            )

            # 轨迹点序列
            trajectory = []
            for _, row in df.iterrows():
                trajectory.append({
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'timestamp': row['timestamp'] if 'timestamp' in row else None
                })

            matched_candidates = matcher.match(trajectory)
            if not matched_candidates:
                return self._match_simple(df)

            # 输出 matched 点坐标 (投影点优先)
            matched_points = []
            for cand in matched_candidates:
                if 'proj_point' in cand and cand['proj_point'] is not None:
                    p = cand['proj_point']
                    matched_points.append({'lat': p.y, 'lon': p.x})
                elif 'lat' in cand and 'lon' in cand:
                    matched_points.append({'lat': cand['lat'], 'lon': cand['lon']})

            if len(matched_points) < 1:
                return self._match_simple(df)

            return pd.DataFrame(matched_points), "✅ IVMM 匹配成功"

        except Exception as e:
            print(f"❌ [IVMM] 异常: {e}")
            return self._match_simple(df)

    def quality_check(self, df):
        """简单质检"""
        return {"point_count": len(df)}