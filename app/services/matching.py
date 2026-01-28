import os
import pandas as pd
from app.services.road_network import road_network_service
from app.services.ivmm import SimpleIVMM

# 引入 LeuvenMapMatching
try:
    from leuvenmapmatching.matcher.distance import DistanceMatcher
    from leuvenmapmatching.map.inmem import InMemMap
    HAS_LEUVEN = True
except ImportError:
    HAS_LEUVEN = False
    print("⚠️ 警告: 未检测到 leuvenmapmatching 库，HMM 功能不可用。请运行 pip install leuvenmapmatching scipy")


def map_match(df_input, algorithm='Simple', config=None):
    """路径匹配入口"""
    print(f"\n>>> [路径匹配] 算法: {algorithm}")
    config = config or {}

    if not road_network_service.is_loaded:
        return df_input.copy(), "❌ 路网未加载"

    if algorithm == 'HMM':
        res, msg = _match_hmm_leuven(df_input, config)
    elif algorithm == 'IVMM':
        res, msg = _match_ivmm(df_input, config)
    else:
        res, msg = _match_simple(df_input)

    print(f">>> [路径匹配] 结束")
    return res, msg


def _match_hmm_leuven(df, config=None):
    """
    使用 LeuvenMapMatching 进行 HMM 匹配
    """
    if not HAS_LEUVEN:
        return df, "❌ 未安装 leuvenmapmatching"

    config = config or {}
    debug_match = bool(config.get('debug_match', False))

    if not road_network_service.is_loaded or road_network_service.hmm_map is None:
        return df, "❌ 路网尚未初始化"

    print(f"    -> [Leuven] 开始匹配...")
    try:
        map_con = road_network_service.hmm_map

        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        path = list(zip(df['lat'], df['lon']))

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

        matcher = DistanceMatcher(
            map_con,
            max_dist=500,
            obs_noise=30,
            obs_noise_ne=50,
            max_lattice_width=50,
        )

        print(f"    -> [Leuven] 执行 Viterbi 算法...")
        states, _ = matcher.match(path)
        # print(f"    -> [Leuven] states: {states}")

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
            return _match_simple(df)

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
            return _match_simple(df, radius=200)

        return pd.DataFrame(matched_points), "✅ HMM 匹配成功"

    except Exception as e:
        print(f"❌ [Leuven] 异常: {e}")
        import traceback
        traceback.print_exc()
        return _match_simple(df)


def _match_simple(df, radius=50):
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


def _match_ivmm(df, config=None):
    """IVMM 匹配"""
    config = config or {}
    print(f"    -> [IVMM] 开始匹配...")

    try:
        search_radius = float(config.get('ivmm_search_radius', 100))
        w_dist = float(config.get('ivmm_w_dist', 0.6))
        w_heading = float(config.get('ivmm_w_heading', 0.4))

        matcher = SimpleIVMM(
            search_radius=search_radius,
            w_dist=w_dist,
            w_heading=w_heading,
        )

        trajectory = []
        for _, row in df.iterrows():
            trajectory.append({
                'lat': row['lat'],
                'lon': row['lon'],
                'timestamp': row['timestamp'] if 'timestamp' in row else None,
            })

        matched_candidates = matcher.match(trajectory)
        if not matched_candidates:
            return _match_simple(df)

        matched_points = []
        for cand in matched_candidates:
            if 'proj_point' in cand and cand['proj_point'] is not None:
                p = cand['proj_point']
                matched_points.append({'lat': p.y, 'lon': p.x})
            elif 'lat' in cand and 'lon' in cand:
                matched_points.append({'lat': cand['lat'], 'lon': cand['lon']})

        if len(matched_points) < 1:
            return _match_simple(df)

        return pd.DataFrame(matched_points), "✅ IVMM 匹配成功"

    except Exception as e:
        print(f"❌ [IVMM] 异常: {e}")
        return _match_simple(df)
