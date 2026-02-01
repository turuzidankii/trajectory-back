import time
import pandas as pd
from app.services.road_network import road_network_service


def _approx_dist_m(lat1, lon1, lat2, lon2):
    dx = (lon2 - lon1) * 111000 * 0.76
    dy = (lat2 - lat1) * 111000
    return (dx * dx + dy * dy) ** 0.5


def _choose_node_for_candidate(cand):
    s_node = cand.get('s_node')
    e_node = cand.get('e_node')
    if not s_node and not e_node:
        return None
    if not e_node:
        return s_node
    if not s_node:
        return e_node
    geom = cand.get('geometry')
    proj_point = cand.get('proj_point')
    if geom is None or proj_point is None:
        return s_node
    try:
        start_lon, start_lat = geom.coords[0]
        end_lon, end_lat = geom.coords[-1]
        d1 = _approx_dist_m(proj_point.y, proj_point.x, start_lat, start_lon)
        d2 = _approx_dist_m(proj_point.y, proj_point.x, end_lat, end_lon)
        return s_node if d1 <= d2 else e_node
    except Exception:
        return s_node


def fill_path_astar(df, config=None):
    """使用 A* 在相邻匹配点之间补全路径。"""
    config = config or {}
    if df is None or df.empty:
        return df, "A* 补全跳过(空数据)"

    print("    -> [A*] 开始补全路径...")
    start_time = time.perf_counter()

    road_network_service.ensure_graph()

    min_dist_m = float(config.get('astar_min_dist_m', 150))
    cand_radius = float(config.get('astar_candidate_radius', 80))
    dedup_m = float(config.get('astar_dedup_m', 3))
    force_astar = bool(config.get('astar_force', False))

    has_ts = 'timestamp' in df.columns
    work_df = df.sort_values('timestamp') if has_ts else df

    points = []
    for _, row in work_df.iterrows():
        lat = row['lat']
        lon = row['lon']
        ts = row['timestamp'] if has_ts else None
        cands = road_network_service.get_candidates_with_nodes(lat, lon, radius=cand_radius)
        best = min(cands, key=lambda x: x['dist_m']) if cands else None
        node_id = _choose_node_for_candidate(best) if best else None
        points.append({'lat': lat, 'lon': lon, 'timestamp': ts, 'node': node_id})

    filled = []
    last_point = None

    def _append_point(lat, lon, ts):
        nonlocal last_point
        if last_point is not None:
            if _approx_dist_m(last_point[0], last_point[1], lat, lon) <= dedup_m:
                return
        filled.append({'lat': lat, 'lon': lon, 'timestamp': ts})
        last_point = (lat, lon)

    for i in range(len(points) - 1):
        curr = points[i]
        next_p = points[i + 1]

        _append_point(curr['lat'], curr['lon'], curr['timestamp'])

        if not curr['node'] or not next_p['node']:
            continue

        seg_dist = _approx_dist_m(curr['lat'], curr['lon'], next_p['lat'], next_p['lon'])
        if not force_astar and seg_dist < min_dist_m:
            continue

        edges = road_network_service.get_astar_path_edges(curr['node'], next_p['node'])
        if not edges:
            continue

        for u, v in edges:
            geom = road_network_service.get_edge_geometry(u, v)
            if geom is None:
                continue
            coords = list(geom.coords)

            try:
                u_key = road_network_service.normalize_node_id(u)
                u_coord = road_network_service.node_coord_map.get(u_key)
                if u_coord:
                    start_lon, start_lat = coords[0]
                    end_lon, end_lat = coords[-1]
                    d_start = _approx_dist_m(u_coord[0], u_coord[1], start_lat, start_lon)
                    d_end = _approx_dist_m(u_coord[0], u_coord[1], end_lat, end_lon)
                    if d_end < d_start:
                        coords = list(reversed(coords))
            except Exception:
                pass

            for lon, lat in coords:
                _append_point(lat, lon, None)

    last = points[-1]
    _append_point(last['lat'], last['lon'], last['timestamp'])

    cost_s = (time.perf_counter() - start_time)
    print(f"    -> [A*] 补全完成, 耗时: {cost_s:.2f} s")
    return pd.DataFrame(filled), "A* 补全完成"
