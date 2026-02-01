import pandas as pd
import numpy as np
import time


def preprocess_pipeline(df, config: dict):
    """
    数据清洗管道：停留点聚类 -> 卡尔曼滤波
    """
    print(f"\n>>> [预处理] 开始执行...")
    start_time = time.time()
    df = df.copy()

    if 'lat' not in df.columns or 'lon' not in df.columns:
        return df
    
    stop_cluster_algo = config.get('stop_cluster_algo', 'basic')

    if stop_cluster_algo == 'spatiotemporal':
        print(f"    -> 执行: 时空阈值聚类")
        df = apply_stop_cluster_basic(df, config)
    elif stop_cluster_algo == 'density':
        print(f"    -> 执行: 密度聚类")
        df = apply_stop_cluster_density(df, config)

    denoise_algo = config.get('denoise_algo', 'kalman')

    if denoise_algo == 'median':
        print(f"    -> 执行: 中值滤波")
        df = apply_median_filter(df, config)
    elif denoise_algo == 'rts':
        print(f"    -> 执行: RTS平滑")
        df = apply_rts(df, config)
    else:
        print(f"    -> 执行: 卡尔曼滤波")
        df = apply_kalman_filter(df, config)

    print(f">>> [预处理] 结束，耗时 {time.time() - start_time:.2f}s")
    return df


def apply_median_filter(df, config: dict):
    """中值滤波：对 lat/lon 进行滚动中值平滑。"""
    try:
        window = int(config.get('median_window', 3))
        if window < 1:
            return df
        if window % 2 == 0:
            window += 1

        df = df.copy()
        df['lat'] = df['lat'].rolling(window=window, center=True, min_periods=1).median()
        df['lon'] = df['lon'].rolling(window=window, center=True, min_periods=1).median()
        return df
    except Exception as e:
        print(f"❌ 中值滤波出错: {e}")
        return df


def _approx_dist_m(lat1, lon1, lat2, lon2):
    dx = (lon2 - lon1) * 111000 * 0.76
    dy = (lat2 - lat1) * 111000
    return (dx * dx + dy * dy) ** 0.5


def apply_stop_cluster_basic(df, config: dict):
    """时空阈值聚类：按相邻点距离/时间阈值合并停留段。"""
    try:
        radius = float(config.get('stop_radius', 5.0))
        duration = float(config.get('stop_duration', 30.0))

        df = df.copy()
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
            if col in df.columns:
                agg_rules[col] = 'first'
        df = df.groupby('group_id', as_index=False).agg(agg_rules)
        return df
    except Exception as e:
        print(f"❌ 停留点聚类(阈值)出错: {e}")
        return df


def apply_stop_cluster_density(df, config: dict):
    """密度聚类：基于空间(可选时间)邻域进行停留点聚类。"""
    try:
        eps_m = float(config.get('stop_eps_m', 10.0))
        min_samples = int(config.get('stop_min_samples', 3))
        time_eps = config.get('stop_time_eps', None)
        if time_eps is not None:
            time_eps = float(time_eps)

        df = df.copy()
        n = len(df)
        if n == 0:
            return df

        lats = df['lat'].to_numpy()
        lons = df['lon'].to_numpy()
        has_ts = 'timestamp' in df.columns
        ts_vals = df['timestamp'].astype('int64').to_numpy() / 1e9 if has_ts else None

        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        def _neighbors(i):
            neigh = []
            for j in range(n):
                if i == j:
                    continue
                if _approx_dist_m(lats[i], lons[i], lats[j], lons[j]) <= eps_m:
                    if time_eps is not None and has_ts:
                        if abs(ts_vals[j] - ts_vals[i]) > time_eps:
                            continue
                    neigh.append(j)
            return neigh

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = _neighbors(i)
            if len(neighbors) + 1 < min_samples:
                labels[i] = -1
                continue
            labels[i] = cluster_id
            seeds = neighbors[:]
            while seeds:
                j = seeds.pop()
                if not visited[j]:
                    visited[j] = True
                    j_neighbors = _neighbors(j)
                    if len(j_neighbors) + 1 >= min_samples:
                        for item in j_neighbors:
                            if item not in seeds:
                                seeds.append(item)
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1

        df['cluster_id'] = labels

        def _cluster_row(idx_list):
            subset = df.loc[idx_list]
            row = subset.iloc[0].copy()
            row['lat'] = subset['lat'].mean()
            row['lon'] = subset['lon'].mean()
            return row

        clusters = {}
        for idx, cid in enumerate(labels):
            if cid >= 0:
                clusters.setdefault(cid, []).append(idx)

        rows = []
        emitted = set()
        for idx, cid in enumerate(labels):
            if cid < 0:
                rows.append(df.iloc[idx])
                continue
            if cid in emitted:
                continue
            rows.append(_cluster_row(clusters[cid]))
            emitted.add(cid)

        result = pd.DataFrame(rows).reset_index(drop=True)
        result = result.drop(columns=['cluster_id'], errors='ignore')
        return result
    except Exception as e:
        print(f"❌ 停留点聚类(密度)出错: {e}")
        return df


def apply_kalman_filter(df, config: dict):
    """卡尔曼滤波：对 lat/lon 进行平滑。"""
    try:
        user_R = float(config.get('kalman_R', 0.01))
        user_Q = float(config.get('kalman_Q', 500.0))

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
                x = F @ x
                P = F @ P @ F.T + Q
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                y = z - H @ x
                x = x + K @ y
                P = (np.eye(dim_x) - K @ H) @ P
                smoothed.append({'lat': x[0], 'lon': x[1]})

            res_df = pd.DataFrame(smoothed)
            df = df.copy()
            df['lat'] = res_df['lat']
            df['lon'] = res_df['lon']
        return df
    except Exception as e:
        print(f"❌ 卡尔曼滤波出错: {e}")
        return df


def apply_rts(df, config: dict):
    """RTS 平滑：先前向卡尔曼滤波，再后向平滑。"""
    try:
        user_R = float(config.get('rts_R', 0.01))
        user_Q = float(config.get('rts_Q', 500.0))

        dim_x, dim_z, dt = 4, 2, 1.0
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        Q = np.eye(dim_x) * user_Q
        R = np.eye(dim_z) * user_R

        if len(df) == 0:
            return df

        x = np.array([df.iloc[0]['lat'], df.iloc[0]['lon'], 0., 0.])
        P = np.eye(dim_x) * 1000.0

        xs = []
        Ps = []
        x_preds = []
        P_preds = []

        for _, row in df.iterrows():
            z = np.array([row['lat'], row['lon']])
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            y = z - H @ x_pred
            x = x_pred + K @ y
            P = (np.eye(dim_x) - K @ H) @ P_pred

            x_preds.append(x_pred)
            P_preds.append(P_pred)
            xs.append(x)
            Ps.append(P)

        xs_smooth = xs[:]
        Ps_smooth = Ps[:]

        for k in range(len(xs) - 2, -1, -1):
            P_pred = P_preds[k + 1]
            if np.linalg.cond(P_pred) > 1e12:
                Ck = Ps[k] @ F.T @ np.linalg.pinv(P_pred)
            else:
                Ck = Ps[k] @ F.T @ np.linalg.inv(P_pred)

            xs_smooth[k] = xs[k] + Ck @ (xs_smooth[k + 1] - x_preds[k + 1])
            Ps_smooth[k] = Ps[k] + Ck @ (Ps_smooth[k + 1] - P_pred) @ Ck.T

        res_df = pd.DataFrame(xs_smooth, columns=['lat', 'lon', 'v_lat', 'v_lon'])
        df = df.copy()
        df['lat'] = res_df['lat']
        df['lon'] = res_df['lon']
        return df
    except Exception as e:
        print(f"❌ RTS平滑出错: {e}")
        return df
