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
            if col in df.columns:
                agg_rules[col] = 'first'
        df = df.groupby('group_id', as_index=False).agg(agg_rules)

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
                    x = F @ x
                    P = F @ P @ F.T + Q
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
