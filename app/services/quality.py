import pandas as pd
import numpy as np


def check_quality(df, config: dict):
    """
    质量检测模块
    返回:
    1. summary: 总体评分和统计
    2. details: 包含每个点状态的列表
    """
    df = df.copy()
    total_points = len(df)

    weights = config.get('quality_weights', {'time': 0.25, 'integrity': 0.25, 'speed': 0.25, 'angle': 0.25})
    max_speed = float(config.get('qc_max_speed', 33.3))
    max_angle = float(config.get('qc_max_angle', 60.0))
    max_time_gap = float(config.get('qc_max_time_gap', 60.0))
    min_turn_dist = float(config.get('qc_min_turn_dist', 2.0))

    df['prev_lat'] = df['lat'].shift(1)
    df['prev_lon'] = df['lon'].shift(1)
    lat_diff = (df['lat'] - df['prev_lat']) * 111000
    lon_diff = (df['lon'] - df['prev_lon']) * 111000 * 0.76
    df['dist_m'] = np.sqrt(lat_diff**2 + lon_diff**2).fillna(0)

    df['next_lat'] = df['lat'].shift(-1)
    df['next_lon'] = df['lon'].shift(-1)
    lat_diff_next = (df['next_lat'] - df['lat']) * 111000
    lon_diff_next = (df['next_lon'] - df['lon']) * 111000 * 0.76
    df['dist_next_m'] = np.sqrt(lat_diff_next**2 + lon_diff_next**2).fillna(0)

    if 'timestamp' in df.columns:
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    else:
        df['time_diff'] = 1.0

    df['speed_mps'] = (df['dist_m'] / df['time_diff'].replace(0, 0.001)).fillna(0)

    df['heading_prev'] = np.degrees(np.arctan2(lat_diff, lon_diff))
    df['heading_next'] = np.degrees(np.arctan2(lat_diff_next, lon_diff_next))

    raw_diff = (df['heading_next'] - df['heading_prev']).abs()
    df['heading_diff'] = raw_diff.apply(lambda x: min(x, 360 - x))

    invalid_turn = (df['dist_m'] < min_turn_dist) | (df['dist_next_m'] < min_turn_dist)
    df.loc[invalid_turn, 'heading_diff'] = 0.0

    mask_integrity = (
        (df['road'] == '')
        | (df['status'] == '')
        | (df['lat'].isnull())
        | (df['lon'].isnull())
        | ((df['lat'] == 0) & (df['lon'] == 0))
    )
    mask_time = df['time_diff'] > max_time_gap
    mask_speed = df['speed_mps'] > max_speed
    mask_angle = (
        (df['heading_diff'] > max_angle)
        & (df['dist_m'] >= min_turn_dist)
        & (df['dist_next_m'] >= min_turn_dist)
    )

    df['qc_status'] = '正常'
    df['qc_tags'] = [[] for _ in range(len(df))]

    anomalies = {
        'time': df[mask_time].index.tolist(),
        'integrity': df[mask_integrity].index.tolist(),
        'speed': df[mask_speed].index.tolist(),
        'angle': df[mask_angle].index.tolist(),
    }

    for idx in df[mask_integrity].index:
        df.at[idx, 'qc_tags'].append('缺失/零值')
    for idx in df[mask_time].index:
        df.at[idx, 'qc_tags'].append('时间断裂')
    for idx in df[mask_speed].index:
        df.at[idx, 'qc_tags'].append(f'速度过快({df.at[idx, "speed_mps"]:.1f}m/s)')
    for idx in df[mask_angle].index:
        df.at[idx, 'qc_tags'].append(f'急转弯({df.at[idx, "heading_diff"]:.0f}°)')

    def format_status(tags):
        return " | ".join(tags) if tags else "正常"

    df['qc_desc'] = df['qc_tags'].apply(format_status)
    df['is_abnormal'] = df['qc_tags'].apply(lambda x: len(x) > 0)

    s_integrity = len(anomalies['integrity']) / total_points if total_points else 0
    s_time = len(anomalies['time']) / total_points if total_points else 0
    s_speed = len(anomalies['speed']) / total_points if total_points else 0
    s_angle = len(anomalies['angle']) / total_points if total_points else 0

    deduction = (
        s_time * weights.get('time', 0.25)
        + s_integrity * weights.get('integrity', 0.25)
        + s_speed * weights.get('speed', 0.25)
        + s_angle * weights.get('angle', 0.25)
    )
    score = max(0, 100 * (1 - deduction))

    details_list = []
    for idx, row in df.iterrows():
        details_list.append({
            "id": idx,
            "road": row['road'],
            "situation": row['status'],
            "timestamp": row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row.get('timestamp')) else '-',
            "lat": row['lat'],
            "lon": row['lon'],
            "speed": round(row['speed_mps'], 2),
            "angle_diff": round(row['heading_diff'], 1),
            "status": row['qc_desc'],
            "is_error": row['is_abnormal'],
        })

    summary = {
        "score": round(score, 1),
        "counts": {
            "time": len(anomalies['time']),
            "integrity": len(anomalies['integrity']),
            "speed": len(anomalies['speed']),
            "angle": len(anomalies['angle']),
        },
    }

    return summary, details_list
