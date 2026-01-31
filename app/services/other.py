import io
import pandas as pd
from app.services.quality import check_quality


def parse_timestamps(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors='coerce', format='ISO8601')
    if parsed.isna().any():
        parsed = pd.to_datetime(series, errors='coerce', format='mixed')
    return parsed


def parse_upload_points(content: bytes):
    # 1. 读取 CSV (优先无表头)
    try:
        df = pd.read_csv(
            io.BytesIO(content),
            header=None,
            names=['road', 'status', 'distance', 'duration', 'polyline'],
        )
    except Exception:
        df = pd.read_csv(io.BytesIO(content))

    # required_cols = ['road', 'status', 'distance', 'duration', 'polyline']
    # missing_cols = [c for c in required_cols if c not in df.columns]
    # if missing_cols:
    #     raise ValueError(f"缺少必要字段: {','.join(missing_cols)}")

    # missing_mask = df[required_cols].isna()
    # empty_mask = df[required_cols].astype(str).apply(lambda s: s.str.strip() == '')
    # any_missing = (missing_mask | empty_mask).any(axis=1)
    # if any_missing.any():
    #     missing_rows = int(any_missing.sum())
    #     raise ValueError(f"存在缺失值的行数: {missing_rows}")

    df = df.fillna("")
    points = []

    # 2. 增强解析逻辑：利用 duration 计算时间
    current_base_time = pd.Timestamp('2024-01-01 08:00:00')

    if 'polyline' in df.columns:
        for _, row in df.iterrows():
            polyline_str = str(row['polyline'])
            if not polyline_str or polyline_str.lower() == 'nan':
                continue

            raw_points = []
            parts = polyline_str.split('|')
            for pt_str in parts:
                if '-' in pt_str:
                    try:
                        lon_str, lat_str = pt_str.split('-')
                        lon, lat = float(lon_str), float(lat_str)
                        raw_points.append((lat, lon))
                    except Exception:
                        continue

            if not raw_points:
                continue

            segment_duration = 0
            try:
                segment_duration = float(row.get('duration', 0))
            except Exception:
                segment_duration = 0

            n_pts = len(raw_points)
            dt = 0
            if n_pts > 1 and segment_duration > 0:
                dt = segment_duration / (n_pts - 1)

            for i, (lat, lon) in enumerate(raw_points):
                pt_time = current_base_time + pd.Timedelta(seconds=i * dt)

                points.append({
                    'id': len(points),
                    'lat': lat,
                    'lon': lon,
                    'timestamp': pt_time,
                    'road': str(row.get('road', '')),
                    'status': str(row.get('status', '')),
                    'orig_duration': segment_duration if i == 0 else 0,
                })

            step = segment_duration if segment_duration > 0 else 1.0
            current_base_time += pd.Timedelta(seconds=step)

    elif 'lat' in df.columns and 'lon' in df.columns:
        if 'timestamp' in df.columns:
            df['timestamp'] = parse_timestamps(df['timestamp'])
            df = df.sort_values('timestamp')

        for idx, row in df.iterrows():
            points.append({
                'id': idx,
                'lat': row['lat'],
                'lon': row['lon'],
                'timestamp': row['timestamp'] if 'timestamp' in row else pd.Timestamp('2024-01-01') + pd.Timedelta(seconds=idx),
                'road': str(row.get('road', '')),
                'status': str(row.get('status', '')),
            })

    print(f">>> 解析完成，提取了 {len(points)} 个点")

    df_qc = pd.DataFrame(points)
    if 'timestamp' in df_qc.columns:
        df_qc['timestamp'] = parse_timestamps(df_qc['timestamp'])

    default_config = {
        'qc_max_speed': 33.3,
        'qc_max_angle': 60.0,
        'qc_max_time_gap': 60.0,
    }

    qc_summary, qc_details = check_quality(df_qc, default_config)

    for p in points:
        if isinstance(p.get('timestamp'), pd.Timestamp):
            p['timestamp'] = p['timestamp'].isoformat()

    return points, qc_summary, qc_details
