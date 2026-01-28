from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np  # 确保引入 numpy
import io
import math
from road_network import road_network_service
from algorithms import TrajectoryProcessor

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> 系统启动...")
    success, msg = road_network_service.load_local_file()
    print(f">>> 路网加载状态: {msg}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/road_network/status")
async def get_road_status():
    count = 0
    if road_network_service.is_loaded and road_network_service.gdf is not None:
        count = len(road_network_service.gdf)
    return {"loaded": road_network_service.is_loaded, "nodes": count}

@app.get("/road_network/nearby")
async def get_nearby_roads(min_lat: float, min_lon: float, max_lat: float, max_lon: float):
    if not road_network_service.is_loaded:
        return {"status": "error", "data": []}
    segments = road_network_service.query_roads_in_bounds(min_lat, min_lon, max_lat, max_lon)
    return {"status": "success", "data": segments}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        # 1. 读取 CSV (优先无表头)
        try:
            df = pd.read_csv(io.BytesIO(content), header=None, 
                             names=['road', 'status', 'distance', 'duration', 'polyline'])
        except:
            df = pd.read_csv(io.BytesIO(content))
        
        df = df.fillna("")
        points = []
        
        # 2. 增强解析逻辑：利用 duration 计算时间
        # 初始基准时间
        current_base_time = pd.Timestamp('2024-01-01 08:00:00')

        if 'polyline' in df.columns:
            for idx, row in df.iterrows():
                polyline_str = str(row['polyline'])
                if not polyline_str or polyline_str.lower() == 'nan':
                    continue

                # 解析所有点
                raw_points = []
                parts = polyline_str.split('|')
                for pt_str in parts:
                    if '-' in pt_str:
                        try:
                            lon_str, lat_str = pt_str.split('-')
                            lon, lat = float(lon_str), float(lat_str)
                            raw_points.append((lat, lon))
                        except:
                            continue
                
                if not raw_points:
                    continue

                # 时间分配逻辑
                segment_duration = 0
                try:
                    segment_duration = float(row.get('duration', 0))
                except:
                    segment_duration = 0

                # 如果该段有多个点，将 duration 均匀分配
                n_pts = len(raw_points)
                dt = 0
                if n_pts > 1 and segment_duration > 0:
                    dt = segment_duration / (n_pts - 1)
                
                # 生成点对象
                for i, (lat, lon) in enumerate(raw_points):
                    pt_time = current_base_time + pd.Timedelta(seconds=i * dt)
                    
                    points.append({
                        'id': len(points), # 全局序号
                        'lat': lat,
                        'lon': lon,
                        'timestamp': pt_time,
                        'road': str(row.get('road', '')),
                        'status': str(row.get('status', '')),
                        'orig_duration': segment_duration if i == 0 else 0 # 标记段开始
                    })
                
                # 更新下一段的基准时间 (假设轨迹是连续的)
                # 如果 duration 为 0，手动增加一点时间防止重叠
                step = segment_duration if segment_duration > 0 else 1.0
                current_base_time += pd.Timedelta(seconds=step)

        # 兼容标准格式
        elif 'lat' in df.columns and 'lon' in df.columns:
            # ... (保持原有逻辑) ...
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            for idx, row in df.iterrows():
                points.append({
                    'id': idx,
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'timestamp': row['timestamp'] if 'timestamp' in row else pd.Timestamp('2024-01-01') + pd.Timedelta(seconds=idx)
                })

        print(f">>> 解析完成，提取了 {len(points)} 个点")
        
        df_qc = pd.DataFrame(points)
        if 'timestamp' in df_qc.columns:
            df_qc['timestamp'] = pd.to_datetime(df_qc['timestamp'])
        
        # 使用默认配置进行检测
        default_config = {
            'qc_max_speed': 33.3, 
            'qc_max_angle': 60.0,
            'qc_max_time_gap': 60.0
        }
        
        processor = TrajectoryProcessor(df_qc)
        qc_summary, qc_details = processor.check_quality(default_config)
        
        # 序列化 timestamp 用于 JSON 返回
        for p in points:
            if isinstance(p.get('timestamp'), pd.Timestamp):
                p['timestamp'] = p['timestamp'].isoformat()

        return {
            "status": "success", 
            "count": len(points), 
            "data": points,
            # 返回质检结果
            "qc_summary": qc_summary,
            "qc_details": qc_details
        }
        
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/process")
async def process(data: dict):
    raw_df = pd.DataFrame(data['trajectory'])
    config = data['config']
    
    # 转换回 datetime 对象以便处理
    if 'timestamp' in raw_df.columns:
        ts_raw = raw_df['timestamp']
        parsed = pd.to_datetime(ts_raw, errors='coerce', format='ISO8601')
        if parsed.isna().any():
            parsed = pd.to_datetime(ts_raw, errors='coerce', format='mixed')
        raw_df['timestamp'] = parsed

    processor = TrajectoryProcessor(raw_df)
    
    # 0. 质量检测 (在清洗前进行)
    # quality_report = processor.check_quality(config)

    # 1. 预处理
    df_cleaned = processor.preprocess_pipeline(config)
    
    # 2. 匹配
    df_matched, msg = processor.map_match(df_cleaned, config.get('match_algo', 'HMM'), config)
    
    # 3. 简单统计
    # simple_report = processor.quality_check(df_cleaned)
    
    # 合并报告
    # final_report = {**simple_report, **quality_report}

    return {
        "trajectory_processed": df_cleaned.to_dict(orient='records'),
        "trajectory_matched": df_matched.to_dict(orient='records'),
        "message": msg
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)