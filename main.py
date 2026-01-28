from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import io
import math
from road_network import road_network_service
from algorithms import TrajectoryProcessor

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> 系统启动...")
    # 启动时加载路网
    success, msg = road_network_service.load_local_file()
    print(f">>> 路网加载状态: {msg}")
    yield

app = FastAPI(lifespan=lifespan)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/road_network/status")
async def get_road_status():
    # 🔥 修复点：移除了对 .graph 的引用
    count = 0
    if road_network_service.is_loaded and road_network_service.gdf is not None:
        count = len(road_network_service.gdf)
        
    return {
        "loaded": road_network_service.is_loaded,
        "nodes": count
    }

@app.get("/road_network/nearby")
async def get_nearby_roads(min_lat: float, min_lon: float, max_lat: float, max_lon: float):
    """
    获取可视区域内的路网供前端绘制
    """
    if not road_network_service.is_loaded:
        return {"status": "error", "data": []}
    
    # 调用 road_network 中的空间查询
    segments = road_network_service.query_roads_in_bounds(min_lat, min_lon, max_lat, max_lon)
    return {"status": "success", "data": segments}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        # 1. 尝试读取 CSV
        # 优先尝试读取无表头格式 (因为您的文件看起来没有标准英文表头)
        try:
            # 假设前5列是: road_name, status, distance, duration, polyline
            df = pd.read_csv(io.BytesIO(content), header=None, 
                             names=['road', 'status', 'distance', 'duration', 'polyline'])
        except:
            # 如果失败，尝试自动推断
            df = pd.read_csv(io.BytesIO(content))
        
        df = df.fillna("")
        points = []

        # 2. 解析逻辑 (针对 "lon-lat|lon-lat" 格式)
        # 您的数据格式示例: "116.573884-39.78614|116.574103-39.786246"
        
        # 检查是否包含关键列
        if 'polyline' in df.columns:
            for idx, row in df.iterrows():
                polyline_str = str(row['polyline'])
                if not polyline_str or polyline_str.lower() == 'nan':
                    continue

                # 仅取第一个点，保持原逻辑
                first_point_str = polyline_str.split('|')[0] if '|' in polyline_str else polyline_str

                # 解析 "lon-lat" (注意您的数据是用减号分隔经纬度的)
                if '-' in first_point_str:
                    try:
                        parts = first_point_str.split('-')
                        if len(parts) >= 2:
                            lon = float(parts[0])
                            lat = float(parts[1])

                            # 简单的有效性检查
                            if not (0 <= lon <= 180 and 0 <= lat <= 90):
                                continue

                            points.append({
                                'id': idx,
                                'lat': lat,
                                'lon': lon,
                                # 伪造一个时间戳，保证顺序 (因为HMM需要)
                                # 假设数据是按时间顺序记录的，每行间隔 5 秒
                                'timestamp': pd.Timestamp('2024-01-01 08:00:00') + pd.Timedelta(seconds=idx * 5),
                                'road': str(row.get('road', '')),
                                'status': str(row.get('status', ''))
                            })
                    except ValueError:
                        continue
                        
        # 3. 兼容标准 GPS 格式 (如果有 lat, lon 列)
        elif 'lat' in df.columns and 'lon' in df.columns:
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

        # 4. 返回结果
        print(f">>> 解析完成，提取了 {len(points)} 个点")
        return {
            "status": "success", 
            "count": len(points), 
            "data": points # 前端会收到这个数组
        }
        
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/process")
async def process(data: dict):
    # 接收前端数据
    raw_df = pd.DataFrame(data['trajectory'])
    config = data['config']
    
    processor = TrajectoryProcessor(raw_df)
    
    # 1. 预处理
    df_cleaned = processor.preprocess_pipeline(config)
    
    # 2. 匹配
    df_matched, msg = processor.map_match(df_cleaned, config.get('match_algo', 'HMM'), config)
    
    # 3. 质检
    report = processor.quality_check(df_cleaned)
    
    return {
        "trajectory_processed": df_cleaned.to_dict(orient='records'),
        "trajectory_matched": df_matched.to_dict(orient='records'),
        "quality_report": report,
        "message": msg
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)