from fastapi import APIRouter, UploadFile, File
import pandas as pd
from app.services.road_network import road_network_service
from app.services.preprocess import preprocess_pipeline
from app.services.matching import map_match
from app.services.other import parse_upload_points, parse_timestamps


router = APIRouter()


@router.get("/road_network/status")
async def get_road_status():
    count = 0
    if road_network_service.is_loaded and road_network_service.gdf is not None:
        count = len(road_network_service.gdf)
    return {"loaded": road_network_service.is_loaded, "nodes": count}


@router.get("/road_network/nearby")
async def get_nearby_roads(min_lat: float, min_lon: float, max_lat: float, max_lon: float):
    if not road_network_service.is_loaded:
        return {"status": "error", "data": []}
    segments = road_network_service.query_roads_in_bounds(min_lat, min_lon, max_lat, max_lon)
    return {"status": "success", "data": segments}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        points, qc_summary, qc_details = parse_upload_points(content)
        return {
            "status": "success",
            "count": len(points),
            "data": points,
            "qc_summary": qc_summary,
            "qc_details": qc_details,
        }
    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/process")
async def process(data: dict):
    raw_df = pd.DataFrame(data['trajectory'])
    config = data.get('config', {})

    if 'timestamp' in raw_df.columns:
        raw_df['timestamp'] = parse_timestamps(raw_df['timestamp'])

    df_cleaned = preprocess_pipeline(raw_df, config)
    df_matched, msg = map_match(df_cleaned, config.get('match_algo', 'HMM'), config)

    return {
        "trajectory_processed": df_cleaned.to_dict(orient='records'),
        "trajectory_matched": df_matched.to_dict(orient='records'),
        "message": msg,
    }
