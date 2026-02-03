from fastapi import APIRouter, UploadFile, File
import os
import pandas as pd
from app.services.road_network import road_network_service
from app.services.preprocess import preprocess_pipeline
from app.services.matching import map_match
from app.services.other import parse_upload_points, parse_timestamps
from app.services.quality import check_quality


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

    skip_preprocess = config.get('denoise_algo') is None and config.get('stop_cluster_algo') is None
    if skip_preprocess:
        print("⚠️ [预处理] 已跳过（denoise_algo/stop_cluster_algo 为 null）")
        df_cleaned = raw_df.copy()
    else:
        df_cleaned = preprocess_pipeline(raw_df, config)
    qc_summary = None
    qc_details = None
    try:
        df_qc = df_cleaned.copy()
        if 'timestamp' in df_qc.columns:
            df_qc['timestamp'] = parse_timestamps(df_qc['timestamp'])
        qc_summary, qc_details = check_quality(df_qc, config)
    except Exception as qc_e:
        print(f"⚠️ [预处理] 质量检测失败: {qc_e}")
    try:
        output_path = os.path.join(os.path.dirname(__file__), '../../output', 'preprocess_result.csv')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_cleaned.to_csv(output_path, index=False)
        print(f"✅ [预处理] 已输出结果: {output_path}")
    except Exception as output_e:
        print(f"⚠️ [预处理] 输出失败: {output_e}")
        
    match_algo = config.get('match_algo', 'HMM')
    if match_algo is None:
        print("⚠️ [匹配] 已跳过（match_algo 为 null）")
        df_matched, msg = df_cleaned.copy(), "⚠️ 已跳过匹配"
    else:
        df_matched, msg = map_match(df_cleaned, match_algo, config)

    return {
        "trajectory_processed": df_cleaned.to_dict(orient='records'),
        "trajectory_matched": df_matched.to_dict(orient='records'),
        "skipped_preprocess": skip_preprocess,
        "skipped_match": match_algo is None,
        "qc_pre_summary": qc_summary,
        "qc_pre_details": qc_details,
        "message": msg,
    }
