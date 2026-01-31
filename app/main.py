import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services.road_network import road_network_service
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> 系统启动...")
    lazy = str(os.environ.get("ROAD_NETWORK_LAZY", "1")).lower() in ("1", "true", "yes")
    success, msg = road_network_service.load_local_file(lazy=lazy)
    print(f">>> 路网加载状态: {msg} (lazy={lazy})")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
