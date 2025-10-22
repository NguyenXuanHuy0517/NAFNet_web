"""
NAFNet Web Application
FastAPI application cho xử lý ảnh với mô hình NAFNet
"""

import os
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.routers import image

# Cấu hình logging - chỉ ghi vào file, không hiển thị terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        # Bỏ StreamHandler để không hiển thị trên terminal
    ]
)
logger = logging.getLogger(__name__)

# Tạo logger riêng cho từng module
app_logger = logging.getLogger("app.main")
image_logger = logging.getLogger("app.image")
model_logger = logging.getLogger("app.model")
service_logger = logging.getLogger("app.service")


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager cho FastAPI app
    Xử lý khởi tạo và cleanup khi app start/stop
    """
    # Startup
    app_logger.info("🚀 Khởi động NAFNet Web Application...")
    app_logger.info("📋 Phiên bản: 1.0.0")
    app_logger.info("🔧 Môi trường: Development")

    # Tạo các thư mục cần thiết
    directories = [
        "app/static/uploads",
        "app/static/results", 
        "models",
        "models/custom"
    ]

    app_logger.info("📁 Đang tạo các thư mục cần thiết...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        app_logger.info(f"   ✓ Đã tạo thư mục: {directory}")

    # Kiểm tra model files
    app_logger.info("🔍 Đang kiểm tra model files...")
    model_files = list(Path("models").glob("*.pth")) + list(Path("models").glob("*.pt"))
    if model_files:
        app_logger.info(f"🤖 Tìm thấy {len(model_files)} model file(s):")
        for model_file in model_files:
            file_size = model_file.stat().st_size / (1024 * 1024)  # MB
            app_logger.info(f"   • {model_file.name} ({file_size:.2f} MB)")
    else:
        app_logger.warning("⚠️  Không tìm thấy model file. Sẽ sử dụng model mặc định.")

    # Kiểm tra thư mục static
    upload_files = list(Path("app/static/uploads").glob("*"))
    result_files = list(Path("app/static/results").glob("*"))
    app_logger.info(f"📊 Thống kê files: {len(upload_files)} uploads, {len(result_files)} results")

    app_logger.info("✅ Ứng dụng đã sẵn sàng!")
    app_logger.info("🌐 Server sẽ chạy tại: http://localhost:8000")
    app_logger.info("📚 API Documentation: http://localhost:8000/docs")

    yield

    # Shutdown
    app_logger.info("🛑 Đang tắt NAFNet Web Application...")
    app_logger.info("🧹 Đang dọn dẹp resources...")
    
    # Thống kê cuối cùng
    upload_files = list(Path("app/static/uploads").glob("*"))
    result_files = list(Path("app/static/results").glob("*"))
    app_logger.info(f"📊 Files còn lại: {len(upload_files)} uploads, {len(result_files)} results")
    
    app_logger.info("✅ Ứng dụng đã được tắt thành công.")


# Tạo FastAPI app
app = FastAPI(
    title="NAFNet Image Restoration API",
    description="API để xử lý và khôi phục ảnh sử dụng mô hình NAFNet",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên hạn chế origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(image.router)


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Trang chủ - redirect đến trang upload"""
    return image.templates.TemplateResponse("index.html", {"request": request})


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NAFNet Web Application",
        "version": "1.0.0"
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "Endpoint không tồn tại",
            "path": str(request.url.path)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Có lỗi xảy ra trên server",
            "path": str(request.url.path)
        }
    )


# Middleware để log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware để log tất cả requests"""
    start_time = time.time()
    
    # Lấy thông tin client
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Log request
    app_logger.info(f"📥 {request.method} {request.url.path}")
    app_logger.info(f"   👤 Client: {client_ip}")
    app_logger.info(f"   🌐 User-Agent: {user_agent[:50]}...")

    # Process request
    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    status_emoji = "✅" if response.status_code < 400 else "❌" if response.status_code < 500 else "🔥"
    app_logger.info(f"📤 {status_emoji} {response.status_code} - {process_time:.3f}s")

    return response


if __name__ == "__main__":
    import uvicorn

    # Cấu hình server
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"

    logger.info(f"🌐 Khởi động server tại http://{host}:{port}")
    logger.info(f"📚 API docs tại http://{host}:{port}/docs")
    logger.info(f"🔄 Auto-reload: {reload}")

    # Chạy server
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
