#!/usr/bin/env python3
"""
Script khởi chạy NAFNet Web Application
"""

import os
import sys
import uvicorn
import logging
from pathlib import Path

# Thêm thư mục app vào Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

# Cấu hình logging - chỉ ghi vào file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Khởi chạy ứng dụng"""
    
    # Cấu hình từ environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info")
    
    logger.info("🚀 Khởi động NAFNet Web Application...")
    logger.info(f"🌐 Server: http://{host}:{port}")
    logger.info(f"📚 API Docs: http://{host}:{port}/docs")
    logger.info(f"🔄 Auto-reload: {reload}")
    logger.info(f"📝 Log level: {log_level}")
    logger.info("-" * 50)
    
    # Khởi chạy server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()
