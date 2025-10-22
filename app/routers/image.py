"""
FastAPI Routes cho xử lý ảnh với NAFNet
"""

import os
import uuid
from pathlib import Path
from typing import Optional
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

from app.services.model_loader import get_model_handler
from ..services.image_processor import get_image_processor

# Cấu hình logging - chỉ ghi vào file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/api/v1", tags=["image"])

# Templates
templates = Jinja2Templates(directory="app/templates")

# Đường dẫn thư mục
UPLOAD_DIR = Path("app/static/uploads")
RESULT_DIR = Path("app/static/results")

# Tạo thư mục nếu chưa có
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Khởi tạo services
model_handler = None
image_processor = None

def initialize_services():
    """Khởi tạo các services"""
    global model_handler, image_processor
    
    if model_handler is None:
        # Tìm model file
        model_path = None
        for ext in ['.pth', '.pt', '.pkl']:
            model_files = list(Path("models").glob(f"*{ext}"))
            if model_files:
                model_path = model_files[0]
                break
        
        model_handler = get_model_handler(model_path)
        logger.info(f"Model handler đã được khởi tạo: {model_handler.get_model_info()}")
    
    if image_processor is None:
        image_processor = get_image_processor(max_size=1024)
        logger.info("Image processor đã được khởi tạo")

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Trang chủ với form upload ảnh"""
    logger.info("🏠 Truy cập trang chủ")
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/upload")
async def upload_and_process_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None)
):
    """
    Upload và xử lý ảnh với NAFNet
    
    Args:
        file: File ảnh upload
        model_name: Tên model (optional)
    
    Returns:
        JSON response với thông tin xử lý
    """
    try:
        logger.info(f"📤 Nhận request upload file: {file.filename}")
        logger.info(f"📏 Kích thước file: {file.size if hasattr(file, 'size') else 'unknown'} bytes")
        
        # Khởi tạo services nếu chưa có
        logger.info("🔧 Đang khởi tạo services...")
        initialize_services()
        
        # Kiểm tra file
        if not file.filename:
            logger.error("❌ Không có file được chọn")
            raise HTTPException(status_code=400, detail="Không có file được chọn")
        
        # Kiểm tra extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        file_ext = Path(file.filename).suffix
        if file_ext not in allowed_extensions:
            logger.error(f"❌ Định dạng file không được hỗ trợ: {file_ext}")
            raise HTTPException(
                status_code=400, 
                detail=f"Định dạng file không được hỗ trợ. Chỉ chấp nhận: {', '.join(allowed_extensions)}"
            )
        
        logger.info(f"✅ File hợp lệ: {file.filename} ({file_ext})")
        
        # Tạo tên file unique
        file_id = str(uuid.uuid4())
        input_filename = f"{file_id}{file_ext}"
        output_filename = f"{file_id}_processed{file_ext}"
        
        input_path = UPLOAD_DIR / input_filename
        output_path = RESULT_DIR / output_filename
        
        logger.info(f"🆔 File ID: {file_id}")
        logger.info(f"📁 Input path: {input_path}")
        logger.info(f"📁 Output path: {output_path}")
        
        # Lưu file upload
        logger.info("💾 Đang lưu file upload...")
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"✅ File đã được lưu: {input_path} ({file_size_mb:.2f} MB)")
        
        # Kiểm tra tính hợp lệ của ảnh
        logger.info("🔍 Đang kiểm tra tính hợp lệ của ảnh...")
        if not image_processor.validate_image(input_path):
            # Xóa file nếu không hợp lệ
            input_path.unlink(missing_ok=True)
            logger.error("❌ File ảnh không hợp lệ, đã xóa file")
            raise HTTPException(status_code=400, detail="File ảnh không hợp lệ")
        
        logger.info("✅ Ảnh hợp lệ, bắt đầu xử lý...")
        
        # Xử lý ảnh
        logger.info(f"🔄 Bắt đầu xử lý ảnh: {input_filename}")
        result = image_processor.process_image_file(
            input_path=input_path,
            output_path=output_path,
            model_handler=model_handler
        )
        
        if not result["success"]:
            # Xóa file nếu xử lý thất bại
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            logger.error(f"❌ Xử lý ảnh thất bại: {result.get('error', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {result.get('error', 'Unknown error')}")
        
        # Log thông tin xử lý
        logger.info(f"✅ Xử lý ảnh hoàn thành: {file_id}")
        logger.info(f"⏱️  Thời gian xử lý: {result['total_time']:.3f}s")
        logger.info(f"📐 Kích thước gốc: {result['original_size']}")
        logger.info(f"📐 Kích thước sau xử lý: {result['processed_size']}")
        logger.info(f"🤖 Model: {result['model_info']['model_name']}")
        
        # Trả về kết quả
        response_data = {
            "success": True,
            "file_id": file_id,
            "original_filename": file.filename,
            "input_url": f"/static/uploads/{input_filename}",
            "output_url": f"/static/results/{output_filename}",
            "processing_info": {
                "original_size": result["original_size"],
                "processed_size": result["processed_size"],
                "preprocessing_time": result["preprocessing_time"],
                "model_processing_time": result["model_processing_time"],
                "total_time": result["total_time"]
            },
            "model_info": result["model_info"]
        }
        
        logger.info(f"🎉 Trả về kết quả thành công cho file: {file_id}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")

@router.get("/result/{file_id}")
async def get_result_image(file_id: str):
    """
    Lấy ảnh kết quả theo file_id
    
    Args:
        file_id: ID của file đã xử lý
    
    Returns:
        File ảnh kết quả
    """
    try:
        logger.info(f"📥 Request lấy ảnh kết quả: {file_id}")
        
        # Tìm file kết quả
        result_files = list(RESULT_DIR.glob(f"{file_id}_processed.*"))
        
        if not result_files:
            logger.warning(f"⚠️  Không tìm thấy ảnh kết quả cho file_id: {file_id}")
            raise HTTPException(status_code=404, detail="Không tìm thấy ảnh kết quả")
        
        result_file = result_files[0]
        
        if not result_file.exists():
            logger.warning(f"⚠️  File kết quả không tồn tại: {result_file}")
            raise HTTPException(status_code=404, detail="File kết quả không tồn tại")
        
        file_size_mb = result_file.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Trả về ảnh kết quả: {result_file.name} ({file_size_mb:.2f} MB)")
        
        return FileResponse(
            path=str(result_file),
            media_type="image/jpeg",
            filename=f"processed_{file_id}.jpg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy ảnh kết quả: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")

@router.get("/model/info")
async def get_model_info():
    """
    Lấy thông tin model hiện tại
    
    Returns:
        Thông tin model
    """
    try:
        logger.info("📥 Request lấy thông tin model")
        initialize_services()
        model_info = model_handler.get_model_info()
        logger.info(f"✅ Trả về thông tin model: {model_info['model_name']}")
        return {
            "success": True,
            "model_info": model_info
        }
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy thông tin model: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")

@router.post("/model/reload")
async def reload_model(model_path: Optional[str] = Form(None)):
    """
    Tải lại model
    
    Args:
        model_path: Đường dẫn đến model mới (optional)
    
    Returns:
        Thông tin model mới
    """
    try:
        global model_handler
        
        if model_path:
            # Tải model từ đường dẫn mới
            from ..services.model_loader import load_model
            model_handler = load_model(model_path)
        else:
            # Tải lại model hiện tại
            initialize_services()
        
        logger.info("Model đã được tải lại")
        return {
            "success": True,
            "message": "Model đã được tải lại thành công",
            "model_info": model_handler.get_model_info()
        }
        
    except Exception as e:
        logger.error(f"Lỗi khi tải lại model: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")

@router.delete("/cleanup")
async def cleanup_files():
    """
    Dọn dẹp các file tạm thời
    
    Returns:
        Thông tin dọn dẹp
    """
    try:
        logger.info("🧹 Bắt đầu dọn dẹp files...")
        
        # Đếm file trong thư mục uploads và results
        upload_count = len(list(UPLOAD_DIR.glob("*")))
        result_count = len(list(RESULT_DIR.glob("*")))
        
        logger.info(f"📊 Tìm thấy {upload_count} file uploads và {result_count} file results")
        
        # Xóa tất cả file
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                logger.info(f"🗑️  Đã xóa upload file: {file_path.name}")
        
        for file_path in RESULT_DIR.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                logger.info(f"🗑️  Đã xóa result file: {file_path.name}")
        
        total_deleted = upload_count + result_count
        logger.info(f"✅ Dọn dẹp hoàn thành: {total_deleted} files đã được xóa")
        
        return {
            "success": True,
            "message": "Dọn dẹp hoàn thành",
            "deleted_files": {
                "uploads": upload_count,
                "results": result_count
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi dọn dẹp: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Kiểm tra trạng thái hệ thống
    
    Returns:
        Trạng thái hệ thống
    """
    try:
        logger.info("🏥 Health check request")
        initialize_services()
        
        # Kiểm tra thư mục
        upload_exists = UPLOAD_DIR.exists()
        result_exists = RESULT_DIR.exists()
        
        # Đếm files
        upload_count = len(list(UPLOAD_DIR.glob("*"))) if upload_exists else 0
        result_count = len(list(RESULT_DIR.glob("*"))) if result_exists else 0
        
        health_status = {
            "status": "healthy",
            "services": {
                "model_handler": model_handler is not None,
                "image_processor": image_processor is not None
            },
            "directories": {
                "uploads_exists": upload_exists,
                "results_exists": result_exists
            },
            "file_counts": {
                "uploads": upload_count,
                "results": result_count
            }
        }
        
        logger.info(f"✅ Health check passed: {health_status}")
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }
