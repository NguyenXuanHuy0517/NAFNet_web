"""
Image Processor cho NAFNet
Xử lý ảnh đầu vào và đầu ra cho model NAFNet
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Tuple, Union, Optional
import time

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Class xử lý ảnh cho NAFNet model
    """
    
    def __init__(self, max_size: int = 1024, device: str = "auto"):
        """
        Khởi tạo ImageProcessor
        
        Args:
            max_size: Kích thước tối đa của ảnh (pixels)
            device: Device để xử lý tensor
        """
        self.max_size = max_size
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
            transforms.ToPILImage()
        ])
    
    def preprocess_image(self, image_path: Union[str, Path]) -> Tuple[torch.Tensor, dict]:
        """
        Tiền xử lý ảnh đầu vào
        
        Args:
            image_path: Đường dẫn đến ảnh
            
        Returns:
            Tuple (tensor, metadata)
        """
        start_time = time.time()
        
        try:
            # Đọc ảnh
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # Resize nếu ảnh quá lớn
            if max(original_size) > self.max_size:
                ratio = self.max_size / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
                logger.info(f"Resize ảnh từ {original_size} xuống {new_size}")
            
            # Chuyển đổi sang tensor
            image_tensor = self.transform(image).unsqueeze(0)  # Thêm batch dimension
            
            # Metadata
            metadata = {
                "original_size": original_size,
                "processed_size": image.size,
                "tensor_shape": image_tensor.shape,
                "preprocessing_time": time.time() - start_time
            }
            
            logger.info(f"Tiền xử lý ảnh hoàn thành: {metadata['preprocessing_time']:.3f}s")
            return image_tensor, metadata
            
        except Exception as e:
            logger.error(f"Lỗi khi tiền xử lý ảnh: {e}")
            raise
    
    def postprocess_image(self, tensor: torch.Tensor, metadata: dict) -> Image.Image:
        """
        Hậu xử lý tensor thành ảnh
        
        Args:
            tensor: Tensor ảnh đã xử lý
            metadata: Metadata từ preprocessing
            
        Returns:
            PIL Image
        """
        start_time = time.time()
        
        try:
            # Chuyển tensor về CPU nếu cần
            if tensor.is_cuda:
                tensor = tensor.cpu()
            
            # Chuyển đổi tensor thành PIL Image
            tensor = tensor.squeeze(0)  # Bỏ batch dimension
            image = self.inverse_transform(tensor)
            
            # Resize về kích thước gốc nếu cần
            if metadata["processed_size"] != metadata["original_size"]:
                image = image.resize(metadata["original_size"], Image.LANCZOS)
                logger.info(f"Resize ảnh kết quả về kích thước gốc: {metadata['original_size']}")
            
            processing_time = time.time() - start_time
            logger.info(f"Hậu xử lý ảnh hoàn thành: {processing_time:.3f}s")
            
            return image
            
        except Exception as e:
            logger.error(f"Lỗi khi hậu xử lý ảnh: {e}")
            raise
    
    def save_image(self, image: Image.Image, output_path: Union[str, Path], 
                   quality: int = 95) -> bool:
        """
        Lưu ảnh ra file
        
        Args:
            image: PIL Image
            output_path: Đường dẫn lưu file
            quality: Chất lượng ảnh (1-100)
            
        Returns:
            True nếu thành công
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Lưu ảnh
            image.save(output_path, quality=quality, optimize=True)
            
            logger.info(f"Ảnh đã được lưu: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu ảnh: {e}")
            return False
    
    def process_image_file(self, input_path: Union[str, Path], 
                          output_path: Union[str, Path],
                          model_handler) -> dict:
        """
        Xử lý ảnh hoàn chỉnh từ file đầu vào đến file đầu ra
        
        Args:
            input_path: Đường dẫn ảnh đầu vào
            output_path: Đường dẫn ảnh đầu ra
            model_handler: ModelHandler instance
            
        Returns:
            Dict chứa thông tin xử lý
        """
        total_start_time = time.time()
        
        try:
            # Tiền xử lý
            image_tensor, metadata = self.preprocess_image(input_path)
            
            # Xử lý với model
            model_start_time = time.time()
            processed_tensor = model_handler.process(image_tensor)
            model_time = time.time() - model_start_time
            
            # Hậu xử lý
            processed_image = self.postprocess_image(processed_tensor, metadata)
            
            # Lưu ảnh
            save_success = self.save_image(processed_image, output_path)
            
            total_time = time.time() - total_start_time
            
            result = {
                "success": save_success,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "original_size": metadata["original_size"],
                "processed_size": metadata["processed_size"],
                "preprocessing_time": metadata["preprocessing_time"],
                "model_processing_time": model_time,
                "total_time": total_time,
                "model_info": model_handler.get_model_info()
            }
            
            logger.info(f"Xử lý ảnh hoàn thành: {total_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý ảnh: {e}")
            return {
                "success": False,
                "error": str(e),
                "input_path": str(input_path),
                "output_path": str(output_path)
            }
    
    def validate_image(self, image_path: Union[str, Path]) -> bool:
        """
        Kiểm tra tính hợp lệ của ảnh
        
        Args:
            image_path: Đường dẫn ảnh
            
        Returns:
            True nếu ảnh hợp lệ
        """
        try:
            with Image.open(image_path) as img:
                # Kiểm tra format
                if img.format not in ['JPEG', 'PNG', 'JPG']:
                    logger.warning(f"Format ảnh không được hỗ trợ: {img.format}")
                    return False
                
                # Kiểm tra kích thước
                if img.size[0] < 32 or img.size[1] < 32:
                    logger.warning(f"Ảnh quá nhỏ: {img.size}")
                    return False
                
                # Kiểm tra mode
                if img.mode != 'RGB':
                    logger.warning(f"Mode ảnh không phải RGB: {img.mode}")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra ảnh: {e}")
            return False

# Global processor instance
_image_processor = None

def get_image_processor(max_size: int = 1024) -> ImageProcessor:
    """
    Singleton pattern để lấy image processor
    
    Args:
        max_size: Kích thước tối đa của ảnh
        
    Returns:
        ImageProcessor instance
    """
    global _image_processor
    
    if _image_processor is None:
        _image_processor = ImageProcessor(max_size=max_size)
    
    return _image_processor
