"""
Model Loader cho NAFNet
Tải và quản lý mô hình NAFNet để xử lý ảnh
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional, Union
import os

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NAFNet(nn.Module):
    """
    Simplified NAFNet architecture cho image restoration
    Đây là phiên bản đơn giản hóa của NAFNet
    """
    def __init__(self, in_channels=3, out_channels=3, width=64, num_blks=16):
        super(NAFNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
        # Middle processing blocks
        self.middle_blocks = nn.ModuleList([
            self._make_block(width) for _ in range(num_blks)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(width, width, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, out_channels, 3, 1, 1)
        )
        
    def _make_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc = self.encoder(x)
        
        # Middle processing
        for block in self.middle_blocks:
            enc = enc + block(enc)  # Residual connection
        
        # Decoder
        out = self.decoder(enc)
        
        # Skip connection
        return x + out

class ModelHandler:
    """
    Handler class để quản lý mô hình NAFNet
    """
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None, device: str = "auto"):
        """
        Khởi tạo ModelHandler
        
        Args:
            model_path: Đường dẫn đến file model (.pth)
            device: Device để chạy model ("cpu", "cuda", hoặc "auto")
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.model_name = "NAFNet"
        
        # Tải model nếu có đường dẫn
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Tạo model mặc định nếu không có pretrained model
            self._create_default_model()
    
    def _get_device(self, device: str) -> torch.device:
        """Xác định device phù hợp"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _create_default_model(self):
        """Tạo model mặc định"""
        logger.info("Tạo model NAFNet mặc định...")
        self.model = NAFNet(in_channels=3, out_channels=3, width=64, num_blks=16)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model đã được tạo và chuyển đến {self.device}")
    
    def load_model(self, model_path: Union[str, Path]):
        """
        Tải model từ file checkpoint
        
        Args:
            model_path: Đường dẫn đến file model
        """
        try:
            model_path = Path(model_path)
            logger.info(f"Đang tải model từ {model_path}")
            
            # Tải checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Tạo model
            self.model = NAFNet(in_channels=3, out_channels=3, width=64, num_blks=16)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.model_path = model_path
            logger.info(f"Model đã được tải thành công từ {model_path}")
            
        except Exception as e:
            logger.error(f"Lỗi khi tải model: {e}")
            logger.info("Sử dụng model mặc định...")
            self._create_default_model()
    
    def process(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Xử lý ảnh với model
        
        Args:
            image_tensor: Tensor ảnh đầu vào (shape: [1, 3, H, W])
            
        Returns:
            Tensor ảnh đã được xử lý
        """
        if self.model is None:
            raise RuntimeError("Model chưa được tải!")
        
        try:
            with torch.no_grad():
                # Chuyển tensor đến device phù hợp
                image_tensor = image_tensor.to(self.device)
                
                # Xử lý ảnh
                output = self.model(image_tensor)
                
                # Clamp values về range [0, 1]
                output = torch.clamp(output, 0, 1)
                
                return output
                
        except Exception as e:
            logger.error(f"Lỗi khi xử lý ảnh: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Trả về thông tin model"""
        return {
            "model_name": self.model_name,
            "model_path": str(self.model_path) if self.model_path else "Default Model",
            "device": str(self.device),
            "is_loaded": self.model is not None
        }

# Global model instance
_model_handler = None

def get_model_handler(model_path: Optional[Union[str, Path]] = None) -> ModelHandler:
    """
    Singleton pattern để lấy model handler
    
    Args:
        model_path: Đường dẫn đến model (chỉ cần thiết lần đầu)
        
    Returns:
        ModelHandler instance
    """
    global _model_handler
    
    if _model_handler is None:
        _model_handler = ModelHandler(model_path)
    
    return _model_handler

def load_model(model_path: Union[str, Path]) -> ModelHandler:
    """
    Tải model mới
    
    Args:
        model_path: Đường dẫn đến file model
        
    Returns:
        ModelHandler instance mới
    """
    global _model_handler
    _model_handler = ModelHandler(model_path)
    return _model_handler
