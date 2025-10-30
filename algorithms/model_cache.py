"""
DVP涂层光谱异常检测系统 - 模型缓存管理器
提供高效的模型加载、缓存和管理功能

Author: MiniMax Agent
Date: 2025-10-30
"""

import os
import pickle
import joblib
import json
import time
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import logging
from threading import Lock
from dataclasses import dataclass
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """模型信息数据类"""
    name: str
    version: str
    model_type: str
    file_path: str
    load_time: float
    last_accessed: datetime
    access_count: int = 0
    metadata: Dict[str, Any] = None

class ModelCacheManager:
    """模型缓存管理器"""
    
    def __init__(self, cache_dir: str = "models/cache", max_cache_size: int = 10):
        """
        初始化模型缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
            max_cache_size: 最大缓存模型数量
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size
        self.cache_lock = Lock()
        self.loaded_models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.metadata_cache: Dict[str, Dict] = {}
        
        logger.info(f"模型缓存管理器初始化完成，缓存目录: {self.cache_dir}")
    
    def load_model(self, 
                   model_name: str, 
                   model_path: str, 
                   model_type: str = "autoencoder",
                   force_reload: bool = False) -> Tuple[Any, Dict]:
        """
        加载模型（带缓存）
        
        Args:
            model_name: 模型名称
            model_path: 模型文件路径
            model_type: 模型类型
            force_reload: 是否强制重新加载
            
        Returns:
            Tuple[Any, Dict]: (模型对象, 元数据)
        """
        with self.cache_lock:
            # 检查是否已缓存且未强制重新加载
            if not force_reload and model_name in self.loaded_models:
                self._update_access_info(model_name)
                logger.info(f"从缓存加载模型: {model_name}")
                return self.loaded_models[model_name], self.metadata_cache[model_name]
            
            # 检查缓存大小，必要时清理
            if len(self.loaded_models) >= self.max_cache_size:
                self._evict_oldest_model()
            
            try:
                start_time = time.time()
                
                # 根据模型类型加载
                if model_type == "autoencoder":
                    model = self._load_autoencoder_model(model_path)
                elif model_type == "scaler":
                    model = self._load_scaler_model(model_path)
                else:
                    raise ValueError(f"不支持的模型类型: {model_type}")
                
                load_time = time.time() - start_time
                
                # 加载元数据
                metadata = self._load_metadata(model_path)
                
                # 缓存模型和元数据
                self.loaded_models[model_name] = model
                self.metadata_cache[model_name] = metadata
                
                # 记录模型信息
                self.model_info[model_name] = ModelInfo(
                    name=model_name,
                    version=metadata.get("version", "unknown"),
                    model_type=model_type,
                    file_path=model_path,
                    load_time=load_time,
                    last_accessed=datetime.now(),
                    access_count=1,
                    metadata=metadata
                )
                
                logger.info(f"模型加载完成: {model_name}, 耗时: {load_time:.3f}s")
                return model, metadata
                
            except Exception as e:
                logger.error(f"加载模型失败 {model_name}: {str(e)}")
                raise
    
    def _load_autoencoder_model(self, model_path: str) -> Dict[str, Any]:
        """
        加载自编码器模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            Dict: 包含encoder, decoder, scaler的字典
        """
        model_dir = Path(model_path)
        
        # 加载各个组件
        encoder_path = model_dir / "encoder.joblib"
        decoder_path = model_dir / "decoder.joblib"
        scaler_path = model_dir / "scaler.joblib"
        
        if not all(p.exists() for p in [encoder_path, decoder_path, scaler_path]):
            raise FileNotFoundError(f"模型组件文件不存在: {model_path}")
        
        encoder = joblib.load(encoder_path)
        decoder = joblib.load(decoder_path)
        scaler = joblib.load(scaler_path)
        
        return {
            "encoder": encoder,
            "decoder": decoder,
            "scaler": scaler
        }
    
    def _load_scaler_model(self, model_path: str):
        """
        加载标准化器模型
        
        Args:
            model_path: 标准化器路径
            
        Returns:
            标准化器对象
        """
        return joblib.load(model_path)
    
    def _load_metadata(self, model_path: str) -> Dict:
        """
        加载模型元数据
        
        Args:
            model_path: 模型路径
            
        Returns:
            Dict: 元数据字典
        """
        metadata_path = Path(model_path) / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"元数据文件不存在: {metadata_path}")
            return {}
    
    def _update_access_info(self, model_name: str):
        """更新模型访问信息"""
        if model_name in self.model_info:
            self.model_info[model_name].last_accessed = datetime.now()
            self.model_info[model_name].access_count += 1
    
    def _evict_oldest_model(self):
        """清理最久未使用的模型"""
        if not self.model_info:
            return
        
        # 找到最久未访问的模型
        oldest_model = min(self.model_info.keys(), 
                          key=lambda x: self.model_info[x].last_accessed)
        
        # 从缓存中移除
        del self.loaded_models[oldest_model]
        del self.metadata_cache[oldest_model]
        removed_info = self.model_info.pop(oldest_model)
        
        logger.info(f"从缓存中移除模型: {oldest_model}")
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """
        获取已缓存的模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型对象或None
        """
        with self.cache_lock:
            if model_name in self.loaded_models:
                self._update_access_info(model_name)
                return self.loaded_models[model_name]
            return None
    
    def get_metadata(self, model_name: str) -> Optional[Dict]:
        """
        获取模型元数据
        
        Args:
            model_name: 模型名称
            
        Returns:
            元数据字典或None
        """
        return self.metadata_cache.get(model_name)
    
    def unload_model(self, model_name: str) -> bool:
        """
        从缓存中卸载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否成功卸载
        """
        with self.cache_lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                del self.metadata_cache[model_name]
                if model_name in self.model_info:
                    del self.model_info[model_name]
                logger.info(f"模型已从缓存卸载: {model_name}")
                return True
            return False
    
    def clear_cache(self):
        """清空所有缓存"""
        with self.cache_lock:
            self.loaded_models.clear()
            self.metadata_cache.clear()
            self.model_info.clear()
            logger.info("模型缓存已清空")
    
    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息
        
        Returns:
            Dict: 统计信息
        """
        with self.cache_lock:
            total_models = len(self.loaded_models)
            total_accesses = sum(info.access_count for info in self.model_info.values())
            
            if self.model_info:
                avg_load_time = sum(info.load_time for info in self.model_info.values()) / len(self.model_info)
                most_accessed = max(self.model_info.keys(), 
                                  key=lambda x: self.model_info[x].access_count)
            else:
                avg_load_time = 0
                most_accessed = "N/A"
            
            return {
                "cached_models": total_models,
                "max_cache_size": self.max_cache_size,
                "total_accesses": total_accesses,
                "average_load_time": avg_load_time,
                "most_accessed_model": most_accessed,
                "cache_utilization": total_models / self.max_cache_size if self.max_cache_size > 0 else 0
            }
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        获取模型详细信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            ModelInfo: 模型信息对象
        """
        return self.model_info.get(model_name)
    
    def list_cached_models(self) -> Dict[str, Dict]:
        """
        列出所有缓存的模型信息
        
        Returns:
            Dict: 模型信息字典
        """
        with self.cache_lock:
            return {
                name: {
                    "name": info.name,
                    "version": info.version,
                    "model_type": info.model_type,
                    "access_count": info.access_count,
                    "last_accessed": info.last_accessed.isoformat(),
                    "load_time": info.load_time
                }
                for name, info in self.model_info.items()
            }
    
    def preload_models(self, model_configs: list):
        """
        预加载模型
        
        Args:
            model_configs: 模型配置列表，每个配置包含name, path, type
        """
        logger.info(f"开始预加载 {len(model_configs)} 个模型")
        
        for config in model_configs:
            try:
                self.load_model(
                    model_name=config["name"],
                    model_path=config["path"],
                    model_type=config.get("type", "autoencoder")
                )
            except Exception as e:
                logger.error(f"预加载模型失败 {config['name']}: {str(e)}")
        
        logger.info("模型预加载完成")
    
    def save_cache_state(self, filepath: str):
        """
        保存缓存状态到文件
        
        Args:
            filepath: 保存路径
        """
        cache_state = {
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                name: {
                    "name": info.name,
                    "version": info.version,
                    "model_type": info.model_type,
                    "file_path": info.file_path,
                    "access_count": info.access_count,
                    "last_accessed": info.last_accessed.isoformat()
                }
                for name, info in self.model_info.items()
            },
            "stats": self.get_cache_stats()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_state, f, ensure_ascii=False, indent=2)
        
        logger.info(f"缓存状态已保存到: {filepath}")
    
    def warm_up_cache(self, model_dir: str, coating_name: str = "DVP"):
        """
        预热缓存 - 加载指定涂层类型的模型
        
        Args:
            model_dir: 模型目录
            coating_name: 涂层名称
        """
        model_path = Path(model_dir) / coating_name
        
        if not model_path.exists():
            logger.warning(f"模型目录不存在: {model_path}")
            return
        
        # 查找所有版本的模型
        version_dirs = [d for d in model_path.iterdir() if d.is_dir()]
        
        for version_dir in version_dirs:
            try:
                # 尝试加载该版本的模型
                self.load_model(
                    model_name=f"{coating_name}_{version_dir.name}",
                    model_path=str(version_dir),
                    model_type="autoencoder"
                )
            except Exception as e:
                logger.warning(f"预热模型失败 {version_dir}: {str(e)}")
        
        logger.info(f"缓存预热完成，加载了 {len(version_dirs)} 个模型版本")

# 使用示例
if __name__ == "__main__":
    # 创建缓存管理器
    cache_manager = ModelCacheManager(cache_dir="models/cache")
    
    # 预热缓存
    cache_manager.warm_up_cache("models", "DVP")
    
    # 获取缓存统计
    stats = cache_manager.get_cache_stats()
    print("=== 缓存统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 列出缓存的模型
    cached_models = cache_manager.list_cached_models()
    print("\n=== 缓存的模型 ===")
    for name, info in cached_models.items():
        print(f"模型: {name}")
        print(f"  版本: {info['version']}")
        print(f"  类型: {info['model_type']}")
        print(f"  访问次数: {info['access_count']}")
        print(f"  最后访问: {info['last_accessed']}")
        print("-" * 30)