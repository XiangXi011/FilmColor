"""
DVP涂层光谱异常检测系统 - FastAPI服务
提供实时光谱分析API接口

Author: MiniMax Agent
Date: 2025-10-30
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# 导入项目模块
from algorithms.decision_engine import DecisionEngine, DecisionThresholds, DecisionResult
from algorithms.model_cache import ModelCacheManager
from algorithms.similarity_evaluator import SimilarityEvaluator
from data.data_loader import load_dvp_standard_curve

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建日志目录
Path("logs").mkdir(exist_ok=True)

# FastAPI应用初始化
app = FastAPI(
    title="DVP涂层光谱异常检测API",
    description="基于Quality Score和Stability Score的实时光谱分析服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
decision_engine = None
model_cache = None
similarity_evaluator = None
dvp_standard_spectrum = None

# Pydantic模型定义
class SpectrumRequest(BaseModel):
    """光谱数据请求模型"""
    wavelengths: List[float] = Field(..., description="波长数据列表")
    spectrum: List[float] = Field(..., description="光谱强度数据列表")
    coating_name: str = Field(default="DVP", description="涂层类型名称")
    
    @validator('wavelengths', 'spectrum')
    def validate_arrays(cls, v):
        if len(v) == 0:
            raise ValueError("数据列表不能为空")
        return v
    
    @validator('wavelengths', 'spectrum')
    def validate_length_match(cls, v, values):
        if 'wavelengths' in values and 'spectrum' in values:
            if len(v) != len(values['wavelengths']):
                raise ValueError("波长和光谱数据长度必须一致")
        return v

class BatchSpectrumRequest(BaseModel):
    """批量光谱数据请求模型"""
    spectra: List[SpectrumRequest] = Field(..., description="光谱数据列表")
    coating_name: str = Field(default="DVP", description="涂层类型名称")
    
    @validator('spectra')
    def validate_spectra_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("光谱列表不能为空")
        return v

class AnalysisResponse(BaseModel):
    """分析结果响应模型"""
    quality_score: float = Field(..., description="质量评分")
    stability_score: float = Field(..., description="稳定性评分")
    decision: str = Field(..., description="决策结果")
    confidence: float = Field(..., description="决策置信度")
    reasoning: str = Field(..., description="决策推理")
    recommendations: List[str] = Field(..., description="建议列表")
    processing_time: float = Field(..., description="处理时间（秒）")
    timestamp: str = Field(..., description="时间戳")

class BatchAnalysisResponse(BaseModel):
    """批量分析结果响应模型"""
    results: List[AnalysisResponse] = Field(..., description="分析结果列表")
    total_processing_time: float = Field(..., description="总处理时间")
    average_processing_time: float = Field(..., description="平均处理时间")
    decision_summary: Dict[str, int] = Field(..., description="决策统计")
    timestamp: str = Field(..., description="时间戳")

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    timestamp: str = Field(..., description="检查时间")
    version: str = Field(..., description="API版本")
    components: Dict[str, str] = Field(..., description="组件状态")

class CacheStatsResponse(BaseModel):
    """缓存统计响应模型"""
    cached_models: int = Field(..., description="缓存模型数量")
    max_cache_size: int = Field(..., description="最大缓存大小")
    total_accesses: int = Field(..., description="总访问次数")
    average_load_time: float = Field(..., description="平均加载时间")
    cache_utilization: float = Field(..., description="缓存利用率")

# 依赖注入函数
async def get_decision_engine() -> DecisionEngine:
    """获取决策引擎依赖"""
    global decision_engine
    if decision_engine is None:
        raise HTTPException(status_code=503, detail="决策引擎未初始化")
    return decision_engine

async def get_model_cache() -> ModelCacheManager:
    """获取模型缓存依赖"""
    global model_cache
    if model_cache is None:
        raise HTTPException(status_code=503, detail="模型缓存未初始化")
    return model_cache

async def get_similarity_evaluator() -> SimilarityEvaluator:
    """获取相似度评估器依赖"""
    global similarity_evaluator
    if similarity_evaluator is None:
        raise HTTPException(status_code=503, detail="相似度评估器未初始化")
    return similarity_evaluator

# 初始化函数
def initialize_components():
    """初始化系统组件"""
    global decision_engine, model_cache, similarity_evaluator, dvp_standard_spectrum
    
    try:
        logger.info("开始初始化系统组件...")
        
        # 初始化决策引擎
        decision_engine = DecisionEngine()
        logger.info("决策引擎初始化完成")
        
        # 初始化模型缓存管理器
        model_cache = ModelCacheManager(cache_dir="models/cache")
        logger.info("模型缓存管理器初始化完成")
        
        # 预热缓存
        if Path("models/DVP").exists():
            model_cache.warm_up_cache("models", "DVP")
            logger.info("模型缓存预热完成")
        
        # 初始化相似度评估器
        similarity_evaluator = SimilarityEvaluator()
        logger.info("相似度评估器初始化完成")
        
        # 加载DVP标准光谱
        dvp_standard_spectrum = load_dvp_standard_curve()
        logger.info("DVP标准光谱加载完成")
        
        logger.info("所有组件初始化完成")
        
    except Exception as e:
        logger.error(f"组件初始化失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    try:
        initialize_components()
        logger.info("API服务器启动成功")
    except Exception as e:
        logger.error(f"API服务器启动失败: {str(e)}")
        raise

# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("API服务器正在关闭...")
    if model_cache:
        model_cache.save_cache_state("logs/cache_state.json")
    logger.info("API服务器已关闭")

# 错误处理装饰器
def error_handler(func):
    """错误处理装饰器"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"API调用失败 {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")
    return wrapper

# API端点
@app.get("/", response_model=Dict[str, str])
@error_handler
async def root():
    """根路径"""
    return {
        "service": "DVP涂层光谱异常检测API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
@error_handler
async def health_check():
    """健康检查"""
    components = {
        "decision_engine": "healthy" if decision_engine else "unavailable",
        "model_cache": "healthy" if model_cache else "unavailable", 
        "similarity_evaluator": "healthy" if similarity_evaluator else "unavailable",
        "dvp_standard_spectrum": "healthy" if dvp_standard_spectrum is not None else "unavailable"
    }
    
    all_healthy = all(status == "healthy" for status in components.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        components=components
    )

@app.post("/analyze", response_model=AnalysisResponse)
@error_handler
async def analyze_spectrum(
    request: SpectrumRequest,
    engine: DecisionEngine = Depends(get_decision_engine),
    cache: ModelCacheManager = Depends(get_model_cache),
    evaluator: SimilarityEvaluator = Depends(get_similarity_evaluator)
):
    """分析单个光谱"""
    start_time = datetime.now()
    
    try:
        # 转换输入数据
        wavelengths = np.array(request.wavelengths)
        spectrum = np.array(request.spectrum)
        
        # 加载模型
        model_name = f"{request.coating_name}_autoencoder"
        model_data, metadata = cache.load_model(
            model_name=model_name,
            model_path=f"models/{request.coating_name}",
            model_type="autoencoder"
        )
        
        # 计算Quality Score
        quality_result = evaluator.evaluate(
            y1=spectrum,
            y2=dvp_standard_spectrum[1],  # 标准光谱数据
            wavelengths=wavelengths,
            coating_name=request.coating_name
        )
        quality_score = quality_result['similarity_score']
        
        # 计算Stability Score
        # 标准化输入数据
        scaled_spectrum = model_data["scaler"].transform(spectrum.reshape(1, -1))
        
        # 重构光谱
        encoded = model_data["encoder"].predict(scaled_spectrum)
        decoded = model_data["decoder"].predict(encoded)
        reconstructed = model_data["scaler"].inverse_transform(decoded).flatten()
        
        # 计算重构误差
        reconstruction_error = np.mean((spectrum - reconstructed) ** 2)
        
        # 获取阈值
        threshold = metadata.get("threshold", 0.01)
        stability_score = max(0, 1 - (reconstruction_error / threshold))
        
        # 做出决策
        decision_result = engine.make_decision(
            quality_score=quality_score,
            stability_score=stability_score,
            coating_name=request.coating_name
        )
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResponse(
            quality_score=quality_score,
            stability_score=stability_score,
            decision=decision_result.decision.value,
            confidence=decision_result.confidence,
            reasoning=decision_result.reasoning,
            recommendations=decision_result.recommendations,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"光谱分析失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"光谱分析失败: {str(e)}")

@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
@error_handler
async def analyze_spectra_batch(
    request: BatchSpectrumRequest,
    engine: DecisionEngine = Depends(get_decision_engine),
    cache: ModelCacheManager = Depends(get_model_cache),
    evaluator: SimilarityEvaluator = Depends(get_similarity_evaluator)
):
    """批量分析光谱"""
    start_time = datetime.now()
    results = []
    
    try:
        for i, spectrum_request in enumerate(request.spectra):
            try:
                # 转换输入数据
                wavelengths = np.array(spectrum_request.wavelengths)
                spectrum = np.array(spectrum_request.spectrum)
                
                # 加载模型
                model_name = f"{spectrum_request.coating_name}_autoencoder"
                model_data, metadata = cache.load_model(
                    model_name=model_name,
                    model_path=f"models/{spectrum_request.coating_name}",
                    model_type="autoencoder"
                )
                
                # 计算Quality Score
                quality_result = evaluator.evaluate(
                    y1=spectrum,
                    y2=dvp_standard_spectrum[1],
                    wavelengths=wavelengths,
                    coating_name=spectrum_request.coating_name
                )
                quality_score = quality_result['similarity_score']
                
                # 计算Stability Score
                scaled_spectrum = model_data["scaler"].transform(spectrum.reshape(1, -1))
                encoded = model_data["encoder"].predict(scaled_spectrum)
                decoded = model_data["decoder"].predict(encoded)
                reconstructed = model_data["scaler"].inverse_transform(decoded).flatten()
                
                reconstruction_error = np.mean((spectrum - reconstructed) ** 2)
                threshold = metadata.get("threshold", 0.01)
                stability_score = max(0, 1 - (reconstruction_error / threshold))
                
                # 做出决策
                decision_result = engine.make_decision(
                    quality_score=quality_score,
                    stability_score=stability_score,
                    coating_name=spectrum_request.coating_name
                )
                
                results.append(AnalysisResponse(
                    quality_score=quality_score,
                    stability_score=stability_score,
                    decision=decision_result.decision.value,
                    confidence=decision_result.confidence,
                    reasoning=decision_result.reasoning,
                    recommendations=decision_result.recommendations,
                    processing_time=0.0,  # 批量处理中不计算单个处理时间
                    timestamp=datetime.now().isoformat()
                ))
                
            except Exception as e:
                logger.error(f"第{i+1}个光谱分析失败: {str(e)}")
                # 添加错误结果
                results.append(AnalysisResponse(
                    quality_score=0.0,
                    stability_score=0.0,
                    decision="error",
                    confidence=0.0,
                    reasoning=f"分析失败: {str(e)}",
                    recommendations=["检查输入数据", "重新运行分析"],
                    processing_time=0.0,
                    timestamp=datetime.now().isoformat()
                ))
        
        # 计算统计信息
        total_processing_time = (datetime.now() - start_time).total_seconds()
        average_processing_time = total_processing_time / len(request.spectra) if request.spectra else 0
        
        # 统计决策分布
        decision_summary = {}
        for result in results:
            decision = result.decision
            decision_summary[decision] = decision_summary.get(decision, 0) + 1
        
        return BatchAnalysisResponse(
            results=results,
            total_processing_time=total_processing_time,
            average_processing_time=average_processing_time,
            decision_summary=decision_summary,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"批量分析失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"批量分析失败: {str(e)}")

@app.get("/cache/stats", response_model=CacheStatsResponse)
@error_handler
async def get_cache_stats(cache: ModelCacheManager = Depends(get_model_cache)):
    """获取缓存统计信息"""
    stats = cache.get_cache_stats()
    return CacheStatsResponse(**stats)

@app.get("/cache/models", response_model=Dict[str, Dict])
@error_handler
async def list_cached_models(cache: ModelCacheManager = Depends(get_model_cache)):
    """列出缓存的模型"""
    return cache.list_cached_models()

@app.post("/cache/clear")
@error_handler
async def clear_cache(cache: ModelCacheManager = Depends(get_model_cache)):
    """清空缓存"""
    cache.clear_cache()
    return {"message": "缓存已清空", "timestamp": datetime.now().isoformat()}

@app.post("/cache/preload")
@error_handler
async def preload_models(
    coating_name: str = "DVP",
    cache: ModelCacheManager = Depends(get_model_cache)
):
    """预加载模型"""
    cache.warm_up_cache("models", coating_name)
    return {
        "message": f"已预加载{coating_name}涂层模型", 
        "timestamp": datetime.now().isoformat()
    }

@app.get("/decision/thresholds", response_model=Dict[str, float])
@error_handler
async def get_decision_thresholds(engine: DecisionEngine = Depends(get_decision_engine)):
    """获取决策阈值"""
    return {
        "quality_threshold": engine.thresholds.quality_threshold,
        "stability_threshold": engine.thresholds.stability_threshold
    }

@app.post("/decision/thresholds")
@error_handler
async def update_decision_thresholds(
    quality_threshold: float,
    stability_threshold: float,
    engine: DecisionEngine = Depends(get_decision_engine)
):
    """更新决策阈值"""
    if not (0 <= quality_threshold <= 1):
        raise HTTPException(status_code=400, detail="质量阈值必须在0-1范围内")
    if not (0 <= stability_threshold <= 1):
        raise HTTPException(status_code=400, detail="稳定性阈值必须在0-1范围内")
    
    engine.thresholds = DecisionThresholds(
        quality_threshold=quality_threshold,
        stability_threshold=stability_threshold
    )
    
    return {
        "message": "阈值更新成功",
        "quality_threshold": quality_threshold,
        "stability_threshold": stability_threshold,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/decision/visualize")
@error_handler
async def visualize_decision_space(engine: DecisionEngine = Depends(get_decision_engine)):
    """生成决策空间可视化"""
    save_path = f"logs/decision_space_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    engine.visualize_decision_space(save_path)
    
    return {
        "message": "决策空间可视化已生成",
        "file_path": save_path,
        "timestamp": datetime.now().isoformat()
    }

# 异常处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "内部服务器错误",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    # 创建日志目录
    Path("logs").mkdir(exist_ok=True)
    
    # 启动服务器
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )