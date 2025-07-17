import logging
import time
import json
import psutil
import tiktoken
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import threading
from dataclasses import dataclass, asdict

@dataclass
class MetricsData:
    timestamp: str
    operation: str
    latency_ms: float
    tokens_input: int
    tokens_output: int
    cost_estimate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    context_quality: Optional[str] = None
    num_sources: int = 0
    web_search_used: bool = False

class MetricsLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup loggers
        self.performance_logger = self._setup_logger("performance", "performance.log")
        self.usage_logger = self._setup_logger("usage", "usage.log")
        self.error_logger = self._setup_logger("errors", "errors.log")
        
        # Token encoding for cost estimation
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
        
        # Cost per token (adjust based on your UFL AI pricing)
        self.input_cost_per_token = 0.0001  # $0.0001 per input token
        self.output_cost_per_token = 0.0002  # $0.0002 per output token
        
        # Metrics aggregation
        self.metrics_buffer = []
        self.buffer_lock = threading.Lock()
        
    def _setup_logger(self, name: str, filename: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_dir / filename)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def count_tokens(self, text: str) -> int:
        if self.encoding:
            return len(self.encoding.encode(text))
        return len(text.split())  # Rough approximation
    
    def get_system_metrics(self) -> Dict[str, float]:
        return {
            "memory_usage_mb": psutil.virtual_memory().used / 1024 / 1024,
            "cpu_usage_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }
    
    def start_operation(self, operation: str) -> Dict[str, Any]:
        return {
            "operation": operation,
            "start_time": time.time(),
            "start_metrics": self.get_system_metrics()
        }
    
    def log_operation(self, context: Dict[str, Any], input_text: str = "", 
                     output_text: str = "", success: bool = True, 
                     error_message: str = None, **kwargs) -> None:
        
        end_time = time.time()
        latency_ms = (end_time - context["start_time"]) * 1000
        
        tokens_input = self.count_tokens(input_text)
        tokens_output = self.count_tokens(output_text)
        
        cost_estimate = (
            tokens_input * self.input_cost_per_token + 
            tokens_output * self.output_cost_per_token
        )
        
        current_metrics = self.get_system_metrics()
        
        metrics_data = MetricsData(
            timestamp=datetime.now().isoformat(),
            operation=context["operation"],
            latency_ms=latency_ms,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_estimate=cost_estimate,
            memory_usage_mb=current_metrics["memory_usage_mb"],
            cpu_usage_percent=current_metrics["cpu_usage_percent"],
            success=success,
            error_message=error_message,
            **kwargs
        )
        
        # Log to appropriate loggers
        if success:
            self.performance_logger.info(json.dumps(asdict(metrics_data)))
            self.usage_logger.info(
                f"Operation: {context['operation']}, "
                f"Latency: {latency_ms:.2f}ms, "
                f"Tokens: {tokens_input + tokens_output}, "
                f"Cost: ${cost_estimate:.6f}"
            )
        else:
            self.error_logger.error(
                f"Operation: {context['operation']}, "
                f"Error: {error_message}, "
                f"Latency: {latency_ms:.2f}ms"
            )
        
        # Add to buffer for aggregation
        with self.buffer_lock:
            self.metrics_buffer.append(metrics_data)
            if len(self.metrics_buffer) > 100:  # Keep last 100 entries
                self.metrics_buffer.pop(0)
    
    def get_aggregated_metrics(self, minutes: int = 60) -> Dict[str, Any]:
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        
        with self.buffer_lock:
            recent_metrics = [
                m for m in self.metrics_buffer 
                if datetime.fromisoformat(m.timestamp).timestamp() > cutoff_time
            ]
        
        if not recent_metrics:
            return {"message": "No recent metrics available"}
        
        total_operations = len(recent_metrics)
        successful_operations = sum(1 for m in recent_metrics if m.success)
        
        avg_latency = sum(m.latency_ms for m in recent_metrics) / total_operations
        total_tokens = sum(m.tokens_input + m.tokens_output for m in recent_metrics)
        total_cost = sum(m.cost_estimate for m in recent_metrics)
        
        return {
            "time_window_minutes": minutes,
            "total_operations": total_operations,
            "success_rate": successful_operations / total_operations,
            "avg_latency_ms": avg_latency,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_cost_per_operation": total_cost / total_operations if total_operations > 0 else 0
        }