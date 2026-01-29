"""
Middleware Configuration
"""
import time
import json
import logging
from typing import Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from ..config import settings


logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Counter(
    'http_requests_active',
    'Active HTTP requests'
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get request info
        method = request.method
        url = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        
        # Log request
        logger.info(f"Request: {method} {url} from {client_ip}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response: {method} {url} - {response.status_code} "
            f"({latency:.3f}s)"
        )
        
        # Add latency header
        response.headers["X-Process-Time"] = str(latency)
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting Prometheus metrics"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)
        
        method = request.method
        endpoint = request.url.path
        
        # Increment active requests
        ACTIVE_REQUESTS.inc()
        
        # Measure latency
        start_time = time.time()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise e
        finally:
            latency = time.time() - start_time
            
            # Update metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status=status_code
            ).inc()
            
            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint
            ).observe(latency)
            
            # Decrement active requests
            ACTIVE_REQUESTS.dec()
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # CSP header (customize for your needs)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:;"
        )
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting"""
    
    def __init__(self, app, redis_client):
        super().__init__(app)
        self.redis = redis_client
    
    async def dispatch(self, request: Request, call_next):
        # Get API key from header
        api_key = request.headers.get(settings.API_KEY_HEADER)
        if not api_key:
            # No API key, check IP
            client_ip = request.client.host if request.client else "unknown"
            identifier = f"ip:{client_ip}"
        else:
            identifier = f"key:{api_key}"
        
        # Check rate limit
        current_minute = int(time.time() / 60)
        minute_key = f"ratelimit:{identifier}:minute:{current_minute}"
        hour_key = f"ratelimit:{identifier}:hour:{int(time.time() / 3600)}"
        
        # Get current counts
        minute_count = await self.redis.get(minute_key) or 0
        hour_count = await self.redis.get(hour_key) or 0
        
        # Check limits
        if (int(minute_count) >= settings.RATE_LIMIT_PER_MINUTE or
            int(hour_count) >= settings.RATE_LIMIT_PER_HOUR):
            
            return Response(
                content=json.dumps({
                    "error": "Rate limit exceeded",
                    "limits": {
                        "per_minute": settings.RATE_LIMIT_PER_MINUTE,
                        "per_hour": settings.RATE_LIMIT_PER_HOUR,
                    }
                }),
                status_code=429,
                media_type="application/json",
            )
        
        # Increment counters
        pipeline = self.redis.pipeline()
        pipeline.incr(minute_key)
        pipeline.expire(minute_key, 120)  # 2 minutes expiry
        
        pipeline.incr(hour_key)
        pipeline.expire(hour_key, 7200)  # 2 hours expiry
        
        await pipeline.execute()
        
        # Add headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit-Minute"] = str(settings.RATE_LIMIT_PER_MINUTE)
        response.headers["X-RateLimit-Limit-Hour"] = str(settings.RATE_LIMIT_PER_HOUR)
        response.headers["X-RateLimit-Remaining-Minute"] = str(
            settings.RATE_LIMIT_PER_MINUTE - int(minute_count) - 1
        )
        response.headers["X-RateLimit-Remaining-Hour"] = str(
            settings.RATE_LIMIT_PER_HOUR - int(hour_count) - 1
        )
        
        return response


def setup_middleware(app):
    """Setup all middleware"""
    # Add session middleware if needed
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.SECRET_KEY,
        session_cookie="consciousnessx_session",
        max_age=3600,
    )
    
    # Add custom middleware
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Rate limiting will be added after Redis initialization
    # app.add_middleware(RateLimitMiddleware, redis_client=redis_client)
    
    # Add metrics endpoint
    @app.get("/metrics")
    async def metrics_endpoint():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
  )
