"""
FastAPI Application Entry Point
"""
import logging
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from contextlib import asynccontextmanager

from .routes import api_router
from .middleware import setup_middleware
from .dependencies import get_settings, verify_api_key
from ..core.agents.base import AgentOrchestrator
from ..config import settings, Environment


# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('consciousnessx.log'),
    ]
)
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    logger.info(f"Starting ConsciousnessX v{settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Initialize agent orchestrator
    app.state.agent_orchestrator = AgentOrchestrator()
    await app.state.agent_orchestrator.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down ConsciousnessX")
    if hasattr(app.state, 'agent_orchestrator'):
        await app.state.agent_orchestrator.close()


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Production-Reide AI Consciousness System",
    docs_url="/docs" if settings.ENVIRONMENT != Environment.PRODUCTION else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != Environment.PRODUCTION else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT != Environment.PRODUCTION else None,
    lifespan=lifespan,
)

# Setup middleware
setup_middleware(app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware
if settings.ENVIRONMENT == Environment.PRODUCTION:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["consciousnessx.com", "api.consciousnessx.com"],  # Add your domains
    )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "docs": "/docs" if settings.ENVIRONMENT != Environment.PRODUCTION else None,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/status", tags=["Status"])
async def status_check(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    settings=Depends(get_settings),
):
    """Detailed status check (requires authentication)"""
    if not verify_api_key(credentials, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    return {
        "status": "operational",
        "agents": list(app.state.agent_orchestrator.agents.keys()),
        "tools": len(app.state.agent_orchestrator.tool_registry.available_tools()),
        "uptime": "TODO",  # Add uptime tracking
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": request.url.path,
            "method": request.method,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle generic exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "path": request.url.path,
            "method": request.method,
        },
    )


# Include API routes
app.include_router(api_router, prefix=settings.API_PREFIX)


if __name__ == "__main__":
    uvicorn.run(
        "consciousnessx.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.ENVIRONMENT == Environment.DEVELOPMENT,
        log_level=settings.LOG_LEVEL.lower(),
    )
