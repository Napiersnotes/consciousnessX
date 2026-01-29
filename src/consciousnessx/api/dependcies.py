"""
Dependency Injection System
"""
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import redis.asyncio as redis

from ..config import settings


security = HTTPBearer(auto_error=False)


# Dependencies
async def get_settings():
    """Get application settings"""
    return settings


async def get_redis_client():
    """Get Redis client"""
    redis_client = await redis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    try:
        yield redis_client
    finally:
        await redis_client.close()


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = None,
    settings=Depends(get_settings),
) -> bool:
    """Verify API key from header"""
    if not credentials:
        return False
    
    # In production, validate against database
    # This is a simplified example
    api_key = credentials.credentials
    
    # Check if it's a user token
    if api_key.startswith("user_"):
        # Validate user token
        try:
            # Extract user ID and validate signature
            parts = api_key.split("_")
            if len(parts) != 3:
                return False
            
            user_id, timestamp_str, signature = parts
            
            # Check timestamp (valid for 30 days)
            timestamp = datetime.fromtimestamp(int(timestamp_str))
            if datetime.utcnow() - timestamp > timedelta(days=30):
                return False
            
            # Verify signature
            expected_signature = hmac.new(
                settings.SECRET_KEY.encode(),
                f"{user_id}_{timestamp_str}".encode(),
                hashlib.sha256
            ).hexdigest()[:16]
            
            return hmac.compare_digest(signature, expected_signature)
            
        except:
            return False
    
    # For now, accept any non-empty key in development
    if settings.ENVIRONMENT != "production":
        return bool(api_key)
    
    return False


async def rate_limit_check(
    api_key: str,
    redis_client: redis.Redis = Depends(get_redis_client),
) -> bool:
    """Check rate limit for API key"""
    # Simplified rate limiting
    # In production, use more sophisticated logic
    
    current_minute = int(datetime.utcnow().timestamp() / 60)
    key = f"ratelimit:{api_key}:{current_minute}"
    
    try:
        current_count = await redis_client.get(key) or 0
        if int(current_count) >= 60:  # 60 requests per minute
            return False
        
        # Increment counter
        await redis_client.incr(key)
        await redis_client.expire(key, 120)  # Expire after 2 minutes
        
        return True
        
    except Exception as e:
        # If Redis fails, allow request (fail open)
        return True


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    settings=Depends(get_settings),
) -> Optional[Dict[str, Any]]:
    """Get current user from JWT token"""
    if not credentials:
        return None
    
    token = credentials.credentials
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        
        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "roles": payload.get("roles", []),
            "permissions": payload.get("permissions", []),
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


async def get_agent_orchestrator(request):
    """Get the agent orchestrator instance"""
    return request.app.state.agent_orchestrator


# Factory functions for different agent types
async def get_reasoning_agent(orchestrator=Depends(get_agent_orchestrator)):
    """Get reasoning agent"""
    return orchestrator.agents.get("reasoner")


async def get_planning_agent(orchestrator=Depends(get_agent_orchestrator)):
    """Get planning agent"""
    return orchestrator.agents.get("planner")


async def get_tool_registry(orchestrator=Depends(get_agent_orchestrator)):
    """Get tool registry"""
    return orchestrator.tool_registry
