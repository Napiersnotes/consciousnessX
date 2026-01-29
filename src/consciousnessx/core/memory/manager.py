"""
Memory Management System with Vector Storage
"""
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Text, JSON, DateTime, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from ...config import settings


logger = logging.getLogger(__name__)
Base = declarative_base()


class MemoryType(str, Enum):
    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    TASK = "task"
    REFLECTION = "reflection"


@dataclass
class Memory:
    id: str
    content: str
    memory_type: MemoryType
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    importance: float = 0.5
    created_at: datetime = None
    accessed_at: datetime = None
    access_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.accessed_at is None:
            self.accessed_at = self.created_at
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        return cls(**data)


# SQLAlchemy Models
class MemoryModel(Base):
    __tablename__ = "memories"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content = Column(Text, nullable=False)
    memory_type = Column(String(50), nullable=False)
    embedding = Column(JSON)  # Store as JSON for simplicity
    metadata = Column(JSON, default={})
    importance = Column(Integer, default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow)
    accessed_at = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    user_id = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    is_archived = Column(Boolean, default=False)


class MemoryManager:
    """Manages memory storage and retrieval"""
    
    def __init__(self):
        self.redis = None
        self.qdrant = None
        self.db_engine = None
        self.async_session = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize all storage backends"""
        if self._initialized:
            return
        
        # Initialize Redis
        self.redis = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        
        # Initialize Qdrant for vector storage
        self.qdrant = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        
        # Ensure collection exists
        try:
            collections = self.qdrant.get_collections()
            if "memories" not in [c.name for c in collections.collections]:
                self.qdrant.create_collection(
                    collection_name="memories",
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI embedding size
                        distance=Distance.COSINE,
                    ),
                )
        except Exception as e:
            logger.warning(f"Could not initialize Qdrant: {e}")
        
        # Initialize PostgreSQL
        self.db_engine = create_async_engine(
            str(settings.DATABASE_URL).replace("postgresql://", "postgresql+asyncpg://"),
            echo=False,
        )
        
        self.async_session = sessionmaker(
            self.db_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        # Create tables
        async with self.db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        self._initialized = True
        logger.info("Memory Manager initialized")
    
    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Memory:
        """Store a new memory"""
        await self.initialize()
        
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            embedding=embedding,
            metadata=metadata or {},
            importance=importance,
        )
        
        # Store in PostgreSQL
        async with self.async_session() as session:
            db_memory = MemoryModel(
                id=memory.id,
                content=content,
                memory_type=memory_type.value,
                embedding=embedding,
                metadata=metadata or {},
                importance=importance,
                user_id=user_id,
                session_id=session_id,
            )
            session.add(db_memory)
            await session.commit()
        
        # Cache in Redis
        cache_key = f"memory:{memory.id}"
        await self.redis.setex(
            cache_key,
            timedelta(hours=24),
            json.dumps(memory.to_dict()),
        )
        
        # Store embedding in Qdrant if available
        if embedding and self.qdrant:
            try:
                point = PointStruct(
                    id=memory.id,
                    vector=embedding,
                    payload={
                        "content": content[:500],  # Store snippet
                        "type": memory_type.value,
                        "importance": importance,
                        "user_id": user_id,
                        "created_at": memory.created_at.isoformat(),
                    }
                )
                self.qdrant.upsert(
                    collection_name="memories",
                    points=[point],
                )
            except Exception as e:
                logger.error(f"Failed to store in Qdrant: {e}")
        
        logger.info(f"Stored memory: {memory.id}")
        return memory
    
    async def retrieve_memory(
        self,
        memory_id: str,
        update_access: bool = True,
    ) -> Optional[Memory]:
        """Retrieve a memory by ID"""
        await self.initialize()
        
        # Try cache first
        cache_key = f"memory:{memory_id}"
        cached = await self.redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            memory = Memory.from_dict(data)
            
            if update_access:
                await self.update_access(memory_id)
            
            return memory
        
        # Fallback to database
        async with self.async_session() as session:
            result = await session.get(MemoryModel, memory_id)
            if result:
                memory = Memory(
                    id=result.id,
                    content=result.content,
                    memory_type=MemoryType(result.memory_type),
                    embedding=result.embedding,
                    metadata=result.metadata,
                    importance=result.importance,
                    created_at=result.created_at,
                    accessed_at=result.accessed_at,
                    access_count=result.access_count,
                )
                
                # Cache for future
                await self.redis.setex(
                    cache_key,
                    timedelta(hours=24),
                    json.dumps(memory.to_dict()),
                )
                
                if update_access:
                    await self.update_access(memory_id)
                
                return memory
        
        return None
    
    async def search_memories(
        self,
        query: str,
        query_embedding: List[float],
        memory_type: Optional[MemoryType] = None,
        user_id: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> List[Tuple[Memory, float]]:
        """Search memories by semantic similarity"""
        await self.initialize()
        
        if not self.qdrant:
            return await self.keyword_search(query, limit=limit)
        
        # Build filter
        filters = []
        if memory_type:
            filters.append(
                FieldCondition(
                    key="type",
                    match=MatchValue(value=memory_type.value),
                )
            )
        if user_id:
            filters.append(
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            )
        
        filter_obj = Filter(must=filters) if filters else None
        
        # Search in Qdrant
        try:
            search_result = self.qdrant.search(
                collection_name="memories",
                query_vector=query_embedding,
                query_filter=filter_obj,
                limit=limit,
                score_threshold=threshold,
            )
            
            # Retrieve full memories
            results = []
            for hit in search_result:
                memory = await self.retrieve_memory(hit.id, update_access=False)
                if memory:
                    results.append((memory, hit.score))
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return await self.keyword_search(query, limit=limit)
    
    async def keyword_search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        user_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Tuple[Memory, float]]:
        """Fallback keyword search"""
        await self.initialize()
        
        async with self.async_session() as session:
            # Simple keyword matching
            # In production, use full-text search
            query_terms = query.lower().split()
            
            # This is simplified - use PostgreSQL full-text search in production
            memories = []
            
            return memories
    
    async def update_access(self, memory_id: str):
        """Update access time and count"""
        async with self.async_session() as session:
            memory = await session.get(MemoryModel, memory_id)
            if memory:
                memory.accessed_at = datetime.utcnow()
                memory.access_count += 1
                await session.commit()
                
                # Update cache
                cache_key = f"memory:{memory_id}"
                cached = await self.redis.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    data["accessed_at"] = memory.accessed_at.isoformat()
                    data["access_count"] = memory.access_count
                    await self.redis.setex(
                        cache_key,
                        timedelta(hours=24),
                        json.dumps(data),
                    )
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Memory]:
        """Get conversation history for a session"""
        await self.initialize()
        
        cache_key = f"conversation:{session_id}:{limit}:{offset}"
        cached = await self.redis.get(cache_key)
        if cached:
            return [Memory.from_dict(data) for data in json.loads(cached)]
        
        async with self.async_session() as session:
            # Simplified query - use proper pagination in production
            results = await session.execute(
                """
                SELECT * FROM memories 
                WHERE session_id = :session_id 
                AND memory_type = 'conversation'
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
                """,
                {"session_id": session_id, "limit": limit, "offset": offset}
            )
            
            memories = []
            for row in results:
                memory = Memory(
                    id=row.id,
                    content=row.content,
                    memory_type=MemoryType(row.memory_type),
                    embedding=row.embedding,
                    metadata=row.metadata,
                    importance=row.importance,
                    created_at=row.created_at,
                    accessed_at=row.accessed_at,
                    access_count=row.access_count,
                )
                memories.append(memory)
            
            # Cache results
            await self.redis.setex(
                cache_key,
                timedelta(minutes=5),
                json.dumps([m.to_dict() for m in memories]),
            )
            
            return memories
    
    async def cleanup_old_memories(self, days_old: int = 30):
        """Clean up old memories (archival)"""
        await self.initialize()
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        async with self.async_session() as session:
            await session.execute(
                """
                UPDATE memories 
                SET is_archived = TRUE 
                WHERE created_at < :cutoff_date 
                AND is_archived = FALSE
                """,
                {"cutoff_date": cutoff_date}
            )
            await session.commit()
        
        logger.info(f"Cleaned up memories older than {days_old} days")
    
    async def close(self):
        """Cleanup connections"""
        if self.redis:
            await self.redis.close()
        if self.db_engine:
            await self.db_engine.dispose()
