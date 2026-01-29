"""
Tool Registry and Management System
"""
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
import httpx
from datetime import datetime, timedelta
import aiofiles
import os

from ...config import settings


logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    SEARCH = "search"
    CALCULATION = "calculation"
    DATABASE = "database"
    FILE = "file"
    NETWORK = "network"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class ToolParameter:
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class Tool:
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter]
    function: Callable[..., Coroutine[Any, Any, Dict[str, Any]]]
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the tool"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description,
                        "enum": param.enum,
                    }
                    for param in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required],
            },
        }


class BaseTool(ABC):
    """Base class for all tools"""
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass


class WebSearchTool(BaseTool):
    """Tool for searching the web"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def search(
        self,
        query: str,
        num_results: int = 5,
        search_engine: str = "google"
    ) -> Dict[str, Any]:
        """Search the web for information"""
        
        # In production, integrate with actual search APIs like:
        # - Google Custom Search API
        # - SerpAPI
        # - DuckDuckGo
        
        # This is a simplified implementation
        search_urls = {
            "google": "https://www.google.com/search",
            "bing": "https://www.bing.com/search",
        }
        
        if search_engine not in search_urls:
            return {
                "success": False,
                "error": f"Unsupported search engine: {search_engine}",
                "results": [],
            }
        
        try:
            # Mock implementation - replace with real API calls
            results = [
                {
                    "title": f"Result {i+1} for: {query}",
                    "url": f"https://example.com/result{i+1}",
                    "snippet": f"This is a sample result for your query about {query}.",
                }
                for i in range(num_results)
            ]
            
            return {
                "success": True,
                "query": query,
                "num_results": len(results),
                "results": results,
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
            }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        return await self.search(**kwargs)


class CalculatorTool(BaseTool):
    """Tool for mathematical calculations"""
    
    async def calculate(
        self,
        expression: str,
        precision: int = 2
    ) -> Dict[str, Any]:
        """Evaluate a mathematical expression"""
        try:
            # Security: Use a safe eval or math parser
            # This is simplified - use a proper math library in production
            import math
            import re
            
            # Simple safe evaluation
            allowed_names = {
                k: v for k, v in math.__dict__.items() 
                if not k.startswith("_")
            }
            
            # Basic safety check
            if re.search(r'[^0-9+\-*/().\s]', expression):
                return {
                    "success": False,
                    "error": "Invalid characters in expression",
                    "result": None,
                }
            
            # Evaluate safely
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            # Round if numeric
            if isinstance(result, (int, float)):
                result = round(result, precision)
            
            return {
                "success": True,
                "expression": expression,
                "result": result,
                "precision": precision,
            }
            
        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "result": None,
            }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        return await self.calculate(**kwargs)


class FileReadTool(BaseTool):
    """Tool for reading files"""
    
    def __init__(self, base_path: str = settings.UPLOAD_DIR):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    async def read_file(
        self,
        file_path: str,
        max_lines: int = 1000
    ) -> Dict[str, Any]:
        """Read a file with safety limits"""
        try:
            # Security: Validate file path
            abs_path = os.path.abspath(os.path.join(self.base_path, file_path))
            if not abs_path.startswith(os.path.abspath(self.base_path)):
                return {
                    "success": False,
                    "error": "Access denied: Path outside allowed directory",
                }
            
            if not os.path.exists(abs_path):
                return {
                    "success": False,
                    "error": "File not found",
                }
            
            # Check file size
            file_size = os.path.getsize(abs_path)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                return {
                    "success": False,
                    "error": "File too large (max 10MB)",
                }
            
            # Read file
            async with aiofiles.open(abs_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Truncate if too many lines
            lines = content.split('\n')
            if len(lines) > max_lines:
                content = '\n'.join(lines[:max_lines])
                truncated = True
            else:
                truncated = False
            
            return {
                "success": True,
                "file_path": file_path,
                "content": content,
                "truncated": truncated,
                "total_lines": len(lines),
                "size_bytes": file_size,
            }
            
        except Exception as e:
            logger.error(f"File read failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        return await self.read_file(**kwargs)


class DatabaseQueryTool(BaseTool):
    """Tool for querying databases"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.DATABASE_URL
    
    async def query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Execute a SQL query safely"""
        # IMPORTANT: This is a simplified implementation
        # In production, use proper parameterized queries and access controls
        
        try:
            # Security: Check for dangerous operations
            query_lower = query.lower().strip()
            dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter']
            
            if any(keyword in query_lower for keyword in dangerous_keywords):
                return {
                    "success": False,
                    "error": "Dangerous operations not allowed",
                }
            
            # Add LIMIT if not present
            if 'limit' not in query_lower:
                query = f"{query} LIMIT {limit}"
            
            # Execute query (simplified)
            # In production, use async database driver
            import asyncpg  # Example
            
            conn = await asyncpg.connect(self.database_url)
            try:
                result = await conn.fetch(query, *(parameters or {}).values())
                columns = [dict(row) for row in result]
                
                return {
                    "success": True,
                    "query": query,
                    "row_count": len(columns),
                    "columns": list(columns[0].keys()) if columns else [],
                    "data": columns,
                }
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        return await self.query(**kwargs)


class ToolRegistry:
    """Registry for managing tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools"""
        # Web Search Tool
        web_search_tool = Tool(
            name="web_search",
            description="Search the web for current information",
            category=ToolCategory.SEARCH,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="num_results",
                    type="number",
                    description="Number of results to return",
                    required=False,
                    default=5,
                ),
            ],
            function=WebSearchTool().execute,
        )
        self.register_tool(web_search_tool)
        
        # Calculator Tool
        calculator_tool = Tool(
            name="calculator",
            description="Perform mathematical calculations",
            category=ToolCategory.CALCULATION,
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate",
                    required=True,
                ),
                ToolParameter(
                    name="precision",
                    type="number",
                    description="Decimal precision for results",
                    required=False,
                    default=2,
                ),
            ],
            function=CalculatorTool().execute,
        )
        self.register_tool(calculator_tool)
        
        # File Read Tool
        file_read_tool = Tool(
            name="read_file",
            description="Read contents of a file",
            category=ToolCategory.FILE,
            parameters=[
                ToolParameter(
                    name="file_path",
                    type="string",
                    description="Path to the file to read",
                    required=True,
                ),
                ToolParameter(
                    name="max_lines",
                    type="number",
                    description="Maximum number of lines to read",
                    required=False,
                    default=1000,
                ),
            ],
            function=FileReadTool().execute,
        )
        self.register_tool(file_read_tool)
        
        logger.info("Default tools registered")
    
    def register_tool(self, tool: Tool):
        """Register a new tool"""
        if tool.name in self.tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting")
        
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_custom_tool(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        function: Callable[..., Coroutine[Any, Any, Dict[str, Any]]],
        category: ToolCategory = ToolCategory.CUSTOM,
    ):
        """Register a custom tool"""
        tool = Tool(
            name=name,
            description=description,
            category=category,
            parameters=parameters,
            function=function,
        )
        self.register_tool(tool)
    
    def available_tools(self) -> Dict[str, Tool]:
        """Get all available tools"""
        return self.tools.copy()
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a specific tool by name"""
        return self.tools.get(name)
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool with given arguments"""
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool not found: {tool_name}",
            }
        
        try:
            # Validate parameters
            for param in tool.parameters:
                if param.required and param.name not in arguments:
                    return {
                        "success": False,
                        "error": f"Missing required parameter: {param.name}",
                    }
            
            # Execute tool
            result = await tool.function(**arguments)
            result["tool"] = tool_name
            
            logger.info(f"Executed tool: {tool_name}")
            return result
            
        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
            }
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get JSON schema for all tools"""
        return [tool.get_schema() for tool in self.tools.values()]
