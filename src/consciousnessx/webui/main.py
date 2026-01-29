"""
Streamlit Web Interface
"""
import streamlit as st
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import httpx
import pandas as pd
import plotly.graph_objects as go

from ..config import settings


# Page configuration
st.set_page_config(
    page_title="ConsciousnessX",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #F1F8E9;
        border-left: 4px solid #4CAF50;
    }
    .system-message {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


class ConsciousnessXUI:
    """Main UI class for ConsciousnessX"""
    
    def __init__(self):
        self.api_base = "http://localhost:8000"  # Adjust for production
        self.api_key = None
        self.session_id = None
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if "api_key" not in st.session_state:
            st.session_state.api_key = ""
        
        self.session_id = st.session_state.session_id
    
    async def make_api_request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request to API"""
        url = f"{self.api_base}{endpoint}"
        
        default_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        if headers:
            default_headers.update(headers)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "POST":
                response = await client.post(url, json=data, headers=default_headers)
            elif method == "GET":
                response = await client.get(url, headers=default_headers)
            elif method == "DELETE":
                response = await client.delete(url, headers=default_headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
    
    def render_header(self):
        """Render page header"""
        st.markdown('<h1 class="main-header">🧠 ConsciousnessX</h1>', unsafe_allow_html=True)
        st.markdown("### Production-Ready AI Consciousness System")
        
        # API Key input in sidebar
        with st.sidebar:
            st.header("Configuration")
            api_key = st.text_input(
                "API Key",
                type="password",
                value=st.session_state.api_key,
                help="Enter your API key to use the service"
            )
            
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key
                st.rerun()
            
            self.api_key = api_key
            
            st.divider()
            
            # Agent selection
            st.subheader("Agent Selection")
            self.selected_agent = st.selectbox(
                "Choose Agent",
                ["reasoner", "planner", "analyzer"],
                index=0,
                help="Select which agent to interact with"
            )
            
            # Session management
            st.divider()
            st.subheader("Session")
            st.text(f"Session ID: {self.session_id[:8]}...")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 New Session"):
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.messages = []
                    st.rerun()
            
            with col2:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.messages = []
                    st.rerun()
    
    def render_chat_interface(self):
        """Render main chat interface"""
        st.header("💬 Chat Interface")
        
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Show metadata if available
                    if "metadata" in message and message["metadata"]:
                        with st.expander("Details"):
                            st.json(message["metadata"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            if not self.api_key:
                st.error("Please enter an API key in the sidebar")
                return
            
            # Add user message to chat
            user_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Make API call
                        response = asyncio.run(self.make_api_request(
                            endpoint="/api/v1/chat",
                            data={
                                "message": prompt,
                                "session_id": self.session_id,
                                "agent": self.selected_agent,
                                "stream": False,
                            }
                        ))
                        
                        # Display response
                        st.markdown(response["content"])
                        
                        # Add to messages
                        assistant_message = {
                            "role": "assistant",
                            "content": response["content"],
                            "metadata": {
                                "model": response.get("model"),
                                "agent": self.selected_agent,
                                "timestamp": response.get("created_at"),
                            }
                        }
                        st.session_state.messages.append(assistant_message)
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    def render_dashboard(self):
        """Render analytics dashboard"""
        st.header("📊 Analytics Dashboard")
        
        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Requests</h3>
                <h1>1,234</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Active Sessions</h3>
                <h1>42</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Response Time</h3>
                <h1>1.2s</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Success Rate</h3>
                <h1>99.8%</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Requests Over Time")
            # Sample data
            data = pd.DataFrame({
                'time': pd.date_range('2024-01-01', periods=30, freq='D'),
                'requests': np.random.randint(100, 500, 30)
            })
            st.line_chart(data.set_index('time'))
        
        with col2:
            st.subheader("Agent Usage Distribution")
            agent_data = pd.DataFrame({
                'agent': ['Reasoner', 'Planner', 'Analyzer', 'Executor'],
                'usage': [45, 25, 20, 10]
            })
            st.bar_chart(agent_data.set_index('agent'))
    
    def render_memory_explorer(self):
        """Render memory explorer"""
        st.header("🧠 Memory Explorer")
        
        tab1, tab2, tab3 = st.tabs(["Search", "Browse", "Statistics"])
        
        with tab1:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                search_query = st.text_input("Search memories")
            
            with col2:
                limit = st.number_input("Results", min_value=1, max_value=100, value=10)
            
            if search_query and self.api_key:
                if st.button("Search", type="primary"):
                    with st.spinner("Searching..."):
                        try:
                            results = asyncio.run(self.make_api_request(
                                endpoint="/api/v1/memories/search",
                                data={
                                    "query": search_query,
                                    "limit": limit,
                                }
                            ))
                            
                            for result in results:
                                with st.expander(f"{result['memory_type']} - Score: {result['score']:.3f}"):
                                    st.text(f"ID: {result['memory_id']}")
                                    st.text(f"Created: {result['created_at']}")
                                    st.markdown(f"**Content:** {result['content']}")
                                    st.divider()
                            
                        except Exception as e:
                            st.error(f"Search failed: {str(e)}")
        
        with tab2:
            st.info("Memory browsing coming soon...")
        
        with tab3:
            st.info("Memory statistics coming soon...")
    
    def render_tools_interface(self):
        """Render tools interface"""
        st.header("🛠️ Tools Interface")
        
        if not self.api_key:
            st.warning("Enter API key to use tools")
            return
        
        # Get available tools
        try:
            tools = asyncio.run(self.make_api_request(
                endpoint="/api/v1/tools",
                method="GET"
            ))
            
            # Display tools
            for tool in tools:
                with st.expander(f"🔧 {tool['name']}"):
                    st.markdown(f"**Description:** {tool['description']}")
                    
                    # Show parameters
                    st.markdown("**Parameters:**")
                    params = tool['parameters']['properties']
                    required = tool['parameters'].get('required', [])
                    
                    for param_name, param_spec in params.items():
                        is_required = param_name in required
                        req_text = "**(required)**" if is_required else "(optional)"
                        st.text(f"- {param_name}: {param_spec['type']} {req_text}")
                        if 'description' in param_spec:
                            st.text(f"  {param_spec['description']}")
                    
                    # Execute tool
                    st.divider()
                    st.subheader("Execute Tool")
                    
                    # Dynamically create input fields
                    tool_args = {}
                    for param_name, param_spec in params.items():
                        param_type = param_spec['type']
                        
                        if param_type == "string":
                            value = st.text_input(
                                f"{param_name}",
                                help=param_spec.get('description', '')
                            )
                            tool_args[param_name] = value
                        
                        elif param_type == "number":
                            value = st.number_input(
                                f"{param_name}",
                                help=param_spec.get('description', '')
                            )
                            tool_args[param_name] = value
                        
                        elif param_type == "boolean":
                            value = st.checkbox(
                                f"{param_name}",
                                help=param_spec.get('description', '')
                            )
                            tool_args[param_name] = value
                    
                    if st.button(f"Execute {tool['name']}", key=f"execute_{tool['name']}"):
                        with st.spinner(f"Executing {tool['name']}..."):
                            try:
                                result = asyncio.run(self.make_api_request(
                                    endpoint="/api/v1/tools/execute",
                                    data={
                                        "tool_name": tool['name'],
                                        "arguments": tool_args,
                                        "session_id": self.session_id,
                                    }
                                ))
                                
                                if result['success']:
                                    st.success("Tool executed successfully!")
                                    st.json(result['result'])
                                else:
                                    st.error(f"Tool failed: {result['error']}")
                                    
                            except Exception as e:
                                st.error(f"Execution failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Failed to load tools: {str(e)}")
    
    def run(self):
        """Main run method"""
        # Render sidebar and header
        self.render_header()
        
        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Chat", 
            "📊 Dashboard", 
            "🧠 Memory", 
            "🛠️ Tools"
        ])
        
        with tab1:
            self.render_chat_interface()
        
        with tab2:
            self.render_dashboard()
        
        with tab3:
            self.render_memory_explorer()
        
        with tab4:
            self.render_tools_interface()
        
        # Footer
        st.divider()
        st.markdown(
            f"<p style='text-align: center; color: gray;'>"
            f"ConsciousnessX v{settings.VERSION} | "
            f"Environment: {settings.ENVIRONMENT}</p>",
            unsafe_allow_html=True
        )


def main():
    """Main entry point for Streamlit app"""
    app = ConsciousnessXUI()
    app.run()


if __name__ == "__main__":
    main()
