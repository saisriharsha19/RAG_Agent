# Add this to your project as src/config_helper.py

import os
import streamlit as st
from typing import Dict, Any

def check_configuration() -> Dict[str, Any]:
    """Check system configuration and return status"""
    
    config_status = {
        "ufl_llm": {
            "configured": False,
            "missing": [],
            "status": "not_configured"
        },
        "web_search": {
            "configured": False,
            "missing": [],
            "status": "not_configured"
        },
        "vector_store": {
            "configured": True,
            "status": "ready"
        },
        "overall_status": "incomplete"
    }
    
    # Check UFL LLM configuration
    ufl_vars = ['UFL_AI_BASE_URL', 'UFL_AI_MODEL', 'UFL_AI_API_KEY']
    missing_ufl = []
    
    for var in ufl_vars:
        if not os.getenv(var):
            missing_ufl.append(var)
    
    if not missing_ufl:
        config_status["ufl_llm"]["configured"] = True
        config_status["ufl_llm"]["status"] = "ready"
    else:
        config_status["ufl_llm"]["missing"] = missing_ufl
        config_status["ufl_llm"]["status"] = "missing_variables"
    
    # Check web search configuration (optional)
    web_vars = ['SEARCH_API_KEY', 'SEARCH_ENGINE_ID']  # Example variables
    missing_web = []
    
    for var in web_vars:
        if not os.getenv(var):
            missing_web.append(var)
    
    if not missing_web:
        config_status["web_search"]["configured"] = True
        config_status["web_search"]["status"] = "ready"
    else:
        config_status["web_search"]["missing"] = missing_web
        config_status["web_search"]["status"] = "optional"
    
    # Overall status
    if config_status["ufl_llm"]["configured"]:
        config_status["overall_status"] = "ready"
    else:
        config_status["overall_status"] = "needs_llm_config"
    
    return config_status

def display_configuration_help():
    """Display configuration help in Streamlit"""
    
    config = check_configuration()
    
    st.markdown("## 🔧 System Configuration")
    
    # Overall status
    if config["overall_status"] == "ready":
        st.success("✅ System is properly configured and ready to use!")
    else:
        st.warning("⚠️ System configuration incomplete. Please complete the setup below.")
    
    # UFL LLM Configuration
    st.markdown("### 🤖 UFL LLM Client")
    
    if config["ufl_llm"]["configured"]:
        st.success("✅ UFL LLM client is configured")
    else:
        st.error("❌ UFL LLM client configuration missing")
        st.markdown("**Missing environment variables:**")
        for var in config["ufl_llm"]["missing"]:
            st.code(f"export {var}=your_value_here")
        
        st.markdown("""
        **Setup Instructions:**
        1. Set the required environment variables:
           - `UFL_AI_BASE_URL`: Your UFL AI service URL
           - `UFL_AI_MODEL`: The model name to use
           - `UFL_AI_API_KEY`: Your API key
        
        2. Restart the application after setting environment variables
        
        3. You can also create a `.env` file in your project root:
        ```
        UFL_AI_BASE_URL=https://your-ufl-ai-service.com
        UFL_AI_MODEL=your-model-name
        UFL_AI_API_KEY=your-api-key
        ```
        """)
    
    # Web Search Configuration
    st.markdown("### 🌐 Web Search (Optional)")
    
    if config["web_search"]["configured"]:
        st.success("✅ Web search is configured")
    else:
        st.info("ℹ️ Web search is not configured (optional feature)")
        if config["web_search"]["missing"]:
            st.markdown("**To enable web search, set:**")
            for var in config["web_search"]["missing"]:
                st.code(f"export {var}=your_value_here")
    
    # Vector Store Configuration
    st.markdown("### 🗄️ Vector Store")
    st.success("✅ Vector store is ready (uses local storage)")
    
    return config

def show_configuration_sidebar():
    """Show configuration status in sidebar"""
    
    config = check_configuration()
    
    st.sidebar.markdown("### 🔧 Configuration")
    
    # Overall status
    if config["overall_status"] == "ready":
        st.sidebar.success("✅ System Ready")
    else:
        st.sidebar.error("❌ Setup Needed")
        if st.sidebar.button("🔧 Show Setup Guide"):
            st.session_state.show_config_help = True
    
    # Component status
    components = {
        "🤖 LLM": config["ufl_llm"]["status"],
        "🌐 Web": config["web_search"]["status"],
        "🗄️ Vector": config["vector_store"]["status"]
    }
    
    for component, status in components.items():
        if status == "ready":
            st.sidebar.success(f"{component} ✅")
        elif status == "optional":
            st.sidebar.info(f"{component} ℹ️")
        else:
            st.sidebar.error(f"{component} ❌")

# Environment variable helper
def load_env_file():
    """Load environment variables from .env file if it exists"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # python-dotenv not installed
        pass
    except Exception as e:
        st.warning(f"Could not load .env file: {e}")

# Configuration validation
def validate_ufl_client():
    """Validate UFL client configuration"""
    try:
        from src.ufl_llm_client import create_llm_client
        client = create_llm_client()
        return True, "UFL client created successfully"
    except Exception as e:
        return False, f"UFL client creation failed: {str(e)}"

def get_environment_info():
    """Get information about the current environment"""
    import sys
    import platform
    
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "streamlit_version": st.__version__,
        "working_directory": os.getcwd(),
        "environment_variables": {
            "UFL_AI_BASE_URL": "✅" if os.getenv("UFL_AI_BASE_URL") else "❌",
            "UFL_AI_MODEL": "✅" if os.getenv("UFL_AI_MODEL") else "❌",
            "UFL_AI_API_KEY": "✅" if os.getenv("UFL_AI_API_KEY") else "❌"
        }
    }
    
    return info