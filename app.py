import streamlit as st
import asyncio
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import shutil
# Import our modules
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Enhanced RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cohesive theming
def load_custom_css():
    st.markdown("""
    <style>
    /* Root variables for theming */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #ff9800;
        --error-color: #d62728;
        --info-color: #17a2b8;
        --border-radius: 8px;
        --box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        --transition: all 0.3s ease;
    }
    
    /* Dark mode overrides */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #4a9eff;
            --secondary-color: #ffa726;
            --success-color: #4caf50;
            --warning-color: #ff9800;
            --error-color: #f44336;
            --info-color: #29b6f6;
            --box-shadow: 0 2px 4px rgba(255,255,255,0.1);
        }
    }
    
    /* Main container styling */
    .main-container {
        background: var(--background-color);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: var(--box-shadow);
        transition: var(--transition);
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, var(--primary-color)10, var(--secondary-color)05);
        border: 1px solid var(--primary-color)20;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        transition: var(--transition);
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Status indicators */
    .status-healthy { color: var(--success-color); }
    .status-warning { color: var(--warning-color); }
    .status-error { color: var(--error-color); }
    .status-info { color: var(--info-color); }
    
    /* Button styling */
    .stButton > button {
        border-radius: var(--border-radius);
        transition: var(--transition);
        border: 1px solid var(--primary-color);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Danger button */
    .danger-button {
        background-color: var(--error-color) !important;
        color: white !important;
    }
    
    /* Success button */
    .success-button {
        background-color: var(--success-color) !important;
        color: white !important;
    }
    
    /* Metric styling */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: var(--border-radius);
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Document item styling */
    .doc-item {
        background: rgba(255,255,255,0.03);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--primary-color);
    }
    
    .doc-item:hover {
        background: rgba(255,255,255,0.08);
    }
    
    /* Chat message styling */
    .chat-message {
        border-radius: var(--border-radius);
        padding: 1rem;
        margin: 0.5rem 0;
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Source styling */
    .source-item {
        background: rgba(255,255,255,0.05);
        border-radius: var(--border-radius);
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-left: 3px solid var(--info-color);
    }
    
    /* Settings panel */
    .settings-panel {
        background: rgba(255,255,255,0.02);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: rgba(255,255,255,0.02);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Animation for loading states */
    .loading-animation {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            padding: 0.5rem;
        }
        
        .info-card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the RAG system components"""
    # Create necessary directories
    os.makedirs("data/documents", exist_ok=True)
    os.makedirs("vector_db", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize components
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    rag_pipeline = RAGPipeline(vector_store)
    
    return doc_processor, vector_store, rag_pipeline

def save_chat_history(chat_history, filename="chat_history.json"):
    """Save chat history to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(chat_history, f, indent=2, default=str)

def load_chat_history(filename="chat_history.json"):
    """Load chat history from a JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def get_document_list():
    """Get list of processed documents with metadata"""
    docs_path = Path("data/documents")
    if not docs_path.exists():
        return []
    
    documents = []
    for file_path in docs_path.iterdir():
        if file_path.is_file():
            stat = file_path.stat()
            documents.append({
                "name": file_path.name,
                "path": str(file_path),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "type": file_path.suffix.lower()
            })
    
    return sorted(documents, key=lambda x: x["modified"], reverse=True)

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def sidebar_controls():
    """Enhanced sidebar with better controls"""
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # System status
        st.markdown("#### 📊 System Status")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh", key="refresh_status"):
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear All", key="clear_all_sidebar"):
                    st.session_state.show_clear_confirmation = True
        
        # Document quick stats
        documents = get_document_list()
        vector_stats = st.session_state.get('vector_stats', {"total_documents": 0})
        
        st.markdown("#### 📚 Quick Stats")
        st.metric("📄 Files", len(documents))
        st.metric("🔍 Indexed", vector_stats.get("total_documents", 0))
        st.metric("💬 Chats", len(st.session_state.get('chat_history', [])))
        
        # Quick settings
        st.markdown("#### ⚙️ Quick Settings")
        
        # Theme toggle (visual indicator)
        theme_mode = st.selectbox(
            "🎨 Theme Preference",
            ["Auto", "Light", "Dark"],
            help="Theme preference (requires browser support)"
        )
        
        # Quick web search toggle
        web_search_enabled = st.checkbox(
            "🌐 Enable Web Search",
            value=st.session_state.get('user_preferences', {}).get('web_search', 'auto') != 'disabled',
            help="Enable or disable web search functionality"
        )
        
        # Quick safety level
        safety_level = st.selectbox(
            "🛡️ Safety Level",
            ["Relaxed", "Standard", "Strict"],
            index=1,
            help="Content filtering strictness"
        )
        
        # Update preferences
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {}
        
        st.session_state.user_preferences.update({
            'web_search': 'auto' if web_search_enabled else 'disabled',
            'safety_level': safety_level.lower(),
            'theme_mode': theme_mode.lower()
        })

def display_metrics_dashboard(rag_pipeline):
    """Enhanced metrics dashboard with guardrails information"""
    st.markdown("## 📊 System Metrics Dashboard")
    
    # Get system health
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        health_data = loop.run_until_complete(rag_pipeline.get_system_health())
        loop.close()
        
        # System overview cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            status = health_data.get("status", "unknown")
            status_emoji = {"healthy": "🟢", "warning": "🟡", "error": "🔴"}.get(status, "⚪")
            st.markdown(f"""
            <div class="metric-card">
                <h3>{status_emoji} System Status</h3>
                <p class="status-{status}">{status.title()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            vector_info = health_data.get("vector_store", {})
            doc_count = vector_info.get("total_documents", 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>📄 Documents</h3>
                <p>{doc_count}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            perf_info = health_data.get("recent_performance", {})
            success_rate = perf_info.get("success_rate", 0) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>✅ Success Rate</h3>
                <p>{success_rate:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            blocked_queries = perf_info.get("blocked_queries", 0)
            block_rate = perf_info.get("block_rate", 0) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>🛡️ Blocked Queries</h3>
                <p>{blocked_queries} ({block_rate:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            guardrails_violations = perf_info.get("guardrails_violations", 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>⚠️ Violations</h3>
                <p>{guardrails_violations}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Guardrails Status Section
        st.markdown("### 🛡️ Guardrails Status")
        
        components = health_data.get("components", {})
        guardrails_status = components.get("guardrails", "unavailable")
        guardrails_details = components.get("guardrails_details", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_emoji = "✅" if guardrails_status == "active" else "❌"
            st.markdown(f"""
            <div class="metric-card">
                <h4>🛡️ Guardrails System</h4>
                <p>{status_emoji} {guardrails_status.title()}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            pattern_count = guardrails_details.get("custom_patterns", 0)
            st.markdown(f"""
            <div class="metric-card">
                <h4>📋 Detection Patterns</h4>
                <p>{pattern_count}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            nemo_available = guardrails_details.get("nemo_available", False)
            nemo_emoji = "✅" if nemo_available else "❌"
            st.markdown(f"""
            <div class="metric-card">
                <h4>🤖 NeMo Guardrails</h4>
                <p>{nemo_emoji} {"Available" if nemo_available else "Unavailable"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed guardrails statistics
        if hasattr(rag_pipeline, 'get_guardrails_stats'):
            guardrails_stats = rag_pipeline.get_guardrails_stats()
            
            if guardrails_stats.get("status") == "active":
                st.markdown("### 📊 Guardrails Details")
                
                # Violation breakdown
                violation_types = guardrails_stats.get("violation_types", {})
                if violation_types:
                    st.markdown("#### 🔍 Violation Types")
                    
                    # Create a simple bar chart using st.columns
                    for vtype, count in violation_types.items():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{vtype.replace('_', ' ').title()}**")
                        with col2:
                            st.write(f"{count} violations")
                        
                        # Simple progress bar
                        max_violations = max(violation_types.values()) if violation_types else 1
                        progress = count / max_violations
                        st.progress(progress)
                
                # Guardrails controls
                st.markdown("#### 🎛️ Guardrails Controls")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🧹 Clear Violation Log"):
                        rag_pipeline.clear_violation_log()
                        st.success("✅ Violation log cleared")
                        st.rerun()
                
                with col2:
                    current_status = guardrails_stats.get("status") == "active"
                    if st.button(f"{'🔴 Disable' if current_status else '🟢 Enable'} Guardrails"):
                        rag_pipeline.toggle_guardrails(not current_status)
                        st.success(f"✅ Guardrails {'disabled' if current_status else 'enabled'}")
                        st.rerun()
                
                with col3:
                    if st.button("📊 Refresh Stats"):
                        st.rerun()
        
        # Performance metrics
        if perf_info.get("total_operations", 0) > 0:
            st.markdown("### 📈 Performance Details")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_queries = perf_info.get("total_queries", 0)
                st.metric("🔄 Total Queries", total_queries)
            
            with col2:
                successful_queries = perf_info.get("successful_queries", 0)
                st.metric("✅ Successful", successful_queries)
            
            with col3:
                blocked_queries = perf_info.get("blocked_queries", 0)
                st.metric("🛡️ Blocked", blocked_queries)
            
            with col4:
                uptime_hours = perf_info.get("uptime_hours", 0)
                st.metric("⏱️ Uptime (hrs)", f"{uptime_hours:.1f}")
        
        # Component health
        st.markdown("### 🔧 Component Health")
        components = health_data.get("components", {})
        
        if components:
            cols = st.columns(min(len(components), 4))
            for i, (component, status) in enumerate(components.items()):
                with cols[i % 4]:
                    if component == "guardrails":
                        status_emoji = "🛡️" if status == "active" else "❌"
                    elif component == "llm_client":
                        status_emoji = "🤖" if status in ["available", "ready"] else "❌"
                    elif component == "web_search":
                        status_emoji = "🌐" if status == "available" else "❌"
                    else:
                        status_emoji = "✅" if status in ["active", "connected", "available"] else "❌"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{component.replace('_', ' ').title()}</h4>
                        <p>{status_emoji} {status}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Failed to load metrics: {str(e)}")
        st.markdown("### 🔧 Troubleshooting")
        st.markdown("""
        **If metrics are not loading:**
        1. Check that the RAG pipeline is properly initialized
        2. Verify that all components are running
        3. Check the logs for detailed error messages
        4. Try refreshing the page
        """)


def document_management_tab(doc_processor, vector_store):
    """Enhanced document management with finer controls"""
    st.markdown("## 📁 Document Management")
    
    # Document upload section
    st.markdown("### 📤 Upload Documents")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'xlsx', 'json', 'csv', 'md'],
            help="Supported formats: PDF, DOCX, TXT, XLSX, JSON, CSV, MD"
        )
    
    with col2:
        if uploaded_files:
            st.markdown("#### Upload Options")
            replace_existing = st.checkbox("Replace existing files", value=False)
            process_immediately = st.checkbox("Process immediately", value=True)
    
    if uploaded_files and st.button("📤 Upload Files", type="primary"):
        upload_progress = st.progress(0)
        status_text = st.empty()
        
        uploaded_count = 0
        for i, uploaded_file in enumerate(uploaded_files):
            file_path = Path(f"data/documents/{uploaded_file.name}")
            
            # Check if file exists
            if file_path.exists() and not replace_existing:
                st.warning(f"⚠️ Skipped {uploaded_file.name} (already exists)")
                continue
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            uploaded_count += 1
            status_text.text(f"Uploaded {uploaded_file.name}")
            upload_progress.progress((i + 1) / len(uploaded_files))
        
        if uploaded_count > 0:
            st.success(f"✅ Successfully uploaded {uploaded_count} files")
            
            if process_immediately:
                with st.spinner("Processing uploaded documents..."):
                    documents = doc_processor.load_documents("data/documents")
                    chunks = doc_processor.split_documents(documents)
                    vector_store.add_documents(chunks)
                    st.success(f"✅ Processed {len(documents)} documents into {len(chunks)} chunks")
        
        st.rerun()
    
    # Document list and management
    st.markdown("### 📋 Document Library")
    
    documents = get_document_list()
    
    if not documents:
        st.info("📭 No documents found. Upload some documents to get started.")
        return
    
    # Document filters and search
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input("🔍 Search documents", placeholder="Search by filename...")
    
    with col2:
        file_type_filter = st.selectbox(
            "📄 File Type",
            ["All"] + list(set(doc["type"] for doc in documents))
        )
    
    with col3:
        sort_by = st.selectbox("📊 Sort by", ["Modified", "Name", "Size"])
    
    # Filter documents
    filtered_docs = documents
    if search_term:
        filtered_docs = [doc for doc in filtered_docs if search_term.lower() in doc["name"].lower()]
    
    if file_type_filter != "All":
        filtered_docs = [doc for doc in filtered_docs if doc["type"] == file_type_filter]
    
    # Sort documents
    if sort_by == "Name":
        filtered_docs.sort(key=lambda x: x["name"])
    elif sort_by == "Size":
        filtered_docs.sort(key=lambda x: x["size"], reverse=True)
    
    # Bulk operations
    st.markdown("#### 🔧 Bulk Operations")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Reprocess All", help="Reprocess all documents"):
            with st.spinner("Reprocessing all documents..."):
                documents = doc_processor.load_documents("data/documents")
                chunks = doc_processor.split_documents(documents)
                vector_store.add_documents(chunks)
                st.success(f"✅ Reprocessed {len(documents)} documents")
    
    with col2:
        if st.button("🗑️ Clear Vector Store", help="Clear all indexed documents"):
            if st.button("⚠️ Confirm Clear Vector Store", key="confirm_clear_vector"):
                try:
                    if hasattr(vector_store, 'clear_collection'):
                        vector_store.clear_collection()
                        st.success("✅ Vector store cleared")
                    elif hasattr(vector_store, 'clear'):
                        vector_store.clear()
                        st.success("✅ Vector store cleared")
                    else:
                        # Fallback: recreate vector store
                        st.warning("⚠️ Using fallback clear method")
                        vector_store = VectorStore()
                        st.success("✅ Vector store recreated")
                    
                    # Reset stats
                    st.session_state.vector_stats = {"total_documents": 0}
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing vector store: {str(e)}")
                    logger.error(f"Error clearing vector store: {e}")
    
    with col3:
        if st.button("📊 Update Stats", help="Refresh document statistics"):
            stats = vector_store.get_collection_stats()
            st.session_state.vector_stats = stats
            st.success("✅ Statistics updated")
    
    with col4:
        if st.button("🗂️ Export Document List", help="Export document list as CSV"):
            df = pd.DataFrame(documents)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Individual document management
    st.markdown("#### 📚 Document Details")
    
    # Document selection for operations
    selected_docs = []
    
    for i, doc in enumerate(filtered_docs):
        with st.expander(f"📄 {doc['name']} ({format_file_size(doc['size'])})", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="doc-item">
                    <strong>📁 Path:</strong> {doc['path']}<br>
                    <strong>📏 Size:</strong> {format_file_size(doc['size'])}<br>
                    <strong>🕐 Modified:</strong> {doc['modified'].strftime('%Y-%m-%d %H:%M:%S')}<br>
                    <strong>📋 Type:</strong> {doc['type'].upper()}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Individual document controls
                if st.button(f"🔄 Reprocess", key=f"reprocess_{i}"):
                    with st.spinner(f"Reprocessing {doc['name']}..."):
                        try:
                            # Load and process single document
                            if hasattr(doc_processor, 'load_single_document'):
                                single_doc = doc_processor.load_single_document(doc['path'])
                                if single_doc:
                                    chunks = doc_processor.split_documents([single_doc])
                                    vector_store.add_documents(chunks)
                                    st.success(f"✅ Reprocessed {doc['name']}")
                                else:
                                    st.error(f"❌ Failed to load {doc['name']}")
                            else:
                                st.error("❌ Single document processing not available")
                        except Exception as e:
                            st.error(f"❌ Error reprocessing: {str(e)}")
                            logger.error(f"Error reprocessing {doc['name']}: {e}")
                
                if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                    if st.button(f"⚠️ Confirm Delete {doc['name']}", key=f"confirm_delete_{i}"):
                        try:
                            os.remove(doc['path'])
                            st.success(f"✅ Deleted {doc['name']}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error deleting: {str(e)}")
                
                # Document selection for chat
                include_in_chat = st.checkbox(
                    "Include in chat",
                    value=True,
                    key=f"include_{i}",
                    help="Include this document in chat responses"
                )
                
                if include_in_chat:
                    selected_docs.append(doc['name'])
    
    # Save selected documents to session state
    st.session_state.selected_documents = selected_docs
    
    # Display selection summary
    if selected_docs:
        st.success(f"✅ {len(selected_docs)} documents selected for chat")
    else:
        st.warning("⚠️ No documents selected - chat will use web search only")

def settings_tab():
    """Enhanced settings with better organization"""
    st.markdown("## ⚙️ System Configuration")
    
    # Settings organization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛡️ Safety & Content")
        with st.container():
            st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
            
            safety_level = st.selectbox(
                "Safety Level",
                ["Relaxed", "Standard", "Strict"],
                index=1,
                help="Controls content filtering strictness"
            )
            
            content_filter = st.multiselect(
                "Content Filters",
                ["Profanity", "Violence", "Adult Content", "Spam"],
                default=["Profanity", "Violence"] if safety_level != "Relaxed" else [],
                help="Select content filters to apply"
            )
            
            response_style = st.selectbox(
                "Response Style",
                ["Concise", "Balanced", "Detailed"],
                index=1,
                help="Default response length and detail level"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 📚 Document Settings")
        with st.container():
            st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
            
            retrieval_k = st.slider(
                "Documents to Retrieve",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of document chunks to retrieve per query"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="Minimum confidence score for document relevance"
            )
            
            chunk_size = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=500,
                help="Size of document chunks for processing"
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=200,
                value=50,
                help="Overlap between consecutive chunks"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🌐 Web Search Settings")
        with st.container():
            st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
            
            web_search_mode = st.selectbox(
                "Web Search Mode",
                ["Disabled", "Fallback Only", "Always On", "Auto"],
                index=3,
                help="When to use web search"
            )
            
            search_engines = st.multiselect(
                "Search Engines",
                ["Google", "Bing", "DuckDuckGo", "Brave"],
                default=["Google", "Bing"],
                help="Available search engines (in order of preference)"
            )
            
            max_web_results = st.slider(
                "Max Web Results",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of web results to retrieve"
            )
            
            web_timeout = st.slider(
                "Web Search Timeout (seconds)",
                min_value=5,
                max_value=30,
                value=15,
                help="Timeout for web search requests"
            )
            
            enable_fallback = st.checkbox(
                "Enable Search Fallback",
                value=True,
                help="Try alternative search engines on failure"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🎨 Interface Settings")
        with st.container():
            st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
            
            theme_preference = st.selectbox(
                "Theme Preference",
                ["Auto", "Light", "Dark"],
                help="Visual theme preference"
            )
            
            auto_save = st.checkbox(
                "Auto-save Chat History",
                value=True,
                help="Automatically save chat history"
            )
            
            show_sources = st.checkbox(
                "Show Sources by Default",
                value=True,
                help="Expand sources section by default"
            )
            
            enable_animations = st.checkbox(
                "Enable Animations",
                value=True,
                help="Enable UI animations and transitions"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Save settings
    st.session_state.user_preferences = {
        "safety_level": safety_level.lower(),
        "content_filters": content_filter,
        "response_style": response_style.lower(),
        "retrieval_k": retrieval_k,
        "confidence_threshold": confidence_threshold,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "web_search_mode": web_search_mode.lower().replace(" ", "_"),
        "search_engines": search_engines,
        "max_web_results": max_web_results,
        "web_timeout": web_timeout,
        "enable_fallback": enable_fallback,
        "theme_preference": theme_preference.lower(),
        "auto_save": auto_save,
        "show_sources": show_sources,
        "enable_animations": enable_animations
    }
    
    # Settings management
    st.markdown("### 💾 Settings Management")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export Settings"):
            settings_json = json.dumps(st.session_state.user_preferences, indent=2)
            st.download_button(
                label="💾 Download Settings",
                data=settings_json,
                file_name=f"rag_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_settings = st.file_uploader(
            "📤 Import Settings",
            type=['json'],
            key="settings_upload"
        )
        
        if uploaded_settings:
            try:
                settings = json.load(uploaded_settings)
                st.session_state.user_preferences.update(settings)
                st.success("✅ Settings imported successfully")
            except Exception as e:
                st.error(f"❌ Error importing settings: {str(e)}")
    
    with col3:
        if st.button("🔄 Reset to Defaults"):
            if st.button("⚠️ Confirm Reset", key="confirm_reset_settings"):
                # Reset to default preferences
                st.session_state.user_preferences = {
                    "safety_level": "standard",
                    "web_search_mode": "auto",
                    "retrieval_k": 5,
                    "confidence_threshold": 0.7
                }
                st.success("✅ Settings reset to defaults")
                st.rerun()

def clear_all_data():
    """Clear all system data with confirmation"""
    if st.session_state.get('show_clear_confirmation', False):
        st.markdown("### ⚠️ Clear All Data")
        st.warning("This action will permanently delete:")
        st.markdown("""
        - 📄 All uploaded documents
        - 🔍 Vector store index
        - 💬 Chat history
        - 📊 System metrics
        - ⚙️ Settings (optional)
        """)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            include_settings = st.checkbox("Include Settings", value=False)
        
        with col2:
            if st.button("✅ Confirm Clear All", type="primary"):
                try:
                    # Clear documents
                    if os.path.exists("data/documents"):
                        shutil.rmtree("data/documents")
                        os.makedirs("data/documents", exist_ok=True)
                    
                    # Clear vector store
                    if os.path.exists("vector_db"):
                        shutil.rmtree("vector_db")
                        os.makedirs("vector_db", exist_ok=True)
                    
                    # Clear chat history
                    if os.path.exists("chat_history.json"):
                        os.remove("chat_history.json")
                    
                    # Clear session state
                    st.session_state.chat_history = []
                    st.session_state.vector_stats = {"total_documents": 0}
                    st.session_state.selected_documents = []
                    
                    if include_settings:
                        st.session_state.user_preferences = {}
                    
                    st.session_state.show_clear_confirmation = False
                    st.success("✅ All data cleared successfully")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error clearing data: {str(e)}")
        
        with col3:
            if st.button("❌ Cancel"):
                st.session_state.show_clear_confirmation = False
                st.rerun()

def enhanced_chat_interface(rag_pipeline):
    """Enhanced chat interface with better controls"""
    st.markdown("## 💬 Chat Interface")
    
    # Chat controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 🎛️ Chat Controls")
    
    with col2:
        # Document selection summary
        selected_docs = st.session_state.get('selected_documents', [])
        if selected_docs:
            st.success(f"📚 {len(selected_docs)} docs selected")
        else:
            st.warning("⚠️ No docs selected")
    
    with col3:
        # Web search status
        web_mode = st.session_state.get('user_preferences', {}).get('web_search_mode', 'auto')
        web_emoji = {"disabled": "🚫", "fallback_only": "🔄", "always_on": "🌐", "auto": "🤖"}
        st.info(f"{web_emoji.get(web_mode, '🤖')} Web: {web_mode.replace('_', ' ').title()}")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    
    # Initialize user preferences if not set
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            "web_search_mode": "auto",
            "safety_level": "standard",
            "response_style": "balanced",
            "retrieval_k": 5,
            "show_sources": True
        }
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        for i, chat in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f"""
            <div class="chat-message">
                <strong>👤 You:</strong><br>
                {chat['user_message']}
            </div>
            """, unsafe_allow_html=True)
            
            # Assistant response
            st.markdown(f"""
            <div class="chat-message">
                <strong>🤖 Assistant:</strong><br>
                {chat['response']}
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced source display
            if chat.get('sources'):
                show_sources = st.session_state.user_preferences.get('show_sources', True)
                
                with st.expander("📚 Sources & Details", expanded=show_sources):
                    # Quick metrics
                    metrics = chat.get('metrics', {})
                    if metrics:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            doc_conf = metrics.get('doc_confidence', 0)
                            conf_color = "🟢" if doc_conf > 0.7 else "🟡" if doc_conf > 0.5 else "🔴"
                            st.metric("📄 Doc Confidence", f"{conf_color} {doc_conf:.2f}")
                        
                        with col2:
                            sources_found = metrics.get('sources_found', 0)
                            st.metric("🔍 Sources Found", sources_found)
                        
                        with col3:
                            web_used = "✅" if chat.get('web_search_used') else "❌"
                            st.metric("🌐 Web Search", web_used)
                        
                        with col4:
                            quality = chat.get('context_quality', 'unknown')
                            quality_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(quality, "⚪")
                            st.metric("📊 Quality", f"{quality_emoji} {quality}")
                    
                    # Document sources
                    doc_sources = chat['sources'].get('documents', [])
                    if doc_sources:
                        st.markdown("#### 📄 Document Sources")
                        for j, doc in enumerate(doc_sources[:3]):
                            confidence_color = "🟢" if doc['score'] > 0.7 else "🟡" if doc['score'] > 0.5 else "🔴"
                            st.markdown(f"""
                            <div class="source-item">
                                <strong>{confidence_color} {j+1}. {doc['metadata'].get('filename', 'Unknown')}</strong> 
                                (Confidence: {doc['score']:.2f})<br>
                                <em>{doc['content'][:200]}...</em>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Web sources
                    web_sources = chat['sources'].get('web_results', [])
                    if web_sources:
                        st.markdown("#### 🌐 Web Sources")
                        web_info = chat.get('web_search_info', {})
                        
                        if web_info.get('engine_used') != 'none':
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Search Engine:** {web_info.get('engine_used', 'Unknown')}")
                            with col2:
                                if web_info.get('fallback_attempts', 0) > 0:
                                    st.write(f"**Fallback Attempts:** {web_info['fallback_attempts']}")
                        
                        for j, result in enumerate(web_sources):
                            confidence_color = "🟢" if result.get('confidence', 0.5) > 0.7 else "🟡"
                            st.markdown(f"""
                            <div class="source-item">
                                <strong>{confidence_color} {j+1}. {result['title']}</strong><br>
                                <em>{result['content'][:200]}...</em><br>
                                <a href="{result.get('url', '#')}" target="_blank">🔗 Source</a>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Guardrails warnings and violations
                    warnings = chat.get('guardrails_warnings', [])
                    violations = chat.get('guardrails_violations', [])
                    
                    if warnings or violations:
                        st.markdown("#### 🛡️ Guardrails Information")
                        
                        if violations:
                            st.markdown("**Violations Detected:**")
                            for violation in violations:
                                severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(violation.get('severity', 'medium'), "⚪")
                                st.warning(f"{severity_emoji} {violation.get('type', 'Unknown')}: {violation.get('pattern', 'Pattern match')}")
                        
                        if warnings:
                            st.markdown("**Warnings:**")
                            for warning in warnings:
                                st.info(warning)
                        
                        # Show if query was blocked
                        if chat.get('blocked', False):
                            st.error("🚫 This query was blocked by safety guardrails")
                    
                    # Legacy guardrails warnings (for backward compatibility)
                    legacy_warnings = chat.get('guardrails_warnings', [])
                    if legacy_warnings and not warnings and not violations:
                        st.markdown("#### ⚠️ Safety Notices")
                        for warning in legacy_warnings:
                            st.warning(warning)
            
            st.divider()
    
    # Chat input
    user_input = st.chat_input("Ask a question about your documents or any topic...")
    
    if user_input:
        # Process the query
        with st.spinner("🤔 Processing your question..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Modify preferences based on selected documents
                current_prefs = st.session_state.user_preferences.copy()
                
                # If no documents selected, force web search
                if not st.session_state.get('selected_documents'):
                    current_prefs['web_search_mode'] = 'always_on'
                
                result = loop.run_until_complete(
                    rag_pipeline.process_query(
                        user_input,
                        user_preferences=current_prefs,
                        selected_documents=st.session_state.get('selected_documents', []),
                        k=current_prefs.get('retrieval_k', 5)
                    )
                )
                
                # Save to chat history
                chat_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'user_message': user_input,
                    'response': result['answer'],
                    'sources': result['sources'],
                    'context_quality': result['context_quality'],
                    'web_search_used': result.get('web_search_used', False),
                    'web_search_info': result.get('web_search_info', {}),
                    'metrics': result.get('metrics', {}),
                    'guardrails_warnings': result.get('guardrails_warnings', []),
                    'selected_documents': st.session_state.get('selected_documents', [])
                }
                
                st.session_state.chat_history.append(chat_entry)
                
                # Auto-save if enabled
                if st.session_state.user_preferences.get('auto_save', True):
                    save_chat_history(st.session_state.chat_history)
                
                # Log successful interaction
                logger.info(f"Processed query: '{user_input[:50]}...' - Quality: {result['context_quality']}")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing your request: {str(e)}")
                logger.error(f"Error processing query '{user_input}': {str(e)}")
            
            finally:
                loop.close()
    
    # Chat management
    st.markdown("### 🔧 Chat Management")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📥 Export Chat", help="Export chat history as JSON"):
            if st.session_state.chat_history:
                chat_data = json.dumps(st.session_state.chat_history, indent=2, default=str)
                st.download_button(
                    label="💾 Download JSON",
                    data=chat_data,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("No chat history to export")
    
    with col2:
        if st.button("📊 Chat Stats", help="Show chat statistics"):
            if st.session_state.chat_history:
                total_chats = len(st.session_state.chat_history)
                web_searches = sum(1 for chat in st.session_state.chat_history if chat.get('web_search_used'))
                avg_quality = sum(1 for chat in st.session_state.chat_history if chat.get('context_quality') == 'high') / total_chats
                
                st.markdown(f"""
                <div class="info-card">
                    <h4>📊 Chat Statistics</h4>
                    <p><strong>Total Chats:</strong> {total_chats}</p>
                    <p><strong>Web Searches:</strong> {web_searches} ({(web_searches/total_chats)*100:.0f}%)</p>
                    <p><strong>High Quality:</strong> {avg_quality*100:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No chat history available")
    
    with col3:
        if st.button("🔄 Refresh Chat", help="Refresh chat interface"):
            st.rerun()
    
    with col4:
        if st.button("🗑️ Clear Chat", help="Clear all chat history"):
            if st.button("⚠️ Confirm Clear Chat", key="confirm_clear_chat"):
                st.session_state.chat_history = []
                save_chat_history([])
                st.success("✅ Chat history cleared")
                st.rerun()

def main():
    """Main application with enhanced UI and configuration checking"""
    # Load custom CSS
    load_custom_css()
    
    # Load environment variables
    try:
        from src.config_helper import load_env_file, check_configuration, display_configuration_help, show_configuration_sidebar
        load_env_file()
    except ImportError:
        st.warning("⚠️ Configuration helper not available. Please ensure all modules are properly installed.")
    
    # Check configuration
    try:
        config_status = check_configuration()
    except:
        config_status = {"overall_status": "unknown"}
    
    # App header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🤖 Enhanced RAG Chatbot</h1>
        <p style="font-size: 1.2em; color: var(--secondary-color);">
            AI-powered document Q&A with advanced controls and safety features
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show configuration status
    if config_status.get("overall_status") != "ready":
        st.warning("⚠️ System configuration incomplete. Some features may not work properly.")
        if st.button("🔧 Show Configuration Guide"):
            st.session_state.show_config_help = True
    
    # Configuration help modal
    if st.session_state.get('show_config_help', False):
        try:
            display_configuration_help()
        except:
            st.error("Configuration helper not available")
        
        if st.button("✅ Close Configuration Guide"):
            st.session_state.show_config_help = False
            st.rerun()
        return
    
    # Initialize system
    try:
        doc_processor, vector_store, rag_pipeline = initialize_system()
        
        # Update vector stats
        if 'vector_stats' not in st.session_state:
            st.session_state.vector_stats = vector_store.get_collection_stats()
        
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {str(e)}")
        st.markdown("### 🔧 Troubleshooting")
        st.markdown("""
        **Common issues:**
        1. **Missing dependencies**: Install required packages
        2. **Configuration**: Check environment variables
        3. **File permissions**: Ensure write access to data directories
        
        **Quick fixes:**
        - Restart the application
        - Check the logs for detailed error messages
        - Verify your UFL AI configuration
        """)
        return
    
    # Sidebar controls
    try:
        show_configuration_sidebar()
    except:
        pass
    
    sidebar_controls()
    
    # Handle clear all confirmation
    if st.session_state.get('show_clear_confirmation', False):
        clear_all_data()
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📁 Documents", "📊 Metrics", "⚙️ Settings"])
    
    with tab1:
        enhanced_chat_interface(rag_pipeline)
    
    with tab2:
        document_management_tab(doc_processor, vector_store)
    
    with tab3:
        display_metrics_dashboard(rag_pipeline)
    
    with tab4:
        settings_tab()
    
    # Footer with system info
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("ℹ️ System Info"):
            st.session_state.show_system_info = not st.session_state.get('show_system_info', False)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; color: var(--secondary-color);">
            <p>Enhanced RAG Chatbot v2.0 | Built with Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Quick status indicator
        if config_status.get("overall_status") == "ready":
            st.success("🟢 Ready")
        else:
            st.error("🔴 Setup Needed")
    
    # System info display
    if st.session_state.get('show_system_info', False):
        try:
            from src.config_helper import get_environment_info
            info = get_environment_info()
            
            st.markdown("### 🔍 System Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Environment:**")
                st.code(f"Python: {info['python_version']}")
                st.code(f"Platform: {info['platform']}")
                st.code(f"Streamlit: {info['streamlit_version']}")
            
            with col2:
                st.markdown("**Configuration:**")
                for var, status in info['environment_variables'].items():
                    st.code(f"{var}: {status}")
                    
        except Exception as e:
            st.error(f"Could not load system info: {e}")

if __name__ == "__main__":
    main()