# 🤖 Enhanced RAG Chatbot

*AI-powered document Q&A with advanced controls and safety features*

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)]()

A sophisticated Retrieval-Augmented Generation (RAG) chatbot that combines document analysis, web search, and advanced AI safety measures to provide accurate, contextual responses while maintaining the highest standards of content safety and user experience.

## ✨ Key Features

### 🔍 **Intelligent Document Processing**
- **Multi-format Support**: PDF, DOCX, TXT, XLSX, JSON, CSV, Markdown
- **Advanced Text Chunking**: Optimized for context preservation
- **Vector Storage**: ChromaDB with sentence transformers for semantic search
- **Selective Processing**: Choose specific documents for targeted queries

### 🌐 **Hybrid Information Retrieval**
- **Smart Web Search**: DuckDuckGo integration with fallback mechanisms
- **Context-Aware Decisions**: Automatically determines when to use web search
- **Multi-Engine Support**: Built-in fallback to multiple search engines
- **Quality Assessment**: Intelligent evaluation of search result relevance

### 🛡️ **Advanced Safety & Guardrails**
- **Multi-Layer Protection**: Custom patterns + NeMo Guardrails integration
- **Comprehensive Detection**: Profanity, hate speech, violence, illegal content
- **Anti-Jailbreak**: Advanced protection against prompt injection attacks
- **Multi-Prompt Detection**: Identifies disguised multi-topic requests
- **Output Sanitization**: Real-time response filtering and content masking

### 🎨 **Professional Interface**
- **Modern UI**: Dark/light themes with smooth animations
- **Real-time Metrics**: Performance monitoring and system health dashboard
- **Interactive Controls**: Granular settings for personalized experience
- **Mobile Responsive**: Optimized for all device sizes

### 📊 **Enterprise-Grade Monitoring**
- **Comprehensive Metrics**: Query success rates, response times, cost tracking
- **Violation Logging**: Detailed security event tracking
- **System Health**: Component status monitoring and diagnostics
- **Usage Analytics**: Token consumption and cost analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- UFL AI access credentials
- 4GB RAM minimum (8GB recommended)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/enhanced-rag-chatbot.git
   cd enhanced-rag-chatbot
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   
   Create a `.env` file in the project root:
   ```env
   # Required: UFL AI Configuration
   UFL_AI_BASE_URL=https://your-ufl-ai-endpoint.com
   UFL_AI_MODEL=your-model-name
   UFL_AI_API_KEY=your-api-key
   
   # Optional: Web Search Enhancement
   SERPER_API_KEY=your-serper-key  # For Google search via serper.dev
   SEARCH_API_KEY=your-search-key  # Alternative search API
   ```

4. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

5. **Access the Interface**
   
   Open your browser to `http://localhost:8501`

## 📖 Usage Guide

### 🗂️ **Document Management**

**Upload Documents**
- Drag and drop files or use the upload interface
- Supported formats: PDF, DOCX, TXT, XLSX, JSON, CSV, MD
- Automatic processing and indexing
- Real-time status updates

**Document Organization**
- Search and filter documents by name or type
- Selective inclusion in chat sessions
- Bulk operations for efficiency
- Export capabilities for documentation

### 💬 **Chat Interface**

**Query Processing**
- Natural language questions about your documents
- Automatic context retrieval from relevant sources
- Web search integration for current information
- Multi-source citation and verification

**Response Quality**
- Confidence scoring for document matches
- Source attribution with clickable links
- Context quality assessment
- Expandable source details

### ⚙️ **Settings & Configuration**

**Safety Controls**
- Adjustable safety levels (Relaxed/Standard/Strict)
- Content filter customization
- Response style preferences
- Real-time guardrails monitoring

**Search Preferences**
- Web search modes (Auto/Always/Fallback/Disabled)
- Search engine priority settings
- Result quantity and timeout controls
- Quality threshold adjustments

**Interface Customization**
- Theme preferences
- Animation controls
- Auto-save settings
- Display options

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                   │
├─────────────────────────────────────────────────────────┤
│  📁 Document Management  │  💬 Chat Interface          │
│  📊 Metrics Dashboard    │  ⚙️  Settings Panel         │
└─────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                        │
├─────────────────────────────────────────────────────────┤
│  🛡️  Guardrails Manager  │  🔍 Vector Store            │
│  🌐 Web Searcher         │  🤖 UFL LLM Client          │
│  📈 Metrics Logger       │  📝 Document Processor      │
└─────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────┐
│                 External Services                      │
├─────────────────────────────────────────────────────────┤
│  🔗 UFL AI API           │  🌐 Search APIs              │
│  📦 ChromaDB             │  🔒 NeMo Guardrails          │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input Processing**: User query → Guardrails check → Intent analysis
2. **Information Retrieval**: Document search + Web search (if needed)
3. **Response Generation**: Context preparation → LLM generation → Output filtering
4. **Quality Assurance**: Response validation → Source citation → Metrics logging

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `UFL_AI_BASE_URL` | ✅ | UFL AI service endpoint | `https://api.ufl-ai.com` |
| `UFL_AI_MODEL` | ✅ | Model identifier | `llama-3.3-70b` |
| `UFL_AI_API_KEY` | ✅ | Authentication token | `sk-...` |
| `SERPER_API_KEY` | ❌ | Google search via Serper | `abc123...` |
| `SEARCH_API_KEY` | ❌ | Alternative search API | `def456...` |

### System Requirements

**Minimum**
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB
- Network: Stable internet connection

**Recommended**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+ SSD
- Network: High-speed broadband

## 🛡️ Safety & Security

### Guardrails System

The application implements a comprehensive multi-layer safety system:

**Input Filtering**
- Profanity detection with leetspeak and unicode variants
- Jailbreak attempt identification
- Multi-prompt injection prevention
- Hate speech and violence detection

**Content Monitoring**
- Real-time violation logging
- Severity-based response protocols
- Automatic content sanitization
- User behavior analytics

**Output Validation**
- Response safety verification
- Sensitive information masking
- Factual accuracy checks
- Source credibility assessment

### Privacy Protection

- No data persistence beyond session scope
- Anonymized logging for safety events
- Configurable data retention policies
- GDPR-compliant data handling

## 📊 Monitoring & Analytics

### Real-time Metrics

**Performance Monitoring**
- Query processing times
- Success/failure rates
- Resource utilization
- Cost tracking

**Safety Analytics**
- Violation frequency and types
- Blocked query statistics
- Guardrails effectiveness
- User behavior patterns

**System Health**
- Component status monitoring
- API connectivity checks
- Database performance metrics
- Error rate tracking

### Dashboard Features

- Interactive charts and graphs
- Exportable reports
- Configurable alerts
- Historical trend analysis

## 🔍 Troubleshooting

### Common Issues

**🚫 UFL AI Connection Failed**
```bash
# Check environment variables
echo $UFL_AI_BASE_URL
echo $UFL_AI_MODEL

# Verify API key
curl -H "Authorization: Bearer $UFL_AI_API_KEY" $UFL_AI_BASE_URL/health
```

**📁 Document Processing Errors**
- Ensure file formats are supported
- Check file permissions and sizes
- Verify available disk space
- Review error logs in the interface

**🌐 Web Search Not Working**
- Check internet connectivity
- Verify search API keys (if used)
- Review search preferences in settings
- Check firewall/proxy settings

**🐛 Performance Issues**
- Monitor system resources (CPU/RAM)
- Clear browser cache
- Restart the application
- Check for conflicting processes

### Health Check

Run the built-in health check script:
```bash
python scripts/health_check.py
```

### Log Analysis

Application logs are stored in the `logs/` directory:
- `performance.log`: Performance metrics
- `usage.log`: Usage statistics
- `errors.log`: Error events

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

### Development Setup

1. **Fork the Repository**
2. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```
4. **Run Tests**
   ```bash
   pytest tests/
   ```
5. **Submit Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for public functions
- Maintain test coverage above 80%

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **LangChain** for document processing capabilities
- **ChromaDB** for vector storage and search
- **Sentence Transformers** for embedding generation
- **NeMo Guardrails** for advanced safety features
- **UFL AI** for language model services

## 📞 Support

### Getting Help

- 📖 Check the [documentation](docs/)
- 🐛 Report bugs via [GitHub Issues](https://github.com/your-username/enhanced-rag-chatbot/issues)
- 💬 Join our [Discord community](https://discord.gg/your-server)
- 📧 Email support: support@your-domain.com

### Commercial Support

For enterprise deployments and commercial support:
- 🏢 Enterprise consulting available
- 🔧 Custom integration services
- 📈 Scalability optimization
- 🛡️ Advanced security auditing

---

<div align="center">

**Built with ❤️ by [Sai Sri Harsha Guddati](https://github.com/your-username)**

⭐ Star this repo if you find it helpful!

</div>