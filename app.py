"""
RAG Chatbot Streamlit Application
A conversational AI interface for document question-answering using RAG architecture.
"""

import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
from typing import List, Dict, Any

# Import backend modules
from backend.embeddings import DocumentProcessor, create_vector_store
from backend.retriever import DocumentRetriever, HybridRetriever
from backend.generator import AnswerGenerator

# Load environment variables
load_dotenv()


class RAGChatbot:
    """Main RAG Chatbot application class."""
    
    def __init__(self):
        """Initialize the RAG chatbot."""
        self.setup_page_config()
        self.initialize_session_state()
        self.setup_api_key()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="RAG Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #667eea;
        }
        
        .user-message {
            background-color: #f0f2f6;
            border-left-color: #667eea;
        }
        
        .assistant-message {
            background-color: #e8f4fd;
            border-left-color: #1f77b4;
        }
        
        .source-info {
            background-color: #f9f9f9;
            padding: 0.5rem;
            border-radius: 5px;
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        
        if 'documents' not in st.session_state:
            st.session_state.documents = None
        
        if 'retriever' not in st.session_state:
            st.session_state.retriever = None
        
        if 'generator' not in st.session_state:
            st.session_state.generator = None
        
        if 'document_processed' not in st.session_state:
            st.session_state.document_processed = False
    
    def setup_api_key(self):
        """Setup OpenAI and OpenRouter API keys from environment or user input."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        
        # Filter out placeholder/invalid OpenAI keys
        if self.openai_api_key and (
            self.openai_api_key.startswith("your_") or
            self.openai_api_key == "your_openai_api_key_here" or
            len(self.openai_api_key) < 20
        ):
            self.openai_api_key = None
        
        # Check if at least one API key is available
        if not self.openai_api_key and not self.openrouter_api_key:
            st.sidebar.error("⚠️ No API Keys found!")
            st.sidebar.info("Please add at least one API key to continue.")
            st.sidebar.markdown("""
            **For FREE usage:**
            - Get OpenRouter API key at [openrouter.ai](https://openrouter.ai)
            - Use free models without payment required
            
            **For Premium usage:**
            - Get OpenAI API key for higher quality responses
            """)
            
            # API key inputs
            st.sidebar.subheader("🔑 API Configuration")
            
            openai_key_input = st.sidebar.text_input(
                "OpenAI API Key:",
                type="password",
                help="For OpenAI models (GPT-3.5, GPT-4)"
            )
            
            openrouter_key_input = st.sidebar.text_input(
                "OpenRouter API Key:",
                type="password", 
                help="For free models (Mistral, etc.)"
            )
            
            if openai_key_input:
                self.openai_api_key = openai_key_input
            if openrouter_key_input:
                self.openrouter_api_key = openrouter_key_input
            
            if not (openai_key_input or openrouter_key_input):
                st.stop()
            else:
                if openai_key_input:
                    st.sidebar.success("✅ OpenAI API Key provided!")
                if openrouter_key_input:
                    st.sidebar.success("✅ OpenRouter API Key provided!")
        else:
            if self.openai_api_key:
                st.sidebar.success("✅ OpenAI API Key loaded")
            if self.openrouter_api_key:
                st.sidebar.success("✅ OpenRouter API Key loaded")
    
    def render_header(self):
        """Render the application header."""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 RAG Chatbot</h1>
            <p>Upload documents and ask questions using AI-powered retrieval</p>
            <p style="font-size: 0.9em; opacity: 0.8;">Supports OpenAI & OpenRouter (Free Models)</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with file upload and settings."""
        st.sidebar.title("📁 Document Upload")
        
        # File upload
        uploaded_file = st.sidebar.file_uploader(
            "Choose a document",
            type=['pdf', 'txt'],
            help="Upload a PDF or text file to create a knowledge base"
        )
        
        if uploaded_file is not None and not st.session_state.document_processed:
            if st.sidebar.button("Process Document", type="primary"):
                self.process_document(uploaded_file)
        
        # Settings
        st.sidebar.title("⚙️ Settings")
        
        # Retrieval settings
        st.sidebar.subheader("Retrieval Settings")
        num_docs = st.sidebar.slider(
            "Number of documents to retrieve:",
            min_value=1,
            max_value=10,
            value=5,
            help="More documents provide more context but may slow response"
        )
        
        use_hybrid_search = st.sidebar.checkbox(
            "Use Hybrid Search",
            value=False,
            help="Combine semantic and keyword search for better results"
        )
        
        # Generation settings
        st.sidebar.subheader("Generation Settings")
        
        # Model provider selection
        available_providers = []
        if self.openai_api_key:
            available_providers.append("OpenAI")
        if self.openrouter_api_key:
            available_providers.append("OpenRouter (Free Models)")
        
        if len(available_providers) > 1:
            provider_choice = st.sidebar.selectbox(
                "Model Provider:",
                available_providers,
                help="Choose between OpenAI or OpenRouter (free models)"
            )
        elif len(available_providers) == 1:
            provider_choice = available_providers[0]
            st.sidebar.info(f"Using: {provider_choice}")
        else:
            st.sidebar.error("No API keys available!")
            st.stop()
        
        # Model selection based on provider
        if provider_choice == "OpenAI":
            model_choice = st.sidebar.selectbox(
                "OpenAI Model:",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
                index=0
            )
            selected_provider = "openai"
        else:  # OpenRouter
            model_choice = st.sidebar.selectbox(
                "OpenRouter Model:",
                [
                    "mistralai/mistral-7b-instruct:free",
                    "huggingfaceh4/zephyr-7b-beta:free", 
                    "openchat/openchat-7b:free",
                    "gryphe/mythomist-7b:free",
                    "nousresearch/nous-capybara-7b:free"
                ],
                index=0,
                help="Free models available through OpenRouter"
            )
            selected_provider = "openrouter"
        
        temperature = st.sidebar.slider(
            "Response Creativity:",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Higher values make responses more creative but less focused"
        )
        
        # Store settings in session state
        st.session_state.num_docs = num_docs
        st.session_state.use_hybrid_search = use_hybrid_search
        st.session_state.model_choice = model_choice
        st.session_state.temperature = temperature
        st.session_state.selected_provider = selected_provider
        
        # Update generator if settings changed
        self.update_generator_if_needed()
        
        # Document info
        if st.session_state.document_processed:
            st.sidebar.success("✅ Document processed!")
            # Show embedding type information
            if not self.openai_api_key and self.openrouter_api_key:
                st.sidebar.info("🆓 Using free local embeddings")
            elif self.openai_api_key:
                st.sidebar.info("🔑 Using OpenAI embeddings")
            
            if st.sidebar.button("Clear Document"):
                self.clear_document()
        
        # Export chat
        if st.session_state.messages:
            st.sidebar.subheader("💾 Export")
            
            # Export format selection
            export_format = st.sidebar.selectbox(
                "Export Format:",
                ["Markdown (.md)", "Plain Text (.txt)", "JSON (.json)"],
                help="Choose the format for your chat export"
            )
            
            # Generate export data based on format
            if export_format == "Markdown (.md)":
                export_data = self.generate_chat_export("markdown")
                file_extension = "md"
                mime_type = "text/markdown"
            elif export_format == "Plain Text (.txt)":
                export_data = self.generate_chat_export("text")
                file_extension = "txt"
                mime_type = "text/plain"
            else:  # JSON
                export_data = self.generate_chat_export("json")
                file_extension = "json"
                mime_type = "application/json"
            
            # Create download button
            st.sidebar.download_button(
                label="📥 Download Chat History",
                data=export_data,
                file_name=f"rag_chatbot_history_{st.session_state.get('export_timestamp', 'export')}.{file_extension}",
                mime=mime_type,
                help="Download your conversation in the selected format"
            )
            
            # Export preview
            if st.sidebar.expander("👁️ Preview Export"):
                if export_format == "JSON (.json)":
                    try:
                        import json
                        # Parse and limit JSON for preview
                        json_data = json.loads(export_data)
                        preview_data = {
                            "metadata": json_data.get("metadata", {}),
                            "messages": json_data.get("messages", [])[:2],  # Show only first 2 messages
                            "note": f"Showing first 2 messages. Full export contains {len(json_data.get('messages', []))} messages."
                        }
                        st.sidebar.json(preview_data)
                    except:
                        st.sidebar.text("JSON preview error")
                else:
                    preview_text = export_data[:400] + "..." if len(export_data) > 400 else export_data
                    st.sidebar.text_area("Preview:", preview_text, height=120, disabled=True)
            
            # Show export statistics
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            msgs_with_sources = len([m for m in st.session_state.messages if m.get("sources")])
            
            st.sidebar.caption(f"💬 {len(st.session_state.messages)} messages ({user_msgs} user, {assistant_msgs} assistant)")
            if msgs_with_sources > 0:
                st.sidebar.caption(f"📚 {msgs_with_sources} messages include source references")
            
            # Clear chat history button
            if st.sidebar.button("🗑️ Clear Chat History", help="Delete all chat messages"):
                st.session_state.messages = []
                st.rerun()
        
        # Debug info (only show if document is processed)
        if st.sidebar.expander("🔧 Debug Info"):
            valid_openai = (self.openai_api_key and 
                           len(self.openai_api_key) > 20 and 
                           not self.openai_api_key.startswith("your_"))
            
            st.sidebar.text(f"OpenAI Key: {'✅ Valid' if valid_openai else '❌ Invalid/Missing'}")
            st.sidebar.text(f"OpenRouter Key: {'✅' if self.openrouter_api_key else '❌'}")
            st.sidebar.text(f"Will use: {'🆓 Local Embeddings' if not valid_openai else '🔑 OpenAI Embeddings'}")
            
            if st.session_state.document_processed:
                st.sidebar.text(f"Provider: {st.session_state.get('selected_provider', 'None')}")
                st.sidebar.text(f"Model: {st.session_state.get('model_choice', 'None')}")
                if hasattr(st.session_state, 'generator') and st.session_state.generator:
                    st.sidebar.text(f"Generator Provider: {st.session_state.generator.provider}")
                    
                # Show embedding type used
                if hasattr(st.session_state, 'embeddings'):
                    if hasattr(st.session_state.embeddings, 'model'):  # LocalEmbeddings
                        st.sidebar.text("📍 Using: Local Embeddings")
                    else:  # OpenAI
                        st.sidebar.text("📍 Using: OpenAI Embeddings")
    
    def process_document(self, uploaded_file):
        """Process uploaded document and create vector store."""
        try:
            with st.spinner("Processing document... This may take a moment."):
                # Always use local embeddings if no valid OpenAI key
                valid_openai_key = (self.openai_api_key and 
                                  len(self.openai_api_key) > 20 and 
                                  not self.openai_api_key.startswith("your_"))
                
                use_local = not valid_openai_key
                
                if use_local:
                    st.info("🔄 Using free local embeddings (sentence-transformers)")
                    # Initialize document processor with local embeddings
                    processor = DocumentProcessor(use_local_embeddings=True)
                else:
                    st.info("🔑 Using OpenAI embeddings")
                    # Initialize document processor with OpenAI embeddings
                    processor = DocumentProcessor(self.openai_api_key)
                
                # Process the uploaded file
                documents = processor.process_uploaded_file(uploaded_file)
                
                # Create embeddings and vector store - the function will auto-detect and use local if needed
                index, texts, docs, embeddings = create_vector_store(
                    documents, 
                    self.openai_api_key, 
                    use_local_embeddings=use_local
                )
                
                # Store in session state
                st.session_state.vector_store = index
                st.session_state.documents = docs
                st.session_state.embeddings = embeddings
                
                # Initialize retriever
                if st.session_state.get('use_hybrid_search', False):
                    st.session_state.retriever = HybridRetriever(index, docs, embeddings)
                else:
                    st.session_state.retriever = DocumentRetriever(index, docs, embeddings)
                
                # Initialize generator
                # Determine provider based on available API keys
                if st.session_state.get('selected_provider'):
                    provider = st.session_state.get('selected_provider')
                elif not self.openai_api_key and self.openrouter_api_key:
                    provider = 'openrouter'
                elif self.openai_api_key:
                    provider = 'openai'
                else:
                    raise ValueError("No valid API key available for generator")
                
                # Set default model based on provider
                if provider == 'openrouter':
                    default_model = 'mistralai/mistral-7b-instruct:free'
                else:
                    default_model = 'gpt-3.5-turbo'
                
                st.session_state.generator = AnswerGenerator(
                    openai_api_key=self.openai_api_key,
                    openrouter_api_key=self.openrouter_api_key,
                    model_name=st.session_state.get('model_choice', default_model),
                    temperature=st.session_state.get('temperature', 0.1),
                    provider=provider
                )
                
                st.session_state.document_processed = True
                
                st.success(f"✅ Successfully processed {len(documents)} document chunks from {uploaded_file.name}")
                st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
    
    def update_generator_if_needed(self):
        """Update generator if model settings have changed."""
        if (st.session_state.document_processed and 
            st.session_state.generator is not None):
            
            # Determine current provider properly
            if st.session_state.get('selected_provider'):
                current_provider = st.session_state.get('selected_provider')
            elif not self.openai_api_key and self.openrouter_api_key:
                current_provider = 'openrouter'
            elif self.openai_api_key:
                current_provider = 'openai'
            else:
                return  # Can't update without valid API key
            
            # Set appropriate defaults based on provider
            if current_provider == 'openrouter':
                default_model = 'mistralai/mistral-7b-instruct:free'
            else:
                default_model = 'gpt-3.5-turbo'
            
            current_model = st.session_state.get('model_choice', default_model)
            current_temperature = st.session_state.get('temperature', 0.1)
            
            # Check if generator needs updating
            generator = st.session_state.generator
            if (generator.provider != current_provider or 
                generator.model_name != current_model or 
                generator.temperature != current_temperature):
                
                # Reinitialize generator with new settings
                st.session_state.generator = AnswerGenerator(
                    openai_api_key=self.openai_api_key,
                    openrouter_api_key=self.openrouter_api_key,
                    model_name=current_model,
                    temperature=current_temperature,
                    provider=current_provider
                )
    
    def clear_document(self):
        """Clear processed document and reset session state."""
        st.session_state.vector_store = None
        st.session_state.documents = None
        st.session_state.retriever = None
        st.session_state.generator = None
        st.session_state.document_processed = False
        st.session_state.messages = []
        st.success("Document cleared successfully!")
        st.rerun()
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        if not st.session_state.document_processed:
            st.info("👈 Please upload and process a document to start chatting!")
            
            # Show setup status
            if not self.openai_api_key and self.openrouter_api_key:
                st.success("🆓 **Free Mode Active!** Using OpenRouter models with local embeddings - no costs for embeddings or chat generation.")
            elif self.openai_api_key and self.openrouter_api_key:
                st.info("🔄 **Dual Mode Available!** You can choose between OpenAI (premium) and OpenRouter (free) models.")
            elif self.openai_api_key and not self.openrouter_api_key:
                st.info("💎 **Premium Mode!** Using OpenAI models with OpenAI embeddings.")
            
            return
        
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for message in st.session_state.messages:
                self.display_message(message)
        
        # Chat input
        if query := st.chat_input("Ask a question about your document..."):
            self.handle_user_query(query)
    
    def display_message(self, message: Dict[str, Any]):
        """Display a single chat message."""
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        
        else:  # assistant
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                
                # Display sources if available
                if "sources" in message:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"""
                            **Source {i+1}** (Score: {source['similarity_score']})
                            
                            {source['content_preview']}
                            """)
                
                # Display follow-up questions if available
                if "follow_ups" in message:
                    st.markdown("**Suggested follow-up questions:**")
                    for question in message["follow_ups"]:
                        if st.button(question, key=f"followup_{hash(question)}"):
                            self.handle_user_query(question)
    
    def handle_user_query(self, query: str):
        """Handle user query and generate response."""
        try:
            from datetime import datetime
            
            # Add user message with timestamp
            st.session_state.messages.append({
                "role": "user", 
                "content": query,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Retrieve relevant documents
                    num_docs = st.session_state.get('num_docs', 5)
                    context = st.session_state.retriever.get_context_for_query(query, k=num_docs)
                    
                    # Get detailed results for sources
                    detailed_results = st.session_state.retriever.get_detailed_results(query, k=num_docs)
                    
                    # Generate answer
                    answer = st.session_state.generator.generate_conversational_answer(
                        query, 
                        context,
                        st.session_state.messages[-10:]  # Last 10 messages for context
                    )
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(detailed_results):
                            st.markdown(f"""
                            **Source {i+1}** (Similarity: {source['similarity_score']:.3f})
                            
                            {source['content_preview']}
                            """)
                    
                    # Generate follow-up questions
                    follow_ups = st.session_state.generator.suggest_follow_up_questions(
                        query, answer, context
                    )
                    
                    if follow_ups:
                        st.markdown("**💡 Suggested follow-up questions:**")
                        for question in follow_ups:
                            if st.button(question, key=f"followup_{len(st.session_state.messages)}_{hash(question)}"):
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": answer,
                                    "sources": detailed_results,
                                    "follow_ups": follow_ups,
                                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                })
                                self.handle_user_query(question)
                                return
            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": detailed_results,
                "follow_ups": follow_ups,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
    
    def generate_chat_export(self, format_type="markdown"):
        """Generate chat history export in different formats."""
        try:
            from datetime import datetime
            import json
            
            # Update export timestamp in session state
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.export_timestamp = current_time
            
            # Get metadata
            valid_openai = (self.openai_api_key and 
                           len(self.openai_api_key) > 20 and 
                           not self.openai_api_key.startswith("your_"))
            
            metadata = {
                "export_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_messages": len(st.session_state.messages),
                "embeddings_used": "OpenAI" if valid_openai else "Local (sentence-transformers)",
                "model_provider": st.session_state.get('selected_provider', 'Unknown'),
                "model": st.session_state.get('model_choice', 'Unknown')
            }
            
            if format_type == "json":
                # JSON format
                export_data = {
                    "metadata": metadata,
                    "messages": []
                }
                
                for i, message in enumerate(st.session_state.messages, 1):
                    message_data = {
                        "message_id": i,
                        "role": message["role"],
                        "content": message["content"],
                        "timestamp": message.get("timestamp", "unknown")
                    }
                    
                    if "sources" in message and message["sources"]:
                        message_data["sources"] = message["sources"]
                    
                    export_data["messages"].append(message_data)
                
                return json.dumps(export_data, indent=2, ensure_ascii=False)
            
            elif format_type == "text":
                # Plain text format
                chat_text = "RAG Chatbot Conversation Export\n"
                chat_text += "=" * 40 + "\n\n"
                chat_text += f"Export Date: {metadata['export_date']}\n"
                chat_text += f"Total Messages: {metadata['total_messages']}\n"
                chat_text += f"Embeddings Used: {metadata['embeddings_used']}\n"
                chat_text += f"Model Provider: {metadata['model_provider']}\n"
                chat_text += f"Model: {metadata['model']}\n\n"
                chat_text += "-" * 40 + "\n\n"
                
                for i, message in enumerate(st.session_state.messages, 1):
                    role = "USER" if message["role"] == "user" else "ASSISTANT"
                    chat_text += f"[{i}] {role}:\n"
                    chat_text += f"{message['content']}\n\n"
                    
                    if "sources" in message and message["sources"]:
                        chat_text += "SOURCES:\n"
                        for j, source in enumerate(message["sources"], 1):
                            chat_text += f"  {j}. (Score: {source['similarity_score']:.3f}) {source['content_preview']}\n"
                        chat_text += "\n"
                    
                    chat_text += "-" * 40 + "\n\n"
                
                return chat_text
            
            else:
                # Markdown format (default)
                chat_text = f"# RAG Chatbot Conversation Export\n\n"
                chat_text += f"**Export Date:** {metadata['export_date']}\n"
                chat_text += f"**Total Messages:** {metadata['total_messages']}\n"
                chat_text += f"**Embeddings Used:** {metadata['embeddings_used']}\n"
                chat_text += f"**Model Provider:** {metadata['model_provider']}\n"
                chat_text += f"**Model:** {metadata['model']}\n\n"
                chat_text += "---\n\n"
                
                for i, message in enumerate(st.session_state.messages, 1):
                    if message["role"] == "user":
                        chat_text += f"## Message {i} - User\n\n"
                        chat_text += f"{message['content']}\n\n"
                    else:
                        chat_text += f"## Message {i} - Assistant\n\n"
                        chat_text += f"{message['content']}\n\n"
                        
                        if "sources" in message and message["sources"]:
                            chat_text += "### Sources Referenced:\n\n"
                            for j, source in enumerate(message["sources"], 1):
                                chat_text += f"**Source {j}** (Similarity: {source['similarity_score']:.3f}):\n"
                                chat_text += f"```\n{source['content_preview']}\n```\n\n"
                    
                    chat_text += "---\n\n"
                
                return chat_text
            
        except Exception as e:
            error_msg = f"Error generating chat export: {str(e)}"
            if format_type == "json":
                return json.dumps({"error": error_msg})
            else:
                return f"# Export Error\n\n{error_msg}"
    
    def run(self):
        """Run the main application."""
        self.render_header()
        self.render_sidebar()
        self.render_chat_interface()


def main():
    """Main application entry point."""
    try:
        app = RAGChatbot()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check your configuration and try again.")


if __name__ == "__main__":
    main()