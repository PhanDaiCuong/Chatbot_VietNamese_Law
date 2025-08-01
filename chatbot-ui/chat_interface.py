import json
import logging
import time
import requests
import streamlit as st
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    API_BASE_URL: str = "http://127.0.0.1:8002"
    DEFAULT_USER_ID: str = "streamlit_user"
    DEFAULT_BOT_ID: str = "vietnamese_law_bot"
    REQUEST_TIMEOUT: int = 30
    ENABLE_STREAMING: bool = False

config = Config()

# Page configuration
st.set_page_config(
    page_title="Vietnamese Law Chatbot",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Vietnamese Law Chatbot")
st.markdown("*Trợ lý AI cho các câu hỏi pháp luật Việt Nam*")

class ChatbotAPI:
    """API client for chatbot backend"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        
    def send_chat_request(self, user_message: str, user_id: str = None, bot_id: str = None, stream: bool = False) -> Dict[Any, Any]:
        """Send chat request to backend"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "user_message": user_message,
            "user_id": user_id or config.DEFAULT_USER_ID,
            "bot_id": bot_id or config.DEFAULT_BOT_ID,
            "stream": stream
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        try:
            logger.info(f"Sending request to {url}")
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            raise Exception("Yêu cầu quá thời gian chờ. Vui lòng thử lại.")
        except requests.exceptions.ConnectionError:
            raise Exception("Không thể kết nối đến server. Vui lòng kiểm tra server có đang chạy không.")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Lỗi server: {e}")
        except Exception as e:
            raise Exception(f"Lỗi không xác định: {e}")
    
    def check_health(self) -> Dict[Any, Any]:
        """Check backend health"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except:
            return {"status": "unhealthy"}

# Initialize API client
api_client = ChatbotAPI(config.API_BASE_URL)

def response_generator(user_message: str):
    """Generate streaming response for chat"""
    try:
        chat_response = api_client.send_chat_request(user_message)
        content = chat_response.get('content', '')
        
        # Simulate streaming effect
        words = content.split()
        for word in words:
            yield word + " "
            time.sleep(0.03)
            
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        yield f"❌ Lỗi: {str(e)}"

def get_chat_response(user_message: str) -> str:
    """Get complete chat response"""
    try:
        response = api_client.send_chat_request(user_message)
        return response.get('content', 'Không có phản hồi từ bot.')
    except Exception as e:
        logger.error(f"Error getting chat response: {e}")
        return f"❌ Lỗi: {str(e)}"

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    # Health check
    health_status = api_client.check_health()
    if health_status.get("status") == "healthy":
        st.success("✅ Server đang hoạt động")
        if "model" in health_status:
            st.info(f"Model: {health_status['model']}")
    else:
        st.error("❌ Server không hoạt động")
    
    # User configuration
    user_id = st.text_input("User ID", value=config.DEFAULT_USER_ID)
    bot_id = st.text_input("Bot ID", value=config.DEFAULT_BOT_ID)
    
    # Streaming option
    enable_streaming = st.checkbox("Bật hiệu ứng streaming", value=config.ENABLE_STREAMING)
    
    # Clear chat button
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Nhập câu hỏi pháp luật của bạn..."):
    # Validate input
    if not prompt.strip():
        st.warning("Vui lòng nhập câu hỏi.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        if enable_streaming:
            # Streaming response
            response = st.write_stream(response_generator(prompt))
        else:
            # Direct response
            with st.spinner("Đang xử lý..."):
                response = get_chat_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>Vietnamese Law Chatbot - Powered by AI</small>
    </div>
    """, 
    unsafe_allow_html=True
)
