#!/usr/bin/env python3
"""
Streamlit UI for DispatchTrack Customer Support Bot.

Run with: streamlit run streamlit_app.py
"""

import os
import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# Load environment variables
dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    load_dotenv()

# Configuration
DEFAULT_MODEL = "gpt-4o"
DEFAULT_SEED_ROLE = "system"
DEFAULT_VECTOR_STORE_IDS = ["vs_68bcc5297c1c8191a1c3fe60cd775b88"]

# Load prompt from file
_here = os.path.dirname(__file__)
_prompt_path = os.path.join(_here, "prompt.txt")

DEFAULT_PROMPT = ""
if os.path.exists(_prompt_path):
    try:
        with open(_prompt_path, "r", encoding="utf-8") as f:
            DEFAULT_PROMPT = f.read().strip()
    except Exception:
        pass

if not DEFAULT_PROMPT:
    DEFAULT_PROMPT = (
        "You are DispatchTrack Customer Support Bot. Your job is to give clear, "
        "efficient, and friendly answers for customers of DT based on documentation."
    )


def init_openai_client():
    """Initialize OpenAI client with error handling."""
    try:
        return OpenAI()
    except Exception as e:
        st.error(f"OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
        st.stop()


def build_tools(vector_store_ids: Optional[List[str]]) -> Optional[List[Dict[str, Any]]]:
    """Build tools for file search if vector store IDs are provided."""
    if not vector_store_ids:
        return None
    return [
        {
            "type": "file_search",
            "vector_store_ids": vector_store_ids,
        }
    ]


def make_message(role: str, text: str) -> Dict[str, Any]:
    """Create a message in the format expected by the OpenAI API."""
    content_type = "output_text" if role == "assistant" else "input_text"
    return {
        "role": role,
        "content": [
            {
                "type": content_type,
                "text": text,
            }
        ],
    }


def extract_output_text(response: Any) -> str:
    """Extract assistant text from OpenAI response."""
    # Try convenience property first
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text
    
    # Try structured extraction
    try:
        outputs = getattr(response, "output", None) or getattr(response, "outputs", None)
        if outputs:
            for item in outputs:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for c in content:
                        c_type = getattr(c, "type", None)
                        c_text = getattr(c, "text", None)
                        if c_type in ("output_text", "text") and isinstance(c_text, str) and c_text:
                            return c_text
    except Exception:
        pass
    
    return ""


def get_knowledge_base_files(client: OpenAI, vector_store_id: str) -> List[Dict[str, str]]:
    """Get list of files in the knowledge base vector store."""
    try:
        # Try beta client first (most likely)
        if hasattr(client, 'beta') and hasattr(client.beta, 'vector_stores'):
            files = client.beta.vector_stores.files.list(vector_store_id=vector_store_id)
        # Fallback to direct access
        elif hasattr(client, 'vector_stores'):
            files = client.vector_stores.files.list(vector_store_id=vector_store_id)
        else:
            return []
        
        file_list = []
        for file in files.data:
            try:
                # Get file details
                file_details = client.files.retrieve(file.id)
                file_list.append({
                    "name": file_details.filename,
                    "status": file.status,
                    "id": file.id
                })
            except Exception:
                file_list.append({
                    "name": f"File {file.id[:8]}...",
                    "status": file.status,
                    "id": file.id
                })
        
        return file_list
    except Exception:
        return []


def create_chat_request(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create the request payload for OpenAI chat completion."""
    return {
        "model": DEFAULT_MODEL,
        "input": messages,
        "text": {
            "format": {"type": "text"},
            "verbosity": "medium",
        },
        "tools": build_tools(DEFAULT_VECTOR_STORE_IDS),
        "store": True,
    }


def generate_llm_email_analysis(messages: List[Dict[str, str]], customer_name: str = "Customer") -> Dict[str, str]:
    """Use LLM to intelligently analyze conversation and generate email subject and content."""
    try:
        client = OpenAI()
        
        # Format conversation for analysis
        conversation_text = ""
        for msg in messages:
            role = "Customer" if msg["role"] == "user" else "MaxBot"
            conversation_text += f"{role}: {msg['content']}\n"
        
        analysis_prompt = f"""
You are analyzing a customer support escalation conversation. Please provide:

1. A concise subject line (3-5 words max) that captures the specific issue
2. A brief issue summary (1-2 sentences)
3. Priority level (High/Medium/Low)
4. Recommended action (1 sentence)

Customer Name: {customer_name}

Conversation:
{conversation_text}

Please respond in this exact JSON format:
{{
    "subject_line": "brief issue description",
    "issue_summary": "detailed summary of the problem",
    "priority": "High|Medium|Low",
    "recommended_action": "what the support agent should do"
}}
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a customer support analysis expert. Analyze conversations and provide actionable insights for support agents."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        # Parse JSON response (handle markdown code blocks)
        import json
        import re
        try:
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without markdown blocks
                json_match = re.search(r'(\{.*?\})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = content
            
            analysis = json.loads(json_str)
            return {
                "subject_line": analysis.get("subject_line", "Support Request"),
                "issue_summary": analysis.get("issue_summary", "Customer needs assistance"),
                "priority": analysis.get("priority", "Medium"),
                "recommended_action": analysis.get("recommended_action", "Contact customer for resolution"),
                "total_messages": len(messages),
                "customer_messages": len([msg for msg in messages if msg["role"] == "user"])
            }
        except (json.JSONDecodeError, AttributeError):
            # Fallback if JSON parsing fails
            content = response.choices[0].message.content.strip()
            return {
                "subject_line": "Support Request",
                "issue_summary": content[:200] + "..." if len(content) > 200 else content,
                "priority": "Medium", 
                "recommended_action": "Review conversation and contact customer",
                "total_messages": len(messages),
                "customer_messages": len([msg for msg in messages if msg["role"] == "user"])
            }
            
    except Exception as e:
        # Fallback analysis if LLM fails
        print(f"LLM analysis failed: {e}")
        customer_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        first_issue = customer_messages[0] if customer_messages else "Support needed"
        
        return {
            "subject_line": "Support Request",
            "issue_summary": first_issue[:150] + "..." if len(first_issue) > 150 else first_issue,
            "priority": "Medium",
            "recommended_action": "Review conversation and contact customer",
            "total_messages": len(messages),
            "customer_messages": len(customer_messages)
        }


def format_conversation_for_email(messages: List[Dict[str, str]], customer_name: str = "Customer") -> str:
    """Format the conversation history for email with improved structure."""
    # Use LLM to analyze the conversation
    issue_analysis = generate_llm_email_analysis(messages, customer_name)
    current_time = datetime.now()
    
    # Enhanced email format
    formatted_email = f"""
🚨 ESCALATION ALERT - Action Required
{'='*60}

📋 ISSUE SUMMARY
Customer: {customer_name}
Issue: {issue_analysis['issue_summary']}
Priority: {issue_analysis['priority']}
Escalated: {current_time.strftime('%Y-%m-%d at %H:%M:%S')}
Interface: Web Chat
Messages: {issue_analysis['total_messages']} total ({issue_analysis['customer_messages']} from customer)

🎯 RECOMMENDED ACTION
{issue_analysis['recommended_action']}

🎯 QUICK CHECKLIST
□ Contact {customer_name} directly within 2 hours
□ Review conversation context below
□ Implement recommended action above
□ Update customer on status and next steps
□ Close escalation ticket when resolved

💬 FULL CONVERSATION HISTORY
{'='*60}
"""
    
    # Format conversation with better readability
    for i, message in enumerate(messages, 1):
        role = "👤 CUSTOMER" if message["role"] == "user" else "🤖 MAXBOT"
        message_time = current_time.strftime('%H:%M')
        
        formatted_email += f"\n[{message_time}] {role}:\n"
        formatted_email += f"{message['content']}\n"
        
        # Add separator between messages
        if i < len(messages):
            formatted_email += "-" * 40 + "\n"
    
    # Footer with next steps
    formatted_email += f"""
{'='*60}

⚡ ESCALATION REASON
This conversation was automatically escalated because MaxBot couldn't provide adequate assistance based on available documentation.

📞 NEXT STEPS
1. Contact {customer_name} directly within 2 hours
2. {issue_analysis['recommended_action']}  
3. Provide detailed resolution steps
4. Follow up to ensure satisfaction
5. Update knowledge base if this is a common issue

🔧 INTERNAL NOTES
- Escalated from: Web Interface
- Bot Version: MaxBot v1.0
- Auto-generated on: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
- Vector Store: Active with latest documentation
- AI Analysis: Priority {issue_analysis['priority']}

---
Automated Escalation System | DispatchTrack Customer Support
"""
    
    return formatted_email


def send_escalation_email(conversation_history: str, customer_name: str = "Customer", messages: List[Dict[str, str]] = None) -> bool:
    """Send escalation email to support team."""
    try:
        # Email configuration from environment variables
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        email_user = os.getenv("SUPPORT_EMAIL_USER")
        email_password = os.getenv("SUPPORT_EMAIL_PASSWORD")
        support_email = os.getenv("SUPPORT_EMAIL_TO", "support@dispatchtrack.com")
        
        if not email_user or not email_password:
            st.error("Email configuration not found. Please configure SUPPORT_EMAIL_USER and SUPPORT_EMAIL_PASSWORD in your .env file.")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = support_email
        # Create intelligent subject line using LLM analysis
        if messages:
            analysis = generate_llm_email_analysis(messages, customer_name)
            priority_emoji = "🚨" if analysis['priority'] == "High" else "⚠️" if analysis['priority'] == "Medium" else "📋"
            msg['Subject'] = f"{priority_emoji} {analysis['subject_line']} - {customer_name} - {datetime.now().strftime('%m/%d %H:%M')}"
        else:
            msg['Subject'] = f"📋 Support Request - {customer_name} - {datetime.now().strftime('%m/%d %H:%M')}"
        
        # Use the enhanced conversation history as the email body
        body = conversation_history
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(email_user, email_password)
        text = msg.as_string()
        server.sendmail(email_user, support_email, text)
        server.quit()
        
        return True
        
    except Exception as e:
        st.error(f"Failed to send escalation email: {str(e)}")
        return False


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="DT Customer Support Bot",
        page_icon="🤖",
        layout="centered"
    )
    
    st.title("🤖 DT Customer Support Bot")
    st.markdown("*DispatchTrack Customer Support Assistant*")
    
    # Initialize OpenAI client
    client = init_openai_client()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add proactive introduction message
        intro_message = "Hi! I'm MaxBot, and I'm here to help you with any DT Agent issues you're experiencing. May I have your name so I can assist you better?"
        st.session_state.messages.append({
            "role": "assistant",
            "content": intro_message
        })
    if "customer_name" not in st.session_state:
        st.session_state.customer_name = ""
    if "escalation_sent" not in st.session_state:
        st.session_state.escalation_sent = False
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("How can I help you with DispatchTrack?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Extract customer name if this looks like a name response
        if not st.session_state.customer_name and len(st.session_state.messages) <= 3:
            # Enhanced name detection: look for name-like responses
            support_keywords = ["issue", "problem", "error", "broken", "not working", "help", "support", "agent", "dt", "dispatch"]
            name_indicators = ["i'm", "my name is", "i am", "call me", "it's"]
            
            # Clean the prompt for name extraction
            clean_prompt = prompt.lower().strip()
            
            # Check if it's likely a name response
            is_likely_name = (
                len(prompt.split()) <= 4 and  # Short response (up to 4 words)
                not any(keyword in clean_prompt for keyword in support_keywords) and  # No support keywords
                (
                    any(indicator in clean_prompt for indicator in name_indicators) or  # Has name indicators
                    (len(prompt.split()) <= 2 and prompt.replace(" ", "").isalpha())  # 1-2 words, all letters
                )
            )
            
            if is_likely_name:
                # Extract the actual name
                if "my name is" in clean_prompt:
                    st.session_state.customer_name = prompt.split("my name is", 1)[1].strip().title()
                elif "i'm" in clean_prompt:
                    st.session_state.customer_name = prompt.split("i'm", 1)[1].strip().title()
                elif "i am" in clean_prompt:
                    st.session_state.customer_name = prompt.split("i am", 1)[1].strip().title()
                elif "call me" in clean_prompt:
                    st.session_state.customer_name = prompt.split("call me", 1)[1].strip().title()
                else:
                    # Just use the whole response as name if it looks like a name
                    st.session_state.customer_name = prompt.strip().title()
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Prepare messages for OpenAI API
        # Add customer name context to system prompt if we have it
        system_prompt = DEFAULT_PROMPT
        if st.session_state.customer_name:
            system_prompt += f"\n\nCustomer's name: {st.session_state.customer_name}. Make sure to use their name naturally throughout the conversation."
        
        api_messages = [make_message(DEFAULT_SEED_ROLE, system_prompt)]
        for msg in st.session_state.messages:
            api_messages.append(make_message(msg["role"], msg["content"]))
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    request_payload = create_chat_request(api_messages)
                    response = client.responses.create(**request_payload)
                    assistant_response = extract_output_text(response)
                    
                    if not assistant_response:
                        assistant_response = "I apologize, but I'm having trouble generating a response. Please try again."
                    
                    # Check if response contains escalation trigger
                    if "ESCALATE_TO_HUMAN" in assistant_response:
                        # Clean the response for display (remove trigger phrase)
                        clean_response = assistant_response.replace("ESCALATE_TO_HUMAN", "").strip()
                        
                        # Auto-escalate to human
                        customer_name = st.session_state.customer_name or "Customer"
                        
                        # Send escalation email in background
                        with st.spinner("Notifying human support team..."):
                            # Include the escalation response in conversation history
                            escalation_messages = st.session_state.messages + [{
                                "role": "assistant", 
                                "content": clean_response
                            }]
                            
                            conversation_history = format_conversation_for_email(
                                escalation_messages, 
                                customer_name
                            )
                            
                            if send_escalation_email(conversation_history, customer_name, escalation_messages):
                                st.session_state.escalation_sent = True
                                # Add escalation notification to the response
                                clean_response += "\n\n🚨 **I've automatically escalated your issue to a human agent who will follow up with you soon.**"
                            else:
                                clean_response += "\n\n⚠️ I tried to escalate this to a human agent, but there was an issue. Please contact support directly."
                        
                        st.write(clean_response)
                        
                        # Add cleaned response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": clean_response
                        })
                    else:
                        # Normal response - no escalation needed
                        st.write(assistant_response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": assistant_response
                        })
                    
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_message
                    })
    
    # Sidebar with info
    with st.sidebar:
        st.markdown("### Knowledge Base")
        
        # Get and display knowledge base files
        if DEFAULT_VECTOR_STORE_IDS:
            kb_files = get_knowledge_base_files(client, DEFAULT_VECTOR_STORE_IDS[0])
            if kb_files:
                for file_info in kb_files:
                    status_emoji = "✅" if file_info["status"] == "completed" else "⏳" if file_info["status"] == "in_progress" else "❌"
                    st.text(f"{status_emoji} {file_info['name']}")
            else:
                st.text("📭 No files found")
        else:
            st.text("No vector store configured")
        
        st.markdown("### Configuration")
        st.text(f"Model: {DEFAULT_MODEL}")
        st.text(f"Vector Store: {DEFAULT_VECTOR_STORE_IDS[0][:20]}...")
        
        st.markdown("### Actions")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            # Re-add the introduction message after clearing
            intro_message = "Hi! I'm MaxBot, and I'm here to help you with any DT Agent issues you're experiencing. May I have your name so I can assist you better?"
            st.session_state.messages.append({
                "role": "assistant",
                "content": intro_message
            })
            st.session_state.customer_name = ""
            st.session_state.escalation_sent = False
            st.rerun()
            
        st.markdown("---")
        
        # Auto-Escalation Status
        if "escalation_sent" in st.session_state and st.session_state.escalation_sent:
            st.markdown("### 🚨 Escalated to Human")
            st.success("This conversation has been automatically escalated to a human agent who will follow up with you soon.")
            st.info("You can continue chatting here while waiting for human assistance.")
        
        st.markdown("### About")
        st.markdown("""
        This bot helps with DispatchTrack customer support questions.
        It uses your uploaded documentation to provide accurate answers.
        """)


if __name__ == "__main__":
    main()
