#!/usr/bin/env python3
"""
Interactive CLI for communicating with a Custom GPT via the OpenAI Responses API.

Features:
- Loads API key from a .env file (OPENAI_API_KEY)
- Supports interactive chat loop or single-shot prompt mode
- Allows configuring model, seed prompt role, reasoning effort, verbosity
- Optional file_search tool with one or more vector store IDs
- Optional streaming output via --stream flag
"""

import argparse
import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import termios
from select import select
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI


DEFAULT_MODEL = "gpt-4o"

# Load customer support seed prompt from prompt.txt
_here = os.path.dirname(__file__)
_default_seed_path = os.path.normpath(os.path.join(_here, "prompt.txt"))

DEFAULT_DEVELOPER_PROMPT = ""
if os.path.exists(_default_seed_path):
    try:
        with open(_default_seed_path, "r", encoding="utf-8") as _f:
            DEFAULT_DEVELOPER_PROMPT = _f.read().strip()
    except Exception:
        pass

# Fallback if file not found
if not DEFAULT_DEVELOPER_PROMPT:
    DEFAULT_DEVELOPER_PROMPT = (
        "You are DispatchTrack Customer Support Bot. Your job is to give clear, "
        "efficient, and friendly answers for customers of DT based on documentation."
    )


def load_api_key_or_exit() -> None:
    """Load OPENAI_API_KEY from .env or environment, exit with a clear message if missing."""
    # Search for a .env file starting from CWD and walking up
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        # Still try default behavior; load_dotenv() is a no-op if no file found
        load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        sys.stderr.write(
            "ERROR: OPENAI_API_KEY is not set. Create a .env file with OPENAI_API_KEY=... or export it in your shell.\n"
        )
        sys.exit(1)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_tools(vector_store_ids: Optional[List[str]]) -> Optional[List[Dict[str, Any]]]:
    if not vector_store_ids:
        return None
    return [
        {
            "type": "file_search",
            "vector_store_ids": vector_store_ids,
        }
    ]


def extract_output_text(response: Any) -> str:
    """Best-effort extraction of model text from Responses API result."""
    # New SDKs expose a convenience property
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text:
        return text

    # Fallbacks
    try:
        # Response output structure is typically a list of items with content
        outputs = getattr(response, "output", None) or getattr(response, "outputs", None)
        if outputs:
            for item in outputs:
                content = item.get("content") if isinstance(item, dict) else None
                if isinstance(content, list):
                    for c in content:
                        if c.get("type") in ("output_text", "text") and c.get("text"):
                            return c["text"]
    except Exception:
        pass

    # Last resort: repr
    return str(response)


def create_response(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    verbosity: str,
    effort: str,
    vector_store_ids: Optional[List[str]],
    store: bool,
    include_fields: Optional[List[str]],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    disable_reasoning: bool = False,
) -> Any:
    request: Dict[str, Any] = {
        "model": model,
        "input": messages,
        "text": {
            "format": {"type": "text"},
            "verbosity": verbosity,
        },
        "store": store,
    }

    if not disable_reasoning:
        request["reasoning"] = {
            "effort": effort,
            "summary": "auto",
        }

    tools = build_tools(vector_store_ids)
    if tools:
        request["tools"] = tools

    if include_fields:
        request["include"] = include_fields

    if temperature is not None:
        request["temperature"] = temperature
    if top_p is not None:
        request["top_p"] = top_p
    if max_output_tokens is not None:
        request["max_output_tokens"] = max_output_tokens

    return client.responses.create(**request)


def stream_and_collect(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    verbosity: str,
    effort: str,
    vector_store_ids: Optional[List[str]],
    store: bool,
    include_fields: Optional[List[str]],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    disable_reasoning: bool = False,
) -> str:
    request: Dict[str, Any] = {
        "model": model,
        "input": messages,
        "text": {
            "format": {"type": "text"},
            "verbosity": verbosity,
        },
        "store": store,
    }

    if not disable_reasoning:
        request["reasoning"] = {
            "effort": effort,
            "summary": "auto",
        }

    tools = build_tools(vector_store_ids)
    if tools:
        request["tools"] = tools

    if include_fields:
        request["include"] = include_fields

    if temperature is not None:
        request["temperature"] = temperature
    if top_p is not None:
        request["top_p"] = top_p
    if max_output_tokens is not None:
        request["max_output_tokens"] = max_output_tokens

    collected_parts: List[str] = []
    try:
        with client.responses.stream(**request) as stream:
            for event in stream:
                etype = getattr(event, "type", None)
                # Only emit assistant-visible text, never reasoning deltas
                if etype == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        print(delta, end="", flush=True)
                        collected_parts.append(delta)
                elif etype == "response.refusal.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        print(delta, end="", flush=True)
                        collected_parts.append(delta)
                elif etype == "response.error":
                    err = getattr(event, "error", None)
                    sys.stderr.write(f"\n[stream error] {err}\n")
                elif etype == "response.completed":
                    # Ensure a newline after completion
                    print(flush=True)
            # Obtain final response for fallback extraction if needed
            try:
                final = stream.get_final_response()
                if not collected_parts:
                    text = extract_output_text(final)
                    if text:
                        print(text, end="", flush=True)
                        collected_parts.append(text)
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Streaming failed: {exc}\n")
        return ""

    return "".join(collected_parts)


def make_message(role: str, text: str) -> Dict[str, Any]:
    # In Responses API, assistant turns must use 'output_text' or 'refusal'.
    # User/developer/system turns should use 'input_text'.
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


@contextmanager
def suppress_user_input_echo():
    """Temporarily disable stdin echo so user can't see typed keys during streaming."""
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    try:
        old_attrs = termios.tcgetattr(fd)
    except Exception:
        yield
        return
    try:
        new_attrs = termios.tcgetattr(fd)
        new_attrs[3] = new_attrs[3] & ~termios.ECHO  # lflags: turn off ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, new_attrs)
        yield
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        except Exception:
            pass


def drain_stdin_nonblocking():
    """Discard any keystrokes entered while streaming so they don't appear next prompt."""
    if not sys.stdin.isatty():
        return
    try:
        fd = sys.stdin.fileno()
        r, _, _ = select([sys.stdin], [], [], 0)
        while r:
            try:
                os.read(fd, 4096)
            except Exception:
                break
            r, _, _ = select([sys.stdin], [], [], 0)
    except Exception:
        return


def run_once(
    *,
    prompt: str,
    model: str,
    developer_prompt: str,
    seed_role: str,
    verbosity: str,
    effort: str,
    vector_store_ids: Optional[List[str]],
    store: bool,
    include_fields: Optional[List[str]],
    stream: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    disable_reasoning: bool = False,
) -> int:
    client = OpenAI()
    messages: List[Dict[str, Any]] = [
        make_message(seed_role, developer_prompt),
        make_message("user", prompt),
    ]
    if stream:
        _ = stream_and_collect(
            client,
            model=model,
            messages=messages,
            verbosity=verbosity,
            effort=effort,
            vector_store_ids=vector_store_ids,
            store=store,
            include_fields=include_fields,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            disable_reasoning=disable_reasoning,
        )
    else:
        try:
            response = create_response(
                client,
                model=model,
                messages=messages,
                verbosity=verbosity,
                effort=effort,
                vector_store_ids=vector_store_ids,
                store=store,
                include_fields=include_fields,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                disable_reasoning=disable_reasoning,
            )
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"Request failed: {exc}\n")
            return 2

        output_text = extract_output_text(response)
        print(output_text)
    return 0


def generate_cli_llm_email_analysis(messages: List[Dict[str, Any]], customer_name: str = "CLI User") -> Dict[str, str]:
    """Use LLM to intelligently analyze CLI conversation and generate email subject and content."""
    try:
        client = OpenAI()
        
        # Format conversation for analysis
        conversation_text = ""
        for msg in messages:
            role = "Customer" if msg["role"] == "user" else "MaxBot"
            conversation_text += f"{role}: {msg['content']}\n"
        
        analysis_prompt = f"""
You are analyzing a CLI customer support escalation conversation. Please provide:

1. A concise subject line (3-5 words max) that captures the specific issue
2. A brief issue summary (1-2 sentences)
3. Priority level (High/Medium/Low)
4. Recommended action (1 sentence)

Customer Name: {customer_name}
Interface: Command Line Interface

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
                {"role": "system", "content": "You are a customer support analysis expert. Analyze CLI conversations and provide actionable insights for support agents."},
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
                "subject_line": analysis.get("subject_line", "CLI Support Request"),
                "issue_summary": analysis.get("issue_summary", "Customer needs CLI assistance"),
                "priority": analysis.get("priority", "Medium"),
                "recommended_action": analysis.get("recommended_action", "Contact customer for resolution"),
                "total_messages": len(messages),
                "customer_messages": len([msg for msg in messages if msg["role"] == "user"])
            }
        except (json.JSONDecodeError, AttributeError):
            # Fallback if JSON parsing fails
            content = response.choices[0].message.content.strip()
            return {
                "subject_line": "CLI Support Request",
                "issue_summary": content[:200] + "..." if len(content) > 200 else content,
                "priority": "Medium", 
                "recommended_action": "Review CLI conversation and contact customer",
                "total_messages": len(messages),
                "customer_messages": len([msg for msg in messages if msg["role"] == "user"])
            }
            
    except Exception as e:
        # Fallback analysis if LLM fails
        print(f"CLI LLM analysis failed: {e}")
        customer_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        first_issue = customer_messages[0] if customer_messages else "CLI support needed"
        
        return {
            "subject_line": "CLI Support Request",
            "issue_summary": first_issue[:150] + "..." if len(first_issue) > 150 else first_issue,
            "priority": "Medium",
            "recommended_action": "Review CLI conversation and contact customer",
            "total_messages": len(messages),
            "customer_messages": len(customer_messages)
        }


def format_cli_conversation_for_email(history: List[Dict[str, Any]], customer_name: str = "CLI User") -> str:
    """Format CLI conversation history for email using LLM analysis."""
    # Convert history to the format expected by LLM analysis
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in history[1:]]  # Skip system prompt
    
    if not messages:
        return "CLI Escalation - No conversation history available", "CLI Support Request"
    
    # Use LLM to analyze the conversation
    analysis = generate_cli_llm_email_analysis(messages, customer_name)
    current_time = datetime.now()
    
    # Enhanced CLI email format with LLM insights
    formatted_email = f"""
🚨 CLI ESCALATION ALERT - Action Required
{'='*60}

📋 ISSUE SUMMARY
Customer: {customer_name}
Issue: {analysis['issue_summary']}
Priority: {analysis['priority']}
Escalated: {current_time.strftime('%Y-%m-%d at %H:%M:%S')}
Interface: Command Line Interface (CLI)
Messages: {analysis['total_messages']} total ({analysis['customer_messages']} from customer)

🎯 RECOMMENDED ACTION
{analysis['recommended_action']}

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
This CLI conversation was automatically escalated because MaxBot couldn't provide adequate assistance based on available documentation.

📞 NEXT STEPS
1. {analysis['recommended_action']}
2. Contact {customer_name} directly within 2 hours  
3. Provide detailed resolution steps
4. Follow up to ensure satisfaction
5. Update knowledge base if this is a common issue

🔧 INTERNAL NOTES
- Escalated from: Command Line Interface
- Bot Version: MaxBot v1.0
- Auto-generated on: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
- Vector Store: Active with latest documentation
- AI Analysis: Priority {analysis['priority']}

---
Automated Escalation System | DispatchTrack Customer Support
"""
    
    return formatted_email, analysis['subject_line']


def send_cli_escalation_email(conversation_history: str, customer_name: str = "CLI User", issue_keywords: str = None) -> bool:
    """Send escalation email from CLI to support team."""
    try:
        # Email configuration from environment variables
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        email_user = os.getenv("SUPPORT_EMAIL_USER")
        email_password = os.getenv("SUPPORT_EMAIL_PASSWORD")
        support_email = os.getenv("SUPPORT_EMAIL_TO", "support@dispatchtrack.com")
        
        if not email_user or not email_password:
            print("⚠️  Email configuration not found for escalation.")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = support_email
        # Create intelligent subject line using LLM analysis
        if issue_keywords:
            # Determine priority emoji (issue_keywords actually contains analysis results now)
            try:
                # Extract priority from the conversation for emoji selection
                priority_emoji = "🚨"  # Default to high priority for CLI escalations
                msg['Subject'] = f"{priority_emoji} {issue_keywords} - {customer_name} - {datetime.now().strftime('%m/%d %H:%M')}"
            except:
                msg['Subject'] = f"🚨 {issue_keywords} - {customer_name} - {datetime.now().strftime('%m/%d %H:%M')}"
        else:
            msg['Subject'] = f"🚨 CLI Support Request - {customer_name} - {datetime.now().strftime('%m/%d %H:%M')}"
        
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
        print(f"❌ Failed to send escalation email: {str(e)}")
        return False


def run_interactive(
    *,
    model: str,
    developer_prompt: str,
    seed_role: str,
    verbosity: str,
    effort: str,
    vector_store_ids: Optional[List[str]],
    store: bool,
    include_fields: Optional[List[str]],
    stream: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    disable_reasoning: bool = False,
) -> int:
    client = OpenAI()
    history: List[Dict[str, Any]] = [make_message(seed_role, developer_prompt)]
    escalation_sent = False
    customer_name = ""
    print("Interactive mode. Type :q or /exit to quit. Press Enter to send.")
    
    # Proactive introduction
    intro_message = "Hi! I'm MaxBot, and I'm here to help you with any DT Agent issues you're experiencing. May I have your name so I can assist you better?"
    print(f"Assistant: {intro_message}\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text in {":q", "/exit", "/quit"}:
            break

        # Append user message to history and request a response
        history.append(make_message("user", user_text))
        
        # Try to extract customer name if we don't have one yet
        if not customer_name and len(history) <= 4:  # Allow a few exchanges to detect name
            support_keywords = ["issue", "problem", "error", "broken", "not working", "help", "support", "agent", "dt", "dispatch"]
            name_indicators = ["i'm", "my name is", "i am", "call me", "it's"]
            
            clean_input = user_text.lower().strip()
            
            # Check if it's likely a name response
            is_likely_name = (
                len(user_text.split()) <= 4 and  # Short response
                not any(keyword in clean_input for keyword in support_keywords) and
                (
                    any(indicator in clean_input for indicator in name_indicators) or
                    (len(user_text.split()) <= 2 and user_text.replace(" ", "").isalpha())
                )
            )
            
            if is_likely_name:
                # Extract the actual name
                if "my name is" in clean_input:
                    customer_name = user_text.split("my name is", 1)[1].strip().title()
                elif "i'm" in clean_input:
                    customer_name = user_text.split("i'm", 1)[1].strip().title()
                elif "i am" in clean_input:
                    customer_name = user_text.split("i am", 1)[1].strip().title()
                elif "call me" in clean_input:
                    customer_name = user_text.split("call me", 1)[1].strip().title()
                else:
                    customer_name = user_text.strip().title()
                
                print(f"[Detected name: {customer_name}]")

        assistant_text: str
        if stream:
            # Prefix for readability in streaming
            print("Assistant: ", end="", flush=True)
            with suppress_user_input_echo():
                assistant_text = stream_and_collect(
                    client,
                    model=model,
                    messages=history,
                    verbosity=verbosity,
                    effort=effort,
                    vector_store_ids=vector_store_ids,
                    store=store,
                    include_fields=include_fields,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                    disable_reasoning=disable_reasoning,
                )
            # Remove any accidental user keystrokes pressed during streaming
            drain_stdin_nonblocking()
            print(flush=True)
        else:
            # Prepare messages with customer name context if we have one
            context_messages = history.copy()
            if customer_name:
                # Update system prompt with customer name
                updated_system_prompt = developer_prompt + f"\n\nCustomer's name: {customer_name}. Make sure to use their name naturally throughout the conversation."
                context_messages[0] = make_message(seed_role, updated_system_prompt)
            
            try:
                response = create_response(
                    client,
                    model=model,
                    messages=context_messages,
                    verbosity=verbosity,
                    effort=effort,
                    vector_store_ids=vector_store_ids,
                    store=store,
                    include_fields=include_fields,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                    disable_reasoning=disable_reasoning,
                )
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"Request failed: {exc}\n")
                # Remove last user message if the request failed so state remains consistent
                history.pop()
                continue

            assistant_text = extract_output_text(response)
            
            # Check for escalation trigger
            if "ESCALATE_TO_HUMAN" in assistant_text:
                # Clean the response for display
                clean_response = assistant_text.replace("ESCALATE_TO_HUMAN", "").strip()
                print(f"Assistant: {clean_response}\n")
                
                # Auto-escalate if not already done
                if not escalation_sent:
                    print("🚨 Escalating to human support...")
                    
                    # Include this response in escalation
                    escalation_history = history + [make_message("assistant", clean_response)]
                    escalation_customer_name = customer_name if customer_name else "CLI User"
                    formatted_conversation, issue_keywords = format_cli_conversation_for_email(escalation_history, escalation_customer_name)
                    
                    if send_cli_escalation_email(formatted_conversation, escalation_customer_name, issue_keywords):
                        escalation_sent = True
                        print("✅ Your issue has been automatically escalated to a human agent who will follow up with you soon.\n")
                    else:
                        print("⚠️  Unable to send escalation email. Please contact support directly.\n")
                
                # Add cleaned response to history
                history.append(make_message("assistant", clean_response))
            else:
                # Normal response
                print(f"Assistant: {assistant_text}\n")
                # Add assistant message to history to preserve context
                history.append(make_message("assistant", assistant_text))

    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI to chat with a Custom GPT using the OpenAI Responses API",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--once",
        action="store_true",
        help="Run a single prompt and print the response, then exit.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to send when using --once. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--developer-prompt",
        type=str,
        default=None,
        help="Developer prompt text to seed the assistant.",
    )
    parser.add_argument(
        "--developer-prompt-file",
        type=str,
        default=None,
        help="Path to a text file containing the developer prompt.",
    )
    parser.add_argument(
        "--seed-role",
        type=str,
        choices=["developer", "system"],
        default="developer",
        help="Role to use for the seed prompt (developer or system).",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Text verbosity preference.",
    )
    parser.add_argument(
        "--effort",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Reasoning effort setting.",
    )
    parser.add_argument(
        "--vector-store-id",
        action="append",
        dest="vector_store_ids",
        default=None,
        help="Vector store ID for file_search tool. Can be passed multiple times.",
    )
    parser.add_argument(
        "--include",
        action="append",
        dest="include_fields",
        default=None,
        help=(
            "Fields to include in response (advanced). Can be passed multiple times. "
            "Example: --include reasoning.encrypted_content"
        ),
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Do not store the response server-side.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens as they arrive.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (e.g., 0.0-2.0).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling probability (0-1).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum number of output tokens.",
    )
    parser.add_argument(
        "--disable-reasoning",
        action="store_true",
        help="Disable reasoning field in requests.",
    )

    args = parser.parse_args(argv)

    if args.developer_prompt_file and args.developer_prompt:
        parser.error("Use either --developer-prompt or --developer-prompt-file, not both.")

    return args


def resolve_developer_prompt(args: argparse.Namespace) -> str:
    if args.developer_prompt_file:
        return read_text_file(args.developer_prompt_file)
    if args.developer_prompt:
        return args.developer_prompt
    return DEFAULT_DEVELOPER_PROMPT


def model_supports_reasoning(model: str) -> bool:
    """Heuristic: enable reasoning only for models that support it.

    Currently, gpt-5 family supports reasoning; gpt-4o does not.
    Also allow any model name that explicitly includes "-reasoning".
    """
    model_lower = model.lower()
    if model_lower.startswith("gpt-5"):
        return True
    if "-reasoning" in model_lower:
        return True
    return False


def main(argv: Optional[List[str]] = None) -> int:
    load_api_key_or_exit()
    args = parse_args(argv)
    developer_prompt = resolve_developer_prompt(args)

    store = not args.no_store
    # Auto-disable reasoning for models that do not support it, unless the user explicitly disabled it
    auto_disable_reasoning = args.disable_reasoning or (not model_supports_reasoning(args.model))

    if args.once:
        prompt = args.prompt
        if prompt is None:
            # Read entire stdin if no prompt provided
            prompt = sys.stdin.read().strip()
            if not prompt:
                sys.stderr.write("No prompt provided. Use --prompt or pipe text via stdin.\n")
                return 2
        return run_once(
            prompt=prompt,
            model=args.model,
            developer_prompt=developer_prompt,
            seed_role=args.seed_role,
            verbosity=args.verbosity,
            effort=args.effort,
            vector_store_ids=args.vector_store_ids,
            store=store,
            include_fields=args.include_fields,
            stream=args.stream,
            temperature=args.temperature,
            top_p=args.top_p,
            max_output_tokens=args.max_output_tokens,
            disable_reasoning=auto_disable_reasoning,
        )

    return run_interactive(
        model=args.model,
        developer_prompt=developer_prompt,
        seed_role=args.seed_role,
        verbosity=args.verbosity,
        effort=args.effort,
        vector_store_ids=args.vector_store_ids,
        store=store,
        include_fields=args.include_fields,
        stream=args.stream,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        disable_reasoning=auto_disable_reasoning,
    )


if __name__ == "__main__":
    raise SystemExit(main())


