#!/usr/bin/env python3
"""
Utility script to manage vector stores and upload files for the DispatchTrack Customer Support Bot.

Usage:
    # Create a new vector store and upload files
    python3 manage_vector_store.py create --name "DT Documentation" --files "doc1.pdf" "doc2.txt" "doc3.md"
    
    # Upload files to existing vector store
    python3 manage_vector_store.py upload --vector-store-id "vs_abc123" --files "new_doc.pdf"
    
    # List files in a vector store
    python3 manage_vector_store.py list --vector-store-id "vs_abc123"
    
    # Update bot configuration with new vector store ID
    python3 manage_vector_store.py update-config --vector-store-id "vs_abc123"
"""

import argparse
import os
import sys
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI


def load_api_key_or_exit() -> None:
    """Load OPENAI_API_KEY from .env or environment, exit with a clear message if missing."""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        sys.stderr.write(
            "ERROR: OPENAI_API_KEY is not set. Create a .env file with OPENAI_API_KEY=... or export it in your shell.\n"
        )
        sys.exit(1)


def create_vector_store(client: OpenAI, name: str, file_paths: List[str]) -> str:
    """Create a new vector store and upload files to it."""
    print(f"Creating vector store: {name}")
    
    try:
        # Try beta client first (most likely)
        if hasattr(client, 'beta') and hasattr(client.beta, 'vector_stores'):
            vector_store = client.beta.vector_stores.create(name=name)
        # Fallback to direct access
        elif hasattr(client, 'vector_stores'):
            vector_store = client.vector_stores.create(name=name)
        else:
            print("❌ Vector stores API not found in client")
            return ""
        
        vector_store_id = vector_store.id
        print(f"✅ Created vector store: {vector_store_id}")
        
        # Upload files
        if file_paths:
            upload_files_to_vector_store(client, vector_store_id, file_paths)
        
        return vector_store_id
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return ""


def upload_files_to_vector_store(client: OpenAI, vector_store_id: str, file_paths: List[str]) -> None:
    """Upload files to an existing vector store."""
    print(f"Uploading {len(file_paths)} files to vector store {vector_store_id}")
    
    file_streams = []
    try:
        # Open all files
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
            print(f"📄 Preparing: {file_path}")
            file_streams.append(open(file_path, "rb"))
        
        if not file_streams:
            print("❌ No valid files to upload")
            return
        
        # Upload files to OpenAI
        print("⬆️  Uploading files...")
        if hasattr(client, 'beta') and hasattr(client.beta, 'vector_stores'):
            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store_id, files=file_streams
            )
        elif hasattr(client, 'vector_stores'):
            file_batch = client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store_id, files=file_streams
            )
        else:
            print("❌ Vector stores API not found in client")
            return
        
        print(f"✅ Upload completed. Status: {file_batch.status}")
        print(f"📊 Files processed: {file_batch.file_counts}")
        
    finally:
        # Close all file streams
        for stream in file_streams:
            stream.close()


def list_vector_store_files(client: OpenAI, vector_store_id: str) -> None:
    """List files in a vector store."""
    print(f"Files in vector store {vector_store_id}:")
    
    try:
        # Try beta client first (most likely)
        if hasattr(client, 'beta') and hasattr(client.beta, 'vector_stores'):
            files = client.beta.vector_stores.files.list(vector_store_id=vector_store_id)
        # Fallback to direct access
        elif hasattr(client, 'vector_stores'):
            files = client.vector_stores.files.list(vector_store_id=vector_store_id)
        else:
            print("❌ Vector stores API not found in client")
            print("Available client attributes:", [attr for attr in dir(client) if not attr.startswith('_')])
            return
        
        if not files.data:
            print("📭 No files found in this vector store")
            return
        
        for file in files.data:
            # Get file details
            file_details = client.files.retrieve(file.id)
            status = "✅" if file.status == "completed" else "⏳" if file.status == "in_progress" else "❌"
            print(f"  {status} {file_details.filename} (ID: {file.id}, Status: {file.status})")
            
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        # Additional debugging
        print(f"Client type: {type(client)}")
        print(f"Client attributes: {[attr for attr in dir(client) if not attr.startswith('_')]}")


def update_bot_config(vector_store_id: str) -> None:
    """Update the bot configuration files with new vector store ID."""
    print(f"Updating bot configuration to use vector store: {vector_store_id}")
    
    # Update web app config
    web_app_path = os.path.join(os.path.dirname(__file__), "webui", "app.py")
    if os.path.exists(web_app_path):
        print(f"📝 Add this to your .env file:")
        print(f"UI_VECTOR_STORE_IDS={vector_store_id}")
        print()
        
    print(f"📝 Or use with CLI:")
    print(f"python3 CommandLine.py --vector-store-id {vector_store_id}")
    print()
    
    print("🔄 To make this permanent, update your .env file with:")
    print(f"UI_VECTOR_STORE_IDS={vector_store_id}")


def main():
    parser = argparse.ArgumentParser(description="Manage vector stores for DispatchTrack Customer Support Bot")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new vector store and upload files")
    create_parser.add_argument("--name", required=True, help="Name for the vector store")
    create_parser.add_argument("--files", nargs="+", required=True, help="Files to upload")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload files to existing vector store")
    upload_parser.add_argument("--vector-store-id", required=True, help="Vector store ID")
    upload_parser.add_argument("--files", nargs="+", required=True, help="Files to upload")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List files in vector store")
    list_parser.add_argument("--vector-store-id", required=True, help="Vector store ID")
    
    # Update config command
    config_parser = subparsers.add_parser("update-config", help="Update bot config with vector store ID")
    config_parser.add_argument("--vector-store-id", required=True, help="Vector store ID")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load API key for commands that need it
    if args.command in ["create", "upload", "list"]:
        load_api_key_or_exit()
        client = OpenAI()
    
    # Execute commands
    if args.command == "create":
        vector_store_id = create_vector_store(client, args.name, args.files)
        print(f"\n🎉 Vector store created successfully!")
        update_bot_config(vector_store_id)
        
    elif args.command == "upload":
        upload_files_to_vector_store(client, args.vector_store_id, args.files)
        
    elif args.command == "list":
        list_vector_store_files(client, args.vector_store_id)
        
    elif args.command == "update-config":
        update_bot_config(args.vector_store_id)


if __name__ == "__main__":
    main()
