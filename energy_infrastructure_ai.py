import os
import numpy as np
import pandas as pd
import pickle
from os.path import join as pjoin
import os
from pdb import set_trace
import requests
import math
from pathlib import Path
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import asyncio
from dataclasses import dataclass, field
# import PyPDF2  # or use pdfplumber, pymupdf
from io import BytesIO
from typing import Set, Union
import json
import base64

# from pyproj import Transformer
# from geopy.geocoders import Nominatim
# import utm
# from shapely.geometry import Point
# import geopandas as gpd
# from geopandas import GeoDataFrame
# import geodatasets
import zipfile

# import pydantic_ai
# from pydantic import BaseModel
# from pydantic_ai import Agent, BinaryContent, RunContext
# from pydantic_ai.models.openai import OpenAIChatModel
# from pydantic_ai.providers.ollama import OllamaProvider
# from pydantic_ai.messages import ModelRequest, ToolReturnPart

from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session
import boto3

import gradio as gr
import uuid
# from config import BASE_PATH, HISTORY_FILE
# from schemas import DataSourceTracker, GetWellEntryInput, WellEntryOutput, DataSourceOutput, SeismicAndDrillingInput, SeismicAndDrillingOutput, PlotOutput
# from document_processor import extract_text_from_pdf
# from data_loader import get_coords
# from seismic_analysis_python import SeismicDrillingAnalyzer
# from agent import agent


boto_session = Session()
region = boto_session.region_name

agentcore_runtime = Runtime()
agent_name = "agentcore_pydantic_bedrockclaude_v11"

# Step 1: Configure (generates standard Dockerfile)
response = agentcore_runtime.configure(
    entrypoint="agent.py",
    auto_create_execution_role=True,
    auto_create_ecr=True,
    requirements_file="requirements.txt",
    region=region,
    agent_name=agent_name
)

# Step 2: Modify Dockerfile to add GDAL
dockerfile_content = open("Dockerfile", "r").read()

# Insert GDAL installation after FROM line
gdal_install = """
RUN apt-get update && apt-get install -y \\
    gdal-bin \\
    libgdal-dev \\
    libspatialindex-dev \\
    build-essential \\
    g++ \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

"""

lines = dockerfile_content.split('\n')
new_dockerfile = []
for line in lines:
    new_dockerfile.append(line)
    if line.startswith('FROM '):
        new_dockerfile.append(gdal_install)

with open("Dockerfile", "w") as f:
    f.write('\n'.join(new_dockerfile))

print("✅ Modified Dockerfile with GDAL")

# Step 3: Rezip and upload to S3
s3 = boto3.client('s3', region_name=region)

# Create new zip with modified Dockerfile
with zipfile.ZipFile('/tmp/source_modified.zip', 'w') as zipf:
    for root, dirs, files in os.walk('.'):
        for file in files:
            if not file.endswith('.zip'):
                zipf.write(os.path.join(root, file))

# Upload modified zip
# bucket = f"bedrock-agentcore-{region}-436355390679"  # May vary
bucket = f"bedrock-agentcore-codebuild-sources-436355390679-{region}"
key = f"{agent_name}/source.zip"

try:
    s3.upload_file('/tmp/source_modified.zip', bucket, key)
    print("✅ Uploaded modified source to S3")
except Exception as e:
    print(f"⚠️  Could not upload to S3: {e}")
    print("You may need to find the correct bucket name")

# Step 4: Now launch
launch_result = agentcore_runtime.launch()
print(launch_result)


# === Function to Query Bedrock Agent ===
# def query_agent(user_message, session_id):
#     """
#     Send a prompt to the deployed AWS Bedrock AgentCore runtime.
#     Each user has their own isolated session (using session_id).
#     """
#     try:
#         print(f"🔹 Invoking agent for session {session_id}...")
#         payload = {"prompt": user_message, "session_id": session_id}
#         response = agentcore_runtime.invoke(payload)
#         # set_trace()

#         try:
#             response_json_string = ''.join(response['response'])
#             response_json_string = json.loads(response_json_string)
#             data = json.loads(response_json_string)
#             result = data['report']
#             map_html = data['map_html']
#         except:
#             # No HTML attached, so just return summarized response
#             # AgentCore returns structured dict
#             result = response.get("response", [""])[0]
        
#             # Clean up escape characters
#             result = result.replace("\\n", "\n").replace('\\"', '"')
#             map_html = ""

#         return result, map_html, None

#     except Exception as e:
#         print(f"❌ Error in query_agent: {e}")
#         return f"⚠️ Error contacting the analysis agent: {e}", ""


def query_agent(user_message, session_id):
    """
    Send a prompt to the deployed AWS Bedrock AgentCore runtime.
    Each user has their own isolated session (using session_id).
    """
    try:
        print(f"🔹 Invoking agent for session {session_id}...")
        payload = {"prompt": user_message, "session_id": session_id}
        response = agentcore_runtime.invoke(payload)

        try:
            # Get the response - handle both string and bytes
            # response_string = ''.join(response['response'])

            response_parts = response.get('response', [''])
            
            # Convert any bytes to strings
            response_list = []
            for part in response_parts:
                if isinstance(part, bytes):
                    response_list.append(part.decode('utf-8'))
                else:
                    response_list.append(str(part))
            
            response_string = ''.join(response_list)


            print(f"📝 Raw response length: {len(response_string)} chars")
            print(f"📝 Raw response preview: {response_string[:200]}...")
            
            # Try to parse as JSON (from mcda tool direct return)
            try:
                data = json.loads(response_string)
                # Check if it's the mcda output format
                if isinstance(data, dict) and 'report' in data and 'map_html' in data:
                    result = data['report']
                    map_html = data['map_html']
                    print(f"✅ Parsed mcda JSON output - map_html length: {len(map_html)} chars")
                    
                    # Verify map_html is complete (should have closing </html> tag)
                    if map_html and not map_html.strip().endswith('</html>'):
                        print("⚠️ Warning: map_html appears truncated (no closing </html> tag)")
                    
                    return result, map_html, None
            except json.JSONDecodeError as e:
                print(f"ℹ️ Direct JSON parse failed: {e}")
            
            # Try double-nested JSON parsing (if Agent wrapped it)
            try:
                outer_json = json.loads(response_string)
                inner_json = json.loads(outer_json)
                if isinstance(inner_json, dict) and 'report' in inner_json and 'map_html' in inner_json:
                    result = inner_json['report']
                    map_html = inner_json['map_html']
                    print(f"✅ Parsed double-nested JSON output - map_html length: {len(map_html)} chars")
                    
                    # Verify map_html is complete
                    if map_html and not map_html.strip().endswith('</html>'):
                        print("⚠️ Warning: map_html appears truncated (no closing </html> tag)")
                    
                    return result, map_html, None
            except (json.JSONDecodeError, TypeError) as e:
                print(f"ℹ️ Double-nested JSON parse failed: {e}")
            
            # Fall back to plain text response (summarized output)
            result = response.get("response", [""])[0]
            result = result.replace("\\n", "\n").replace('\\"', '"')
            map_html = ""
            print("ℹ️ Using plain text response (no map)")
            
        except Exception as parse_error:
            print(f"⚠️ Parse error: {parse_error}")
            import traceback
            traceback.print_exc()
            result = response.get("response", [""])[0]
            result = result.replace("\\n", "\n").replace('\\"', '"')
            map_html = ""

        return result, map_html, None

    except Exception as e:
        print(f"❌ Error in query_agent: {e}")
        import traceback
        traceback.print_exc()
        return f"⚠️ Error contacting the analysis agent: {e}", "", None



def generate_data_source_footer(used_sources: Set[str]) -> str:
    if not used_sources:
        return ""
    
    sources = list(used_sources)
    if len(sources) == 1:
        return f"\n\n---\n**Data source:** *{sources[0]} (updated 2025)*"
    else:
        footer = "\n\n---\n**Data sources:**\n"
        for source in sources:
            footer += f"* *{source} (updated 2025)*\n"
        return footer.rstrip()  # Remove trailing newline


# Identify areas in the UK with high seismic survey density but low recent drilling activity
# Perform a multi-criteria decision analysis for the licensing blocks in the UK, rank by safety, environment, technical and economic.
# Find all wells which are within 10 kilometres of existing pipelines
# Next, I want you to identify seismic activities which are within 10 kilometres of licence blocks
# Do a scenario modeling for the licence blocks in the UK, with a focus on safety. 
# What are the available data sources ?
# Show all seismic events which fall into offshore fields

# === Gradio Frontend ===
def create_enhanced_interface():
    # Custom CSS for better styling
    try:
        with open('templates/visual.css', 'r') as file:
            custom_css = file.read()
    except:
        custom_css = "body {background-color: #f7f9fb;}"

    with gr.Blocks(
        css=custom_css,
        title="GeoAnalysis AI",
        theme=gr.themes.Soft()) as demo:
        
        # Session state per user
        session_state = gr.State(value=None)

        # Header with logo
        try:
            with open("logo/logo.png", "rb") as f:
                data = base64.b64encode(f.read()).decode()
            img_tag = f'<img src="data:image/png;base64,{data}" class="logo-img" alt="GeoAnalysis AI Logo">'
            
            with open('templates/header.htmltemplate', 'r') as file:
                header_html = file.read()
            gr.HTML(header_html.format(img_tag=img_tag))
        except:
            gr.Markdown("# 🧭 GeoAnalysis AI\n**Powered by AWS Bedrock + Pydantic AI**")

        # Feature overview cards
        try:
            with open('templates/feature_overview.htmltemplate', 'r') as file:
                feature_overview_html = file.read()
            gr.HTML(feature_overview_html)
        except:
            gr.Markdown("Perform intelligent multi-criteria decision analyses and visualize results interactively.")
        
        with gr.Row():            
            # Main chat interface
            with gr.Column(scale=3):
                # Chat display
                try:
                    chatbot_display = gr.Chatbot(
                        height=500,
                        elem_classes=["chat-container"],
                        avatar_images=("./logo/rig-icon-oil-worker-symbol.png", "./logo/logo.png"),
                        bubble_full_width=False,
                        show_copy_button=True,
                        label="Chat"
                    )
                except:
                    chatbot_display = gr.Chatbot(
                        height=500,
                        label="Chat"
                    )
                
                # Input area
                with gr.Row():
                    textbox = gr.Textbox(
                        placeholder="💬 Ask me about UK energy infrastructure, seismic data, or drilling locations...",
                        container=False,
                        scale=7,
                        show_label=False
                    )
                    send_btn = gr.Button("Send 🚀", scale=1, size="lg")
                
                # Control buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", size="sm")
                    retry_btn = gr.Button("🔄 Retry", size="sm")
            
            # HTML display for plots - matched height
            with gr.Column(scale=3):
                plot_display = gr.HTML(
                    value="<div style='height: 580px; display: flex; align-items: center; justify-content: center; color: #666;'>Analysis visualizations will appear here</div>",
                    label="📊 Generated Analysis Plot"
                )

        # === Chat Functions ===
        def chat_fn(message, history, session_state):
            """Handle chat with proper session management"""
            # Generate or retrieve session ID
            session_id = session_state or str(uuid.uuid4())

            if not message.strip():
                return history, "", gr.HTML(value="<div style='height: 400px; display: flex; align-items: center; justify-content: center; color: #666;'>Analysis visualizations will appear here</div>"), session_id
            
            # Add user message to history
            history = history + [(message, None)]
            
            # Call the agent
            response, html_content, legend_content = query_agent(message, session_id)
            
            print(f"🔍 Response length: {len(response)} chars")
            print(f"🔍 HTML content length: {len(html_content) if html_content else 0} chars")
            print(f"🔍 Has HTML content: {bool(html_content)}")
            
            if html_content and response:
                print("📝 HTML detected - generating summary of full report...")
                summary_prompt = f"""Please provide a concise summary (2-3 sentences) of this analysis report, highlighting the key findings:
QUERY:
{message}
                
REPORT:
{response[:2000]}  

Keep the summary brief and actionable."""
                
                bedrock_client = boto3.client('bedrock-runtime', region_name=region)
                
                # Call Claude directly via Bedrock
                bedrock_response = bedrock_client.converse(
                    modelId='us.anthropic.claude-3-5-haiku-20241022-v1:0',
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": summary_prompt}]
                        }
                    ],
                    inferenceConfig={
                        "maxTokens": 200,
                        "temperature": 0.3
                    }
                )
                
                summary_response = bedrock_response['output']['message']['content'][0]['text']
                
                # Combine: Summary first, then full report
                combined_response = f"**Summary:**\n{summary_response}\n\n---\n\n**Full Report:**\n{response}"
                response = combined_response
                print(f"✅ Added summary. New response length: {len(response)} chars")

            # Add bot response to history
            history[-1] = (message, response)
            
            # Process HTML content
            if html_content:
                print("✅ HTML content detected, processing for display...")
                
                # Save HTML to file for inspection
                import time
                timestamp = int(time.time())
                html_filename = f"debug_map_{session_id}_{timestamp}.html"
                try:
                    with open(html_filename, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    print(f"💾 Saved HTML to {html_filename} for inspection")
                except Exception as e:
                    print(f"⚠️ Could not save HTML file: {e}")
                
                # Fix Folium responsive container issue
                html_content = html_content.replace(
                    'height:0;padding-bottom:60%;', 
                    'height:600px;padding-bottom:0;'
                )
                html_content = html_content.replace('height:0;', 'height:600px;')

                # Use iframe approach for better compatibility with Gradio
                import html as html_module
                escaped_html = html_module.escape(html_content)
                iframe_html = f'''
                <iframe 
                    srcdoc="{escaped_html}" 
                    style="width:100%; height:600px; border:1px solid #ddd; border-radius:4px;"
                    frameborder="0"
                ></iframe>
                '''
                
                print(f"✅ Created iframe wrapper, returning HTML output of length: {len(iframe_html)} chars")
                return history, "", gr.HTML(value=iframe_html), session_id
            else:
                print("ℹ️ No HTML content, returning empty visualization")
                return history, "", gr.HTML(value="<div style='height: 400px; display: flex; align-items: center; justify-content: center; color: #666;'>No visualization generated</div>"), session_id

        def clear_chat():
            """Clear chat history and reset"""
            return [], "", gr.HTML(value="<div style='height: 400px; display: flex; align-items: center; justify-content: center; color: #666;'>Analysis visualizations will appear here</div>"), None

        def retry_last(history, session_state):
            """Retry the last message"""
            if history:
                last_message = history[-1][0]
                history = history[:-1]
                return chat_fn(last_message, history, session_state)
            return history, "", gr.HTML(value="<div style='height: 400px; display: flex; align-items: center; justify-content: center; color: #666;'>No message to retry</div>"), session_state

        # === Event Handlers ===
        send_btn.click(
            chat_fn,
            inputs=[textbox, chatbot_display, session_state],
            outputs=[chatbot_display, textbox, plot_display, session_state]
        )
        
        textbox.submit(
            chat_fn,
            inputs=[textbox, chatbot_display, session_state],
            outputs=[chatbot_display, textbox, plot_display, session_state]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot_display, textbox, plot_display, session_state]
        )
        
        retry_btn.click(
            retry_last,
            inputs=[chatbot_display, session_state],
            outputs=[chatbot_display, textbox, plot_display, session_state]
        )
        
        # Footer
        try:
            with open('templates/footer.htmltemplate', 'r') as file:
                footer_html = file.read()
            gr.HTML(footer_html)
        except:
            gr.Markdown("---\nBuilt with ❤️ using **AWS Bedrock AgentCore + Pydantic AI + Folium**")
    
    return demo

# Launch the app
demo = create_enhanced_interface()
demo.launch(
    favicon_path="logo/logo.png" if os.path.exists("logo/logo.png") else None,
    server_name="0.0.0.0",
    server_port=7860
)


