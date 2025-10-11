from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session
import os
import zipfile
import boto3

import gradio as gr
import json
from pdb import set_trace

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
bucket = f"bedrock-agentcore-{region}-436355390679"  # May vary
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

# What are the available data sources? 
# # invoke_response = agentcore_runtime.invoke({
# #     "prompt": "Perform a multi-criteria decision analysis for the licensing blocks in the UK, rank by safety, environment, technical and economic."
# # })
# # print(invoke_response)

# === Function to Query Bedrock Agent ===
def query_agent(user_message, session_id):
    """
    Send a prompt to the deployed AWS Bedrock AgentCore runtime.
    Each user has their own isolated session (using session_id).
    """
    try:
        print(f"🔹 Invoking agent for session {session_id}...")
        payload = {"prompt": user_message, "session_id": session_id}
        response = agentcore_runtime.invoke(payload)
        # set_trace()

        try:
            response_json_string = ''.join(response['response'])
            response_json_string = json.loads(response_json_string)
            data = json.loads(response_json_string)
            result = data['report']
            map_html = data['map_html']
        except:
            # No HTML attached, so just return summarized response
            # AgentCore returns structured dict
            result = response.get("response", [""])[0]
        
            # Clean up escape characters
            result = result.replace("\\n", "\n").replace('\\"', '"')
            map_html = ""

        return result, map_html

    except Exception as e:
        print(f"❌ Error in query_agent: {e}")
        return f"⚠️ Error contacting the analysis agent: {e}", ""


# === Gradio Frontend ===
with gr.Blocks(css="body {background-color: #f7f9fb;}") as demo:
    gr.Markdown(
        """
        # 🧭 UK Oil & Gas Analysis Agent  
        **Powered by AWS Bedrock + Pydantic AI**  
        Perform intelligent multi-criteria decision analyses and visualize results interactively.
        """
    )

    # Session state per user
    session_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            msg = gr.Textbox(
                label="Your Query",
                placeholder="e.g., Perform MCDA analysis ranking by safety and environment.",
                lines=3,
            )
            submit = gr.Button("Submit", variant="primary")
            clear = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=2):
            output_text = gr.Markdown(label="Analysis Report")
            output_map = gr.HTML(label="Interactive Map", visible=True)

    # === Logic Handlers ===
    def handle_submit(user_message, session_state):
        """Handles user queries while preserving per-user session state."""
        import uuid
        session_id = session_state or str(uuid.uuid4())
        report, map_html = query_agent(user_message, session_id)
        return report, map_html, session_id

    def handle_clear():
        """Reset the output display."""
        return "", "", None

    # === Bind events ===
    submit.click(handle_submit, inputs=[msg, session_state], outputs=[output_text, output_map, session_state])
    clear.click(handle_clear, outputs=[output_text, output_map, session_state])

    gr.Markdown("---")
    gr.Markdown("Built with ❤️ using **AWS Bedrock AgentCore + Pydantic AI + Folium**")

demo.launch(debug=True, server_name="0.0.0.0", server_port=7860)
