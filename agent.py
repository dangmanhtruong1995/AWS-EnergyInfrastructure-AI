import os
region="us-east-1" #set this to AWS region you're using
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

from dataclasses import dataclass, field
from datetime import datetime
import json

from pydantic_ai import Agent, RunContext, Tool
# from pydantic_ai.models.openai import OpenAIChatModel
# from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.settings import ModelSettings
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session

from config import DATASETS, DATASET_LIST

# from schemas import DataSourceTracker

from tools import analyse_and_plot_features_and_nearby_infrastructure,\
    analyse_and_plot_within_op
from schemas import ReportMapOutput
# from tools import analyse_using_mcda_then_plot
    # get_drilling_and_production_data_source,\
    # get_ukcs_licensed_blocks_data_source,\
    # analyze_seismic_and_drilling_data,\
    # plot_seismic_and_drilling_data,\
    # analyze_seismic_events_close_to_licensed_blocks,\
    # plot_seismic_events_close_to_licensed_blocks,\
    # mcda, get_available_data_sources,\
    # analyse_and_plot_features_and_nearby_infrastructure,\
    # analyse_and_plot_within_op,\
    # analyse_using_mcda_then_plot,\
    # perform_scenario_analysis_then_plot

from tools import analyse_and_plot_features_and_nearby_infrastructure,\
    analyse_and_plot_within_op,\
    analyse_using_mcda_then_plot,\
    mcda,\
    perform_scenario_analysis_then_plot,\
    get_scenario_weights

# from config import DATASETS, DATASET_LIST
# from config import DATASET_LIST

# def get_available_data_sources():
#     """Return available UK energy data sources"""
#     return ["seismic", "wells", "licences", "pipelines", "offshore_fields"]

def show_seismic_dataset(run_context: RunContext):
    """ Show the seismic dataset.
    Args:
        None

    Return:
        data: The seismic data
    """
    import pandas as pd
    from config import get_dataset_path
    
    df = pd.read_csv(get_dataset_path("seismic"))
    return df.head().to_dict()



def get_available_data_sources(run_context: RunContext):    
    """ Return the available data sources.
    Args:
        None

    Return:
        data_source_list: A list of strings, showing the available data sources
    """

    print(run_context)
    print()

    data_source_list = DATASET_LIST
    return data_source_list


app = BedrockAgentCoreApp()

# model = BedrockConverseModel('us.anthropic.claude-sonnet-4-20250514-v1:0')
model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
# model = BedrockConverseModel('us.anthropic.claude-3-5-haiku-20241022-v1:0')

model_settings = ModelSettings(
    max_retries=6,  # Retry on throttling
    retry_delay=2.0  # Wait 2 seconds between retries
)

dummy_agent = Agent(
    model=model,
    # deps_type=DataSourceTracker,
    tools=[
        # get_available_data_sources,
        Tool(
            function=get_available_data_sources,
            takes_ctx=True, 
            description="Get list of available datasets"
        ),        
    ],
    # system_prompt=""""You're a helpful assistant. Use the tools available for you to answer questions.""",

    system_prompt="""
You're a helpful assistant. Use the tools available for you to answer questions.

SYSTEM:
Show your reasoning explicitly in <think>...</think> tags.
Keep it concise and structured.

"""
)

conversation_histories = {}

@app.entrypoint
def pydantic_bedrock_claude_main(payload):
# def agent(payload):
    """
    Invoke the agent with a payload
    """
    print("========== ENTRYPOINT CALLED ==========")
    print(f"Payload: {payload}")

    user_input = payload.get("prompt")

    session_id = payload.get("session_id", "default")
    
    # Get or create conversation history
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    # Currently, Pydantic AI does not officially support returning the results of
    # called tool directly (without summarizing). So I followed this workaround: 
    # https://github.com/pydantic/pydantic-ai/pull/142#issuecomment-3158974832
    result = dummy_agent.run_sync(user_input,
        output_type=[
            # mcda,
            analyse_and_plot_features_and_nearby_infrastructure,
            analyse_and_plot_within_op,
            analyse_using_mcda_then_plot,
            perform_scenario_analysis_then_plot,
            str],  # Functions passed here!
        model_settings=model_settings,
        message_history=conversation_histories[session_id],                   
    )   
    
    # Extract thinking and tool calls from messages
    thinking_log = []
    tool_calls_log = []
    seen_tool_calls = set()
    
    import re
    for msg in result.all_messages():
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                # Extract tool calls
                if hasattr(part, 'tool_name'):
                    tool_name = part.tool_name

                    # Remove Pydantic AI added prefix in tool names
                    if tool_name.startswith('final_result_'):
                        tool_name = tool_name.replace('final_result_', '')
                    
                    args = part.args if hasattr(part, 'args') else {}
                    tool_signature = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                    
                    if tool_signature not in seen_tool_calls:
                        seen_tool_calls.add(tool_signature)
                        tool_calls_log.append({'tool_name': tool_name, 'args': args})
                
                # Extract text content (includes <think> tags)
                elif hasattr(part, 'content') and isinstance(part.content, str):
                    # Extract thinking from <think> tags
                    think_matches = re.findall(r'<think>(.*?)</think>', part.content, re.DOTALL)
                    for think_content in think_matches:
                        thinking_log.append(think_content.strip())
    
    # Update conversation history
    conversation_histories[session_id] = result.all_messages()

    print(result.output)
    # return result.output

    # Return structured response with thinking and tool calls
    return {
        "output": result.output,
        "thinking": thinking_log,
        "tool_calls": tool_calls_log,
    }

if __name__ == "__main__":
    app.run()


# def main():
#     pass

# if __name__ == "__main__":
#     main()


