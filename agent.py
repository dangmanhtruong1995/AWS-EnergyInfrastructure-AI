import os
region="us-east-1" #set this to AWS region you're using
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

from dataclasses import dataclass, field
from datetime import datetime
import json
import re

from pydantic_ai import Agent, RunContext, Tool
# from pydantic_ai.models.openai import OpenAIChatModel
# from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.settings import ModelSettings
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session

from config import DATASETS, DATASET_LIST
from schemas import DataSourceTracker, ReportMapOutput
from tools import analyse_and_plot_features_and_nearby_infrastructure,\
    analyse_and_plot_within_op,\
    analyse_using_mcda_then_plot,\
    perform_scenario_analysis_then_plot,\
    get_scenario_weights,\
    geocode_location,\
    assess_seismic_risk_at_location,\
    assess_infrastructure_proximity,\
    calculate_overall_risk_score,\
    create_risk_assessment_map,\
    plan_low_impact_exploration_sites,\
    plan_global_wind_farm_sites,\
    analyze_wind_farm_constraints
    

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
# model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
model = BedrockConverseModel('us.anthropic.claude-3-5-haiku-20241022-v1:0')

model_settings = ModelSettings(
    max_retries=6,  # Retry on throttling
    retry_delay=5.0   # Wait between retries
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
        Tool(
            function=geocode_location,
            takes_ctx=True, 
            description="Convert a location name or address to latitude and longitude coordinates."
        ),
        Tool(
            function=assess_seismic_risk_at_location,
            takes_ctx=True, 
            description="Assess seismic risk by counting earthquake events within a radius of a specific location."
        ),
        Tool(
            function=assess_infrastructure_proximity,
            takes_ctx=True, 
            description="Assess infrastructure proximity risk by counting existing infrastructure within a radius."
        ),

        Tool(
            function=calculate_overall_risk_score,
            takes_ctx=True, 
            description="Calculate overall risk score from individual risk components and provide recommendations."
        ),
        # analyze_wind_farm_constraints,
        
        # calculate_overall_risk_score,
        # create_risk_assessment_map,


    ],
    deps_type=DataSourceTracker,

    system_prompt = """
You're a helpful assistant specialized in energy infrastructure analysis and risk assessment. 

IMPORTANT: When users ask for comprehensive analyses like "assess risk", "evaluate location", or "analyze infrastructure", you should use MULTIPLE tools in sequence to provide complete answers.

TOOL CHAINING PATTERNS:

For "risk assessment" queries:
1. If location is a name/address → geocode_location first
2. Then assess_seismic_risk_at_location 
3. Then assess_infrastructure_proximity
4. Then calculate_overall_risk_score
5. Finally create_risk_assessment_map for visualization

For "infrastructure analysis":
1. Use appropriate analysis tools (analyse_and_plot_features_and_nearby_infrastructure, etc.)
2. Add mapping/visualization tools when helpful

For "multi-criteria analysis":
1. Use MCDA tools (analyse_using_mcda_then_plot, perform_scenario_analysis_then_plot)

CRITICAL: Don't stop after just getting coordinates or one piece of information. The user expects a complete analysis when they ask for "assessment" or "analysis".

When you get coordinates from geocode_location, immediately use those coordinates in subsequent risk assessment tools.

SYSTEM:
Show your reasoning explicitly in <think>...</think> tags.
Keep it concise and structured.
Continue analysis until you've fully answered the user's question.
"""


    # system_prompt=""""You're a helpful assistant. Use the tools available for you to answer questions.""",

#     system_prompt="""
# You're a helpful assistant. Use the tools available for you to answer questions.

# SYSTEM:
# Show your reasoning explicitly in <think>...</think> tags.
# Keep it concise and structured.

# """
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
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []

    deps = DataSourceTracker()

    # Currently, Pydantic AI does not officially support returning the results of
    # called tool directly (without summarizing). So I followed this workaround: 
    # https://github.com/pydantic/pydantic-ai/pull/142#issuecomment-3158974832
    result = dummy_agent.run_sync(user_input,
        deps=deps,
        output_type=[
            analyse_and_plot_features_and_nearby_infrastructure,
            analyse_and_plot_within_op,
            analyse_using_mcda_then_plot,
            perform_scenario_analysis_then_plot,
            create_risk_assessment_map,
            plan_low_impact_exploration_sites,
            plan_global_wind_farm_sites,            
            str],  # Functions passed here!
        model_settings=model_settings,
        message_history=conversation_histories[session_id],                   
    )   
    
    # Extract thinking and tool calls from messages
    thinking_log = []
    tool_calls_log = []
    seen_tool_calls = set()
    
    
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
                    
                    # More robust signature - handle empty args
                    if args:
                        tool_signature = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                    else:
                        tool_signature = f"{tool_name}:no_args"
                    
                    # Debug: print what we're seeing
                    print(f"🔍 Tool detected: {tool_name}, Args: {args}, Signature: {tool_signature}")
                    
                    if tool_name not in seen_tool_calls:
                        seen_tool_calls.add(tool_name)
                        tool_calls_log.append({'tool_name': tool_name, 'args': args})
                        print(f"  ✅ Added to log")
                    else:
                        print(f"  ⏭️ Skipped (duplicate)")
                        
                # Extract text content (includes <think> tags)
                elif hasattr(part, 'content') and isinstance(part.content, str):
                    # Extract thinking from <think> tags
                    think_matches = re.findall(r'<think>(.*?)</think>', part.content, re.DOTALL)
                    for think_content in think_matches:
                        thinking_log.append(think_content.strip())
    
    conversation_histories[session_id] = result.all_messages()
    data_sources = deps.get_sources()

    print(result.output)

    # Return structured response with thinking and tool calls
    return {
        "output": result.output,
        "thinking": thinking_log,
        "tool_calls": tool_calls_log,
        "data_sources": data_sources,
    }

if __name__ == "__main__":
    app.run()


# def main():
#     pass

# if __name__ == "__main__":
#     main()


