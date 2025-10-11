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

from tools import mcda

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
# model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
model = BedrockConverseModel('us.anthropic.claude-3-5-haiku-20241022-v1:0')

model_settings = ModelSettings(
    max_retries=6,  # Retry on throttling
    retry_delay=2.0  # Wait 2 seconds between retries
)

dummy_agent = Agent(
    model=model,
    # deps_type=DataSourceTracker,
    tools=[
        # get_available_data_sources,
        Tool(function=get_available_data_sources, takes_ctx=True)


        # get_seismic_data_source, 
        # get_drilling_and_production_data_source,
        # get_ukcs_licensed_blocks_data_source,
        # analyze_seismic_and_drilling_data,
        # plot_seismic_and_drilling_data,
        # analyze_seismic_events_close_to_licensed_blocks,
        # plot_seismic_events_close_to_licensed_blocks,
        # mcda,
        
        # analyse_and_plot_features_and_nearby_infrastructure,
        # analyse_and_plot_within_op,
        # analyse_using_mcda_then_plot,
        # perform_scenario_analysis_then_plot
        # show_seismic_dataset,
    ],
    # system_prompt=""""You're a helpful assistant. Use the tools available for you to answer questions.""",

    system_prompt="""
You are a data analysis agent. 

When the user asks about available data sources, call get_available_data_sources and provide a natural language summary.

When the user asks for analysis (MCDA, scenario analysis, etc.), call mcda which will return structured results.

IMPORTANT: After calling a tool once and getting its result, provide your answer. Do NOT call the same tool repeatedly."""

# You are a data analysis agent. Your primary function is to execute the available tools to answer the user's request.

# CRITICAL INSTRUCTION: When a tool returns a JSON object containing an 'report' and 'map_html' key, you MUST return the raw, unadulterated JSON string as your final response. DO NOT attempt to summarize, interpret, or modify the JSON. Your final output in this case must be ONLY the raw JSON string."""

#     system_prompt=""""You are a data analyst specialized in oil and gas analysis.

# CRITICAL: You MUST actually call tools, not just describe calling them.

# For seismic/drilling analysis, follow this EXACT sequence:
# 1. Call get_seismic_data_source() - wait for the actual file path result
# 2. Call get_drilling_and_production_data_source() - wait for the actual file path result  
# 3. Call analyze_seismic_and_drilling_data() using the REAL paths from steps 1&2
# 4. Call plot_seismic_and_drilling_data() using the REAL paths from steps 1&2

# For other types of analysis, follow a similar sequence. Namely get the relevant data sources, then call the relevant analysis tool, then call the plot tool.

# DO NOT make up data or locations. DO NOT provide analysis without calling all three tools.
# DO NOT generate fake responses. Only respond after actually executing the analysis tools.
# Return the report in step 3 in full, followed by a summary. DO NOT only return the summary without the report.

# If you cannot get real data from the tools, say "I need to call the data analysis tools first" and stop.""",
)

@app.entrypoint
def pydantic_bedrock_claude_main(payload):
# def agent(payload):
    """
    Invoke the agent with a payload
    """
    print("========== ENTRYPOINT CALLED ==========")
    print(f"Payload: {payload}")

    user_input = payload.get("prompt")
    result = dummy_agent.run_sync(user_input,
            output_type=[mcda, str],  # Functions passed here!
            model_settings=model_settings,                   
        )
    print(result.output)
    # return result.output, result.all_messages # To get output of tools
    # return result
    return result.output

if __name__ == "__main__":
    app.run()


# def main():
#     pass

# if __name__ == "__main__":
#     main()


