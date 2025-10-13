from agent import dummy_agent, conversation_histories
from pydantic_ai.settings import ModelSettings
import json
from pdb import set_trace

from tools import analyse_and_plot_features_and_nearby_infrastructure,\
    analyse_and_plot_within_op,\
    analyse_using_mcda_then_plot,\
    mcda,\
    perform_scenario_analysis_then_plot,\
    get_scenario_weights

model_settings = ModelSettings(max_retries=6, retry_delay=2.0)

def test_local(user_input, session_id="test-session"):
    """Test agent locally"""
    
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    # Run the agent
    result = dummy_agent.run_sync(
        user_input,
        message_history=conversation_histories[session_id],
        output_type=[
            analyse_and_plot_features_and_nearby_infrastructure,
            analyse_and_plot_within_op,
            analyse_using_mcda_then_plot,
            perform_scenario_analysis_then_plot,
            str,
        ],
        model_settings=model_settings,
    )
    
    # Extract thinking and tool calls
    thinking_log = []
    tool_calls_log = []
    seen_tool_calls = set()
    
    import re
    for msg in result.all_messages():
        if hasattr(msg, 'parts'):
            for part in msg.parts:
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
                    
                    # if tool_signature not in seen_tool_calls:
                        # seen_tool_calls.add(tool_signature)
                    if tool_name not in seen_tool_calls:
                        seen_tool_calls.add(tool_name)
                        tool_calls_log.append({'tool_name': tool_name, 'args': args})
                        print(f"  ✅ Added to log")
                    else:
                        print(f"  ⏭️ Skipped (duplicate)")
                
                elif hasattr(part, 'content') and isinstance(part.content, str):
                    think_matches = re.findall(r'<think>(.*?)</think>', part.content, re.DOTALL)
                    for think_content in think_matches:
                        thinking_log.append(think_content.strip())
    
    conversation_histories[session_id] = result.all_messages()
    set_trace()
    return {
        "output": result.output,
        "thinking": thinking_log,
        "tool_calls": tool_calls_log,
    }

if __name__ == "__main__":
    # Test your changes
    response = test_local("Do a scenario modeling for the licence blocks in the UK, with a focus on safety.")
    
    print("="*80)
    print("THINKING:")
    for thought in response['thinking']:
        print(f"  - {thought[:100]}...")
    
    print("\nTOOL CALLS:")
    for tc in response['tool_calls']:
        print(f"  - {tc['tool_name']}")
    
    print("\nOUTPUT:")
    print(response['output'][:500])
