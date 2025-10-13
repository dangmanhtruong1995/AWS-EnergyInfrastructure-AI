from agent import dummy_agent, conversation_histories
from pydantic_ai.settings import ModelSettings
import json

model_settings = ModelSettings(max_retries=6, retry_delay=2.0)

def test_local(user_input, session_id="test-session"):
    """Test agent locally"""
    
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    # Run the agent
    result = dummy_agent.run_sync(
        user_input,
        message_history=conversation_histories[session_id],
        output_type=[str],
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
                    if tool_name.startswith('final_result_'):
                        tool_name = tool_name.replace('final_result_', '')
                    
                    args = part.args if hasattr(part, 'args') else {}
                    tool_signature = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                    
                    if tool_signature not in seen_tool_calls:
                        seen_tool_calls.add(tool_signature)
                        tool_calls_log.append({'tool_name': tool_name, 'args': args})
                
                elif hasattr(part, 'content') and isinstance(part.content, str):
                    think_matches = re.findall(r'<think>(.*?)</think>', part.content, re.DOTALL)
                    for think_content in think_matches:
                        thinking_log.append(think_content.strip())
    
    conversation_histories[session_id] = result.all_messages()
    
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
