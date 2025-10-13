import os
from pdb import set_trace


def format_thinking(thinking):
    thinking_text = ""
    if thinking:
        thinking_text = '<div style="background-color: #f0f7ff; border-left: 4px solid #0066cc; padding: 15px; margin: 10px 0; border-radius: 5px;">\n\n'
        thinking_text += "**🤔 Agent's Reasoning:**\n\n"
        for thought in thinking:
            thinking_text += f"{thought}\n\n"
        thinking_text += '</div>\n\n'

    return thinking_text


def format_tools_called(tool_calls):
    tool_calls_text = ""
    if tool_calls:
        tool_calls_text = '<div style="background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 15px; margin: 15px 0; border-radius: 8px; font-family: system-ui;">\n\n'
        tool_calls_text += "**🔧 Tools Executed:**\n\n"
        
        for i, tc in enumerate(tool_calls, 1):
            tool_name = tc.get('tool_name', 'unknown')
            args = tc.get('args', {})
            
            # Different icons for different tools
            if 'mcda' in tool_name or 'scenario' in tool_name:
                icon = "📊"
            elif 'plot' in tool_name or 'map' in tool_name:
                icon = "🗺️"
            elif 'analyse' in tool_name or 'analyze' in tool_name:
                icon = "🔍"
            elif 'data' in tool_name or 'source' in tool_name:
                icon = "📁"
            else:
                icon = "⚙️"
            
            tool_calls_text += f"{icon} **`{tool_name}`**"
            
            # Show key arguments inline
            if args:
                key_args = []
                if 'scenario_name' in args:
                    key_args.append(f"scenario: `{args['scenario_name']}`")
                if 'layer_1' in args:
                    key_args.append(f"layer_1: `{args['layer_1']}`")
                if 'layer_2' in args:
                    key_args.append(f"layer_2: `{args['layer_2']}`")
                
                if key_args:
                    tool_calls_text += f" ({', '.join(key_args)})"
            
            tool_calls_text += "\n\n"
        
        tool_calls_text += '</div>\n\n'

    return tool_calls_text


def format_data_sources(data_sources):
    sources_text = ""
    if data_sources:
        # Purple/Indigo theme for data sources
        sources_text = '<div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); border-left: 5px solid #9c27b0; padding: 15px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">\n\n'
        sources_text += '<p style="margin: 0 0 10px 0; font-weight: bold; font-size: 14px; color: #6a1b9a;">📚 Data Sources</p>\n\n'
        
        for source in data_sources:
            sources_text += f'<p style="margin: 4px 0; font-size: 12px; color: #4a148c; padding-left: 8px;">• {source}</p>\n'
        
        sources_text += '</div>\n\n'

    return sources_text