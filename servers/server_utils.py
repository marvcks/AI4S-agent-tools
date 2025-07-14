"""
Utility module for AI4S MCP servers.
Provides decorators and helper functions to simplify server development.
"""
import argparse
import inspect
import logging
from typing import Dict, Optional, Callable, Any, List
from functools import wraps

# Server metadata registry
_server_registry: Dict[str, Dict[str, Any]] = {}


def mcp_server(name: str, description: str, author: str, category: Optional[str] = None):
    """
    Enhanced decorator to register MCP server metadata and auto-export tool functions.
    
    Args:
        name: Server name
        description: Server description  
        author: Tool author/contributor
        category: Tool category (e.g., "biology", "chemistry", "materials", "physics", "research")
    
    Usage:
        @mcp_server("PubChem", "Chemical compound data retrieval", author="@deepmodeling", category="chemistry")
        def create_server(host="0.0.0.0", port=50001):
            mcp = FastMCP("pubchem", host=host, port=port)
            # Define tools...
            return mcp
            
    The decorator now automatically exports tool functions defined inside create_server to module level,
    making them importable: from server import tool_function_name
    """
    def decorator(func):
        # Register minimal metadata
        metadata = {
            "name": name,
            "description": description,
            "author": author
        }
        
        # Add category if specified
        if category:
            metadata["category"] = category.lower()
        
        _server_registry[name] = metadata
        
        # Attach metadata to function
        func.__mcp_metadata__ = metadata
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Execute create_server
            mcp_instance = func(*args, **kwargs)
            
            # Try to extract and export tool functions
            try:
                tools_to_export = {}
                
                # Check for FastMCP-style tool manager first
                if hasattr(mcp_instance, '_tool_manager') and hasattr(mcp_instance._tool_manager, '_tools'):
                    tools_dict = mcp_instance._tool_manager._tools
                    for tool_name, tool_obj in tools_dict.items():
                        # FastMCP stores the function in the 'fn' attribute of Tool objects
                        if hasattr(tool_obj, 'fn') and callable(tool_obj.fn):
                            tools_to_export[tool_name] = tool_obj.fn
                else:
                    # Fallback: Try other common attributes where tools might be stored
                    potential_attrs = ['_tools', 'tools', '_handlers', 'handlers', '_registry']
                    
                    for attr in potential_attrs:
                        if hasattr(mcp_instance, attr):
                            tools_dict = getattr(mcp_instance, attr)
                            if isinstance(tools_dict, dict):
                                for tool_name, tool_info in tools_dict.items():
                                    # Extract the actual function
                                    if callable(tool_info):
                                        tools_to_export[tool_name] = tool_info
                                    elif isinstance(tool_info, dict):
                                        # Some MCP implementations wrap functions in dicts
                                        for key in ['handler', 'func', 'function', 'callback', 'fn']:
                                            if key in tool_info and callable(tool_info[key]):
                                                tools_to_export[tool_name] = tool_info[key]
                                                break
                                    elif hasattr(tool_info, 'fn') and callable(tool_info.fn):
                                        # Handle object-style tool storage
                                        tools_to_export[tool_name] = tool_info.fn
                                if tools_to_export:
                                    break
                
                # Export tools to module namespace
                if tools_to_export:
                    frame = inspect.currentframe()
                    if frame and frame.f_back:
                        module_globals = frame.f_back.f_globals
                        
                        # Export each tool function
                        for tool_name, tool_func in tools_to_export.items():
                            if not tool_name.startswith('_'):  # Skip private functions
                                module_globals[tool_name] = tool_func
                        
                        # Update __all__ for proper import behavior
                        exported_names = [n for n in tools_to_export.keys() if not n.startswith('_')]
                        if exported_names:
                            if '__all__' in module_globals:
                                # Extend existing __all__
                                existing = module_globals['__all__']
                                if isinstance(existing, list):
                                    for name in exported_names:
                                        if name not in existing:
                                            existing.append(name)
                            else:
                                # Create new __all__
                                module_globals['__all__'] = exported_names
                            
                            logging.debug(f"Exported {len(exported_names)} tool functions: {', '.join(exported_names)}")
                
            except Exception as e:
                # Don't break the server if export fails
                logging.warning(f"Failed to auto-export tool functions: {e}")
            
            return mcp_instance
        
        return wrapper
    return decorator


def setup_server(create_server_func: Optional[Callable] = None) -> Any:
    """
    Minimal setup for MCP server with command line argument parsing.
    
    Args:
        create_server_func: Function decorated with @mcp_server (optional, auto-detected)
    
    Returns:
        The configured MCP server instance
    
    Usage:
        if __name__ == "__main__":
            setup_server().run()  # One line!
    """
    # Find the decorated function if not provided
    if create_server_func is None:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            for name, obj in frame.f_back.f_globals.items():
                if callable(obj) and hasattr(obj, '__mcp_metadata__'):
                    create_server_func = obj
                    break
    
    if create_server_func is None or not hasattr(create_server_func, '__mcp_metadata__'):
        raise ValueError("No function decorated with @mcp_server found")
    
    metadata = create_server_func.__mcp_metadata__
    
    # Simple argument parser
    parser = argparse.ArgumentParser(description=metadata['description'])
    parser.add_argument('--port', type=int, required=True, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info(f"Starting {metadata['name']} on {args.host}:{args.port}")
    

    return create_server_func(host=args.host, port=args.port)


def get_server_registry() -> Dict[str, Dict[str, Any]]:
    """Get the current server registry for tool discovery."""
    return _server_registry.copy()


