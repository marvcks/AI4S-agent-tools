#!/usr/bin/env python3
"""
Generate TOOLS.json by scanning all MCP servers in the repository.
This script extracts metadata from servers decorated with @mcp_server
and reads additional information from pyproject.toml files.
"""
import json
import os
import sys
import importlib.util
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
import tomllib
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to import servers
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_server_metadata_from_ast(filepath: Path) -> Optional[Dict[str, Any]]:
    """Extract @mcp_server decorator metadata using AST parsing."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        if (isinstance(decorator.func, ast.Name) and 
                            decorator.func.id == 'mcp_server'):
                            # Extract positional arguments (name, description, author)
                            metadata = {}
                            if len(decorator.args) >= 2:
                                if isinstance(decorator.args[0], ast.Constant):
                                    metadata['name'] = decorator.args[0].value
                                if isinstance(decorator.args[1], ast.Constant):
                                    metadata['description'] = decorator.args[1].value
                            
                            # Extract keyword arguments
                            for keyword in decorator.keywords:
                                if keyword.arg == 'author':
                                    if isinstance(keyword.value, ast.Constant):
                                        metadata['author'] = keyword.value.value
                                elif keyword.arg == 'category':
                                    if isinstance(keyword.value, ast.Constant):
                                        # Map human-readable category names to category keys
                                        category_map = {
                                            'materials science': 'materials',
                                            'material science': 'materials',
                                            'material': 'materials',
                                            'chemistry': 'chemistry',
                                            'biology': 'biology',
                                            'physics': 'physics',
                                            'research tools': 'research',
                                            'research': 'research',
                                            'simulation': 'simulation',
                                            'data & analysis': 'data',
                                            'data analysis': 'data',
                                            'data': 'data',
                                            'machine learning': 'machine-learning',
                                            'ml': 'machine-learning',
                                            'general tools': 'general',
                                            'general': 'general'
                                        }
                                        raw_category = keyword.value.value.lower()
                                        metadata['category'] = category_map.get(raw_category, raw_category)
                            
                            if 'name' in metadata:
                                return metadata
    except Exception as e:
        logger.debug(f"Failed to parse {filepath}: {e}")
    
    return None


def extract_legacy_server_info(filepath: Path) -> Optional[Dict[str, Any]]:
    """Extract information from legacy servers without @mcp_server decorator."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for common patterns
        metadata = {}
        
        # Extract server name and port
        import re
        
        # Try to find server initialization
        mcp_pattern = r'(?:FastMCP|CalculationMCPServer)\s*\(\s*["\']([^"\']+)["\']'
        match = re.search(mcp_pattern, content)
        if match:
            metadata['name'] = match.group(1)
        
        # Extract author from pyproject.toml if available
        metadata['author'] = '@unknown'
        
        return metadata if 'name' in metadata else None
        
    except Exception as e:
        logger.debug(f"Failed to extract legacy info from {filepath}: {e}")
    
    return None


def read_pyproject_toml(server_dir: Path) -> Dict[str, Any]:
    """Read pyproject.toml to extract dependencies and other metadata."""
    pyproject_path = server_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return {}
    
    try:
        with open(pyproject_path, 'rb') as f:
            data = tomllib.load(f)
        
        result = {}
        
        # Extract dependencies
        if 'project' in data:
            project = data['project']
            
            if 'dependencies' in project:
                result['dependencies'] = project['dependencies']
            
            if 'description' in project:
                result['description'] = project['description']
            
            if 'name' in project:
                result['package_name'] = project['name']
        
        return result
    except Exception as e:
        logger.debug(f"Failed to read {pyproject_path}: {e}")
        return {}


def extract_tools_from_server(server_dir: Path) -> List[str]:
    """Extract tool names from server implementation."""
    tools = []
    
    # First, try to find tools in server files
    for pattern in ['server.py', '*_server.py', '*_mcp_server.py']:
        for server_file in server_dir.glob(pattern):
            try:
                with open(server_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                # Find functions decorated with @mcp_server
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check if this function has @mcp_server decorator
                        has_mcp_server = False
                        for decorator in node.decorator_list:
                            if isinstance(decorator, ast.Call):
                                if (isinstance(decorator.func, ast.Name) and 
                                    decorator.func.id == 'mcp_server'):
                                    has_mcp_server = True
                                    break
                        
                        # If we found a function with @mcp_server, look for tools inside it
                        if has_mcp_server:
                            # Look for nested function definitions with @mcp.tool() decorator
                            for inner_node in ast.walk(node):
                                if isinstance(inner_node, ast.FunctionDef) and inner_node != node:
                                    for decorator in inner_node.decorator_list:
                                        # Check for @mcp.tool() pattern
                                        if isinstance(decorator, ast.Call):
                                            if (isinstance(decorator.func, ast.Attribute) and 
                                                decorator.func.attr == 'tool' and
                                                isinstance(decorator.func.value, ast.Name)):
                                                tools.append(inner_node.name)
                                        # Also check for @mcp.tool without parentheses
                                        elif isinstance(decorator, ast.Attribute):
                                            if (decorator.attr == 'tool' and
                                                isinstance(decorator.value, ast.Name)):
                                                tools.append(inner_node.name)
            except Exception as e:
                logger.debug(f"Failed to extract tools from {server_file}: {e}")
    
    # Special handling for servers that load tools dynamically (like ABACUS)
    # Check if there's a modules directory with tool definitions
    for mod_dir in server_dir.glob("src/*/modules"):
        if mod_dir.is_dir():
            # Scan all Python files in modules directory
            for py_file in mod_dir.glob("*.py"):
                if py_file.name.startswith("_") or py_file.name in ["utils.py", "comm.py"]:
                    continue
                
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                    
                    # Look for @mcp.tool() decorated functions
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            for decorator in node.decorator_list:
                                # Check for @mcp.tool() pattern
                                if isinstance(decorator, ast.Call):
                                    if (isinstance(decorator.func, ast.Attribute) and 
                                        decorator.func.attr == 'tool' and
                                        isinstance(decorator.func.value, ast.Name) and
                                        decorator.func.value.id == 'mcp'):
                                        tools.append(node.name)
                                # Also check for @mcp.tool without parentheses
                                elif isinstance(decorator, ast.Attribute):
                                    if (decorator.attr == 'tool' and
                                        isinstance(decorator.value, ast.Name) and
                                        decorator.value.id == 'mcp'):
                                        tools.append(node.name)
                except Exception as e:
                    logger.debug(f"Failed to extract tools from module {py_file}: {e}")
    
    return list(set(tools))  # Remove duplicates


def scan_server_directory(server_dir: Path) -> Optional[Dict[str, Any]]:
    """Scan a server directory and extract all metadata."""
    # Find server.py or similar files
    server_files = []
    for pattern in ['server.py', '*_server.py', '*_mcp_server.py']:
        server_files.extend(server_dir.glob(pattern))
    
    if not server_files:
        logger.warning(f"No server files found in {server_dir}")
        return None
    
    # Try to extract metadata from each file
    metadata = None
    for server_file in server_files:
        # First try @mcp_server decorator
        metadata = extract_server_metadata_from_ast(server_file)
        
        # If not found, try legacy extraction
        if not metadata:
            metadata = extract_legacy_server_info(server_file)
        
        if metadata:
            metadata['path'] = f"servers/{server_dir.name}"
            break
    
    if not metadata:
        return None
    
    # Enhance with pyproject.toml data
    pyproject_data = read_pyproject_toml(server_dir)
    
    # Add description from pyproject if not already set
    if 'description' in pyproject_data and 'description' not in metadata:
        metadata['description'] = pyproject_data['description']
    
    # Set defaults
    metadata.setdefault('name', server_dir.name)
    metadata.setdefault('description', f"{server_dir.name} MCP server")
    metadata.setdefault('author', '@unknown')
    
    # Generate start command based on directory name
    metadata['start_command'] = f"cd {metadata['path']} && python server.py --port <PORT>"
    
    return metadata


def load_categories() -> Dict[str, Any]:
    """Load category definitions from config file."""
    config_path = Path(__file__).parent.parent / "config" / "categories.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"categories": {}, "default_category": "research"}


def categorize_tool(name: str, description: str, categories_config: Dict[str, Any]) -> str:
    """Automatically categorize a tool based on its name and description."""
    text = (name + " " + description).lower()
    
    # Keyword-based categorization
    category_keywords = {
        "simulation": ["molecule", "atom", "simulation", "dpa", "deepmd", "dynamics", "md"],
        "materials": ["material", "thermoelectric", "property", "crystal", "structure"],
        "chemistry": ["compound", "pubchem", "chemical", "catalysis", "reaction", "adsorption"],
        "research": ["paper", "search", "arxiv", "document", "rag", "literature"],
        "data": ["data", "analysis", "visualization", "plot", "graph"],
        "machine-learning": ["ml", "machine learning", "ai", "model", "neural", "network"]
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in text for keyword in keywords):
            return category
    
    return categories_config.get("default_category", "research")


def generate_tools_json(root_dir: Path) -> Dict[str, Any]:
    """Generate the complete TOOLS.json structure."""
    servers_dir = root_dir / "servers"
    
    if not servers_dir.exists():
        logger.error(f"Servers directory not found: {servers_dir}")
        return {}
    
    # Load category configuration
    categories_config = load_categories()
    
    tools = []
    
    # Scan each server directory
    for server_dir in sorted(servers_dir.iterdir()):
        if not server_dir.is_dir() or server_dir.name.startswith('_'):
            continue
        
        logger.info(f"Scanning {server_dir.name}...")
        metadata = scan_server_directory(server_dir)
        
        if metadata:
            # Extract available tools
            tools_list = extract_tools_from_server(server_dir)
            metadata['tools'] = tools_list
            
            # Use category from decorator if available, otherwise auto-categorize
            if 'category' not in metadata:
                metadata['category'] = categorize_tool(
                    metadata.get('name', ''),
                    metadata.get('description', ''),
                    categories_config
                )
            
            # Validate category exists in config
            if metadata['category'] not in categories_config['categories']:
                logger.warning(f"  Unknown category '{metadata['category']}' for {metadata['name']}, using default")
                metadata['category'] = categories_config.get('default_category', 'general')
            
            tools.append(metadata)
            logger.info(f"  Found: {metadata['name']} with {len(tools_list)} tools (category: {metadata['category']})")
    
    # Sort tools by name
    tools.sort(key=lambda x: x['name'])
    
    return {
        "version": "1.0.0",
        "description": "AI4S Agent Tools Registry - A collection of MCP servers for scientific computing",
        "categories": categories_config["categories"],
        "tools": tools
    }


def main():
    """Main function to generate TOOLS.json."""
    # Get the repository root
    root_dir = Path(__file__).parent.parent
    
    logger.info(f"Generating TOOLS.json for repository: {root_dir}")
    
    # Generate the data
    tools_data = generate_tools_json(root_dir)
    
    if not tools_data.get('tools'):
        logger.error("No tools found!")
        return 1
    
    # Write to file
    output_path = root_dir / "TOOLS.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tools_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nSuccessfully generated {output_path}")
    logger.info(f"Found {len(tools_data['tools'])} tools")
    
    # Print summary
    print("\nTools Summary:")
    print("-" * 70)
    print(f"{'Name':20} {'Author':20} {'Tools':30}")
    print("-" * 70)
    for tool in tools_data['tools']:
        tools_str = ', '.join(tool.get('tools', [])[:3])
        if len(tool.get('tools', [])) > 3:
            tools_str += '...'
        print(f"{tool['name']:20} {tool.get('author', '@unknown'):20} {tools_str:30}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())