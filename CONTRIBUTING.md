# Contributing to AI4S-agent-tools

Welcome to the DeepModeling community! 🎉

## 🚀 Quick Start: Add Your Scientific Tool in 5 Minutes

### 1. Copy the Template
```bash
cp -r servers/_example servers/your_tool_name
cd servers/your_tool_name
```
### 2. Write Your Tool
```python
# server.py
#!/usr/bin/env python3
"""
Example MCP Server using the new simplified pattern.
This demonstrates how to create a new AI4S tool with tools defined at module level.
"""
import argparse

from mcp.server.fastmcp import FastMCP

def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="DPA Calculator MCP Server")
    parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args


args = parse_args()
mcp = FastMCP("example",host=args.host,port=args.port)

@mcp.tool()
def demo_function(message: str) -> str:
    return f"Echo: {message}"


if __name__ == "__main__":
    mcp.run(transport="sse")
```

### 3. Create Metadata File
```json
# metadata.json
{
    "name": "YourToolName",
    "description": "Brief description of what your tool does",
    "author": "@your-github-username",
    "category": "chemistry"
}
```

### 4. Update Dependencies
```toml
# pyproject.toml
[project]
name = "your-tool-name"
version = "0.1.0"
description = "Your tool description"
requires-python = ">=3.8"
dependencies = [
    "fastmcp>=0.5.0",
    # Add your dependencies here
]
```

Generate lock file:
```bash
uv sync  # This creates uv.lock automatically
```

### 5. Test Your Tool
```bash
# Install dependencies
uv sync

# Run server
python server.py --port 50001

# Check it works
use myinspector tool to Check it works
```

### 6. Submit PR
```bash
git add .
git commit -m "feat: add YourToolName for molecular calculations"
git push origin your-branch
```

## 📦 Available Categories

Choose one when creating your tool (use exact string in metadata.json):

- `materials` - Crystal structures, DFT, MD simulations
- `chemistry` - Molecular properties, reactions, spectroscopy  
- `biology` - Protein structures, genomics, bioinformatics
- `physics` - Quantum mechanics, statistical physics
- `research` - Literature search, data management
- `simulation` - Molecular dynamics, Monte Carlo
- `data` - Processing, visualization, statistics
- `machine-learning` - AI models for science

## 🏗️ Project Structure

```
servers/
├── your_tool/ 
│   ├── utils/            # Utility functions (optional)
│   ├── server.py         # Your MCP server
│   ├── metadata.json     # Tool metadata
│   ├── pyproject.toml    # Dependencies
│   ├── README.md         # Documentation (optional)
│   └── uv.lock          # Lock file (auto-generated)
```

## 📝 Minimal Requirements

1. **Working Code** - Test before submitting
2. **Clear Metadata** - Complete metadata.json file
3. **Basic Docs** - README with usage example
4. **Error Handling** - Wrap logic in try-except

## 🔧 Development Tips

### Running with Debug
```bash
python server.py --port <port> --log-level DEBUG
```


## 🎯 PR Checklist

- [ ] Tool runs without errors
- [ ] Added to correct category
- [ ] README includes usage example
- [ ] Dependencies in pyproject.toml

## 💡 Examples

### Simple Tool (FastMCP)
See: `servers/pubchem/` - Chemical compound data retrieval

### Complex Tool (CalculationMCPServer)  
See: `servers/DPACalculator/` - Deep learning atomistic simulations

### Research Tool
See: `servers/Paper_Search/` - ArXiv paper search

## 🤝 Need Help?

- **Questions?** Open a GitHub issue
- **Bugs?** Include error logs and steps to reproduce
- **Ideas?** Start a discussion

## 🌟 Your Tool on the Showcase

Once merged, your tool automatically appears at:
https://deepmodeling.github.io/AI4S-agent-tools/

---

**Remember**: We're building tools for scientists, by scientists. Keep it simple, make it useful! 🔬