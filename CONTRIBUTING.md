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
from servers.server_utils import mcp_server, setup_server
from mcp.server.fastmcp import FastMCP

@mcp_server("YourToolName", "One-line description", author="@your-github", category="Chemistry")
def create_server(host="0.0.0.0", port=50001):
    mcp = FastMCP("your_tool", host=host, port=port)
    
    @mcp.tool()
    def calculate_something(molecule: str) -> dict:
        """Calculate properties of a molecule."""
        # Your scientific logic here
        return {"result": "value"}
    
    return mcp

if __name__ == "__main__":
    setup_server().run("sse")
```

### 3. Update Dependencies
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

### 4. Test Your Tool
```bash
# Install dependencies
uv sync

# Run server
python server.py --port 50001

# Check it works
use myinspector tool to Check it works
```

### 5. Submit PR
```bash
git add .
git commit -m "feat: add YourToolName for molecular calculations"
git push origin your-branch
```

## 📦 Available Categories

Choose one when creating your tool (use exact string in `category` parameter):

- `"Materials Science"` - Crystal structures, DFT, MD simulations
- `"Chemistry"` - Molecular properties, reactions, spectroscopy  
- `"Biology"` - Protein structures, genomics, bioinformatics
- `"Physics"` - Quantum mechanics, statistical physics
- `"Research Tools"` - Literature search, data management
- `"Simulation"` - Molecular dynamics, Monte Carlo
- `"Data & Analysis"` - Processing, visualization, statistics
- `"Machine Learning"` - AI models for science
- `"General Tools"` - Utilities, converters

## 🏗️ Project Structure

```
servers/
├── your_tool/ 
│   ├── utils             # Utils 
│   ├── server.py         # Your MCP server
│   ├── pyproject.toml    # Dependencies
│   ├── README.md         # Documentation
│   └── uv.lock          # Lock file (auto-generated)
```

## 📝 Minimal Requirements

1. **Working Code** - Test before submitting
2. **Clear Purpose** - One-line description in decorator
3. **Basic Docs** - README with usage example
4. **Error Handling** - Wrap logic in try-except

## 🔧 Development Tips

### Running with Debug
```bash
python server.py --port 50001 --log-level DEBUG
```

### Check Logs
```bash
tail -f ~/.your_tool_name/*.log
```

### Avoid Port Conflicts
Check `TOOLS.json` for used ports (50001-50100 range)

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