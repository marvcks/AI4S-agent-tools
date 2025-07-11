#!/usr/bin/env python3
"""
Generate a simple, Apple-style showcase page for AI4S Agent Tools.
"""
import json
from pathlib import Path
from datetime import datetime

def load_categories():
    """Load category definitions."""
    categories_path = Path(__file__).parent.parent / "config" / "categories.json"
    if categories_path.exists():
        with open(categories_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"categories": {}, "default_category": "research"}

def get_tool_category(tool: dict, categories_config: dict) -> str:
    """Get the category of a tool from its metadata."""
    # Use the category directly from the tool's metadata
    category = tool.get("category", "")
    
    # Validate that the category exists in our configuration
    if category in categories_config.get("categories", {}):
        return category
    
    # Fall back to default category if the tool's category is invalid or missing
    return categories_config.get("default_category", "general")

def generate_showcase():
    """Generate the showcase HTML page."""
    root_dir = Path(__file__).parent.parent
    tools_json_path = root_dir / "TOOLS.json"
    
    # Load tools data
    with open(tools_json_path, 'r', encoding='utf-8') as f:
        tools_data = json.load(f)
    
    # Load categories
    categories_config = load_categories()
    categories = categories_config["categories"]
    
    # Categorize tools using their metadata
    categorized_tools = {}
    for tool in tools_data["tools"]:
        category = get_tool_category(tool, categories_config)
        if category not in categorized_tools:
            categorized_tools[category] = []
        categorized_tools[category].append(tool)
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI4S Agent Tools</title>
    <style>
        :root {{
            --primary: #007AFF;
            --secondary: #5856D6;
            --success: #34C759;
            --warning: #FF9500;
            --danger: #FF3B30;
            --dark: #1C1C1E;
            --light: #F2F2F7;
            --gray: #8E8E93;
            --bg: #FFFFFF;
            --card-bg: #F7F7F7;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--dark);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        header {{
            padding: 60px 0 40px;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        h1 {{
            font-size: 48px;
            font-weight: 600;
            margin-bottom: 10px;
            letter-spacing: -0.5px;
        }}
        
        .subtitle {{
            font-size: 20px;
            font-weight: 300;
            opacity: 0.9;
        }}
        
        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 30px;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 36px;
            font-weight: 600;
        }}
        
        .stat-label {{
            font-size: 14px;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .search-container {{
            padding: 40px 0;
            background: var(--light);
        }}
        
        .search-box {{
            position: relative;
            max-width: 600px;
            margin: 0 auto;
        }}
        
        .search-input {{
            width: 100%;
            padding: 15px 20px 15px 50px;
            font-size: 16px;
            border: none;
            border-radius: 10px;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .search-icon {{
            position: absolute;
            left: 20px;
            top: 50%;
            transform: translateY(-50%);
            opacity: 0.5;
        }}
        
        .categories {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 20px;
        }}
        
        .category-tag {{
            padding: 8px 16px;
            border-radius: 20px;
            background: white;
            color: var(--dark);
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid transparent;
        }}
        
        .category-tag:hover {{
            border-color: var(--primary);
            color: var(--primary);
        }}
        
        .category-tag.active {{
            background: var(--primary);
            color: white;
        }}
        
        .tools-grid {{
            padding: 60px 0;
        }}
        
        .category-section {{
            margin-bottom: 60px;
        }}
        
        .category-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
        }}
        
        .category-icon {{
            font-size: 28px;
        }}
        
        .category-title {{
            font-size: 28px;
            font-weight: 600;
        }}
        
        .tools-row {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }}
        
        .tool-card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            transition: all 0.3s;
            cursor: pointer;
        }}
        
        .tool-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .tool-header {{
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 12px;
        }}
        
        .tool-name {{
            font-size: 20px;
            font-weight: 600;
            color: var(--dark);
        }}
        
        .tool-author {{
            font-size: 14px;
            color: var(--gray);
        }}
        
        .tool-description {{
            color: var(--gray);
            font-size: 15px;
            margin-bottom: 16px;
            line-height: 1.5;
        }}
        
        .tool-features {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }}
        
        .tool-feature {{
            padding: 4px 10px;
            background: white;
            border-radius: 6px;
            font-size: 12px;
            color: var(--dark);
        }}
        
        .tool-count {{
            font-size: 12px;
            color: var(--gray);
            margin-left: 4px;
        }}
        
        footer {{
            padding: 40px 0;
            text-align: center;
            background: var(--light);
            color: var(--gray);
            font-size: 14px;
        }}
        
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            backdrop-filter: blur(4px);
        }}
        
        .modal-content {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }}
        
        .modal-close {{
            position: absolute;
            top: 20px;
            right: 20px;
            font-size: 24px;
            color: var(--gray);
            cursor: pointer;
        }}
        
        .modal-title {{
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 20px;
        }}
        
        .modal-section {{
            margin-bottom: 24px;
        }}
        
        .modal-section-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        
        .code-block {{
            background: var(--light);
            padding: 16px;
            border-radius: 8px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 14px;
            overflow-x: auto;
        }}
        
        @media (max-width: 768px) {{
            h1 {{
                font-size: 36px;
            }}
            
            .stats {{
                gap: 20px;
            }}
            
            .stat-value {{
                font-size: 28px;
            }}
            
            .tools-row {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>AI4S Agent Tools</h1>
            <p class="subtitle">Scientific Computing Tools for Intelligent Agents</p>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{len(tools_data['tools'])}</div>
                    <div class="stat-label">Tools</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{sum(len(tool.get('tools', [])) for tool in tools_data['tools'])}</div>
                    <div class="stat-label">Functions</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len(set(tool.get('author', '@unknown') for tool in tools_data['tools']))}</div>
                    <div class="stat-label">Contributors</div>
                </div>
            </div>
        </div>
    </header>
    
    <div class="search-container">
        <div class="container">
            <div class="search-box">
                <span class="search-icon">🔍</span>
                <input type="text" class="search-input" placeholder="Search tools..." id="searchInput">
            </div>
            <div class="categories">
                <span class="category-tag active" data-category="all">All</span>
"""
    
    for cat_id, cat_info in categories.items():
        if cat_id in categorized_tools:
            html += f"""                <span class="category-tag" data-category="{cat_id}">{cat_info['icon']} {cat_info['name']}</span>
"""
    
    html += """            </div>
        </div>
    </div>
    
    <div class="tools-grid">
        <div class="container">
"""
    
    # Add tools by category
    for cat_id, tools in categorized_tools.items():
        if cat_id in categories:
            cat_info = categories[cat_id]
            html += f"""            <div class="category-section" data-category="{cat_id}">
                <div class="category-header">
                    <span class="category-icon">{cat_info['icon']}</span>
                    <h2 class="category-title">{cat_info['name']}</h2>
                </div>
                <div class="tools-row">
"""
            
            for tool in tools:
                features = tool.get('tools', [])[:3]
                more_count = len(tool.get('tools', [])) - 3
                
                html += f"""                    <div class="tool-card" onclick="showToolDetails('{tool['name']}')">
                        <div class="tool-header">
                            <div>
                                <div class="tool-name">{tool['name']}</div>
                                <div class="tool-author">{tool.get('author', '@unknown')}</div>
                            </div>
                        </div>
                        <p class="tool-description">{tool.get('description', 'No description available')}</p>
                        <div class="tool-features">
"""
                
                for feature in features:
                    html += f"""                            <span class="tool-feature">{feature}</span>
"""
                
                if more_count > 0:
                    html += f"""                            <span class="tool-count">+{more_count} more</span>
"""
                
                html += """                        </div>
                    </div>
"""
            
            html += """                </div>
            </div>
"""
    
    html += f"""        </div>
    </div>
    
    <footer>
        <div class="container">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d')} • Part of the DeepModeling Community</p>
        </div>
    </footer>
    
    <!-- Tool Details Modal -->
    <div id="toolModal" class="modal">
        <div class="modal-content">
            <span class="modal-close" onclick="closeModal()">&times;</span>
            <h2 class="modal-title" id="modalTitle"></h2>
            <div id="modalContent"></div>
        </div>
    </div>
    
    <script>
        const toolsData = {json.dumps(tools_data['tools'], indent=4)};
        
        // Search functionality
        document.getElementById('searchInput').addEventListener('input', function(e) {{
            const searchTerm = e.target.value.toLowerCase();
            filterTools(searchTerm);
        }});
        
        // Category filter
        document.querySelectorAll('.category-tag').forEach(tag => {{
            tag.addEventListener('click', function() {{
                document.querySelectorAll('.category-tag').forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                
                const category = this.dataset.category;
                if (category === 'all') {{
                    document.querySelectorAll('.category-section').forEach(section => {{
                        section.style.display = 'block';
                    }});
                }} else {{
                    document.querySelectorAll('.category-section').forEach(section => {{
                        section.style.display = section.dataset.category === category ? 'block' : 'none';
                    }});
                }}
            }});
        }});
        
        function filterTools(searchTerm) {{
            document.querySelectorAll('.tool-card').forEach(card => {{
                const text = card.textContent.toLowerCase();
                card.style.display = text.includes(searchTerm) ? 'block' : 'none';
            }});
        }}
        
        function showToolDetails(toolName) {{
            const tool = toolsData.find(t => t.name === toolName);
            if (!tool) return;
            
            document.getElementById('modalTitle').textContent = tool.name;
            
            let content = `
                <div class="modal-section">
                    <div class="modal-section-title">Description</div>
                    <p>${{tool.description || 'No description available'}}</p>
                </div>
                
                <div class="modal-section">
                    <div class="modal-section-title">Author</div>
                    <p>${{tool.author || '@unknown'}}</p>
                </div>
                
                <div class="modal-section">
                    <div class="modal-section-title">Installation</div>
                    <div class="code-block">cd ${{tool.path}} && uv sync</div>
                </div>
                
                <div class="modal-section">
                    <div class="modal-section-title">Start Command</div>
                    <div class="code-block">${{tool.start_command}}</div>
                </div>
            `;
            
            if (tool.tools && tool.tools.length > 0) {{
                content += `
                    <div class="modal-section">
                        <div class="modal-section-title">Available Functions (${{tool.tools.length}})</div>
                        <div class="tool-features">
                `;
                tool.tools.forEach(t => {{
                    content += `<span class="tool-feature">${{t}}</span>`;
                }});
                content += `
                        </div>
                    </div>
                `;
            }}
            
            document.getElementById('modalContent').innerHTML = content;
            document.getElementById('toolModal').style.display = 'block';
        }}
        
        function closeModal() {{
            document.getElementById('toolModal').style.display = 'none';
        }}
        
        // Close modal when clicking outside
        window.onclick = function(event) {{
            const modal = document.getElementById('toolModal');
            if (event.target == modal) {{
                modal.style.display = 'none';
            }}
        }}
    </script>
</body>
</html>"""
    
    # Write HTML file
    showcase_dir = root_dir / "showcase"
    showcase_dir.mkdir(exist_ok=True)
    
    with open(showcase_dir / "index.html", 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Generated showcase page at: {showcase_dir / 'index.html'}")
    print(f"📊 Stats: {len(tools_data['tools'])} tools in {len(categorized_tools)} categories")

if __name__ == "__main__":
    generate_showcase()