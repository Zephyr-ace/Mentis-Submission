# Knowledge Graph Visualization

This module provides simple visualization capabilities for your knowledge graph using Graphviz.

## Features

- **GraphVisualizer class**: Main visualization class with customizable colors, shapes, and layouts
- **Global convenience function**: Simple one-line usage via `visualize_knowledge_graph()`
- **Fallback support**: Generates DOT files if system Graphviz is not installed
- **Color-coded nodes**: Different colors and shapes for each entity type:
  - 🔴 **Events** (red ellipses)
  - 🟢 **People** (teal boxes) 
  - 🔵 **Thoughts** (blue diamonds)
  - 🟣 **Emotions** (green circles)
  - 🟡 **Problems** (yellow hexagons)
  - 🟠 **Achievements** (plum stars)
  - 🔶 **Goals** (orange triangles)

## Quick Usage

```python
from visualization.graph_visualizer import visualize_knowledge_graph

# Simple usage - creates visualization/knowledge_graph.png (or .dot)
output_file = visualize_knowledge_graph()

# Custom output path and format
output_file = visualize_knowledge_graph(
    output_path="visualization/my_custom_graph", 
    output_format="svg"
)
```

## Advanced Usage

```python
from visualization.graph_visualizer import GraphVisualizer

# Create custom visualizer
visualizer = GraphVisualizer(
    output_format='pdf',
    engine='neato'  # Different layout algorithm
)

# Generate visualization
output_file = visualizer.create_graph("custom_graph")
```

## Installation Requirements

1. **Python package**: `pip install graphviz`
2. **System Graphviz** (for image generation):
   - **macOS**: `brew install graphviz`
   - **Ubuntu**: `sudo apt-get install graphviz`
   - **Windows**: Download from https://graphviz.org/download/

If system Graphviz is not installed, the module will generate a `.dot` file that you can manually render:

```bash
dot -Tpng my_graph.dot -o my_graph.png
```
