import graphviz
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, deque
from core.vector_db import VectorDB
from config.classes import *


class GraphVisualizer:
    """Simple knowledge graph visualizer using Graphviz."""
    
    def __init__(self, output_format='png', engine='dot'):
        """
        Initialize the visualizer.
        
        Args:
            output_format: Output format (png, svg, pdf, etc.)
            engine: Graphviz engine (dot, neato, fdp, etc.) 
        """
        self.output_format = output_format
        self.engine = engine
        
        # Color scheme for different node types
        self.node_colors = {
            'Event': '#FF6B6B',      # Red
            'Person': '#4ECDC4',     # Teal  
            'ThoughtReflection': '#45B7D1',  # Blue
            'Emotion': '#96CEB4',    # Green
            'Problem': '#FFEAA7',    # Yellow
            'Achievement': '#DDA0DD', # Plum
            'FutureIntention': '#FFB347'  # Orange
        }
        
        # Shape for different node types
        self.node_shapes = {
            'Event': 'ellipse',
            'Person': 'box',
            'ThoughtReflection': 'diamond', 
            'Emotion': 'circle',
            'Problem': 'hexagon',
            'Achievement': 'star',
            'FutureIntention': 'triangle'
        }
    
    def _get_all_objects_from_db(self, db: VectorDB) -> Dict[str, List[BaseModel]]:
        """Retrieve all objects from the database, organized by type."""
        collections = [
            ('ChunkEvent', 'events'),
            ('ChunkPerson', 'people'), 
            ('ChunkThought', 'thoughts'),
            ('ChunkEmotion', 'emotions'),
            ('ChunkProblem', 'problems'),
            ('ChunkAchievement', 'achievements'),
            ('ChunkFutureIntention', 'goals'),
            ('Connection', 'connections')
        ]
        
        all_objects = {}
        
        for collection_name, key in collections:
            try:
                objects = db.get_all_objects(collection_name)
                all_objects[key] = objects
            except Exception as e:
                print(f"Warning: Could not retrieve from {collection_name}: {e}")
                all_objects[key] = []
        
        return all_objects
    
    def _truncate_text(self, text: str, max_length: int = 30) -> str:
        """Truncate text for display in nodes."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def _create_node_label(self, obj: BaseModel) -> str:
        """Create a readable label for a node."""
        obj_type = obj.__class__.__name__
        
        if hasattr(obj, 'title') and obj.title:
            return f"{obj_type}\\n{self._truncate_text(obj.title)}"
        elif hasattr(obj, 'name') and obj.name:
            return f"{obj_type}\\n{self._truncate_text(obj.name)}"
        elif hasattr(obj, 'description') and obj.description:
            return f"{obj_type}\\n{self._truncate_text(obj.description)}"
        else:
            return obj_type
    
    def _find_largest_connected_component(self, all_objects: Dict[str, List[BaseModel]]) -> Tuple[Set[str], Dict[str, BaseModel]]:
        """
        Find the largest connected component in the graph.
        
        Returns:
            Tuple of (node_ids_in_largest_component, id_to_object_mapping)
        """
        # Build adjacency list from connections
        graph = defaultdict(set)
        all_nodes = set()
        id_to_object = {}
        
        # Collect all nodes and build mapping
        for category, objects in all_objects.items():
            if category == 'connections':
                continue
            for obj in objects:
                if hasattr(obj, 'object_id'):
                    all_nodes.add(obj.object_id)
                    id_to_object[obj.object_id] = obj
        
        # Build adjacency list from connections
        for connection in all_objects.get('connections', []):
            if (hasattr(connection, 'source_id') and hasattr(connection, 'target_id') and
                connection.source_id in all_nodes and connection.target_id in all_nodes):
                graph[connection.source_id].add(connection.target_id)
                graph[connection.target_id].add(connection.source_id)
        
        # Find all connected components using BFS
        visited = set()
        components = []
        
        for node in all_nodes:
            if node not in visited:
                # BFS to find this component
                component = set()
                queue = deque([node])
                
                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    
                    visited.add(current)
                    component.add(current)
                    
                    # Add unvisited neighbors to queue
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                components.append(component)
        
        # Return the largest component
        if not components:
            return set(), {}
        
        largest_component = max(components, key=len)
        return largest_component, id_to_object
    
    def _find_smallest_connected_component(self, all_objects: Dict[str, List[BaseModel]]) -> Tuple[Set[str], Dict[str, BaseModel]]:
        """
        Find the smallest connected component in the graph (excluding single isolated nodes).
        
        Returns:
            Tuple of (node_ids_in_smallest_component, id_to_object_mapping)
        """
        # Build adjacency list from connections
        graph = defaultdict(set)
        all_nodes = set()
        id_to_object = {}
        
        # Collect all nodes and build mapping
        for category, objects in all_objects.items():
            if category == 'connections':
                continue
            for obj in objects:
                if hasattr(obj, 'object_id'):
                    all_nodes.add(obj.object_id)
                    id_to_object[obj.object_id] = obj
        
        # Build adjacency list from connections
        for connection in all_objects.get('connections', []):
            if (hasattr(connection, 'source_id') and hasattr(connection, 'target_id') and
                connection.source_id in all_nodes and connection.target_id in all_nodes):
                graph[connection.source_id].add(connection.target_id)
                graph[connection.target_id].add(connection.source_id)
        
        # Find all connected components using BFS
        visited = set()
        components = []
        
        for node in all_nodes:
            if node not in visited:
                # BFS to find this component
                component = set()
                queue = deque([node])
                
                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    
                    visited.add(current)
                    component.add(current)
                    
                    # Add unvisited neighbors to queue
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                components.append(component)
        
        # Filter out single isolated nodes and return the smallest connected component
        connected_components = [comp for comp in components if len(comp) > 1]
        
        if not connected_components:
            return set(), {}
        
        smallest_component = min(connected_components, key=len)
        return smallest_component, id_to_object
    
    def _find_connected_component_by_rank(self, all_objects: Dict[str, List[BaseModel]], rank: str) -> Tuple[Set[str], Dict[str, BaseModel]]:
        """
        Find a connected component by rank (smallest, middle, largest).
        
        Args:
            all_objects: All objects from the database
            rank: One of 'smallest', 'middle', 'largest'
            
        Returns:
            Tuple of (node_ids_in_component, id_to_object_mapping)
        """
        # Build adjacency list from connections
        graph = defaultdict(set)
        all_nodes = set()
        id_to_object = {}
        
        # Collect all nodes and build mapping
        for category, objects in all_objects.items():
            if category == 'connections':
                continue
            for obj in objects:
                if hasattr(obj, 'object_id'):
                    all_nodes.add(obj.object_id)
                    id_to_object[obj.object_id] = obj
        
        # Build adjacency list from connections
        for connection in all_objects.get('connections', []):
            if (hasattr(connection, 'source_id') and hasattr(connection, 'target_id') and
                connection.source_id in all_nodes and connection.target_id in all_nodes):
                graph[connection.source_id].add(connection.target_id)
                graph[connection.target_id].add(connection.source_id)
        
        # Find all connected components using BFS
        visited = set()
        components = []
        
        for node in all_nodes:
            if node not in visited:
                # BFS to find this component
                component = set()
                queue = deque([node])
                
                while queue:
                    current = queue.popleft()
                    if current in visited:
                        continue
                    
                    visited.add(current)
                    component.add(current)
                    
                    # Add unvisited neighbors to queue
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                components.append(component)
        
        # Filter out single isolated nodes for smallest and middle
        if rank in ['smallest', 'middle']:
            components = [comp for comp in components if len(comp) > 1]
        
        if not components:
            return set(), {}
        
        # Sort components by size
        components.sort(key=len)
        
        # Return the requested component
        if rank == 'smallest':
            return components[0], id_to_object
        elif rank == 'largest':
            return components[-1], id_to_object
        elif rank == 'middle':
            if len(components) < 2:
                # If we only have one component, return it
                return components[0], id_to_object
            # Return the middle component (or one of the middle ones if even number)
            middle_idx = len(components) // 2
            return components[middle_idx], id_to_object
        else:
            raise ValueError(f"Invalid rank: {rank}. Must be 'smallest', 'middle', or 'largest'")
    
    def create_graph(self, output_path: str = "visualization/knowledge_graph") -> str:
        """
        Create and render the knowledge graph visualization.
        
        Args:
            output_path: Path for the output file (without extension)
            
        Returns:
            Path to the generated file
        """
        # Create Graphviz graph
        dot = graphviz.Digraph(
            engine=self.engine,
            format=self.output_format,
            graph_attr={
                'rankdir': 'TB',  # Top to bottom layout
                'bgcolor': 'white',
                'fontname': 'Arial',
                'fontsize': '14',
                'label': 'Knowledge Graph Visualization',
                'labelloc': 't'
            },
            node_attr={
                'fontname': 'Arial',
                'fontsize': '10',
                'style': 'filled',
                'margin': '0.1'
            },
            edge_attr={
                'fontname': 'Arial', 
                'fontsize': '8',
                'color': 'gray60'
            }
        )
        
        # Get all data from database
        with VectorDB() as db:
            all_objects = self._get_all_objects_from_db(db)
        
        # Add nodes for each object type
        node_count = 0
        for category, objects in all_objects.items():
            if category == 'connections':
                continue  # Handle connections separately
                
            for obj in objects:
                if hasattr(obj, 'object_id'):
                    obj_type = obj.__class__.__name__
                    color = self.node_colors.get(obj_type, '#CCCCCC')
                    shape = self.node_shapes.get(obj_type, 'ellipse')
                    label = self._create_node_label(obj)
                    
                    dot.node(
                        obj.object_id,
                        label=label,
                        color=color,
                        shape=shape
                    )
                    node_count += 1
        
        # Add edges for connections
        edge_count = 0
        for connection in all_objects.get('connections', []):
            if hasattr(connection, 'source_id') and hasattr(connection, 'target_id'):
                edge_label = getattr(connection, 'type', 'connected to')
                dot.edge(
                    connection.source_id,
                    connection.target_id,
                    label=edge_label
                )
                edge_count += 1
        
        # Add legend
        with dot.subgraph(name='cluster_legend') as legend:
            legend.attr(label='Node Types', fontsize='12', style='dashed')
            legend.attr('node', shape='plaintext', style='')
            
            legend_items = []
            for obj_type, color in self.node_colors.items():
                shape = self.node_shapes.get(obj_type, 'ellipse')
                legend_items.append(f'<TR><TD><FONT COLOR="{color}">●</FONT></TD><TD>{obj_type}</TD></TR>')
            
            legend_html = f'''<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
            <TR><TD COLSPAN="2"><B>Legend</B></TD></TR>
            {"".join(legend_items)}
            </TABLE>
            >'''
            
            legend.node('legend', label=legend_html)
    
        
        # Try to render the graph, fallback to DOT file if system graphviz not available
        try:
            output_file = dot.render(output_path, cleanup=True)
            return output_file
        except Exception as e:
            print(f"⚠️ Could not render to {self.output_format}: {e}")
            print("Falling back to DOT file generation...")
            
            # Save as DOT file instead
            dot_path = f"{output_path}.dot"
            with open(dot_path, 'w') as f:
                f.write(dot.source)
            
            print(f"✅ Graph DOT file saved to: {dot_path}")
            print("To generate image, install system graphviz and run:")
            print(f"  dot -T{self.output_format} {dot_path} -o {output_path}.{self.output_format}")
            return dot_path

    def _create_connected_component_graph(self, rank: str, output_path: str) -> str:
        """
        Generic method to create connected component graphs by rank.
        
        Args:
            rank: One of 'smallest', 'middle', 'largest'
            output_path: Path for the output file (without extension)
            
        Returns:
            Path to the generated visualization file
        """
        
        # Create Graphviz graph with improved styling
        dot = graphviz.Digraph(
            engine='neato',  # Use neato for more square/circular layouts
            format=self.output_format,
            graph_attr={
                'bgcolor': 'white',
                'fontname': 'Helvetica Neue',
                'splines': 'curved',
                'overlap': 'false',
                'sep': '+30',
                'mode': 'major',
                'model': 'mds',
                'start': 'random'
            },
            node_attr={
                'fontname': 'Helvetica Neue',
                'fontsize': '9',
                'shape': 'circle',
                'style': 'filled',
                'fillcolor': 'white',
                'penwidth': '3',
                'margin': '0.08'
            },
            edge_attr={
                'fontname': 'Helvetica Neue', 
                'fontsize': '7',
                'color': 'gray70',
                'penwidth': '1.5',
                'labeldistance': '2.0',
                'labelangle': '0'
            }
        )
        
        # Get all data from database
        with VectorDB() as db:
            all_objects = self._get_all_objects_from_db(db)
        
        # Find the component by rank
        component_ids, id_to_object = self._find_connected_component_by_rank(all_objects, rank)
        
        if not component_ids:
            print(f"⚠️ No connected components found for rank '{rank}'. The graph may contain only isolated nodes.")
            return None
        
        print(f"Found {len(component_ids)} nodes in the {rank} connected component")
        
        # Calculate connection counts for each node to determine size
        connection_counts = {}
        for node_id in component_ids:
            connection_counts[node_id] = 0
        
        for connection in all_objects.get('connections', []):
            if (hasattr(connection, 'source_id') and hasattr(connection, 'target_id') and
                connection.source_id in component_ids and connection.target_id in component_ids):
                connection_counts[connection.source_id] = connection_counts.get(connection.source_id, 0) + 1
                connection_counts[connection.target_id] = connection_counts.get(connection.target_id, 0) + 1
        
        # Add nodes only from the component
        node_count = 0
        for node_id in component_ids:
            if node_id in id_to_object:
                obj = id_to_object[node_id]
                obj_type = obj.__class__.__name__
                color = self.node_colors.get(obj_type, '#CCCCCC')
                
                # Calculate node size based on connections (min 0.8, max 2.2)
                conn_count = connection_counts.get(node_id, 0)
                max_connections = max(connection_counts.values()) if connection_counts else 1
                size_ratio = (conn_count / max_connections) if max_connections > 0 else 0.3
                node_size = 0.8 + (size_ratio * 1.4)  # Range from 0.8 to 2.2
                
                # Adjust text length based on node size (more generous truncation)
                max_chars = int(12 + (size_ratio * 15))  # Range from 12 to 27 characters
                
                # Create simple label with just title/name/description
                label = ""
                if hasattr(obj, 'title') and obj.title:
                    label = self._truncate_text(obj.title, max_chars)
                elif hasattr(obj, 'name') and obj.name:
                    label = self._truncate_text(obj.name, max_chars)
                elif hasattr(obj, 'description') and obj.description:
                    label = self._truncate_text(obj.description, max_chars)
                else:
                    label = self._truncate_text(obj_type, max_chars)
                
                dot.node(
                    obj.object_id,
                    label=label,
                    color=color,
                    width=str(node_size),
                    height=str(node_size),
                    fixedsize='true'
                )
                node_count += 1
        
        # Add edges for connections (only between nodes in the component)
        edge_count = 0
        for connection in all_objects.get('connections', []):
            if (hasattr(connection, 'source_id') and hasattr(connection, 'target_id') and
                connection.source_id in component_ids and connection.target_id in component_ids):
                edge_label = getattr(connection, 'type', 'connected to')
                dot.edge(
                    connection.source_id,
                    connection.target_id,
                    label=edge_label
                )
                edge_count += 1

        # Try to render the graph
        try:
            output_file = dot.render(output_path, cleanup=True)
            print(f"✅ {rank.capitalize()} connected component visualization saved to: {output_file}")
            print(f"   Nodes: {node_count}, Edges: {edge_count}")
            return output_file
        except Exception as e:
            print(f"⚠️ Could not render to {self.output_format}: {e}")
            print("Falling back to DOT file generation...")
            
            # Save as DOT file instead
            dot_path = f"{output_path}.dot"
            with open(dot_path, 'w') as f:
                f.write(dot.source)
            
            print(f"✅ Graph DOT file saved to: {dot_path}")
            print(f"   Nodes: {node_count}, Edges: {edge_count}")
            print("To generate image, install system graphviz and run:")
            print(f"  dot -T{self.output_format} {dot_path} -o {output_path}.{self.output_format}")
            return dot_path

    def create_largest_connected_component_graph(self, output_path: str = "visualization/knowledge_graph_largest") -> str:
        """
        Visualize only the largest connected component of the knowledge graph.
        
        Args:
            output_path: Path for the output file (without extension)
            
        Returns:
            Path to the generated visualization file
        """
        return self._create_connected_component_graph('largest', output_path)

    def create_smallest_connected_component_graph(self, output_path: str = "visualization/knowledge_graph_smallest") -> str:
        """
        Visualize only the smallest connected component of the knowledge graph.
        
        Args:
            output_path: Path for the output file (without extension)
            
        Returns:
            Path to the generated visualization file
        """
        return self._create_connected_component_graph('smallest', output_path)

    def create_middle_connected_component_graph(self, output_path: str = "visualization/knowledge_graph_middle") -> str:
        """
        Visualize the middle-sized connected component of the knowledge graph.
        
        Args:
            output_path: Path for the output file (without extension)
            
        Returns:
            Path to the generated visualization file
        """
        return self._create_connected_component_graph('middle', output_path)

    def _find_chunk_with_most_objects(self, all_objects: Dict[str, List[BaseModel]]) -> Tuple[str, int]:
        """
        Find the chunk_id that has the most objects across all collections.
        
        Returns:
            Tuple of (chunk_id, object_count)
        """
        chunk_object_counts = {}
        
        # Count objects per chunk across all collections (excluding connections)
        for category, objects in all_objects.items():
            if category == 'connections':
                continue
            for obj in objects:
                if hasattr(obj, 'chunk_id'):
                    chunk_id = obj.chunk_id
                    chunk_object_counts[chunk_id] = chunk_object_counts.get(chunk_id, 0) + 1
        
        if not chunk_object_counts:
            return None, 0
        
        # Find chunk with maximum objects
        max_chunk_id = max(chunk_object_counts, key=chunk_object_counts.get)
        max_count = chunk_object_counts[max_chunk_id]
        
        return max_chunk_id, max_count

    def create_largest_local_graph(self, output_path: str = "visualization/knowledge_graph_local") -> str:
        """
        Visualize the local graph for the chunk that contains the most objects.
        Gets all chunk_ids from the database, finds which chunk has the most objects 
        (Person, Event, etc.), and displays all connections in that local graph.
        
        Args:
            output_path: Path for the output file (without extension)
            
        Returns:
            Path to the generated visualization file
        """
        
        # Create Graphviz graph with improved styling (same as largest connected component)
        dot = graphviz.Digraph(
            engine='neato',
            format=self.output_format,
            graph_attr={
                'bgcolor': 'white',
                'fontname': 'Helvetica Neue',
                'splines': 'curved',
                'overlap': 'false',
                'sep': '+30',
                'mode': 'major',
                'model': 'mds',
                'start': 'random'
            },
            node_attr={
                'fontname': 'Helvetica Neue',
                'fontsize': '9',
                'shape': 'circle',
                'style': 'filled',
                'fillcolor': 'white',
                'penwidth': '3',
                'margin': '0.08'
            },
            edge_attr={
                'fontname': 'Helvetica Neue', 
                'fontsize': '7',
                'color': 'gray70',
                'penwidth': '1.5',
                'labeldistance': '2.0',
                'labelangle': '0'
            }
        )
        
        # Get all data from database
        with VectorDB() as db:
            all_objects = self._get_all_objects_from_db(db)
        
        # Find the chunk with the most objects
        largest_chunk_id, object_count = self._find_chunk_with_most_objects(all_objects)
        
        if not largest_chunk_id:
            print("⚠️ No chunks found with objects. The database may be empty.")
            return None
        
        print(f"Found chunk '{largest_chunk_id}' with {object_count} objects")
        
        # Collect all objects that belong to the largest chunk
        chunk_objects = {}  # object_id -> object
        chunk_object_ids = set()
        
        for category, objects in all_objects.items():
            if category == 'connections':
                continue
            for obj in objects:
                if hasattr(obj, 'chunk_id') and hasattr(obj, 'object_id') and obj.chunk_id == largest_chunk_id:
                    chunk_objects[obj.object_id] = obj
                    chunk_object_ids.add(obj.object_id)
        
        # Calculate connection counts for each node to determine size
        connection_counts = {}
        for object_id in chunk_object_ids:
            connection_counts[object_id] = 0
        
        # Count connections within this chunk
        chunk_connections = []
        for connection in all_objects.get('connections', []):
            if (hasattr(connection, 'source_id') and hasattr(connection, 'target_id') and 
                hasattr(connection, 'chunk_id') and connection.chunk_id == largest_chunk_id and
                connection.source_id in chunk_object_ids and connection.target_id in chunk_object_ids):
                chunk_connections.append(connection)
                connection_counts[connection.source_id] = connection_counts.get(connection.source_id, 0) + 1
                connection_counts[connection.target_id] = connection_counts.get(connection.target_id, 0) + 1
        
        # Add nodes only from the largest chunk
        node_count = 0
        for object_id, obj in chunk_objects.items():
            obj_type = obj.__class__.__name__
            color = self.node_colors.get(obj_type, '#CCCCCC')
            
            # Calculate node size based on connections (min 0.8, max 2.2)
            conn_count = connection_counts.get(object_id, 0)
            max_connections = max(connection_counts.values()) if connection_counts else 1
            size_ratio = (conn_count / max_connections) if max_connections > 0 else 0.3
            node_size = 0.8 + (size_ratio * 1.4)  # Range from 0.8 to 2.2
            
            # Adjust text length based on node size (more generous truncation)
            max_chars = int(12 + (size_ratio * 15))  # Range from 12 to 27 characters
            
            # Create simple label with just title/name/description
            label = ""
            if hasattr(obj, 'title') and obj.title:
                label = self._truncate_text(obj.title, max_chars)
            elif hasattr(obj, 'name') and obj.name:
                label = self._truncate_text(obj.name, max_chars)
            elif hasattr(obj, 'description') and obj.description:
                label = self._truncate_text(obj.description, max_chars)
            else:
                label = self._truncate_text(obj_type, max_chars)
            
            dot.node(
                obj.object_id,
                label=label,
                color=color,
                width=str(node_size),
                height=str(node_size),
                fixedsize='true'
            )
            node_count += 1
        
        # Add edges for connections within this chunk
        edge_count = 0
        for connection in chunk_connections:
            edge_label = getattr(connection, 'type', 'connected to')
            dot.edge(
                connection.source_id,
                connection.target_id,
                label=edge_label
            )
            edge_count += 1

        # Try to render the graph
        try:
            output_file = dot.render(output_path, cleanup=True)
            print(f"✅ Largest local graph visualization saved to: {output_file}")
            print(f"   Chunk ID: {largest_chunk_id}")
            print(f"   Nodes: {node_count}, Edges: {edge_count}")
            return output_file
        except Exception as e:
            print(f"⚠️ Could not render to {self.output_format}: {e}")
            print("Falling back to DOT file generation...")
            
            # Save as DOT file instead
            dot_path = f"{output_path}.dot"
            with open(dot_path, 'w') as f:
                f.write(dot.source)
            
            print(f"✅ Graph DOT file saved to: {dot_path}")
            print(f"   Chunk ID: {largest_chunk_id}")
            print(f"   Nodes: {node_count}, Edges: {edge_count}")
            print("To generate image, install system graphviz and run:")
            print(f"  dot -T{self.output_format} {dot_path} -o {output_path}.{self.output_format}")
            return dot_path


def visualize_knowledge_graph(output_path: str = "visualization/knowledge_graph", output_format: str = 'png') -> str:
    """
    Global convenience function to visualize the knowledge graph.
    
    Args:
        output_path: Path for the output file (without extension)
        output_format: Output format (png, svg, pdf, etc.)
        
    Returns:
        Path to the generated visualization file
    """
    visualizer = GraphVisualizer(output_format=output_format)
    return visualizer.create_graph(output_path)