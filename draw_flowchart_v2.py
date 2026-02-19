
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.patches as patches

def draw_mind_architecture():
    # Create figure
    plt.figure(figsize=(20, 14))
    ax = plt.gca()
    
    G = nx.DiGraph()

    # === Define Nodes ===
    # Group: Simulator
    G.add_node('Sim', label='Simulator\n(simulator.py)', pos=(0, 10), color='#eceff1', shape='s')
    G.add_node('Agent', label='MIND Agent\n(agent.py)', pos=(0, 7), color='#e3f2fd', shape='s')
    
    # Group: Planner (The Brain)
    G.add_node('Planner', label='MINDPlanner\n(planner.py)', pos=(0, 4), color='#fff9c4', shape='s')
    
    # Group: Modules
    G.add_node('STG', label='Scenario Tree Gen\n(scenario_tree.py)\n[Intentions]', pos=(-3, 2), color='#f3e5f5', shape='o')
    G.add_node('Risk', label='Ghost Probe\n(utils.py)\n[Perception]', pos=(0, 2), color='#ffebee', shape='o')
    G.add_node('Traj', label='Traj Tree Opt\n(trajectory_tree.py)\n[Execution]', pos=(3, 2), color='#e0f2f1', shape='o')
    
    # Group: Output
    G.add_node('AEB', label='AEB Shield\n(Safety Check)', pos=(0, 0), color='#ffcdd2', shape='d')
    G.add_node('Ctrl', label='Control\n(Acc, Steer)', pos=(0, -2), color='#c8e6c9', shape='s')

    # === Define Edges ===
    edges = [
        ('Sim', 'Agent', 'Observation'),
        ('Agent', 'Planner', 'State & Map'),
        ('Planner', 'STG', 'Init'),
        ('Planner', 'Risk', 'Scan'),
        ('STG', 'Traj', 'Scenario Tree'),
        ('Risk', 'Traj', 'Risk Sources'),
        ('Traj', 'Planner', 'Best Trajectory'),
        ('Planner', 'AEB', 'Planned Ctrl'),
        ('AEB', 'Ctrl', 'Final Command'),
        ('Ctrl', 'Sim', 'Physics Step')
    ]
    
    # === Layout & Drawing ===
    pos = nx.get_node_attributes(G, 'pos')
    colors = [nx.get_node_attributes(G, 'color')[n] for n in G.nodes()]
    labels = nx.get_node_attributes(G, 'label')
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=6000, node_color=colors, edgecolors='#455a64', linewidths=2)
    
    # Draw Edges
    for u, v, t in edges:
        # Curve edges for feedback loop
        connection_style = "arc3,rad=0.0"
        if u == 'Ctrl' and v == 'Sim':
            connection_style = "arc3,rad=-0.4"
            
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                             edge_color='#546e7a', 
                             width=2.0, 
                             arrowsize=25, 
                             arrowstyle='-|>',
                             connectionstyle=connection_style)
        
        # Draw Edge Labels
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        
        # Adjust label pos for feedback
        if u == 'Ctrl' and v == 'Sim':
            mid_x -= 3
        
        plt.text(mid_x, mid_y, t, fontsize=10, ha='center', va='center', 
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Draw Node Labels
    nx.draw_networkx_labels(G, pos, labels, font_size=11, font_weight='bold')

    # === Annotations ===
    plt.title("E.R.A-MIND Architecture Diagram", fontsize=20, pad=20)
    plt.axis('off')
    
    # Save
    output_path = 'mind_architecture_v2.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    draw_mind_architecture()
