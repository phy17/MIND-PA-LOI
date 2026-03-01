
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_architecture():
    G = nx.DiGraph()

    # Define nodes with layers/groups
    nodes = {
        'Entry': 'run_sim.py',
        'Sim': 'Simulator\n(simulator.py)',
        'Loader': 'ArgoAgentLoader\n(loader.py)',
        'Loop': 'Simulation Loop',
        'AgentObs': 'agent.observe()',
        'AgentPlan': 'agent.plan()',
        'AgentUpd': 'agent.update_state()',
        'MIND': 'MINDPlanner\n(planners/mind)',
        'AIME': 'ScenarioTreeGen\n(AIME)',
        'iLQR': 'TrajTreeOpt\n(iLQR)'
    }

    # Add nodes
    for k, v in nodes.items():
        G.add_node(k, label=v)

    # Add edges
    edges = [
        ('Entry', 'Sim'),
        ('Sim', 'Loader'),
        ('Sim', 'Loop'),
        ('Loop', 'AgentObs'),
        ('Loop', 'AgentPlan'),
        ('Loop', 'AgentUpd'),
        ('AgentPlan', 'MIND'),
        ('MIND', 'AIME'),
        ('MIND', 'iLQR'),
        ('AIME', 'iLQR'),  # Flow of data
        ('iLQR', 'AgentPlan') # Return control
    ]
    G.add_edges_from(edges)

    pos = {
        'Entry': (0, 10),
        'Sim': (0, 8),
        'Loader': (-2, 8),
        'Loop': (0, 6),
        'AgentObs': (-2, 4),
        'AgentPlan': (0, 4),
        'AgentUpd': (2, 4),
        'MIND': (0, 2),
        'AIME': (-1.5, 0),
        'iLQR': (1.5, 0)
    }

    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', edgecolors='black')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, width=1.5)
    
    # Draw labels
    labels = {k: v for k, v in nodes.items()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

    plt.title("MIND Code Architecture & Data Flow", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('mind_architecture.png', dpi=300)
    print("Graph saved to mind_architecture.png")

if __name__ == "__main__":
    draw_architecture()
