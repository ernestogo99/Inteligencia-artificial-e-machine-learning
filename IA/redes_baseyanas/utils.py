import matplotlib.pyplot as plt
import networkx as nx

# Visualizar a estrutura da rede
def visualizar_rede(modelo, titulo="Rede Bayesiana"):
    plt.figure(figsize=(10, 6))
    pos = nx.circular_layout(modelo) # Alterado para circular_layout para maior estabilidade

    # 1. Desenhar os nós
    nx.draw_networkx_nodes(modelo, pos, node_color='lightblue', node_size=2000)

    # 2. Desenhar as arestas (explicitamente sem setas para evitar erros)
    nx.draw_networkx_edges(modelo, pos, edge_color='gray', arrows=False)

    # 3. Desenhar os rótulos dos nós
    nx.draw_networkx_labels(modelo, pos, font_size=14, font_weight='bold')

    plt.title(titulo, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
