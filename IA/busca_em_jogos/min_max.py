from time import time

class Node:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children or []

    def is_terminal(self):
        return self.value is not None  

    def get_value(self):
        return self.value

    def get_children(self):
        return self.children


def minimax(node:Node, depth:int, maximizing_player:bool)->float:
    """
    node: estado atual 
    depth: profundidade máxima da busca
    maximizing_player: True se estamos maximizando, False se minimizando
    """


    if depth == 0 or node.is_terminal():
        return node.get_value()


    if maximizing_player:
        max_eval = float('-inf')
        for child in node.get_children():
            eval = minimax(child, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval


    else:
        min_eval = float('inf')
        for child in node.get_children():
            eval = minimax(child, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval



def alpha_beta(node:Node,depth:int,alpha:float,beta:float,maximizing_player:bool)->float:
     """
    node: estado atual 
    depth: profundidade máxima da busca
    alpha: melhor valor garantido para o jogador max até agora
    beta: melhor valor garantido para o jogador min até agora
    maximizing_player: True se estamos maximizando, False se minimizando
    """

     if depth ==0 or node.is_terminal():
         return node.get_value()
     

     if maximizing_player:
         value=float("-inf")
         for child in node.get_children():
             value=max(value,alpha_beta(child,depth-1,alpha,beta,False))
             
             alpha=max(alpha,value)

             if alpha >=beta:
                 break
        
         return value
     else:
        value = float('inf')
        for child in node.get_children():
            value = min(value,
                        alpha_beta(child, depth - 1, alpha, beta, True))

            beta = min(beta, value)

          
            if beta <= alpha:
                break

        return value


def main():

    leaf1 = Node(value=3)
    leaf2 = Node(value=5)
    leaf3 = Node(value=2)
    leaf4 = Node(value=9)
    leaf5 = Node(value=0)
    leaf6 = Node(value=7)


    n1 = Node(children=[leaf1, leaf2])
    n2 = Node(children=[leaf3, leaf4])
    n3 = Node(children=[leaf5, leaf6])


    root = Node(children=[n1, n2, n3])

    start=time()
    result = minimax(root, depth=3, maximizing_player=True)
    end=time()
    print(f"Tempo:{end -start:.6f}")
    print("Resultado do minimax:", result)
    print("="*60)
    start = time()
    result2 = alpha_beta(root, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
    end = time()
    print(f"Tempo alpha-beta: {end - start:.6f}")
    print("Resultado do alpha-beta:", result2)


if __name__ =='__main__':
    main()