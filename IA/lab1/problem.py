from .aima.search import Problem,breadth_first_graph_search,breadth_first_tree_search,depth_first_graph_search,depth_first_tree_search,uniform_cost_search,astar_search,greedy_best_first_graph_search



from itertools import combinations

DIAS = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta']
SLOTS = ['8-10', '10-12', '14-16', '18-20']

class EscalonamentoTarefas(Problem):
    def __init__(self):
    
        empty_day = ('_', '_', '_', '_')
        days_tuple = (empty_day, empty_day, empty_day, empty_day, empty_day)

   
        self.tarefas = {
            'T1': {'nome': 'Estudar IA', 'duracao': 1,
                   'dias_permitidos': [0,1,2,3,4],
                   'slots_permitidos': [0,1], 'slots_proibidos': [],
                   'dias_proibidos': [], 'preferencia': [0,1],
                   'precedencias': ['T3'], 'repeticoes_minimas': 1},

            'T2': {'nome': 'Reunião Projeto', 'duracao': 1,
                   'dias_permitidos': [1,3],
                   'slots_permitidos': [2], 'slots_proibidos': [],
                   'dias_proibidos': [], 'preferencia': [],
                   'precedencias': [], 'repeticoes_minimas': 1},

            'T3': {'nome': 'Laboratório Python', 'duracao': 2,
                   'dias_permitidos': [0,1,2],
                   'slots_permitidos': [0,1,2], 'slots_proibidos': [3],
                   'dias_proibidos': [], 'preferencia': [],
                   'precedencias': [], 'repeticoes_minimas': 1},

            'T4': {'nome': 'Exercícios Matemática', 'duracao': 1,
                   'dias_permitidos': [0,1,2,3,4],
                   'slots_permitidos': [0,1,2], 'slots_proibidos': [],
                   'dias_proibidos': [], 'preferencia': [],
                   'precedencias': [], 'repeticoes_minimas': 1},

            'T5': {'nome': 'Revisão Provas', 'duracao': 1,
                   'dias_permitidos': [3,4],
                   'slots_permitidos': [0,1,2,3], 'slots_proibidos': [],
                   'dias_proibidos': [], 'preferencia': [],
                   'precedencias': [], 'repeticoes_minimas': 1},

            'T6': {'nome': 'Trabalho Escrito', 'duracao': 2,
                   'dias_permitidos': [1,2,3,4],
                   'slots_permitidos': [0,1,2,3], 'slots_proibidos': [],
                   'dias_proibidos': [0], 'preferencia': [0,1],
                   'precedencias': [], 'repeticoes_minimas': 1},

            'T7': {'nome': 'Videoaula Online', 'duracao': 1,
                   'dias_permitidos': [0,1,2,3,4],
                   'slots_permitidos': [3], 'slots_proibidos': [],
                   'dias_proibidos': [], 'preferencia': [3],
                   'precedencias': [], 'repeticoes_minimas': 1},

            'T8': {'nome': 'Atividade Física', 'duracao': 1,
                   'dias_permitidos': [0,1,2,3,4],
                   'slots_permitidos': [0,1,2], 'slots_proibidos': [3],
                   'dias_proibidos': [], 'preferencia': [],
                   'precedencias': [], 'repeticoes_minimas': 2}
        }

        # manter ordem determinística das tarefas para vetor de contagens
        self.task_list = list(self.tarefas.keys())
        self.task_index = {t:i for i,t in enumerate(self.task_list)}

        # estado inclui: (dias_tuple..., counts_tuple)
        initial_counts = tuple(0 for _ in self.task_list)
        initial_state = tuple(days_tuple) + (initial_counts,)
        # global constraints
        self.max_tarefas_distintas_por_dia = 2
        self.min_free_slots_por_dia = 1

        super().__init__(initial=initial_state)

    # ---------- helpers ----------
    def _counts_from_state(self, state):
        return state[5]  # tuple de contadores na mesma ordem de self.task_list

    def _task_completed_by_counts(self, task, counts):
        idx = self.task_index[task]
        return counts[idx] >= self.tarefas[task]['repeticoes_minimas']

    def _distinct_tasks_in_day(self, day_tuple):
        return set(s for s in day_tuple if s != '_')

    # ---------- actions ----------
    def actions(self, state):
        days = state[:5]
        counts = state[5]
        acoes = []

        for tid, meta in self.tarefas.items():
            idx_task = self.task_index[tid]
            # se já alcançou repetições necessárias, não gerar mais ações para essa tarefa
            if counts[idx_task] >= meta['repeticoes_minimas']:
                continue

            dur = meta['duracao']
            for dia in meta['dias_permitidos']:
                day_slots = list(days[dia])

                # available free slots in day
                free_slots_list = [i for i,s in enumerate(day_slots) if s == '_']
                free_count = len(free_slots_list)
                # precisa ter espaço suficiente
                if free_count < dur:
                    continue
                # precisa sobrar pelo menos 1 slot livre após alocar
                if (free_count - dur) < self.min_free_slots_por_dia:
                    continue

                # count distinct tasks already in that day
                distinct_before = self._distinct_tasks_in_day(day_slots)
                distinct_before_count = len(distinct_before)
                # will the placement introduce a new distinct task?
                will_introduce_new_task = (tid not in distinct_before)

                # check max distinct tasks constraint
                if will_introduce_new_task and (distinct_before_count + 1) > self.max_tarefas_distintas_por_dia:
                    continue

                # Geração de posições possíveis (trata dur==1, dur>1 consecutivo)
                if dur == 1:
                    for s in free_slots_list:
                        # slot permitido?
                        if s not in meta['slots_permitidos']:
                            continue
                        if s in meta['slots_proibidos']:
                            continue
                        # precedências: só permitir colocar tid se todas precedentes já completaram suas repetições
                        if not self._precedencias_satisfeitas(tid, counts):
                            continue
                        # tudo ok: ação (task, day, start_slot, dur)
                        acoes.append((tid, dia, s, 1))
                else:
                    # dur > 1 - precisa de slots consecutivos (T3 exige consecutivos)
                    for start in range(0, 4 - dur + 1):
                        slots = list(range(start, start + dur))
                        # todos os slots devem estar livres
                        if not all(day_slots[s] == '_' for s in slots):
                            continue
                        # slots permitidos/proibidos
                        if any(s not in meta['slots_permitidos'] for s in slots):
                            continue
                        if any(s in meta['slots_proibidos'] for s in slots):
                            continue
                        # precedências
                        if not self._precedencias_satisfeitas(tid, counts):
                            continue
                        # ainda reserva 1 slot livre após colocação? já checado por free_count - dur >= min_free
                        acoes.append((tid, dia, start, dur))
        return acoes

    def _precedencias_satisfeitas(self, tarefa, counts):
        precedentes = self.tarefas[tarefa].get('precedencias', [])
        for p in precedentes:
            if not self._task_completed_by_counts(p, counts):
                return False
        return True

    # ---------- result ----------
    def result(self, state, action):
        tid, dia, inicio, dur = action
        days = [list(d) for d in state[:5]]
        for off in range(dur):
            days[dia][inicio + off] = tid
        new_days_tuple = tuple(tuple(d) for d in days)

        counts = list(state[5])
        counts[self.task_index[tid]] += 1
        new_counts = tuple(counts)

        return new_days_tuple + (new_counts,)

    # ---------- goal ----------
    def goal_test(self, state):
        counts = state[5]
        # todas tarefas atingiram repeticoes minimas?
        for tid in self.task_list:
            if not self._task_completed_by_counts(tid, counts):
                return False

        # restrições globais finais:
        # 1) cada dia deve ter <= max_tarefas_distintas_por_dia e >= min_free_slots_por_dia
        for day in state[:5]:
            if len(self._distinct_tasks_in_day(day)) > self.max_tarefas_distintas_por_dia:
                return False
            if list(day).count('_') < self.min_free_slots_por_dia:
                return False

        # 2) precedência T3 antes de T1: precisamos localizar posições
        pos = {}
        for i, day in enumerate(state[:5]):
            for s_idx, val in enumerate(day):
                if val != '_':
                    pos.setdefault(val, []).append((i, s_idx))
        # Se T3 e T1 alocadas, garanta que a última aparição (T3) termina antes de qualquer T1
        if 'T3' in pos and 'T1' in pos:
            # para T3 achar o último slot ocupacional (tem duração 2)
            t3_positions = pos['T3']  # lista de (dia,slot) onde T3 aparece; como T3 é consecutivo, pega min slot para inicio
            # obter primeiro aparição (menor dia, menor slot)
            t3_first = min(t3_positions)
            t3_start_day, t3_start_slot = t3_first
            t3_end_day, t3_end_slot = t3_start_day, t3_start_slot + self.tarefas['T3']['duracao'] - 1

            # obter todas aparicoes de T1 e garantir T1 start é depois
            for (d1, s1) in pos['T1']:
                if (t3_end_day, t3_end_slot) >= (d1, s1):
                    return False

        # 3) T8 deve ocorrer em pelo menos 2 dias diferentes
        dias_com_T8 = sum(1 for d in state[:5] if any(slot == 'T8' for slot in d))
        if dias_com_T8 < 2:
            return False

        return True

    # ---------- cost e heuristica ----------
    def path_cost(self, c, state1, action, state2):
        tid = action[0]
        custo = c + 1
        prefs = self.tarefas[tid].get('preferencia', [])
        if prefs and action[2] in prefs:
            custo -= 2
        return max(0, custo)
    
    def h(self, node):
        # heurística: número de tarefas ainda não atingiram repetição mínima
        counts = node.state[5]
        faltam = 0
        for tid in self.task_list:
            needed = self.tarefas[tid]['repeticoes_minimas']
            idx = self.task_index[tid]
            if counts[idx] < needed:
                faltam += (needed - counts[idx])
        return faltam

# para teste local (gera primeiras ações iniciais)
if __name__ == "__main__":
    prob = EscalonamentoTarefas()
    print("Estado inicial:")
    print(prob.initial)
    print("\nAções iniciais (até 50):")
    a = prob.actions(prob.initial)
    for ac in a[:50]:
        print(ac)
    print("Total ações iniciais:", len(a))
