import numpy as np
import pandas as pd
from typing import List
from collections import defaultdict
from InquirerPy import inquirer
import ast

def check_warp_and_sarp(prices: List[np.ndarray], bundles: List[np.ndarray]) -> None:
    n = len(prices)
    assert len(bundles) == n, "Antallet af prisvektorer og forbrugsbundter skal være det samme"

    cost_matrix = np.array([[np.dot(p, x) for x in bundles] for p in prices])

    df = pd.DataFrame(cost_matrix,
                      columns=[f"x{j+1}" for j in range(n)],
                      index=[f"p{i+1}" for i in range(n)])
    print("\n\033[1mUdgiftstabel (prisvektor i anvendt på bundt j):\033[0m")
    print(df.round(2))

    revealed_pref_graph = defaultdict(list)
    warp_violations = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cost_matrix[i][i] <= cost_matrix[i][j]:
                revealed_pref_graph[i].append(j)
                if cost_matrix[j][j] <= cost_matrix[j][i]:
                    warp_violations.append((i, j))

    if not warp_violations:
        print("\n\033[1mWARP er opfyldt for hele datasættet.\033[0m")
    else:
        print("\n\033[91mWARP er *ikke* opfyldt. Følgende afslørede præferencer er i konflikt:\033[0m")
        for i, j in warp_violations:
            print(f"x{i+1} afsløret foretrukket over x{j+1} ved p{i+1}, og omvendt ved p{j+1}")

    def has_cycle(graph):
        visited = [0] * n
        def dfs(v):
            if visited[v] == 1:
                return True
            if visited[v] == 2:
                return False
            visited[v] = 1
            for neighbor in graph[v]:
                if dfs(neighbor):
                    return True
            visited[v] = 2
            return False
        for v in range(n):
            if visited[v] == 0:
                if dfs(v):
                    return True
        return False

    if has_cycle(revealed_pref_graph):
        print("\n\033[91mSARP er *ikke* opfyldt: Der findes en cykel i de afslørede præferencer.\033[0m")
    else:
        print("\n\033[1mSARP er opfyldt: Ingen cykler i de afslørede præferencer.\033[0m")

def prompt_for_input():
    prices_input = inquirer.text(
        message="Indtast en liste af prisvektorer (fx [[2,1],[1,3],[3,2]]):"
    ).execute()

    bundles_input = inquirer.text(
        message="Indtast en liste af forbrugsbundter (fx [[1,3],[2,1],[1,2]]):"
    ).execute()

    try:
        prices = [np.array(row) for row in ast.literal_eval(prices_input)]
        bundles = [np.array(row) for row in ast.literal_eval(bundles_input)]
    except Exception as e:
        print("\n\033[91mUgyldigt format. Brug korrekt Python-listeformat som vist.\033[0m")
        exit(1)

    return prices, bundles

def main():
    prices, bundles = prompt_for_input()
    check_warp_and_sarp(prices, bundles)

if __name__ == "__main__":
    main()
