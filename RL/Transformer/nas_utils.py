import numpy as np
from collections import deque

def is_connected(adjacency_matrix: np.ndarray) -> bool:
    """
    Checks if a path exists from input node 0 to the final node N
    in an (N+1)x(N+1) upper-triangular adjacency matrix (binary).
    """
    A = (adjacency_matrix > 0).astype(np.uint8)
    num_nodes = A.shape[0]
    src, dst = 0, num_nodes - 1

    q = deque([src])
    seen = {src}
    while q:
        u = q.popleft()
        for v in range(u + 1, num_nodes):
            if A[u, v] == 1 and v not in seen:
                seen.add(v)
                q.append(v)
    return dst in seen
