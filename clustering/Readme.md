# Cluster Analysis

## Profile Embeddings

Each profile in the dataset if represented as $v_j$.

## Formalism


Let ${v_i}$ be the embedding of the $i$-th user profile, where $i \in \{1, 2, \cdots, N\}$. Let ${v_\text{liked}^{(u)}}$ and  ${v_\text{disliked}^{(u)}}$ be the "liked" and "disliked" vectors for user $u$. We obtain a mean vector ${m} = \frac{1}{N} \sum_{i=1}^{N} {v}_i$ and shift the vector cloud: ${v}_i^{\text{centered}} = {v}_i - {m}$ For a given user $u$ let ${L}_u$ be the set of embeddings of profiles the user likes, and ${D}_u$ be the set of embeddings of profiles the user dislikes. We compute the "liked" and "disliked" vectors as the mean of the respective centered embeddings.


```math

     \mathbf{v}_{\text{liked}}^{(u)} = \frac{1}{|\mathcal{L}_u|} \sum_{\mathbf{v} \in \mathcal{L}_u} (\mathbf{v} - \mathbf{m})

```
```math
\mathbf{v}_{\text{disliked}}^{(u)} = \frac{1}{|\mathcal{D}_u|} \sum_{\mathbf{v} \in \mathcal{D}_u} (\mathbf{v} - \mathbf{m})

```

