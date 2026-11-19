"""Utility functions supporting the other modules"""

import torch

def norm_std(x):
  x = x - x.mean()
  s = x.std()
  if s == 0:
    return torch.ones_like(x)*0.5
  else:
    return x / s

def norm_01(x):
  x = x - x.min()
  mx = x.max()
  if mx == 0:
    return torch.ones_like(x)*0.5
  else:
    return x / mx

def norm_img_01(x):
  B,C,H,W = x.shape
  x = x - torch.min(x.reshape(B,C, H*W), dim=-1).values.reshape(B,C,1,1)
  x = x / torch.max(x.reshape(B,C, H*W), dim=-1).values.reshape(B,C,1,1)
  return x

def compute_cos_sim(vec1: torch.FloatTensor,
                    vec2: torch.FloatTensor,
                    softmax: bool = False) -> torch.FloatTensor:
  """Compute cosine similarity between two batches of D dim vectors.
  
  Args:
    vec1: NxC float tensor representing batch of vectors
    vec2: MxC float tensor representing batch of vectors
    softmax: If False, cosine similarity is returned. If True, softmaxed
      probability is returned across the N dimension.
  Returns:
    result: MxN float tensor representing similarity/prob. where result[0,1]
      represents the similarity of vec2[0] with vec1[1].
  """
  N, C1 = vec1.shape
  M, C2 = vec2.shape
  if C1 != C2:
    raise ValueError(f"vec1 feature dimension '{C1}' does not match vec2"
                     f"feature dimension '{C2}'")
  vec1 = vec1 / vec1.norm(dim = -1, keepdim = True)
  text_columns = vec1.transpose(0, 1)

  def score_rows(rows):
    rows = rows / rows.norm(dim=-1, keepdim=True)
    # A 2D GEMM gives the same MxN dot products without broadcasting both
    # inputs into MxNxC copies (several GiB on a growing global voxel map).
    sim = rows @ text_columns
    return torch.softmax(100 * sim, dim=-1) if softmax else sim

  # Chunk only the independent voxel/ray rows. Every row still sees ALL N
  # queries in its softmax. This bounds the normalized feature temporary
  # (64 MiB per chunk for 512-dimensional float32 features), preserving
  # input precision, zero-vector behavior, output order and autograd.
  chunk_rows = 32768
  if M <= chunk_rows:
    return score_rows(vec2)
  return torch.cat([score_rows(vec2[start:start + chunk_rows])
                    for start in range(0, M, chunk_rows)], dim=0)


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    if len(hex_code) == 3:
        hex_code = ''.join([char * 2 for char in hex_code])
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
