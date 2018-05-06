import numpy as np
import pandas as pd

def branching_embedding(Z, method='fix', theta=15):
  """
  This function calculates two-dimensional coordinates from
  a dendrogram, which is named branching  embedding (BE). 

  Parameters
  ----------
  Z : ndarray
    Linkage matrix.
  method : {'fix', 'random', 'even'}, optional
    indicates the way to determine division angles.
    * fix: fixed angles (default)
    * random: random angles
    * even: angles are tuned so that two child clusters have the same
            distance from the sister cluster
  theta : float, optional
    fixed division angle in degree. Default is 15.
    This value is used when method is 'fix'.

  Returns
  -------
  pos_df : DataFrame
    two-dimensional positions of the leaf nodes
  
  Examples
  --------
  >>> from scipy.cluster.hierarchy import distance, linkage
  >>> X = np.random.randn(5,3)                     # data matrix
  >>> Y = distance.pdist(X, metric='euclidean')    # distance matrix
  >>> Z = linkage(Y, method='average')             # linkage matrix
  >>> pos_df = branching_embedding(Z)

  """
  # collect information -------------------------------------
  # child 1, child 2, dissimilarity, size, and sister

  n = len(Z) + 1
  info_df = pd.DataFrame(Z[:, [0,1,3]].astype(int),
                         index=range(n, 2*n-1),
                         columns=['c1', 'c2', 'size'])
  d_sr = pd.Series(Z[:,2], index=range(n, 2*n-1))

  sis_sr = pd.Series(0, index=range(2*n-2))
  for i, (c1, c2, size) in info_df.iterrows():
    sis_sr[c1] = c2
    sis_sr[c2] = c1

  theta_rad = np.pi * theta / 180


  # calculate coordinates  ----------------------------------

  # define functions
  def get_size(i):
    return 1 if i < n else info_df.at[i, 'size']
  def get_actual_theta(n1, n2, l1, l2, L):
    if method == 'fix':
      if (theta < 30) and (theta > 0):
        return theta_rad if n2 > n1 else np.pi + theta_rad
      else:
        return theta_rad
    elif method == 'random':
      return 2 * np.pi * np.random.random()
    elif method == 'even':
      return np.sign(np.random.randn()) * np.arccos((l1-l2)/(2*L))
    else:
      return 0

  pos_df = pd.DataFrame(0, index=range(2*n-2), columns=list('xy'))

  for i in info_df.index[::-1]:
    c1, c2 = info_df.loc[i, ['c1','c2']]
    n1, n2 = (get_size(c1),  get_size(c2))
    d      = d_sr[i]
    l1     = d * n2 / (n1 + n2)
    l2     = d * n1 / (n1 + n2)

    # the first branching
    if i == 2*n - 2:
      pos_df.loc[c1] = (l1, 0)
      pos_df.loc[c2] = (-l2, 0)
    elif d == 0:
      pos_df.loc[c1] = pos_df.loc[i]
      pos_df.loc[c2] = pos_df.loc[i]
    else:
      pos_s  = pos_df.loc[sis_sr[i]]  # sister node
      pos_i  = pos_df.loc[i]
      L      = np.linalg.norm(pos_s - pos_i)
      th     = get_actual_theta(n1, n2, l1, l2, L)
      phi    = np.angle(np.complex(*(pos_i - pos_s)))
      psi    = phi + th - np.pi
      pos_df.loc[c1] = pos_i + [l1 * np.cos(psi), l1 * np.sin(psi)]
      pos_df.loc[c2] = pos_i - [l2 * np.cos(psi), l2 * np.sin(psi)]

      
  return pos_df.iloc[:n]

if __name__ == '__main__':
  from scipy.cluster.hierarchy import distance, linkage
  X = np.random.randn(5,3)                     # data matrix
  Y = distance.pdist(X, metric='euclidean')    # distance matrix
  Z = linkage(Y, method='average')             # linkage matrix
  pos_df = branching_embedding(Z)  
