#!/usr/bin/python

import sys, getopt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools
from itertools import *
from fractions import Fraction
pd.set_option('display.max_columns', 500)
np.set_printoptions(threshold=np.inf)
try:
    from qif import *
except: # install qif if not available (for running in colab, etc)
    #import IPython; IPython.get_ipython().run_line_magic('pip', 'install qif')
    from qif import *

result = []
 
def main(argv):
   graph_file = argv[0]
   corrupted_file = argv[1]
   print('The graph file is: ', graph_file)
   print('The corrupted file is: ', corrupted_file)

   with open(graph_file, "rt") as f:
      X = f.readlines()
      X = [i.strip('\n') for i in X]
      X = [[float(Fraction(x)) for x in line.split(" ")] for line in X]
      matrix = np.array(X)
   
   # N is the number of cryptographers at the table
   N = np.shape(matrix)[0]
   print('\nThe graph is:')
   print(matrix)
   
   corrupted_users = np.loadtxt(corrupted_file)
   # if there is only one corrupted user make the scalar an array
   if corrupted_users.size != 0:
      if corrupted_users.size == 1:
         corrupted_users = np.array([int(corrupted_users)])
   print('\nThe corrupted users are:')
   print(corrupted_users)

   for i in range(N+1):    
      if i % 2 != 0: 
         cols = kbits(N, i)


   x = [[0 for i in range(len(cols))] for j in range(N)]
   for i in range(N):
      x[i][i] = 1
   C = np.asarray(x, dtype=int)
  
   
   # print('\nThe initial channel is:')
   # df0 = pd.DataFrame(C, columns=cols)
   # df0.columns = ['\'' + s[:N] + '\'' for s in df0.columns]
   # print(df0)
   # print(df0.size)
   
   
   for i in range(N):  
      for j in range(N):
         if matrix[i][j] != -1 and j>i:
            p = matrix[i][j]
            df = pd.DataFrame(C, columns=cols)
            cols = df.columns
            pos = [i, j]
            #check if this is a corrupted user
            if j in corrupted_users or i in corrupted_users:
                arr1 = np.array([[bool(int(y)) for y in x] for x in cols])
                arr1[:, pos] = ~arr1[:, pos]
                df1 = pd.DataFrame(C, columns=cols)
                df1.columns = [''.join(str(int(y)) for y in x) for x in arr1]
                df1 = df1[cols]
                cols0 = [s + '0' for s in cols]
                cols1 = [s + '1' for s in cols]
                C_ = df1.values
                pC = p * C
                pC_ = (1-p) * C_
                df0 = pd.DataFrame(pC, columns=cols0)
                df1 = pd.DataFrame(pC_, columns=cols1)
                df = pd.concat([df0[cols0],df1[cols1]], axis=1)
                cols = df.columns
                C = df.values
            else:
                arr = np.array([[bool(int(y)) for y in x] for x in cols])
                arr[:, pos] = ~arr[:, pos]
                df.columns = [''.join(str(int(y)) for y in x) for x in arr]
                df = df[cols]
                C_ = df.values
                C = p * C + (1-p) * C_ 
                df = pd.DataFrame(C, columns=cols)
               # channel = C[:,~np.all(C == 0, axis = 0)]
               # print('\n', channel.tolist())
         
            
   # Deleting the corrupted user
   for k in range(corrupted_users.size):
      df = df.drop(int(corrupted_users[k]))
   df = df.loc[:, (df != 0).any(axis=0)]
   # Deleting all columns that are filled with zeros
   if corrupted_users.size != 0:
      df.columns = ['\'' + s[:N] + '\',\'' + s[N:] + '\'' for s in df.columns]
      print('\nThe final channel -including the announcements and the coins- is:')
   else:
      df.columns = ['\'' + s[:N] + '\'' for s in df.columns]
      print('\nThe final channel is:')
   print(df)
   
   
  # pi = probab.uniform(N)
  # print("\nPrior Bayes vulnerability =", measure.bayes_vuln.prior(pi))
  # print("Posterior Bayes vulnerability =", measure.bayes_vuln.posterior(pi,C))
  # print("Multiplicative Bayes leakage =", measure.bayes_vuln.mult_leakage(pi,C))
  
  
  

   

def kbits(n, k):
    for bits in itertools.combinations(range(n), k):
        s = ['0'] * n
        for bit in bits:
            s[bit] = '1'
        result.append(''.join(s))
    return result

if __name__ == "__main__":
   main(sys.argv[1:])