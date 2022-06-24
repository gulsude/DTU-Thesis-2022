#different encodings here
import pandas as pd
import numpy as np

aminoacidTp = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def encodePeptides(peptides,scheme,bias=False):

       #loading the matrice
       bl50 = pd.read_csv("/home/s202357/thesis/transmut/source/matrices/BLOSUM50", sep="\s+", comment="#", index_col=0)
       bl50 = bl50.loc[aminoacidTp, aminoacidTp]

       #output
       encoded_pep = []

       #converting scheme to list if needed
       if type(scheme) != list:
              scheme = [scheme]

       #encding by peptide/by aa/ by scheme
       for peptide in peptides:
              pos = 0
              seq = []
              for aa in peptide:
                     for sc in scheme:
                            if sc == "blosum":
                                   seq.append(bl50.loc[[aa]].values.tolist()[0])

                            elif sc in aaProperties:
                                   seq.append(aaIndex[aa][sc])

                            elif sc == "sparse":
                                   seq += sp[aa].values.tolist()

                            elif sc == "sparse2":
                                   seq += sp2[aa].values.tolist()

                            elif sc == "sparse3":
                                   seq += sp3[aa].values.tolist()

                            elif sc == "allProperties":
                                   seq += aaIndex[aa].values.tolist()

                            elif sc == "vhse":
                                   seq += vhse[aa].values.tolist()
                                    
                            elif sc == "pssm":
                                   seq.append(pssm[aa][pos])
                            
                            else: 
                                   print("ERROR: No encoding matrix with the name {}".format(sc))
                                   
                     pos = pos + 1
              if bias:
                     seq.append(1)
              encoded_pep.append(seq)
       return encoded_pep