# Riccardo-Bona-InforetProject_P6

-Il repository contiene due files chiamati 'Project.py' e 'GameEvaluation.py'. Il primo contiene l'intero progetto, il secondo contiene una versione ristretta del 
 codice, utile alle sole funzioni che permettono di valutare quanto 'german' o 'american' sia un set di meccaniche arbitrarie.

-Il seguente tutorial si riferisce al file 'GameEvaluation.py'.

-Di seguito la lista degli import necessari:

import json

import pandas as pd

import numpy as np

from gensim.models import Word2Vec

from gensim.models.keyedvectors import KeyedVectors

from sklearn import cluster

from sklearn.decomposition import PCA

import matplotlib

import matplotlib.pyplot as plt

import itertools


-Lo script fa uso di due files esterni contenuti nella cartella 'data': 'ggt3.json', contenente i dati relativi a 18800 boardgames e 'mechanic2vec.model', un modello  
 Word2Vec (gensim) il cui training è stato eseguito sulle meccaniche di tutti i giochi contenuti nel dataframe.
 
-Una volta lanciato lo script è possibile utilizzare tre funzioni:

-    mechanics_ranking(top_n)

     dove 'top_n' è un numero intero: permette di visualizzare un grafico delle prime n meccaniche (in ordine discendente di frequenza con la quale appaiono nei 
     giochi del dataframe) con il relativo score di quanto ogni meccanica sia valutata 'german' e 'american' (compreso tra 0 e 1).
 
-    evaluate_game(mechanics)

     dove 'mechanics' è una lista di meccaniche. I 182 valori possibili delle meccaniche sono elencati nel file di testo presente nel repository 'mech_values.txt'.
     La funzione restituisce il valore di quanto in totale l'insieme di meccaniche appartenga alle categorie 'german' e 'american' e i rispettivi valori di ogni
     singola meccanica.
     La funzione mostra inoltre le medesime informazioni tramite un grafico.
     
     E.g.
     
     In:
     
     new_game = ['Dice Rolling', 'Market', 'Hexagon Grid', 'Role Playing']
     
     evaluate_game(new_game)
     
     Out:
     
     American:  0.6036637042976677 
     German:  0.46544516283490034 
     
     Dice Rolling:  [American:  0.9077027042356596 ], [German:  0.5374427795084435 ]
     
     Hexagon Grid:  [American:  0.5739635620737565 ], [German:  0.35432375483082634 ]
     
     Role Playing:  [American:  0.6765184953935794 ], [German:  0.28141149461690673 ]
     
     Market:  [American:  0.2564700554876753 ], [German:  0.6886026223834245 ]


-    game_in_space(mechanics)
     
     dove 'mechanics' è la lista di meccaniche utilizzata per la funzione precedente.
     La funzione mostra uno scatterplot nel quale vengono mostrate le meccaniche nello spazio e la posizione rispetto ad esse del gioco arbitrario indicato come lista 
     di meccaniche.
