Implementación para Curso de Seminario 2 (2021-2), Universidad de Lima

# Estructura de Modelos:
|BLSTM|Transformer|
|:---:|:---------:|
|<img src="images\BLSTM_100_ModelPlot.png" alt="BLSTM Model" width="350"/>|<img src="images\TRNS_100_ModelPlot.png" alt="Transformer Model" width="350"/>|

<br>

# Matrizes de Confusion
|SeqLen|BLSTM|Transformer|
|:-----------------:|:---:|:---------:|
|100|<img src="images\BLSTM_100_ConfussionMatrix.png" alt="EDNLP Confussion Matrix" width="350"/>|<img src="images\TRNS_100_ConfussionMatrix.png" alt="TRNS Confussion Matrix" width="350"/>|
|20|<img src="images\BLSTM_20_ConfussionMatrix.png" alt="EDNLP Confussion Matrix" width="350"/>|<img src="images\TRNS_20_ConfussionMatrix.png" alt="TRNS Confussion Matrix" width="350"/>|

<!-- CSV to MD Table: https://www.convertcsv.com/csv-to-markdown.htm -->
<br>

# Podio
|         |BLSTM 100|BLSTM 20|TRNS 100 | TRNS 20 |
|:-------:|:-------:|:------:|:-------:|:-------:|
| Valor F |  0.814  | 0.817  |**0.855**|  0.849  |
|Precisión|  0.823  | 0.815  |**0.848**|  0.845  |
|Exactitud|  0.849  | 0.852  |**0.878**|  0.873  |
| Tiempo  |  85.17  |  32.4  | 44.58   |**27.43**|