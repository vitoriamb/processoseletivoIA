👤 Nome Completo: Vitória Maciel Bernardo

1. Resumo da Arquitetura do Modelo

A CNN tem 2 blocos convolucionais (8 e 16 filtros, kernel 3×3), cada um seguido de MaxPooling 2×2. Depois disso, usei uma camada Dense com 32 neurônios e a saída softmax com 10 classes.

Optei por usar apenas 2 blocos convolucionais porque, com convoluções sem padding, após os dois poolings o mapa de características chega a aproximadamente 5×5. A partir desse ponto, adicionar mais uma camada aumentaria o número de parâmetros, mas sem trazer ganho relevante de informação espacial.

A Dense com 32 neurônios foi uma escolha pra deixar o modelo mais leve, sem perder acurácia de forma significativa.

No final, o modelo ficou com 14.410 parâmetros, o que é bem adequado para cenários de Edge AI.

2. Bibliotecas Utilizadas
TensorFlow ≥ 2.12
NumPy
3. Técnica de Otimização do Modelo

Usei duas técnicas de quantização pra explorar o trade-off entre tamanho e precisão:

Dynamic Range Quantization (~20.49 KB)
Converte os pesos para int8 sem precisar de dados de calibração. É a opção mais prática pra CPU, principalmente em dispositivos embarcados, por reduzir bastante o tamanho do modelo.
Float16 Quantization (~33.93 KB)
Converte os pesos para float16, mantendo mais precisão numérica. Funciona melhor em hardware que já tem suporte a esse formato (como algumas GPUs).

De forma geral, para Edge AI em CPU, a quantização Dynamic Range acaba sendo a melhor escolha.

4. Resultados Obtidos
Acurácia no teste: 98,54%
Loss: 0,0462

Analisando por classe, o dígito com menor acurácia foi o 8 (96,1%), provavelmente por confusão com 3 e 9, que têm formatos parecidos. Isso já era esperado considerando o tamanho reduzido do modelo.

5. Comentários Adicionais

A principal decisão foi não aumentar a profundidade do modelo. Em Edge AI, um modelo maior nem sempre compensa — o ganho em acurácia costuma ser pequeno perto do custo em memória e tempo de inferência.

Com 14.410 parâmetros e 98,54% de acurácia, o modelo já atende bem ao objetivo do desafio. A quantização Dynamic Range ainda reduziu o tamanho em cerca de 64% em relação ao modelo original (.h5), com impacto praticamente nulo na acurácia.