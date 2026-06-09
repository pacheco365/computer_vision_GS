# Bio-Orbit Labs — Visão Computacional (ACV) · versão MARCO
### Classificação de resultados de cristalização de proteína (dataset real)

Global Solution 2026 · Indústria Espacial · Engenharia de Software 4º ano — FIAP

**Integrantes:** Sofia Coutinho – RM552534 · Anna Yagyu – RM550360 · Felipe Capriotti - RM98460 · Gustavo Kawamura - RM99679 · Gabriel Pacheco - RM550191

---

## 1. Problema e conexão espacial
Empresas farmacêuticas cultivam cristais de proteína na ISS e em minissatélites porque a
microgravidade gera cristais mais perfeitos, acelerando a descoberta de fármacos. Avaliar
visualmente cada gota de cristalização consome banda de telemetria cara e escassa.
**Nossa solução roda a triagem visual a bordo (Edge Laboratory Computing):** uma CNN classifica
o resultado da gota direto no satélite e envia para a Terra apenas o rótulo + alerta (poucos bytes),
em vez da imagem inteira. **ODS:** 9 (Inovação e infraestrutura) e 3 (Saúde e bem-estar).

A classe `Other`/`Precipitate` (falha) é o elo de integração com as demais disciplinas: é o evento
que aciona o RPA para corrigir os parâmetros do reator.

## 2. Dataset — MARCO (real)
[MARCO](https://marco.ccr.buffalo.edu) (MAchine Recognition of Crystallization Outcomes) reúne
~463 mil imagens reais de experimentos de cristalização de 5 instituições, em 4 classes
normalizadas (índice oficial): **0=Clear, 1=Crystals, 2=Other, 3=Precipitate**. Usamos o
subconjunto público do Kaggle (`grantwiersum/marco-protein-crystal-image-recognition`), distribuído
em **TFRecords** (as imagens JPEG vêm codificadas dentro dos registros). Por viabilidade de treino
do zero no Colab, usamos um subconjunto de **20.000** imagens de treino, **4.000** de validação e
**4.000** de teste; cada imagem é redimensionada para **128×128 RGB**.

> **Dificuldade do problema:** dado real e ruidoso. A concordância entre crystallographers humanos
> é de ~70% (≈93% só para cristais), e os melhores resultados publicados (~94%) usaram redes
> **pré-treinadas** e dezenas de GPUs. Treinando **do zero**, um resultado na faixa de ~71–77% é
> forte e honesto.

## 3. Metodologia
Duas CNNs implementadas **do zero** em TensorFlow/Keras (sem modelos pré-treinados):
- **CNN_Compacta** — 3 blocos convolucionais + `Flatten` + denso (~4,3M parâmetros).
- **CNN_Profunda** — 4 blocos (com convoluções duplas no início) + `GlobalAveragePooling` +
  `Dropout(0.5)` (~0,5M parâmetros): mais profunda **e** muito mais leve, argumento direto para
  execução na borda, a bordo do satélite.

Detalhes do pipeline: imagens trafegam em **uint8** e a normalização ocorre **dentro do modelo**
(camada `Rescaling`), o que reduz drasticamente o uso de RAM; **pesos de classe suaves** (raiz do
inverso da frequência) tratam o desbalanceamento sem sabotar a classe majoritária; data
augmentation (flip, rotação, zoom); otimizador Adam (1e-3) com `EarlyStopping` e `ReduceLROnPlateau`.

**Decisão técnica documentada:** testamos `BatchNormalization` e observamos instabilidade de
convergência neste cenário (acurácia oscilando/colapsando); por isso adotamos `Dropout` +
`ReduceLROnPlateau`. É uma escolha empírica, não um descuido.

## 4. Resultados
Treino do zero, conjunto de teste com 3.072 imagens. Baseline trivial (chutar a classe maior,
`Precipitate`) = **0,476**.

| Modelo | Acurácia de teste | Macro-F1 |
|---|---|---|
| CNN_Compacta | 0,710 | 0,54 |
| **CNN_Profunda (melhor)** | **0,749** | **0,64** |

O melhor modelo fica ~27 pontos acima do baseline trivial, com acerto distribuído nas quatro
classes — ou seja, aprendizado real. As classes grandes (`Clear`, `Precipitate`) têm bom F1
(0,85 e 0,78). A classe `Other` é a mais difícil — categoria "resto", ambígua até para humanos,
o que reflete a concordância humana de ~70%. **Foi justamente `Other` que decidiu a comparação:**
seu recall subiu de 0,05 (Compacta) para 0,33 (Profunda), ou seja, a capacidade extra da rede
profunda rendeu onde o problema era mais difícil.

> **Nota sobre reprodutibilidade:** por ser treino do zero em GPU, há variação estocástica entre
> execuções; o melhor modelo fica estável na faixa de ~71–77% e a análise se sustenta em todos os
> casos (a classe `Other` é sempre o fator decisivo).

Matrizes de confusão, curvas de acurácia/loss e análise de erros estão no notebook.

## 5. Como executar (Google Colab)
1. Abra `Computer_Vision_GS.ipynb` no Colab e selecione **GPU**.
2. `Executar tudo`. Na primeira execução o `kagglehub` pede a credencial Kaggle
   (kaggle.com → Settings → Create New Token → `kaggle.json`).
3. O melhor modelo é salvo em `bio_orbit_acv_marco_best.keras`.

## 6. Estrutura
```
acv/
├── Computer_Vision_GS.ipynb            # notebook principal (MARCO real)
├── bio_orbit_acv_marco_best.keras       # pesos do melhor modelo (gerado ao rodar)
├── requirements.txt
└── README.md
```


## 7. Limitações e trabalhos futuros
O modelo foi treinado **do zero**, sem transfer learning (exigência da disciplina). O trabalho
original do MARCO atinge ~94% usando redes **pré-treinadas** (ex.: Inception-v3) e grande poder
computacional; um próximo passo natural seria aplicar transfer learning, elevando a acurácia para
a faixa de 90%+. Outras melhorias: usar o MARCO completo (~463 mil imagens), oversampling das
classes minoritárias e blocos residuais. A acurácia atual (~75%) é coerente com a dificuldade do
problema — a concordância entre especialistas humanos é de ~70%.

## Links
- 🎥 Vídeo (≤3 min): _(https://youtu.be/qoeRN_6Ou-8)_
