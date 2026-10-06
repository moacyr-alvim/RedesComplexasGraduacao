# Zipf dos candidatos a deputado (TSE 2022)

Gráfico de ranking (eixo x) × votos (eixo y), em escala log-log, dos candidatos
a deputado nas eleições de 2022, e ajuste de modelos à distribuição de votos.

## Dados

Dados Abertos do TSE, conjunto "Votação por candidato por município e zona":

    https://cdn.tse.jus.br/estatistica/sead/odsele/votacao_candidato_munzona/votacao_candidato_munzona_2022.zip

O zip (~640 MB) não está versionado. Os votos são somados por candidato
(`QT_VOTOS_NOMINAIS`) a partir do CSV `BRASIL`.

## Uso

    pip install -r requirements.txt
    python zipf_deputados.py votacao_candidato_munzona_2022.zip resultados

Gera `resultados/votos_por_candidato.csv` (votos por candidato) e os gráficos de
ranking × votos. Os scripts de `modelos/` leem esse CSV e usam apenas
Deputado Federal (Brasil, SP e BA).

## Análises (`modelos/` → `resultados/`)

Ordem em que foram feitas. Cada etapa corrige ou refina a anterior.

| Script | O que faz | Saída |
|---|---|---|
| `nb_sem_corte.py` | Binomial negativa (MLE) e lognormal sobre todos os candidatos | `nb_lognormal_sem_corte.png` |
| `nb_corte5000.py` | Idem, só com candidatos ≥ 5.000 votos (verossimilhança truncada) | `nb_lognormal_corte5000.png` |
| `bins_logaritmicos.py` | Histograma com bins geométricos (razão 1,6) + NB, lognormal e distribuição logarítmica; χ² de aderência | `histograma_bins_logaritmicos.png` |
| `plc_run.py` (usa `plc.py`) | Lei de potência com corte exponencial, método de Clauset–Shalizi–Newman | `lei_de_potencia_com_corte.*` |

    python modelos/nb_sem_corte.py     resultados/votos_por_candidato.csv saida.png
    python modelos/nb_corte5000.py     resultados/votos_por_candidato.csv saida.png
    python modelos/bins_logaritmicos.py resultados/votos_por_candidato.csv saida.png
    python modelos/plc_run.py          resultados/votos_por_candidato.csv saida 200 500

(`plc_run.py` leva ~15 min com 4 núcleos; os dois últimos números são as réplicas
de bootstrap com x_min estimado e com x_min fixo em 5.000.)

## Resultados

- **Binomial negativa:** ajusta mal sem corte (não captura a cauda). Com corte em
  5.000 votos converge para o limite `r → 0`, que é a distribuição logarítmica;
  `r` e `μ` não são identificáveis (ver `bins_logaritmicos.py`, que confere a
  convergência por varredura em grade).
- **Distribuição logarítmica (1 parâmetro):** tão boa quanto a NB e com menor AIC.
- **Lei de potência com corte** `p(x) ∝ x^-α e^(-θx)`: α ≈ 1,3 e escala de corte
  1/θ ≈ 2–3×10⁵ votos, com x_min (KS) de 389–835 votos. Fortemente preferida à lei
  de potência pura, à exponencial e à lognormal na cauda, e a α = 1.
  A aderência (bootstrap) é ainda rejeitada (p < 0,1), exceto marginalmente em SP.

## Ressalvas

- Os candidatos não são independentes (partido, estado), o que afeta os p-valores.
- `nb_sem_corte.py` ajusta a lognormal pelos momentos de `log(votos)`, e não por MLE;
  os demais scripts usam MLE para todos os modelos.
- O ajuste da lei de potência usa a aproximação contínua; verossimilhanças absolutas
  não são comparáveis com as da versão discreta de `bins_logaritmicos.py`.
- O topo do ranking (poucos candidatos mais votados) é mal descrito por todos os modelos.
