# Zipf dos candidatos a deputado (TSE 2022)

Gráfico de ranking (eixo x) × votos (eixo y), em escala log-log, dos candidatos
a deputado nas eleições de 2022.

## Dados

Dados Abertos do TSE, conjunto "Votação por candidato por município e zona":

    https://cdn.tse.jus.br/estatistica/sead/odsele/votacao_candidato_munzona/votacao_candidato_munzona_2022.zip

O zip (~640 MB) não está versionado. Os votos são somados por candidato
(`QT_VOTOS_NOMINAIS`) a partir do CSV `BRASIL`.

## Uso

    pip install -r requirements.txt
    python zipf_deputados.py votacao_candidato_munzona_2022.zip resultados

## Resultados (`resultados/`)

- `votos_por_candidato.csv`: votos por candidato (Federal, Estadual, Distrital)
- `zipf_deputados.png`: Deputado Federal, Brasil e São Paulo
- `zipf_por_uf.png`: uma curva por UF
