# [As funções cognitivas são capazes de predizer a efetividade geral e as contribuições do atleta para a equipe em jogos reduzidos de futebol? Uma abordagem utilizando aprendizado de máquina.](https://bv.fapesp.br/pt/bolsas/209042/as-funcoes-cognitivas-sao-capazes-de-predizer-a-efetividade-geral-e-contribuicao-do-atleta-para-a-eq/)

## Visão geral do projeto

Este repositório contém dados e softwares desenvolvidos em Python para a predição do desempenho de jovens jogadores em jogos reduzidos por meio da análise de funções cognitivas utilizando algoritmos de aprendizado de máquina. Os dados compartilhados são resultado do projeto de mestrado do aluno Rafael Luiz Martins Monteiro com fomento da Fundação de Amparo à Pesquisa do Estado de São Paulo, processo #2021/15134-9. A dissertação é intitulada: "As funções cognitivas são capazes de predizer a efetividade geral e as contribuições do atleta para a equipe em jogos reduzidos de futebol? Uma abordagem utilizando aprendizado de máquina.".

Foram utilizados dados obtidos por meio de testes computadorizados com softwares como PEBL (https://pebl.sourceforge.net/)  e PsychoPy (https://www.psychopy.org/), focando em avaliações de atenção sustentada, memória visuoespacial, impulsividade, flexibilidade cognitiva e capacidade de rastreamento de objetos múltiplos. Os testes utilizados foram: Blocos de Corsi, Go/ No Go e teste de trilhas no PEBL e rastreamento de objetos múltiplos no PsychoPy.

A avaliação em campo foi feita por meio de jogos reduzidos de 3 x 3 jogadores sem goleiros. Cada jogo tinha duração de 4 minutos. Após cada rodada os times eram misturados de forma aleatória, dessa forma os jogadores não jogavam com ou contra os mesmos colegas. O objetivo da avaliação é extrair a performance individual de cada atleta por meio da interação coletiva entre os jogadores. Este protoclo foi proposto e validado por Wilson et al. 2021 (https://onlinelibrary.wiley.com/doi/full/10.1111/sms.13969).

<p align="center">
  <img src="https://github.com/user-attachments/assets/b595a0ce-6425-4b5d-9c8b-78c8d9749c09" alt="Campo" width="45%" />
  <img src="https://github.com/user-attachments/assets/24891f2c-6b51-4c2f-afee-c5f580eb06c9" alt="Gráfico" width="45%" />
</p>

## Acesso aos dados
Todos os códigos utilizados para o processamento de dados podem ser acessados na pasta [src](./src/). Os dados anonimizados utilizados para o processamento e trabalho final podem ser acessados na pasta [data](./data/). Dentro desta pasta tem os dados brutos e os dados processados que foram utilizados para análise no arquivo 'dataset.csv'. Todos os procedimentos de análise, incluindo treinamento dos algoritmos e resultados podem ser acessados na pasta [statistics](./statistics/).


### Agradecimentos
Os autores agradecem imensamente ao pessoal técnico e aos atletas pela participação e contribuições fundamentais para este estudo. Além disso, agradecemos o apoio financeiro da Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP) sob as bolsas [#2021/15134-9](https://bv.fapesp.br/53743), [#2020/14845-6](https://bv.fapesp.br/54220), [#2019/22262-3](https://bv.fapesp.br/51740), [#2019/17729-0](https://bv.fapesp.br/51021) and [#2024/06226-5](https://bv.fapesp.br/52558), da Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) – Código de Financiamento 001, e do Centro Acadêmico Bavariano para a América Latina (BAYLAT).

## Contato
Para mais informações ou dúvidas, por favor, entre em contato pelo e-mail [rafaell_mmonteiro@usp.br](mailto:rafaell_mmonteiro@usp.br).

