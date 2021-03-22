<h1>Logistic Regression</h1>

- Possui, atualmente, duas implementações em octave de regressão logística e um diretório `helpers` com funções auxiliares
- Implementações para features normais e que necessitam de regularização, além de aumentar o grau da função.
> Implementações realizadas durante o curso de Aprendizado de Máquina, oferecido pela Stanford University e disponível na plataforma Coursera.

<h2>Normal</h2>

- Implementação de um modelo em um conjunto de dados de exames de alunos, a fim de ser capaz de prever se o aluno será aprovado ou não.
- O dataset é composto por: `X(1) = avalição 1`, `X(2) = avaliação 2` e `y = aprovado ou não`.
- O arquivo `costFunction` possui as implementações das funções de custo e gradiente descedente.

<p align="center">
    <img src="https://github.com/fdloopes/Praticas_Machine_Learning/blob/main/Octave/Logistic_Regression/normal/result.png"/><br/>
    <em>Resultado aplicado aos dados</em>
</p>

<h2>Multi Features</h2>

- Implementação de um modelo em um conjunto de dados de teste de qualidade de microchip, a fim de ser capaz de prever se um microchip será aprovado ou não no teste de qualidade.
- O dataset é composto por: `X(1) = teste 1`, `X(2) = teste 2` e `y = aprovado ou não`.
- O arquivo `costFunction` possui as implementações das funções de custo e gradiente descedente, onde é utilizado o termo regularizador da função.
- O arquivo `mapFeature` possui a implementação da função responsável por aumentar o grau de entrada dos dados, aumentando de 2 features iniciais para 28 features, com intuito de aumentar a precisão.
- Também é utilizada a função `fminunc` para otimização, a fim de encontrar o theta de forma mais eficiente e precisa.

<p align="center">
    <img src="https://github.com/fdloopes/Praticas_Machine_Learning/blob/main/Octave/Logistic_Regression/regularized/result.png"/><br/>
    <em>Resultado após 50 iterações</em>
</p>
