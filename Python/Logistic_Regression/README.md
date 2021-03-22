<h1>Logistic Regression</h1>

- Possui, atualmente, duas implementações em Python de regressão logística e um diretório `helpers` com funções auxiliares
- Implementação para distribuição normal de dados e que necessitam aumentar o grau da função.

<h2>Normal</h2>

- Implementação de um modelo em um conjunto de dados de exames de alunos, a fim de ser capaz de prever se o aluno será aprovado ou não.
- O dataset é composto por: `X(1) = avalição 1`, `X(2) = avaliação 2` e `y = aprovado ou não`.
- O arquivo `functions` possui as implementações das funções de custo e gradiente descedente.

<p align="center">
    <img src="https://github.com/fdloopes/Praticas_Machine_Learning/blob/main/Python/Logistic_Regression/normal/data_boundary_line.png"/><br/>
    <em>Resultado aplicado aos dados</em>
</p>

<h2>Regularized</h2>

- Implementação de um modelo em um conjunto de dados de teste de qualidade de microchip, a fim de ser capaz de prever se um microchip será aprovado ou não no teste de qualidade.
- O dataset é composto por: `X(1) = teste 1`, `X(2) = teste 2` e `y = aprovado ou não`.
- O arquivo `functions` possui as implementações das funções de custo e gradiente descedente, onde é utilizado o termo regularizador da função.
- Dentro da pasta helpers o arquivo `functions` possui a implementação da função responsável por aumentar o grau de entrada dos dados, aumentando de 2 features iniciais para 28 features, com intuito de aumentar a precisão.
- Também é utilizada a função `minimize` da biblioteca `scipy` para otimização, a fim de encontrar o theta de forma mais eficiente e precisa.

<p align="center">
    <img src="https://github.com/fdloopes/Praticas_Machine_Learning/blob/main/Python/Logistic_Regression/regularized/data_boundary_line.png"/><br/>
    <em>Resultado após 60 iterações</em>
</p>
