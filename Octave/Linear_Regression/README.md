<h1>Linear Regression</h1>

- Possui, atualmente, duas implementações em octave de regressão linear.
- Implementações para uma feature e para multiplas features.
> Implementações realizadas durante o curso de Aprendizado de Máquina, oferecido pela Stanford University e disponível na plataforma Coursera.

<h2>One Feature</h2>

- Implementação de um modelo em um conjunto de eficiência de food truck por cidade, para poder prever onde será mais lucrativo abrir um novo food truck.
- O dataset é composto por: `X = população da cidade em 10.000` e `y = lucro obtido na cidade em 10.000`.
- Possui as implementações das funções de custo e gradiente descedente.

<p align="center">
    <img src="https://github.com/fdloopes/Praticas_Machine_Learning/blob/main/Octave/Linear_Regression/one_feature/plot.png"/><br/>
    <em>Resultado aplicado aos dados</em>
</p>

<h2>Multi Features</h2>

- Implementação de um modelo em um conjunto de preços de imóveis por cidade, de forma a prever quanto ficará o valor de cada imóvel de acordo com tamanho e número de quartos.
- O dataset é composto por: `X(1) = tamano da casa em metros quadrados`, `X(2) = número de quartos` e `y = preço das casas`.
- Possui as implementações das funções:`cost`, `gradient descent`, `feature normalize` e `normal equation`.
- Por o conjunto de dados possuir dados em uma ordem de grandeza bem diferente foi necessário aplicar a normalização.
- A implementação da `normal equation` foi a titulo de comparação, pois ela obtém o mesmo resultado, sem necessidade de aplicar normalização aos dados e de forma mais eficiente.
> Obs: O uso da normal equation é válido apenas para conjuntos de dados que possuam poucas features, pois conforme o número aumenta a eficiência começa a cair, com n >= 1000 começa a perde eficiência. 

<p align="center">
    <img src="https://github.com/fdloopes/Praticas_Machine_Learning/blob/main/Octave/Linear_Regression/multi_features/plot.png"/><br/>
    <em>Resultado descida de gradiente após 50 iterações</em>
</p>
