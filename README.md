# PYTORCH-FORECASTING-SENTIMENT

I. DEFINITIONS AND TIME SCOPE

The project was tested on 200 experiments, with 100 experiments on each of two laptops.
Experiment - forecasting of already existing prices (backtesting) + forecasting of future, unknown prices.
First laptop on old software (operating system + python) and second laptop on newer software.

The experiments were conducted randomly in the period:
from February 2024 to June 17, 2024. Most experiments were conducted in June.
Each experiment ended with a 30-minute price forecast for BTC.

If the last forecast price was higher than the last actual price, it could be concluded that the forecast market sentiment is positive and vice versa.
Only the difference between the forecast and actual prices was taken into account for calculations, without considering the commission for opening a contract.
It was not possible to open positions, because the execution time of one experiment, depending on the laptop, took up to:
20 minutes - first laptop (including backtest up to 12 min.), 15 minutes - second laptop (including backtest up to 10 min.) which made real investing impossible.

___
II. RESULTS

The first laptop showed a higher cumulative theoretical profit.
The maximum cumulative theoretical profit that could be generated during the analyzed period was 14139 USDT.

Taking into account the losses, the cumulative theoretical profit was:

first laptop: 4312 USDT – (30.1%),

second laptop: 1565 USDT – (11.07%).

The share of correct forecasts is for:

first laptop - 61%, 

second laptop – 51%

___
III. CONCLUSIONS AND ASSUMPTIONS

Presumably:

    1.  Such different results are caused by the generation of different pseudo-random numbers on each laptop.
    2.  This problem should not exist for when programmed a large number of epochs. However, the project was tested on a small number of epochs.
       

If we start to consider the results of the projects only after the end of the first period, which generated more than three consecutive incorrect forecasts, the share of accumulated profit will increase for:

first laptop - from 30.1% to 42.4%  (for 72 forecasts),

second laptop - from 11.07% to 25.6%  (for 85 forecasts).

________________________________
THIS PROJECT DOES NOT CONSTITUTE INVESTMENT ADVICE.
IF YOU DRAW INCORRECT CONCLUSIONS FROM AI-GENERATED FORECASTS, YOU CAN LOSE ALL YOUR INVESTED MONEY.
