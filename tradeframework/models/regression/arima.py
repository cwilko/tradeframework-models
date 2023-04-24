import pandas as pd
import numpy as np
from tradeframework.api.core import Model
import statsmodels.api as sm
from typing import Literal

# from tqdm import tqdm


class ARIMA(Model):
    """Creates a predictive model based on an ARIMA timeseries model"""

    def __init__(
        self,
        env,
        AR=0,
        I=0,
        MA=0,
        window=1000,
        fit: Literal["fitAll", "fitWindow", "fitOnce", "inSample"] = "inSample",
        params=None,
        barOnly=True,
        log=True,
    ):
        Model.__init__(self, env, window)
        self.AR = AR
        self.I = I
        self.MA = MA
        self.fit = fit
        self.params = params
        self.barOnly = barOnly
        self.log = log

    def getSignals(self, window, idx=0):
        signals = pd.DataFrame(
            np.zeros((len(window), 2)),
            index=window.index,
            columns=["bar", "gap"],
        )

        predictionSignals = pd.DataFrame()

        # Convert window to log prices
        if self.log:
            window = np.log(window)

        # TODO: Option to convert window to differenced prices?

        if self.fit == "inSample" or self.params is not None:
            # For development and analysis purposes, get "predictions" of in-sample values
            model = sm.tsa.arima.ARIMA(
                window["Close"].values, order=(self.AR, self.I, self.MA)
            )

            if self.params is not None:
                # Re-use provided model paramaters
                self.result = model.filter(params=self.params)
            else:
                self.result = model.fit()

            predictions = self.result.predict(start=1, end=len(window))

            predictionSignals = np.sign(predictions - window["Close"].values)

            predictionSignals = pd.DataFrame(
                np.array([predictionSignals[:-1], predictionSignals[1:]]).T,
                index=window.index[1:],
                columns=["bar", "gap"],
            )

        elif len(window) > self.window:
            dataLen = len(window) - self.window + 1
            self.result = sm.tsa.arima.ARIMA(
                window[: self.window]["Close"].values, order=(self.AR, self.I, self.MA)
            ).fit()

            if self.fit == "fitOnce":
                # Fit to an initial window, then predict all based on fitted model.
                self.result = self.result.append(window[self.window :]["Close"].values)
                predictions = self.result.predict(start=self.window, end=len(window))

            else:
                predictions = np.array([self.result.forecast()[0]])

                # for i in tqdm(range(1, dataLen)):
                for i in range(1, dataLen):
                    if self.fit == "fitAll":
                        # Fit to an ever increasing window and predict next step

                        predictions = np.array([self.result.forecast()[0]])
                        self.result = self.result.append(
                            window[self.window + i - 1 : self.window + i][
                                "Close"
                            ].values,
                            refit=True,
                        )
                    elif self.fit == "fitWindow":
                        # Fit to a rolling window and predict next step
                        self.result = sm.tsa.arima.ARIMA(
                            window[i : i + self.window]["Close"].values,
                            order=(self.AR, self.I, self.MA),
                        ).fit()

                    predictions = np.append(predictions, self.result.forecast()[0])

            predictionSignals = np.sign(
                predictions - window[self.window - 1 :]["Close"].values
            )

            predictionSignals = pd.DataFrame(
                np.array([predictionSignals[:-1], predictionSignals[1:]]).T,
                index=window[self.window :].index,
                columns=["bar", "gap"],
            )

        if not predictionSignals.empty:
            if self.barOnly:
                predictionSignals["gap"] = 0

            signals.update(predictionSignals)

        return signals[idx:]
