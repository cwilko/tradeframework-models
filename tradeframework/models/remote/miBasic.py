import pandas as pd
import numpy as np
import datetime
import quantutils.dataset.ml as mlutils
from tradeframework.api.core import Model
from marketinsights.remote.ml import MIAssembly


class MIBasicModel(Model):
    def __init__(
        self,
        env,
        modelSvr,
        trainingRunId,
        threshold=0,
        barOnly=False,
        credstore=None,
        secret=None,
        window=0,
        debug=False,
    ):
        Model.__init__(self, env, window)

        self.assembly = MIAssembly(
            modelSvr=modelSvr, credentials_store=credstore, secret=secret
        )
        self.trainingRunId = trainingRunId
        self.threshold = threshold
        self.debug = debug
        self.barOnly = barOnly

    # Future
    # Get dataset description, extract feature def and label def
    # Send all data through pipeline and retrieve dataset
    # Retrieve dataset index
    # Add label time entries(s) to index
    # load results into time entries
    #
    # Now - all pipelines should return data with time index of prediction labels.
    # Send all data through pipeline and retrieve dataset
    # Retrieve dataset index and use it for signals
    def getSignals(self, window, idx=0):

        # Obtain the signals for the next n steps from the Market Insights API
        signals = pd.DataFrame(
            np.zeros((len(window), 2)), index=window.index, columns=["bar", "gap"]
        )
        predictions = self.assembly.get_predictions_with_raw_data(
            window, self.trainingRunId, debug=self.debug
        )
        if not predictions.empty:
            predictionSignals = mlutils.toTradeSignals(
                mlutils.onehot(predictions[["y_pred0"]].values), self.threshold
            )
            predictionSignals = pd.DataFrame(
                np.array([predictionSignals, predictionSignals]).T,
                index=predictions.index,
                columns=["bar", "gap"],
            )
            if self.barOnly:
                predictionSignals["gap"] = 0

            if predictions is not None:
                signals.update(predictionSignals)

        return signals[idx:]
