import unittest
import os
import pandas as pd
import numpy as np
import quantutils.dataset.pipeline as ppl
from tradeframework.api.core import Asset
from tradeframework.environments import SandboxEnvironment
import tradeframework.operations.utils as utils
from marketinsights.remote.models import MIModelServer

dir = os.path.dirname(os.path.abspath(__file__))


class FrameworkTest(unittest.TestCase):

    def setUp(self):
        ts = pd.read_csv(dir + '/data/testDOW.csv', parse_dates=True, index_col=0, dayfirst=True)
        #ts = ts.tz_localize("UTC")
        #ts.index = ts.index.tz_convert("US/Eastern")
        self.asset1 = Asset("DOW", ts)
        self.modelsvr = MIModelServer(secret="marketinsights-k8s-cred")

    def test_MIBasic_singleModel(self):

        TRAINING_RUN_ID = "testModel-4b8fcc0053f13d518c4056ba9e1e3cdc"
        DATASET_ID = "4234f0f1b6fcc17f6458696a6cdf5101"

        # Create portfolio
        env = SandboxEnvironment("TradeFair", "US/Eastern")
        asset = env.append(Asset("DOW"))

        p = env.setPortfolio(
            env.createDerivative("MyPortfolio", weightGenerator=env.createOptimizer("EqualWeightsOptimizer"))
            .addAsset(
                env.createDerivative(
                    "Test-MIBasicModel",
                    weightGenerator=env.createModel(
                        "MIBasicModel",
                        opts={"modelSvr": self.modelsvr, "trainingRunId": "testModel-4b8fcc0053f13d518c4056ba9e1e3cdc", "barOnly": True},
                        modelModule="tradeframework.models.remote"))
                .addAsset(asset)
            )
        )

        env.append(self.asset1)
        env.refresh()

        # Check results
        self.assertEqual(ppl.cropTime(p.assets[0].weights, start="15:00", end="16:00").values.sum(), 8.0)
        self.assertTrue(np.allclose(np.prod(utils.getPeriodReturns(p.returns) + 1), 1.001976))

    def test_MIBasic_singleModel_online(self):

        TRAINING_RUN_ID = "testModel-4b8fcc0053f13d518c4056ba9e1e3cdc"
        DATASET_ID = "4234f0f1b6fcc17f6458696a6cdf5101"

        # Create portfolio
        env = SandboxEnvironment("TradeFair", "US/Eastern")
        asset = env.append(Asset("DOW"))

        p = env.setPortfolio(
            env.createDerivative("MyPortfolio", weightGenerator=env.createOptimizer("EqualWeightsOptimizer"))
            .addAsset(
                env.createDerivative(
                    "Test-MIBasicModel",
                    weightGenerator=env.createModel(
                        "MIBasicModel",
                        opts={"window": 2, "modelSvr": self.modelsvr, "trainingRunId": "testModel-4b8fcc0053f13d518c4056ba9e1e3cdc", "barOnly": True, "debug": True},
                        modelModule="tradeframework.models.remote"))
                .addAsset(asset)
            )
        )

        # Extract 3pm indices
        # crop = ppl.cropTime(asset1.values, "15:00", "16:00")
        # idx = [asset1.values.index.get_loc(crop.index[x]) for x in range(len(crop))]
        idx = [19, 43, 67, 90, 114, 138, 162, 186]

        c = 0
        for i in idx:
            env.append(Asset("DOW", self.asset1.values[c:i]), refreshPortfolio=True)
            env.append(Asset("DOW", self.asset1.values[i:i + 1]), refreshPortfolio=True)
            c = i + 1
        env.append(Asset("DOW", self.asset1.values[c:]), refreshPortfolio=True)

        # Check results
        self.assertEqual(ppl.cropTime(p.assets[0].weights, start="15:00", end="16:00").values.sum(), 8.0)
        self.assertTrue(np.allclose(np.prod(utils.getPeriodReturns(p.returns) + 1), 1.001976))
