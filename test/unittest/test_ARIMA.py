import unittest
import os
import pandas as pd
import numpy as np
from tradeframework.api.core import Asset
from tradeframework.environments import SandboxEnvironment
import tradeframework.operations.utils as utils

dir = os.path.dirname(os.path.abspath(__file__))


class FrameworkTest(unittest.TestCase):
    def setUp(self):
        ts = pd.read_csv(
            dir + "/data/testDOW.csv", parse_dates=True, index_col=0, dayfirst=True
        )
        # ts = ts.tz_localize("UTC")
        # ts.index = ts.index.tz_convert("US/Eastern")
        self.asset1 = Asset("DOW", ts)

    def test_ARIMA_singleModel_inSample(self):
        # Create portfolio
        env = SandboxEnvironment("TradeFair", "US/Eastern")
        asset = env.append(Asset("DOW"))

        p = env.setPortfolio(
            env.createDerivative(
                "MyPortfolio",
                weightGenerator=env.createOptimizer("EqualWeightsOptimizer"),
            ).addAsset(
                env.createDerivative(
                    "Test-ARIMA",
                    weightGenerator=env.createModel(
                        "ARIMA",
                        opts={
                            "AR": 3,
                            "I": 1,
                            "MA": 2,
                            "window": 50,
                            "fit": "inSample",
                            "barOnly": False,
                        },
                        modelModule="tradeframework.models.regression",
                    ),
                ).addAsset(asset)
            )
        )

        env.append(self.asset1)
        env.refresh()

        print(np.prod(utils.getPeriodReturns(p.returns) + 1))

        # Check results
        self.assertTrue(
            np.allclose(np.prod(utils.getPeriodReturns(p.returns) + 1), 1.050955)
        )

    def test_ARIMA_singleModel_fitOnce(self):
        # Create portfolio
        env = SandboxEnvironment("TradeFair", "US/Eastern")
        asset = env.append(Asset("DOW"))

        p = env.setPortfolio(
            env.createDerivative(
                "MyPortfolio",
                weightGenerator=env.createOptimizer("EqualWeightsOptimizer"),
            ).addAsset(
                env.createDerivative(
                    "Test-ARIMA",
                    weightGenerator=env.createModel(
                        "ARIMA",
                        opts={
                            "AR": 3,
                            "I": 1,
                            "MA": 2,
                            "window": 50,
                            "fit": "fitOnce",
                            "barOnly": False,
                        },
                        modelModule="tradeframework.models.regression",
                    ),
                ).addAsset(asset)
            )
        )

        env.append(self.asset1)
        env.refresh()

        print(np.prod(utils.getPeriodReturns(p.returns) + 1))

        # Check results
        self.assertTrue(
            np.allclose(np.prod(utils.getPeriodReturns(p.returns) + 1), 1.014392)
        )

    def test_ARIMA_singleModel_fitWindow(self):
        # Create portfolio
        env = SandboxEnvironment("TradeFair", "US/Eastern")
        asset = env.append(Asset("DOW"))

        p = env.setPortfolio(
            env.createDerivative(
                "MyPortfolio",
                weightGenerator=env.createOptimizer("EqualWeightsOptimizer"),
            ).addAsset(
                env.createDerivative(
                    "Test-ARIMA",
                    weightGenerator=env.createModel(
                        "ARIMA",
                        opts={
                            "AR": 3,
                            "I": 1,
                            "MA": 2,
                            "window": 180,
                            "fit": "fitWindow",
                            "barOnly": False,
                        },
                        modelModule="tradeframework.models.regression",
                    ),
                ).addAsset(asset)
            )
        )

        env.append(self.asset1)
        env.refresh()

        print(np.prod(utils.getPeriodReturns(p.returns) + 1))

        # Check results
        self.assertTrue(
            np.allclose(np.prod(utils.getPeriodReturns(p.returns) + 1), 1.000102)
        )

    def test_arima_singleModel_online(self):
        env = SandboxEnvironment("TradeFair", "US/Eastern")
        asset = env.append(Asset("DOW"))

        p = env.setPortfolio(
            env.createDerivative(
                "MyPortfolio",
                weightGenerator=env.createOptimizer("EqualWeightsOptimizer"),
            ).addAsset(
                env.createDerivative(
                    "Test-ARIMA",
                    weightGenerator=env.createModel(
                        "ARIMA",
                        opts={
                            "AR": 3,
                            "I": 1,
                            "MA": 2,
                            "window": 187,
                            "fit": "fitWindow",
                            "log": False,
                            "barOnly": False,
                        },
                        modelModule="tradeframework.models.regression",
                    ),
                ).addAsset(asset)
            )
        )

        for i in range(len(self.asset1.values)):
            env.append(
                Asset("DOW", self.asset1.values[i : i + 1]), refreshPortfolio=True
            )

        print(np.prod(utils.getPeriodReturns(p.returns) + 1))

        # Check results
        self.assertTrue(
            np.allclose(np.prod(utils.getPeriodReturns(p.returns) + 1), 1.000783)
        )

    def test_arima_singleModel_online_partials(self):
        env = SandboxEnvironment("TradeFair", "US/Eastern")
        asset = env.append(Asset("DOW"))

        p = env.setPortfolio(
            env.createDerivative(
                "MyPortfolio",
                weightGenerator=env.createOptimizer("EqualWeightsOptimizer"),
            ).addAsset(
                env.createDerivative(
                    "Test-ARIMA",
                    weightGenerator=env.createModel(
                        "ARIMA",
                        opts={
                            "AR": 3,
                            "I": 1,
                            "MA": 2,
                            "window": 187,
                            "fit": "fitWindow",
                            "log": False,
                            "barOnly": False,
                        },
                        modelModule="tradeframework.models.regression",
                    ),
                ).addAsset(asset)
            )
        )

        for i in range(len(self.asset1.values)):
            slice = self.asset1.values[i : i + 1].copy()
            slice["Close"] = np.nan
            env.append(Asset("DOW", slice), refreshPortfolio=True)
            env.append(
                Asset("DOW", self.asset1.values[i : i + 1]), refreshPortfolio=True
            )

        print(np.prod(utils.getPeriodReturns(p.returns) + 1))

        # Check results
        self.assertTrue(
            np.allclose(np.prod(utils.getPeriodReturns(p.returns) + 1), 1.000783)
        )
