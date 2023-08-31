"""script to train a model on the powerload dataset."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from skops.io import dumps, loads

from powerload.model_selection import TimeSeriesCrossValidation

if TYPE_CHECKING:
    from logging import Logger
    from typing import Literal

    import numpy.typing as npt


def get_logger() -> Logger:
    """Return a logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    return logging.getLogger(__name__)


class ForecastingPipeline:
    """Pipeline for training and forecasting."""

    def __init__(
        self,
        data: pl.DataFrame,
        target: pl.DataFrame,
        *,
        scores: list[str],
        logger: Logger | None = None,
    ) -> None:
        self.logger = logger or get_logger()
        self.raw_train, self.train = data, data.to_numpy()
        self.raw_target, self.target = target, target.to_numpy().ravel()
        self.scores = scores

        self.is_fitted: bool = False

        self._pipeline_singleton: Pipeline | None = None  # type: ignore[no-any-unimported]

    # should cache this one or something, but I liked the idea to separate
    # the model instance creation from the model fitting. To be fair, this class
    # should not do model fitting - that would be responsibility for a Trainer class.
    def _get_pipeline(self) -> Pipeline:  # type: ignore[no-any-unimported]
        """Return an instance of the model, untrained."""
        if self._pipeline_singleton is None:
            categorical_cols = self.raw_train.select(pl.col(pl.Categorical)).columns
            categorical_cols_idx = [
                self.raw_train.find_idx_by_name(col) for col in categorical_cols
            ]

            # get a list of the categories
            categories = [
                self.raw_train[col].cat.get_categories().to_list()
                for col in categorical_cols
            ]

            ordinal_encoder = OrdinalEncoder(categories=categories)

            transformer = ColumnTransformer(
                transformers=[
                    ("categorical_encoder", ordinal_encoder, categorical_cols_idx)
                ],
                remainder="passthrough",
            )

            model = HistGradientBoostingRegressor(
                categorical_features=categorical_cols_idx,
                random_state=42,
            )

            self._pipeline_singleton = Pipeline(
                steps=[("preprocessing", transformer), ("gbrt", model)]
            )

        return self._pipeline_singleton

    @property
    def pipeline_(self) -> Pipeline:  # type: ignore[no-any-unimported]
        """The model artifact."""
        if not self.is_fitted:
            raise RuntimeError("The model is not fitted. Call `self.fit()` first")
        return loads(self._fitted_pipeline, trusted=True)

    def fit(self) -> None:
        """Fit the model."""
        pipeline = self._get_pipeline()

        try:
            pipeline.fit(self.train, self.target)
        except Exception:
            self.logger.exception("Model fitting failed.")
            raise

        self.residuals = self.target - pipeline.predict(self.train)

        # pretend the results are saved to S3
        self.logger.info("Save in-sample residuals to S3...")

        # model is serialised in memory, pretend it is saved to S3/artifact registry
        self.logger.info("Save model artifact to S3...")
        self._fitted_pipeline = dumps(pipeline)
        self.is_fitted = True

    def validate(
        self, window: int, fh: int, strat: Literal["rolling", "expanding"]
    ) -> None:
        """Validate the model."""

        def get_average_error(
            cv_results: dict[str, npt.NDArray]
        ) -> dict[str, tuple[float, float]]:
            """Return average and standard deviation from an array of error metrics."""
            return {
                key: ((x_ := np.abs(metrics)).mean().round(3), x_.std().round(3))
                for key, metrics in cv_results.items()
                if key.startswith("test_")
            }

        pipeline = self._get_pipeline()

        cv = TimeSeriesCrossValidation(
            train_size=window,
            forecasting_horizon=fh,
            strategy=strat,
        )

        try:
            results = cross_validate(
                estimator=pipeline,
                X=self.train,
                y=self.target,
                scoring=self.scores,
                cv=cv,
                n_jobs=-1,
                return_estimator=True,
            )
        except Exception:
            self.logger.info("Cross-validation failed.")
            raise

        # pretend the results are saved to S3
        self.logger.info("Save cross-validation scores to S3...")

        avg_scores = get_average_error(results)
        self.logger.info(avg_scores)

    def predict(self, X: pl.DataFrame) -> npt.NDArray[np.float64]:
        """Generate predictions using the fitted model."""
        X_ = X.to_numpy()

        # pretend the results are saved to S3
        self.logger.info("Log training to S3...")

        try:
            predictions: npt.NDArray[np.floating] = self.pipeline_.predict(X_)
        except Exception:
            self.logger.exception("Model inference failed.")
            raise

        # pretend the results are saved to S3
        self.logger.info("Log predictions to S3...")

        return predictions
