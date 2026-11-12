"""Spatial-transfer protocol: runs end to end, scores only the requested years."""
import numpy as np

from carbon_world_model.eval.configs import make_config
from carbon_world_model.eval.transfer import spatial_transfer


def test_transfer_scores_only_up_to_end_year(panel):
    cfg = make_config("orig", steps=2)
    tr = spatial_transfer(panel, cfg, cutoff=2009, end_year=2012, adapt_steps=2, verbose=False)
    assert tr["holdout_years"] == [2010, 2011, 2012]
    assert len(tr["test_countries"]) == 12 and set(tr["per_country"]) == set(tr["test_countries"])
    for key in ("world_model", "trend_kaya"):
        assert all(np.isfinite(v) for v in tr[key].values())
    assert np.isfinite(tr["train_insample_mape"]) and np.isfinite(tr["adapt_insample_mape"])
