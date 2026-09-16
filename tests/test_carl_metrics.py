import numpy as np
import pytest
from carl_metrics import logged_advantages, policy_alignment


def test_success_failure_and_zero_is_not_positive():
    q = np.ones((4,4,10))
    l = np.repeat(np.array([2.,1.,0.,1.5])[:,None],10,axis=1)
    r = policy_alignment(q,l,[1,1,0,0])
    assert r['success_logged_advantage_mean'] == .5
    assert r['failure_logged_advantage_mean'] == -.25
    assert r['success_failure_advantage_separation'] == .75
    assert r['success_logged_advantage_positive_fraction'] == .5
    assert r['failure_logged_advantage_positive_fraction'] == .5


def test_raw_head_mean_not_min_and_same_state_baseline():
    q = np.arange(80).reshape(2,4,10)
    l = np.arange(20).reshape(2,10)
    np.testing.assert_allclose(logged_advantages(q,l),[-15,-45])


def test_missing_group_remains_undefined():
    r = policy_alignment(np.zeros((1,4,10)),np.ones((1,10)),[1])
    assert r['failure_state_count'] == 0
    assert np.isnan(r['failure_logged_advantage_mean'])
    assert np.isnan(r['success_failure_advantage_separation'])


@pytest.mark.parametrize('labels', [[2], [np.nan], [1,0]])
def test_bad_labels(labels):
    with pytest.raises(ValueError):policy_alignment(np.zeros((1,4,10)),np.zeros((1,10)),labels)


def test_bad_shape_or_nonfinite():
    with pytest.raises(ValueError):logged_advantages(np.zeros((1,4,10)),np.zeros((10,1)))
    with pytest.raises(ValueError):logged_advantages(np.full((1,4,10),np.nan),np.zeros((1,10)))
