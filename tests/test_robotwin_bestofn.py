import numpy as np
import pytest
from robotwin_bestofn import candidate_noise,choose_qmean

def test_candidate_zero_is_original_base_noise():
    expected=np.random.default_rng(np.random.SeedSequence([123,7])).standard_normal((50,32),dtype=np.float32)
    np.testing.assert_array_equal(candidate_noise(123,7,0,50,32),expected)

def test_streams_are_distinct_and_repeatable():
    a=[candidate_noise(123,7,k,50,32) for k in range(4)]
    for k in range(4):np.testing.assert_array_equal(a[k],candidate_noise(123,7,k,50,32))
    assert all(not np.array_equal(a[i],a[j]) for i in range(4) for j in range(i))

def test_mean_not_min_and_ties():
    q=np.zeros((4,10));q[0,0]=20;q[1]=1
    assert choose_qmean(q)==0
    assert choose_qmean(np.ones((4,10)))==0
    assert choose_qmean(np.ones((1,10)))==0

def test_nonfinite_rejected():
    with pytest.raises(ValueError):choose_qmean(np.full((4,10),np.nan))
