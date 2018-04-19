import pyhistdata as phd


def test_calc_samples():
    c, r = phd.calc_samples(9 * 60, 10, True)
    assert(c == 55)
    assert(r == 0)

    c, r = phd.calc_samples(9 * 60 + 5, 10, False)
    assert(c == 54)
    assert(r == 5)
