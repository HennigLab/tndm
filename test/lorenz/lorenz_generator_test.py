import numpy as np
import pytest

from tndm.lorenz.initial_conditions import uniform
from tndm.lorenz import LorenzGenerator, bo


@pytest.mark.unit
def test_lorenz_generate_latent():
    g = LorenzGenerator()
    t, z = g.generate_latent()

    np.testing.assert_equal(t.shape, (np.ceil(1 / 0.006),))
    np.testing.assert_equal(z.shape, (np.ceil(1 / 0.006), 3))
    np.testing.assert_almost_equal(t[0], 0)
    np.testing.assert_almost_equal(t[-1], 0.996)


@pytest.mark.unit
def test_lorenz_generate_rates():
    g = LorenzGenerator()
    t, f, w, z = g.generate_rates(n=5, trials=2)

    np.testing.assert_equal(t.shape, (np.ceil(1 / 0.006),))
    np.testing.assert_equal(f.shape, (2, np.ceil(1 / 0.006), 5))
    np.testing.assert_equal(w.shape, (3, 5))
    np.testing.assert_equal(z.shape, (2, np.ceil(1 / 0.006), 3))
    np.testing.assert_almost_equal(t[0], 0)
    np.testing.assert_almost_equal(t[-1], 0.996)


@pytest.mark.unit
def test_lorenz_generate_spikes():
    g = LorenzGenerator()
    t, s, f, w, z = g.generate_spikes(n=2, trials=5, conditions=10)

    np.testing.assert_equal(t.shape, (np.ceil(1 / 0.006),))
    np.testing.assert_equal(s.shape, (10, 5, np.ceil(1 / 0.006), 2))
    np.testing.assert_equal(f.shape, (10, 5, np.ceil(1 / 0.006), 2))
    np.testing.assert_equal(w.shape, (10, 3, 2))
    np.testing.assert_equal(z.shape, (10, 5, np.ceil(1 / 0.006), 3))
    np.testing.assert_almost_equal(t[0], 0)
    np.testing.assert_almost_equal(t[-1], 0.996)


@pytest.mark.unit
def test_lorenz_generate_spikes_and_behaviour():
    g = LorenzGenerator()
    t, b, s, f, bw, w, z = g.generate_spikes_and_behaviour(
        n=2, trials=5, conditions=10, l=1, b=2, y=6)

    np.testing.assert_equal(t.shape, (np.ceil(1 / 0.006),))
    np.testing.assert_equal(b.shape, (10, 5, np.ceil(1 / 0.006), 6))
    np.testing.assert_equal(s.shape, (10, 5, np.ceil(1 / 0.006), 2))
    np.testing.assert_equal(f.shape, (10, 5, np.ceil(1 / 0.006), 2))
    np.testing.assert_equal(bw.shape, (10, 2, 6))
    np.testing.assert_equal(w.shape, (10, 1, 2))
    np.testing.assert_equal(z.shape, (10, 5, np.ceil(1 / 0.006), 3))
    np.testing.assert_almost_equal(t[0], 0)
    np.testing.assert_almost_equal(t[-1], 0.996)


@pytest.mark.unit
def test_lorenz_generate_spikes_and_behaviour_ones():
    g = LorenzGenerator()
    t, b, s, f, bw, w, z = g.generate_spikes_and_behaviour(
        n=1, trials=1, conditions=1, l=1, b=1, y=1)

    np.testing.assert_equal(t.shape, (np.ceil(1 / 0.006),))
    np.testing.assert_equal(b.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(s.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(f.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(bw.shape, (1, 1, 1))
    np.testing.assert_equal(w.shape, (1, 1, 1))
    np.testing.assert_equal(z.shape, (1, 1, np.ceil(1 / 0.006), 3))
    np.testing.assert_almost_equal(t[0], 0)
    np.testing.assert_almost_equal(t[-1], 0.996)


@pytest.mark.unit
def test_lorenz_generate_spikes_with_warmup():
    g = LorenzGenerator()
    t, b, s, f, bw, w, z = g.generate_spikes_and_behaviour(
        n=1, trials=1, conditions=1, l=1, b=1, y=1, warmup=1000)

    np.testing.assert_equal(t.shape, (np.ceil(1 / 0.006),))
    np.testing.assert_equal(b.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(s.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(f.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(bw.shape, (1, 1, 1))
    np.testing.assert_equal(w.shape, (1, 1, 1))
    np.testing.assert_equal(z.shape, (1, 1, np.ceil(1 / 0.006), 3))
    np.testing.assert_almost_equal(t[0], 0)
    np.testing.assert_almost_equal(t[-1], 0.996)


@pytest.mark.unit
def test_lorenz_generate_spikes_with_overlay():
    g = LorenzGenerator()
    t, b, s, f, bw, w, z = g.generate_spikes_and_behaviour(
        n=1, trials=1, conditions=1, l=1, b=1, y=1, warmup=1000, behaviour_overlay=bo.sine_overlay)

    np.testing.assert_equal(t.shape, (np.ceil(1 / 0.006),))
    np.testing.assert_equal(b.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(s.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(f.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(bw.shape, (1, 1, 1))
    np.testing.assert_equal(w.shape, (1, 1, 1))
    np.testing.assert_equal(z.shape, (1, 1, np.ceil(1 / 0.006), 3))
    np.testing.assert_almost_equal(t[0], 0)
    np.testing.assert_almost_equal(t[-1], 0.996)


@pytest.mark.unit
def test_lorenz_generate_spikes_with_initial_condition():
    g = LorenzGenerator()
    t, b, s, f, bw, w, z = g.generate_spikes_and_behaviour(
        n=1, trials=1, conditions=1, l=1, b=1, y=1, initial_conditions=uniform(
            low=-80, high=80))

    np.testing.assert_equal(t.shape, (np.ceil(1 / 0.006),))
    np.testing.assert_equal(b.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(s.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(f.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(bw.shape, (1, 1, 1))
    np.testing.assert_equal(w.shape, (1, 1, 1))
    np.testing.assert_equal(z.shape, (1, 1, np.ceil(1 / 0.006), 3))
    np.testing.assert_almost_equal(t[0], 0)
    np.testing.assert_almost_equal(t[-1], 0.996)


@pytest.mark.unit
def test_lorenz_generate_spikes_with_seed():
    g = LorenzGenerator()
    t, b, s, f, bw, w, z = g.generate_spikes_and_behaviour(n=1, trials=1, conditions=1, l=1, b=1, y=1,
                                                           initial_conditions=uniform(low=-80, high=80), seed=1234, behaviour_overlay=bo.sine_overlay)

    np.testing.assert_equal(t.shape, (np.ceil(1 / 0.006),))
    np.testing.assert_equal(b.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(s.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(f.shape, (1, 1, np.ceil(1 / 0.006), 1))
    np.testing.assert_equal(bw.shape, (1, 1, 1))
    np.testing.assert_equal(w.shape, (1, 1, 1))
    np.testing.assert_equal(z.shape, (1, 1, np.ceil(1 / 0.006), 3))
    np.testing.assert_almost_equal(t[0], 0)
    np.testing.assert_almost_equal(t[-1], 0.996)
