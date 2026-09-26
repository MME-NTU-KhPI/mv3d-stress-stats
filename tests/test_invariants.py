import numpy as np
import pytest
from mv3d_stress_stats.invariants import (
    i1,
    j2,
    j3,
    von_mises,
    principal_stresses,
    principal_1,
    principal_2,
    principal_3,
    lode_angle,
    lode_parameter,
    triaxiality,
    resolve_quantity,
)


def test_uniaxial_tension():
    # sigma_x = sigma0 > 0, all others 0
    s0 = 120.0
    stress = np.array([s0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    assert i1(stress) == pytest.approx(s0)
    # J2 = (1/3) * s0^2
    assert j2(stress) == pytest.approx((1.0 / 3.0) * s0**2)
    # von Mises = s0
    assert von_mises(stress) == pytest.approx(s0)
    
    # Principal stresses: [s0, 0, 0]
    p = principal_stresses(stress)
    np.testing.assert_allclose(p, [s0, 0.0, 0.0])
    assert principal_1(stress) == pytest.approx(s0)
    assert principal_2(stress) == pytest.approx(0.0)
    assert principal_3(stress) == pytest.approx(0.0)
    
    # Triaxiality: sigma_h / sigma_vm = (s0 / 3) / s0 = 1/3
    assert triaxiality(stress) == pytest.approx(1.0 / 3.0)
    
    # Lode angle: theta = 0 for axisymmetric tension
    assert lode_angle(stress) == pytest.approx(0.0, abs=1e-5)
    # Lode parameter: mu = -1
    assert lode_parameter(stress) == pytest.approx(-1.0, abs=1e-5)


def test_pure_shear():
    # tau_xy = tau0 > 0, all others 0
    t0 = 50.0
    stress = np.array([0.0, 0.0, 0.0, t0, 0.0, 0.0])
    
    assert i1(stress) == pytest.approx(0.0)
    # J2 = t0^2
    assert j2(stress) == pytest.approx(t0**2)
    # von Mises = sqrt(3) * t0
    assert von_mises(stress) == pytest.approx(np.sqrt(3.0) * t0)
    
    # Principal stresses: [t0, 0, -t0]
    p = principal_stresses(stress)
    np.testing.assert_allclose(p, [t0, 0.0, -t0])
    
    # Triaxiality: 0 / vm = 0
    assert triaxiality(stress) == pytest.approx(0.0)
    
    # Lode angle: theta = pi/6 for pure shear
    assert lode_angle(stress) == pytest.approx(np.pi / 6.0, abs=1e-5)
    # Lode parameter: mu = 0
    assert lode_parameter(stress) == pytest.approx(0.0, abs=1e-5)


def test_uniaxial_compression():
    # sigma_x = -s0
    s0 = 100.0
    stress = np.array([-s0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    assert von_mises(stress) == pytest.approx(s0)
    # Principal stresses: [0, 0, -s0]
    p = principal_stresses(stress)
    np.testing.assert_allclose(p, [0.0, 0.0, -s0])
    
    # Lode angle: theta = pi/3 for axisymmetric compression
    assert lode_angle(stress) == pytest.approx(np.pi / 3.0, abs=1e-5)
    # Lode parameter: mu = +1
    assert lode_parameter(stress) == pytest.approx(1.0, abs=1e-5)


def test_resolve_quantity():
    fn_vm = resolve_quantity("von_mises")
    fn_seqv = resolve_quantity("seqv")
    s = np.array([[100.0, 50.0, 20.0, 10.0, 5.0, 0.0]])
    assert fn_vm(s)[0] == fn_seqv(s)[0]
    
    # Custom lambda
    fn_custom = resolve_quantity(lambda st: st[..., 0] + 2.0 * st[..., 1])
    assert fn_custom(s)[0] == pytest.approx(200.0)
