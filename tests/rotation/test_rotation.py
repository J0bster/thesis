import pytest
from sympy import symbols
import galois
from rotation import unmask_polynomial_multi

@pytest.mark.parametrize("poly_expr, variables, x_values, expected", [
    # Linear univariate: f(x) = 3*x + 2 at x=5
    (3*symbols("x") + 2, [symbols("x")], [5], 3*5 + 2),
    # Linear bivariate: f(x, y) = 2*x + 3*y + 1 at x=4, y=7
    (2*symbols("x") + 3*symbols("y") + 1, [symbols("x"), symbols("y")], [4, 7], 2*4 + 3*7 + 1),
    # Linear trivariate: f(x, y, z) = x + y + z at x=2, y=3, z=4
    (symbols("x") + symbols("y") + symbols("z"), [symbols("x"), symbols("y"), symbols("z")], [2, 3, 4], 2 + 3 + 4),
])
def test_unmask_polynomial_multi_linear(poly_expr, variables, x_values, expected):
    p = 17
    GF = galois.GF(p)
    m = 4
    x_values_gf = [GF(v) for v in x_values]
    recovered = unmask_polynomial_multi(poly_expr, variables, x_values_gf, m, GF)
    assert int(recovered) == (expected % p)

def test_unmask_polynomial_multi_quadratic():
    p = 17
    GF = galois.GF(p)
    m = 4
    x = symbols("x")
    poly_expr = x**2 + 2
    x_val = GF(5)
    recovered = unmask_polynomial_multi(poly_expr, [x], [x_val], m, GF)
    expected = (5**2 + 2) % p
    assert int(recovered) == expected

def test_unmask_polynomial_multi_bivariate_quadratic():
    p = 17
    GF = galois.GF(p)
    m = 4
    x, y = symbols("x y")
    poly_expr = x*y + 3
    x_val = GF(4)
    y_val = GF(7)
    recovered = unmask_polynomial_multi(poly_expr, [x, y], [x_val, y_val], m, GF)
    expected = (4*7 + 3) % p
    assert int(recovered) == expected

def test_unmask_polynomial_multi_trivariate():
    p = 17
    GF = galois.GF(p)
    m = 4
    x, y, z = symbols("x y z")
    poly_expr = x*y + y*z + z*x + 1
    x_val = GF(2)
    y_val = GF(3)
    z_val = GF(4)
    recovered = unmask_polynomial_multi(poly_expr, [x, y, z], [x_val, y_val, z_val], m, GF)
    expected = (2*3 + 3*4 + 4*2 + 1) % p
    assert int(recovered) == expected

def test_unmask_polynomial_multi_power():
    p = 17
    GF = galois.GF(p)
    m = 4
    x, y = symbols("x y")
    poly_expr = x**y
    x_val = GF(2)
    y_val = GF(3)
    # Should raise ValueError because x**y is not a polynomial in y
    with pytest.raises(ValueError):
        unmask_polynomial_multi(poly_expr, [x, y], [x_val, y_val], m, GF)

def test_unmask_polynomial_multi_high_degree():
    p = 17
    GF = galois.GF(p)
    m = 4
    x = symbols("x")
    poly_expr = x**4 + 1  # degree 4 == m, should fail
    x_val = GF(2)
    with pytest.raises(ValueError):
        unmask_polynomial_multi(poly_expr, [x], [x_val], m, GF)