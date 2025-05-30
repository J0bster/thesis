import numpy as np
from sympy import symbols, Integer
import galois
import random

def unmask_polynomial_multi(poly_expr, variables, x_values, m, field):
    # Check degree in each variable
    for var in variables:
        poly = poly_expr.as_poly(var)
        if poly is None:
            raise ValueError(f"Expression is not a polynomial in variable {var}.")
        deg = poly.degree()
        if deg >= m:
            raise ValueError(f"Degree of variable {var} is {deg}, which is not less than m={m}")

    constants = {c for c in poly_expr.atoms(Integer)}
    poly_expr_field = poly_expr
    for c in constants:
        poly_expr_field = poly_expr_field.subs(c, field(int(c)))
    for c in poly_expr_field.atoms(int):
        poly_expr_field = poly_expr_field.subs(c, field(c))

    def poly_func(vals):
        subs_dict = {var: val for var, val in zip(variables, vals)}
        return poly_expr_field.subs(subs_dict)

    omega = None
    for g in field.elements[1:]:
        if pow(g, m) == 1 and all(pow(g, k) != 1 for k in range(1, m)):
            omega = g
            print(f"Found primitive root of unity: {omega}")
            break
    assert omega is not None, "No primitive root of unity found"

    taus = [field(random.randint(0, field.order - 1)) for _ in variables]
    print("Random taus:", taus)
    results = []
    for j in range(m):
        print(f" omega^{j} = {omega}^{j} = {pow(omega, j)}")
        rotated = [tau * omega**j + x for tau, x in zip(taus, x_values)]
        eval_j = field(int(poly_func(rotated)) % field.order)
        results.append(eval_j)
        print(f"j={j}: rotated={rotated}, f(rotated)={eval_j}")

    recovered = sum(results, field(0)) / field(m)
    print("Recovered value:", recovered)
    print("Expected value:", field(int(poly_func(x_values)) % field.order))
    return recovered

if __name__ == "__main__":
    p = 17
    GF = galois.GF(p)
    m = 4
    x = symbols("x")
    poly_expr = x**2 + 2
    x_val = GF(5)
    recovered = unmask_polynomial_multi(poly_expr, [x], [x_val], m, GF)
    expected = (5**2 + 2) % p
    assert int(recovered) == expected