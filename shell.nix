let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.matplotlib
      python-pkgs.networkx
      python-pkgs.sympy
      python-pkgs.numpy
      python-pkgs.pytest
      python-pkgs.galois
    ]))
    pkgs.vscode
  ];
}

