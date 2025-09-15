# shell.nix — Python 3.12.11 dev shell

with import (builtins.fetchGit {
  url = "https://github.com/NixOS/nixpkgs";
  rev = "8cbadfa068534bdd8238eea362d2bf0b1d46b7e8"; # commit with 3.12.11
}) { config.allowUnfree = true; };

let
  py       = pkgs.python312;

  # Bind the generic python3Packages set to Python 3.12
  pyPkgsBase = pkgs.python3Packages.override { python = py; };

  # IMPORTANT: stop sphinx from running its flaky test suite
  pyPkgs = pyPkgsBase.overrideScope' (final: prev: {
    sphinx = prev.sphinx.overrideAttrs (_: { doCheck = false; });
  });

  libPath  = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib ];
in
mkShell {
  venvDir = "./_venv";

  buildInputs =
    [
      py                  # Python 3.12.11 interpreter
      pyPkgs.venvShellHook
      pyPkgs.pip
      pyPkgs.wheel
      pkgs.stdenv.cc.cc
    ]
    ++ (import ./system-dependencies.nix { inherit pkgs; });

  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH="${libPath}:$LD_LIBRARY_PATH"
    export PYTHONPATH="$venvDir/${py.sitePackages}:$PYTHONPATH"
  '';

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -r requirements.txt
  '';
}
