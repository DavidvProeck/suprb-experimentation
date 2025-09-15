# shell.nix — force the whole world onto Python 3.12.11, avoid Sphinx test flakiness

with import (builtins.fetchGit {
  url = "https://github.com/NixOS/nixpkgs";
  rev = "8cbadfa068534bdd8238eea362d2bf0b1d46b7e8"; # commit with 3.12.11
}) {
  config.allowUnfree = true;

  # Make *every* reference to python3/python3Packages be 3.12-bound.
  overlays = [
    (final: prev:
      let
        py = prev.python312;
        pyPkgsBase = prev.python3Packages.override { python = py; };
      in {
        # Replace the default python3 (was 3.13 at this rev) with 3.12
        python3 = py;

        # Replace the default package set with one bound to 3.12,
        # and also disable Sphinx checks to avoid flaky builds.
        python3Packages = pyPkgsBase.overrideScope (f: p: {
          sphinx = p.sphinx.overrideAttrs (_: { doCheck = false; });
          pip    = p.pip.overrideAttrs    (_: { doCheck = false; });
        });
      })
  ];
};

let
  py     = pkgs.python3;               # now == python312
  pyPkgs = pkgs.python3Packages;       # now bound to python312
  libPath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib ];
in
pkgs.mkShell {
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
