# nix script for python 3.12.11

with import (builtins.fetchGit {
  url = "https://github.com/NixOS/nixpkgs/";
  rev = "8cbadfa068534bdd8238eea362d2bf0b1d46b7e8";  # nixpkgs commit with Python 3.12.11
}) { config = { allowUnfree = true; }; };

let
  py     = pkgs.python312;
  pyPkgs = pkgs.python3Packages.override { python = py; };  # generic set bound to 3.12
in
mkShell {
  venvDir = "./_venv";

  buildInputs =
    [ py pyPkgs.venvShellHook pyPkgs.wheel pyPkgs.pip ]
    ++ [ pkgs.stdenv.cc.cc ]
    ++ (import ./system-dependencies.nix { inherit pkgs; });

  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]}:${pkgs.lib.makeLibraryPath [ pkgs.zlib ]}:$LD_LIBRARY_PATH"
    export PYTHONPATH=$venvDir/${py.sitePackages}:$PYTHONPATH
  '';

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -r requirements.txt
  '';
}
