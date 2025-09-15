# shell.nix — Python 3.12.11
with import (builtins.fetchGit {
  url = "https://github.com/NixOS/nixpkgs";
  rev = "8cbadfa068534bdd8238eea362d2bf0b1d46b7e8"; # commit with 3.12.11
}) { config.allowUnfree = true; };

let
  py     = pkgs.python312;
  pyPkgs = pkgs.python3Packages.override { python = py; }; # bind generic set to 3.12
in
mkShell {
  venvDir = "./_venv";

  buildInputs =
    [
      py                   # interpreter (3.12.11)
      pyPkgs.venvShellHook # venv hook for this interpreter
      pyPkgs.pip
      pyPkgs.wheel
      pkgs.stdenv.cc.cc
    ]
    ++ (import ./system-dependencies.nix { inherit pkgs; });

  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ ${pkgs.stdenv.cc.cc} ${pkgs.zlib} ]}:$LD_LIBRARY_PATH"
    export PYTHONPATH=$venvDir/${py.sitePackages}:$PYTHONPATH
  '';

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -r requirements.txt
  '';
}
