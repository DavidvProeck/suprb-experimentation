# nix script for python 3.12.11

with import (builtins.fetchGit {
  url = "https://github.com/NixOS/nixpkgs/";
  rev = "8cbadfa068534bdd8238eea362d2bf0b1d46b7e8";  # nixpkgs commit with Python 3.12.11
}) { config = { allowUnfree = true; }; };

mkShell {
  venvDir = "./_venv";

  buildInputs = (with pkgs.python312Packages; [ python312 venvShellHook wheel pip ]) ++ [
    pkgs.stdenv.cc.cc
  ] ++ (import ./system-dependencies.nix { inherit pkgs; });

  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc]}:${pkgs.lib.makeLibraryPath [pkgs.zlib]}:$LD_LIBRARY_PATH"
    export PYTHONPATH=$venvDir/${python312.sitePackages}:$PYTHONPATH
  '';

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -r requirements.txt
  '';
}
