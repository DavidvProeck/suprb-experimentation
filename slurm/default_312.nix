# default_312.nix — Dev shell pinned to Python 3.12.11

with import (builtins.fetchGit {
  url = "https://github.com/NixOS/nixpkgs";
  rev = "8cbadfa068534bdd8238eea362d2bf0b1d46b7e8"; # contains Python 3.12.11
}) { config.allowUnfree = true; };

let
  py      = pkgs.python312;                                       # interpreter we want
  pyPkgs  = pkgs.python3Packages.override { python = py; };       # bind generic set to 3.12
  libPath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib ];
in
mkShell {
  venvDir = "./_venv";

  buildInputs =
    [
      py                   # Python 3.12.11 interpreter
      pyPkgs.venvShellHook # venv hook for *this* interpreter
      pyPkgs.pip
      pyPkgs.wheel
      pkgs.stdenv.cc.cc
    ]
    # adjust this path if your file sits elsewhere
    ++ (import ./system-dependencies.nix { inherit pkgs; });

  # Guard: if a venv exists but is NOT py312, recreate it
  shellHook = ''
    # SLURM sometimes injects PYTHONPATH; we don't want that here
    unset PYTHONPATH
    # Make /tmp safe on clusters
    export TMPDIR="${TMPDIR:-/tmp}"

    want_major=3; want_minor=12
    if [ -x "$venvDir/bin/python" ]; then
      if ! "$venvDir/bin/python" - <<'PY'
import sys
want=(3,12)
raise SystemExit(0 if sys.version_info[:2]==want else 1)
PY
      then
        echo "Recreating venv for Python $want_major.$want_minor..."
        rm -rf "$venvDir"
      fi
    fi
  '';

  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH="${libPath}:$LD_LIBRARY_PATH"
    export PYTHONPATH="$venvDir/${py.sitePackages}:$PYTHONPATH"
    echo "Shell Python: $(python -V) at $(which python)"
  '';

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    # Avoid user site & interactivity; prefer wheels
    export PYTHONNOUSERSITE=1
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_NO_INPUT=1
    # optional: only wheels; relax if something needs a build
    # export PIP_ONLY_BINARY=:all:
    echo "Venv Python: $("$venvDir"/bin/python -V)"
    pip install -r requirements.txt
  '';
}
