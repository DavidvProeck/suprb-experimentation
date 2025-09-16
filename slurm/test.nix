# shell.nix — Python 3.12.11, venv managed manually (no venvShellHook)

with import (builtins.fetchGit {
  url = "https://github.com/NixOS/nixpkgs";
  rev = "8cbadfa068534bdd8238eea362d2bf0b1d46b7e8"; # Python 3.12.11
}) { config.allowUnfree = true; };

let
  py      = pkgs.python312;
  libPath = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.zlib ];
in
pkgs.mkShell {
  # no venvShellHook here — we do it ourselves to guarantee 3.12
  buildInputs = [
    py
    pkgs.stdenv.cc.cc
    pkgs.git
  ] ++ (import ./system-dependencies.nix { inherit pkgs; });

  shellHook = ''
    set -e
    # keep temp files local to avoid permission issues on shared /tmp
    export TMPDIR="$PWD/.nix-tmp"
    mkdir -p "$TMPDIR"

    export LD_LIBRARY_PATH="${libPath}:$LD_LIBRARY_PATH"

    VENV="$PWD/_venv"

    # If venv missing OR wrong python, (re)create it with the pinned interpreter.
    if [ ! -x "$VENV/bin/python" ] || ! "$VENV/bin/python" -c 'import sys; exit(0 if sys.version.startswith("3.12.") else 1)'; then
      echo "[venv] creating fresh venv with ${py}/bin/python ..."
      rm -rf "$VENV"
      ${py}/bin/python -m venv "$VENV"
    fi

    # activate
    . "$VENV/bin/activate"

    # sanity print
    echo -n "[venv] python: " ; which python
    python -V

    # keep venv sys.path visible for nix-provided libs too
    export PYTHONPATH="$VENV/lib/python3.12/site-packages:$PYTHONPATH"

    # install requirements if changed (or first run)
    if [ -f requirements.txt ]; then
      REQ_HASH_FILE="$VENV/.req-hash"
      NEW_HASH="$(sha256sum requirements.txt | awk '{print $1}')"
      OLD_HASH=""
      [ -f "$REQ_HASH_FILE" ] && OLD_HASH="$(cat "$REQ_HASH_FILE")"

      if [ "$NEW_HASH" != "$OLD_HASH" ]; then
        echo "[venv] installing requirements.txt ..."
        # Use the activated python to ensure correct interpreter
        python -m pip install --upgrade pip wheel
        python -m pip install -r requirements.txt
        echo "$NEW_HASH" > "$REQ_HASH_FILE"
      fi
    fi
  '';
}
