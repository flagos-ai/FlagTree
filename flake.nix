{
  description = "FlagTree (Triton fork) — Ascend backend development shell";

  # NOTE: this machine's flake registry already provides `nixpkgs` as a local
  # store path, so we intentionally resolve it through the registry instead of
  # `github:NixOS/nixpkgs` (plain HTTPS to github.com is blocked here).
  inputs.nixpkgs.url = "nixpkgs";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      inherit (pkgs) lib;

      # ---------------------------------------------------------------- LLVM --
      # Triton requires a *patched* LLVM (triton-lang/llvm-project). The revision
      # is pinned in cmake/llvm-hash.txt. Ascend ships an aarch64 build of it; for
      # x86_64 the upstream Triton build server has the matching tarball.
      llvmHash = "sha256-Jx1VLRXGnpA3mnIrhrP5EhVKBfSH0Kxy+LiaSreVcHs=";
      llvm = pkgs.stdenv.mkDerivation {
        pname = "llvm-flagtree";
        version = "7d5de303";

        src = pkgs.fetchurl {
          url = "https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-7d5de303-ubuntu-x64.tar.gz";
          hash = llvmHash;
        };

        # The tarball is built on Ubuntu, so its binaries need to be re-pointed at
        # the Nix dynamic loader / library paths before they can run on NixOS.
        nativeBuildInputs = [
          pkgs.autoPatchelfHook
          pkgs.patchelf
        ];
        buildInputs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
          zstd
          libxml2
          ncurses
          ncurses5
          libedit
          openssl
          curl
          libffi
        ];

        unpackPhase = ''
          runHook preUnpack
          mkdir source
          tar -xzf "$src" -C source --strip-components=1
          runHook postUnpack
        '';
        sourceRoot = "source";

        installPhase = ''
          runHook preInstall
          mkdir -p "$out"
          cp -a . "$out"/
          # MLIR python bindings are not used by the build and their `.so`s tend
          # to drag in a specific libpython; drop them.
          rm -rf "$out"/python_packages || true
          runHook postInstall
        '';

        appendRunpaths = [ "${placeholder "out"}/lib" ];
        autoPatchelfIgnoreMissingDeps = true;
        dontStrip = true;

        meta = {
          description = "Patched LLVM/MLIR used by Triton (FlagTree)";
          platforms = [ "x86_64-linux" ];
        };
      };

      # The Ascend custom-op bitcode is compiled with `ccec`, a compiler that
      # only ships with Huawei's CANN toolkit. FlagTree only needs the bitcode
      # to *run* kernels, so when CANN is not sourced we emit empty modules to
      # let the project build. If a real `ccec` is on PATH, it is used instead.
      ccecShim = pkgs.writeShellScriptBin "ccec" ''
        out=""
        prev=""
        for a in "$@"; do
          if [ "$prev" = "-o" ]; then
            out="$a"
            break
          fi
          prev="$a"
        done

        real="$(type -P ccec 2>/dev/null || true)"
        if [ -n "$real" ] && [ "$(readlink -f "$real" 2>/dev/null)" != "$(readlink -f "$0" 2>/dev/null)" ]; then
          exec "$real" "$@"
        fi

        if [ -z "$out" ]; then
          echo "ccec shim: no -o <output> argument" >&2
          exit 1
        fi

        exec "${llvm}/bin/llvm-as" -o "$out" /dev/null
      '';

      # -------------------------------------------------------------- Python --
      python = pkgs.python311.withPackages (
        ps: with ps; [
          pip
          setuptools
          wheel
          pybind11
          lit
          numpy
          pytest
          pytest-xdist
          pytest-forked
        ]
      );

      # ------------------------------------------------------------ Tooling --
      # clang/clangd that match the LLVM version used to build the project.
      # The prebuilt x86_64 tarball does not ship a compiler (the aarch64
      # Ascend build does), so the compiler comes from nixpkgs.
      clang = pkgs.llvmPackages_22.clang;
      clangTools = pkgs.llvmPackages_22.clang-tools;
    in
    {
      packages.${system} = {
        inherit llvm;
        default = llvm;
      };

      devShells.${system}.default = pkgs.mkShell {
        name = "flagtree-ascend";

        packages = [
          python
          llvm
          pkgs.cmake
          pkgs.ninja
          pkgs.ccache
          pkgs.gnumake
          pkgs.binutils
          pkgs.gdb
          pkgs.bear
          pkgs.git
          pkgs.openssh
          clang
          clangTools
          pkgs.basedpyright
          pkgs.ruff
          pkgs.nixd
          pkgs.nil
          pkgs.nixfmt
          # Needed by MLIRConfig's find_package(ZLIB)/find_package(LibXml2).
          pkgs.zlib
          pkgs.libxml2
          pkgs.zstd
        ];

        # Compilers come from nixpkgs (they are guaranteed to run on NixOS); the
        # fork's tablegen/cmake/lib artifacts come from the `llvm` derivation.
        CC = "clang";
        CXX = "clang++";

        FLAGTREE_BACKEND = "ascend";
        LLVM_SYSPATH = "${llvm}";
        LLVM_INCLUDE_DIRS = "${llvm}/include";
        LLVM_LIBRARY_DIR = "${llvm}/lib";

        # setup.py drives CMake directly (bypassing the nixpkgs cmake hook), so
        # help find_package() locate the C library dependencies.
        CMAKE_PREFIX_PATH = lib.concatStringsSep ":" (
          map (p: "${p}") [
            (lib.getLib pkgs.zlib)
            (lib.getDev pkgs.zlib)
            (lib.getLib pkgs.libxml2)
            (lib.getDev pkgs.libxml2)
            (lib.getLib pkgs.zstd)
            (lib.getDev pkgs.zstd)
          ]
        );

        # Keep the Ascend build lean: Proton/FlagPrism are runtime profilers and
        # pull extra sources we do not need to compile the core.
        TRITON_BUILD_PROTON = "OFF";
        TRITON_BUILD_FLAGPRISM = "OFF";
        TRITON_BUILD_UT = "OFF";
        # AscendNPU-IR calls mlir_configure_python_dev_packages() only when
        # MLIR_ENABLE_BINDINGS_PYTHON is on, and that pulls in nanobind. Triton
        # uses pybind11 directly, so keep MLIR's python bindings off.
        TRITON_APPEND_CMAKE_ARGS = "-DMLIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES=ON -DMLIR_ENABLE_BINDINGS_PYTHON=OFF -DCCEC_EXECUTABLE=${ccecShim}/bin/ccec -DLLVM_LINK_EXECUTABLE=${llvm}/bin/llvm-link";
        # Avoid OOM on a 16 GiB machine; raise if you have headroom.
        MAX_JOBS = "6";

        PYTHONNOUSERSITE = "1";
        PIP_DISABLE_PIP_VERSION_CHECK = "1";

        # setup.py clones flir / AscendNPU-IR over plain HTTPS to github.com,
        # which is blocked on this host. Rewrite those URLs to the SSH alias
        # `ghz` (see ~/.ssh/config) that does work.
        GIT_CONFIG_COUNT = "1";
        GIT_CONFIG_KEY_0 = "url.git@ghz:.insteadOf";
        GIT_CONFIG_VALUE_0 = "https://github.com/";

        shellHook = ''
          # Prefer the fork's tablegen/cmake artifacts, but keep nixpkgs clang in
          # the foreground (it is what actually compiles the project).
          export PATH="${clang}/bin:${llvm}/bin:$PATH"

          # Nix's Python is read-only (PEP 668), so use a writable venv that can
          # still see the nixpkgs-provided deps (pybind11, lit, numpy, pytest).
          if [ -z "''${FLAGTREE_VENV_ACTIVE:-}" ]; then
            if [ ! -d .venv ]; then
              echo "Creating .venv (once)..."
              python -m venv --system-site-packages .venv
            fi
            # shellcheck disable=SC1091
            source .venv/bin/activate
            export FLAGTREE_VENV_ACTIVE=1
          fi

          echo "FlagTree Ascend dev shell"
          echo "  FLAGTREE_BACKEND : $FLAGTREE_BACKEND"
          echo "  LLVM_SYSPATH     : $LLVM_SYSPATH"
          echo "  clang            : $(command -v clang)"
          echo "  clangd           : $(command -v clangd)"
          echo "  python           : $(command -v python)"
          echo
          echo "Build:"
          echo "  python -m pip install -e . --no-build-isolation -v"
          echo
          echo "clangd uses the compile_commands.json symlinked into the repo root"
          echo "after the first CMake configure."
          echo
          echo "CANN is NOT required to build. Without it, the Ascend custom-op"
          echo "bitcode is stubbed out with empty LLVM modules, so kernels cannot"
          echo "actually be launched. Source your CANN set_env.sh to build them for"
          echo "real (the shim then forwards to the real ccec)."
        '';
      };
    };
}
