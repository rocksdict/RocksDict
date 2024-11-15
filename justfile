clean:
    cargo clean

# for macos
develop:
    # conda deactivate
    LIBCLANG_PATH=${HOMEBREW_PREFIX}/opt/llvm/lib \
    CC=${HOMEBREW_PREFIX}/opt/llvm/bin/clang \
    CXX=${HOMEBREW_PREFIX}/opt/llvm/bin/clang++ \
    AR=${HOMEBREW_PREFIX}/opt/llvm/bin/llvm-ar \
    CFLAGS="-flto=thin -O3" \
    CXXFLAGS="-flto=thin -O3" \
    RUSTFLAGS="-Clinker-plugin-lto -Clinker=$PWD/macos-linker.sh -Clink-arg=-fuse-ld=${HOMEBREW_PREFIX}/opt/lld/bin/ld64.lld" \
    maturin develop --release --verbose

# for macos
build:
    # conda deactivate
    LIBCLANG_PATH=${HOMEBREW_PREFIX}/opt/llvm/lib \
    CC=${HOMEBREW_PREFIX}/opt/llvm/bin/clang \
    CXX=${HOMEBREW_PREFIX}/opt/llvm/bin/clang++ \
    AR=${HOMEBREW_PREFIX}/opt/llvm/bin/llvm-ar \
    CFLAGS="-flto=thin -O3" \
    CXXFLAGS="-flto=thin -O3" \
    RUSTFLAGS="-Clinker-plugin-lto -Clinker=$PWD/macos-linker.sh -Clink-arg=-fuse-ld=${HOMEBREW_PREFIX}/opt/lld/bin/ld64.lld" \
    maturin build --release --verbose

bin:
    LIBCLANG_PATH=${HOMEBREW_PREFIX}/opt/llvm/lib \
    CC=${HOMEBREW_PREFIX}/opt/llvm/bin/clang \
    CXX=${HOMEBREW_PREFIX}/opt/llvm/bin/clang++ \
    AR=${HOMEBREW_PREFIX}/opt/llvm/bin/llvm-ar \
    CFLAGS="-flto=thin -O3" \
    CXXFLAGS="-flto=thin -O3" \
    RUSTFLAGS="-Clinker-plugin-lto -Clinker=$PWD/macos-linker.sh -Clink-arg=-fuse-ld=${HOMEBREW_PREFIX}/opt/lld/bin/ld64.lld" \
    cargo build --release --bin create_cf_db
