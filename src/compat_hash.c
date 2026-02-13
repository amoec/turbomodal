/*
 * macOS libc++ compatibility shim.
 *
 * Apple Clang 16+ (Xcode 16 / macOS 15) libc++ headers emit calls to
 * std::__1::__hash_memory, an out-of-line function added in LLVM 18.
 * Conda environments ship an older libc++.dylib that lacks this symbol,
 * causing ImportError at runtime.
 *
 * The C function name _ZNSt3__113__hash_memoryEPKvm gets the standard
 * Mach-O underscore prefix, producing symbol __ZNSt3__113__hash_memoryEPKvm
 * — exactly the one the dynamic linker is looking for.
 *
 * On systems whose libc++ already provides the symbol, the static linker
 * binds our definition into _core.so; at runtime two-level namespaces
 * keep each image using its own copy, so there is no conflict.
 */
#ifdef __APPLE__
#include <stddef.h>

size_t _ZNSt3__113__hash_memoryEPKvm(const void* data, size_t len) {
    /* FNV-1a (64-bit) — fast, reasonable distribution */
    size_t h = 0xcbf29ce484222325ULL;
    const unsigned char* p = (const unsigned char*)data;
    size_t i;
    for (i = 0; i < len; ++i) {
        h ^= (size_t)p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}
#endif
