#include "catlass/catlass.hpp"
#include "../common.h"

#include <cstdint>

extern "C" {
__aicore__ __attribute__((always_inline)) void get_buf_m(uint8_t bufid)
{
    __builtin_cce_get_buf(PIPE_M, bufid, false);
}

__aicore__ __attribute__((always_inline)) void rls_buf_m(uint8_t bufid)
{
    __builtin_cce_rls_buf(PIPE_M, bufid, false);
}

__aicore__ __attribute__((always_inline)) void get_buf_mte1(uint8_t bufid)
{
    __builtin_cce_get_buf(PIPE_MTE1, bufid, false);
}

__aicore__ __attribute__((always_inline)) void rls_buf_mte1(uint8_t bufid)
{
    __builtin_cce_rls_buf(PIPE_MTE1, bufid, false);
}

__aicore__ __attribute__((always_inline)) void get_buf_mte2(uint8_t bufid)
{
    __builtin_cce_get_buf(PIPE_MTE2, bufid, false);
}

__aicore__ __attribute__((always_inline)) void rls_buf_mte2(uint8_t bufid)
{
    __builtin_cce_rls_buf(PIPE_MTE2, bufid, false);
}

__aicore__ __attribute__((always_inline)) void get_buf_mte3(uint8_t bufid)
{
    __builtin_cce_get_buf(PIPE_MTE3, bufid, false);
}

__aicore__ __attribute__((always_inline)) void rls_buf_mte3(uint8_t bufid)
{
    __builtin_cce_rls_buf(PIPE_MTE3, bufid, false);
}

__aicore__ __attribute__((always_inline)) void get_buf_fix(uint8_t bufid)
{
    __builtin_cce_get_buf(PIPE_FIX, bufid, false);
}

__aicore__ __attribute__((always_inline)) void rls_buf_fix(uint8_t bufid)
{
    __builtin_cce_rls_buf(PIPE_FIX, bufid, false);
}
}
