import pytest
import torch
import numpy as np
import torch_npu
import torch_catlass


from common import only_on_2201


def _generate_sparse_data(m, n, k):
    """Generate 4:2 sparse matmul test data matching examples/41_sparse_matmul_tla/sparse_gen_data.py.

    B is generated as a (n, k) matrix where each group of 4 consecutive elements
    along K has exactly 2 non-zero values. It is then compressed into:
      - b_gm: dense (n, k//2) — just the non-zero values
      - index_matrix ND: (n, k//8) — 1 byte per 4 indices, each 2 bits
      - index_matrix NZ: blocked format, fractal (16,8)

    The golden is computed as C[row] = sum(A[:, selected_cols] @ b_gm[row])
    where selected_cols are reconstructed from the index mask.

    Returns:
        a_gm: int8 (m, k) — left input
        b_gm: int8 (k//2, n), ColumnMajor stored — sparse right input
        index_gm: uint8 flat — index in NZ format
        golden: int32 (m, n) — reference result
    """
    np.random.seed(42)

    # A
    a_gm = np.random.randint(-10, 10, (m, k), dtype=np.int8)

    # B sparse with 2 non-zeros per 4-group
    b_sparse = np.zeros((n, k), dtype=np.int8)
    for row in range(n):
        for i in range(0, k, 4):
            pos = np.random.choice(4, 2, replace=False)
            b_sparse[row, i + pos[0]] = np.random.randint(1, 10, dtype=np.int8)
            b_sparse[row, i + pos[1]] = np.random.randint(1, 10, dtype=np.int8)

    # Densify and generate index (ND format)
    dense_b = np.zeros((n, k // 2), dtype=np.int8)
    index_nd = np.zeros((n, k // 8), dtype=np.uint8)
    index_mask = np.zeros((n, k // 2), dtype=np.int32)

    for row in range(n):
        dense_row = []
        index_row = []
        index_mask_row = []

        for i in range(0, k, 4):
            block = b_sparse[row, i:i + 4]
            nonzero_positions = [j for j in range(4) if block[j] != 0]
            if len(nonzero_positions) == 0:
                idx1, idx2 = 0, 0
                index_mask_row.extend([i, i])
            elif len(nonzero_positions) == 1:
                idx1 = nonzero_positions[0] if nonzero_positions[0] < 3 else 0
                idx2 = 0 if nonzero_positions[0] < 3 else 2
                index_mask_row.extend([nonzero_positions[0] + i, i])
            else:
                idx1 = nonzero_positions[0]
                idx2 = nonzero_positions[1] - 1
                index_mask_row.extend([nonzero_positions[0] + i, nonzero_positions[1] + i])

            dense_block = [block[pos] for pos in nonzero_positions[:2]]
            if len(dense_block) < 2:
                dense_block += [0] * (2 - len(dense_block))
            dense_row.extend(dense_block)
            index_row.extend([idx1, idx2])

        # Pack 4 indices (each 2 bits) into 1 byte
        index_bytes = []
        for j in range(0, len(index_row), 4):
            indices = index_row[j:j + 4]
            byte_val = sum((idx << (2 * bit_pos)) for bit_pos, idx in enumerate(indices))
            index_bytes.append(byte_val)

        dense_b[row, :] = dense_row
        index_nd[row, :] = index_bytes
        index_mask[row, :] = index_mask_row

    # Convert index from ND to NZ (fractal 16,8)
    ceil_n = int(np.ceil(n / 16) * 16)
    ceil_k_idx = int(np.ceil(k / 8) / 8 * 8)  # index k-dim = k//8, round to 8
    index_nz = np.zeros((ceil_n, ceil_k_idx), dtype=np.uint8)
    index_nz[:n, :k // 8] = index_nd
    new_shape = (ceil_n // 16, 16, ceil_k_idx // 8, 8)
    index_nz = index_nz.reshape(new_shape).transpose(2, 0, 1, 3).flatten()

    # Golden: C[m, n] = A[m, :] @ sparse_B[:, n] using index_mask
    golden = np.zeros((m, n), dtype=np.int32)
    for r in range(n):
        selected_cols = index_mask[r]
        a_selected = a_gm[:, selected_cols]
        golden[:, r] = np.dot(a_selected.astype(np.int32), dense_b[r].astype(np.int32))

    # B is stored as flat dense_b (n, k//2) C-order bytes, interpreted as ColumnMajor (k//2, n)
    # Reshaping preserves flat byte order while giving the correct shape for the kernel
    b_gm_col = np.ascontiguousarray(dense_b.reshape(k // 2, n))

    return (
        torch.from_numpy(a_gm).npu(),
        torch.from_numpy(b_gm_col).npu(),
        torch.from_numpy(index_nz).npu(),
        torch.from_numpy(golden).npu(),
    )


@only_on_2201
def test_sparse_matmul_tla():
    """Compare the CATLASS 4:2 sparse matmul (TLA) wrapper against a reference computation.

    Golden logic (from examples/41_sparse_matmul_tla/sparse_gen_data.py):
    A is int8 [M, K], B is int8 sparse-compressed [K/2, N] with 4:2 sparsity.
    The index matrix (NZ blocked format) encodes which 2 of every 4 consecutive
    K elements are kept for each N column.
    C = A @ sparse(B) with int32 accumulation.
    """
    m, n, k = 128, 256, 256

    a, b_sparse, idx, expected = _generate_sparse_data(m, n, k)

    result = torch_catlass.sparse_matmul_tla(
        a, b_sparse, idx, "int32", transA=False, transB=True
    )

    assert result.shape == (m, n)
    assert result.dtype == torch.int32
    assert result.device.type == "npu"

    rtol = 0
    atol = 0
    assert torch.allclose(result.cpu().float(), expected.cpu().float(), rtol=rtol, atol=atol), (
        f"Results not close: max diff = {(result.float() - expected.float()).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
