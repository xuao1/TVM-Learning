import tvm
from tvm import te
import numpy as np
from tvm.contrib import nvcc

# 输入和过滤器的大小
batch_size = 256
height = 14
width = 14
in_channels = 256
out_channels = 512
kernel_h = 3
kernel_w = 3
pad_h = 1
pad_w = 1
stride_h = 1
stride_w = 1

# TensorCore shape
block_size = 16

assert batch_size % block_size == 0
assert in_channels % block_size == 0
assert out_channels % block_size == 0

# 输入特征图：（N，H，W，IC，n，ic）
data_shape = (
    batch_size // block_size,
    height,
    width,
    in_channels // block_size,
    block_size,
    block_size,
)
# Kernel: (H, W, IC, OC, ic, oc)
kernel_shape = (
    kernel_h,
    kernel_w,
    in_channels // block_size,
    out_channels // block_size,
    block_size,
    block_size,
)
# 输出特征图：(N, H, W, OC, n, oc)
output_shape = (
    batch_size // block_size,
    height,
    width,
    out_channels // block_size,
    block_size,
    block_size,
)

# Reduction axes
kh = te.reduce_axis((0, kernel_h), name="kh")
kw = te.reduce_axis((0, kernel_w), name="kw")
ic = te.reduce_axis((0, in_channels // block_size), name="ic")
ii = te.reduce_axis((0, block_size), name="ii")

# 算法
A = te.placeholder(data_shape, name="A", dtype="float16")
W = te.placeholder(kernel_shape, name="W", dtype="float16")
Apad = te.compute(
    (
        batch_size // block_size,
        height + 2 * pad_h,
        width + 2 * pad_w,
        in_channels // block_size,
        block_size,
        block_size,
    ),
    lambda n, h, w, i, nn, ii: tvm.tir.if_then_else(
        tvm.tir.all(h >= pad_h, h - pad_h < height, w >= pad_w, w - pad_w < width),
        A[n, h - pad_h, w - pad_w, i, nn, ii],
        tvm.tir.const(0.0, "float16"),
    ),
    name="Apad",
)
Conv = te.compute(
    output_shape,
    lambda n, h, w, o, nn, oo: te.sum(
        Apad[n, h * stride_h + kh, w * stride_w + kw, ic, nn, ii].astype("float32")
        * W[kh, kw, ic, o, ii, oo].astype("float32"),
        axis=[ic, kh, kw, ii],
    ),
    name="Conv",
)

s = te.create_schedule(Conv.op)
s[Apad].compute_inline()

# 指定内存层次结构
AS = s.cache_read(Apad, "shared", [Conv])
WS = s.cache_read(W, "shared", [Conv])
AF = s.cache_read(AS, "wmma.matrix_a", [Conv])
WF = s.cache_read(WS, "wmma.matrix_b", [Conv])
ConvF = s.cache_write(Conv, "wmma.accumulator")

def intrin_wmma_load_matrix(scope):
    n = 16
    A = te.placeholder((n, n), name="A", dtype="float16")
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="shared", data_alignment=32, offset_factor=256)
    C = te.compute((n, n), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope=scope, data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                n,
                n,
                n,
                BC.elem_offset // 256,
                BA.access_ptr("r"),
                n,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})

def intrin_wmma_gemm():
    n = 16
    A = te.placeholder((n, n), name="A", dtype="float16")
    B = te.placeholder((n, n), name="B", dtype="float16")
    k = te.reduce_axis((0, n), name="k")
    C = te.compute(
        (n, n),
        lambda ii, jj: te.sum(A[ii, k].astype("float") * B[k, jj].astype("float"), axis=k),
        name="C",
    )
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="BA", scope="wmma.matrix_a", data_alignment=32, offset_factor=256
    )
    BB = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="BB", scope="wmma.matrix_b", data_alignment=32, offset_factor=256
    )
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, name="BC", scope="wmma.accumulator", data_alignment=32, offset_factor=256
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle", "tir.tvm_fill_fragment", BC.data, n, n, n, BC.elem_offset // 256, 0.0
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    BC.elem_offset // 256,
                    BA.data,
                    BA.elem_offset // 256,
                    BB.data,
                    BB.elem_offset // 256,
                    BC.data,
                    BC.elem_offset // 256,
                )
            )
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})

def intrin_wmma_store_matrix():
    n = 16
    A = te.placeholder((n, n), name="A", dtype="float32")
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="wmma.accumulator", data_alignment=32, offset_factor=256
    )
    C = te.compute((n, n), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope="global", data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                n,
                n,
                n,
                BA.elem_offset // 256,
                BC.access_ptr("w"),
                n,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


# 定义 tile 大小
block_row_warps = 4
block_col_warps = 2
warp_row_tiles = 2
warp_col_tiles = 4
warp_size = 32
chunk = 2

block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
block_z = te.thread_axis("blockIdx.z")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")
thread_z = te.thread_axis("threadIdx.z")

nc, hc, wc, oc, nnc, ooc = Conv.op.axis
block_k = s[Conv].fuse(hc, wc)
s[Conv].bind(block_k, block_z)
nc, nci = s[Conv].split(nc, factor=warp_row_tiles)
block_i, nc = s[Conv].split(nc, factor=block_row_warps)
oc, oci = s[Conv].split(oc, factor=warp_col_tiles)
block_j, oc = s[Conv].split(oc, factor=block_col_warps)
s[Conv].reorder(block_k, block_i, block_j, nc, oc, nci, oci, nnc, ooc)
s[Conv].bind(block_i, block_x)
s[Conv].bind(block_j, block_y)
s[Conv].bind(nc, thread_y)
s[Conv].bind(oc, thread_z)

# 调度本地计算
s[ConvF].compute_at(s[Conv], oc)
n, h, w, o, nnf, oof = ConvF.op.axis
ko, ki = s[ConvF].split(ic, factor=chunk)
s[ConvF].reorder(ko, kh, ki, kw, n, o, nnf, oof, ii)

# 将中间计算移动到每个输出计算块中
s[AF].compute_at(s[ConvF], kw)
s[WF].compute_at(s[ConvF], kw)

# A 的共享内存调度
s[AS].compute_at(s[ConvF], kh)
n, h, w, i, nn, ii = AS.op.axis
tx, xo = s[AS].split(n, nparts=block_row_warps)
ty, yo = s[AS].split(xo, nparts=block_col_warps)
t = s[AS].fuse(nn, ii)
to, ti = s[AS].split(t, factor=warp_size)
s[AS].bind(tx, thread_y)
s[AS].bind(ty, thread_z)
s[AS].bind(ti, thread_x)

# W 的共享内存调度
s[WS].compute_at(s[ConvF], kh)
kh, kw, ic, o, ii, oo = WS.op.axis
tx, xo = s[WS].split(o, nparts=block_row_warps)
ty, yo = s[WS].split(xo, nparts=block_col_warps)
t = s[WS].fuse(ii, oo)
to, ti = s[WS].split(t, nparts=warp_size)
s[WS].bind(tx, thread_y)
s[WS].bind(ty, thread_z)
s[WS].bind(to, thread_x)
s[WS].vectorize(ti)
print("==============================================================\n")
print(tvm.lower(s, [A, W, Conv], simple_mode=True))

s[AF].tensorize(AF.op.axis[-2], intrin_wmma_load_matrix("wmma.matrix_a"))
s[WF].tensorize(WF.op.axis[-2], intrin_wmma_load_matrix("wmma.matrix_b"))
s[Conv].tensorize(nnc, intrin_wmma_store_matrix())
s[ConvF].tensorize(nnf, intrin_wmma_gemm())
print("==============================================================\n")
print(tvm.lower(s, [A, W, Conv], simple_mode=True))

dev = tvm.cuda(0)
if nvcc.have_tensorcore(dev.compute_version):
    with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 16}}):
        func = tvm.build(s, [A, W, Conv], "cuda")
    a_np = np.random.uniform(size=data_shape).astype(A.dtype)
    w_np = np.random.uniform(size=kernel_shape).astype(W.dtype)
    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    c = tvm.nd.array(np.zeros(output_shape, dtype=Conv.dtype), dev)
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print("==============================================================\n")
    print("conv2d with tensor core: %f ms" % (evaluator(a, w, c).mean * 1e3))