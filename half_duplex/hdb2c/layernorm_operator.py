"""Standalone ordered LayerNorm: original CPU source and diagnostic CUDA port.

CUDA execution uses the driver API and NVRTC; it never uses torch.cuda and
never falls back to CPU. This module does not replace a model operator.
"""
from __future__ import annotations

import ctypes as C
import ctypes.util
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'ci/experimental/cpu_parity/ordered_layernorm.cpp'
NVRTC_OPTIONS = ('--std=c++17', '--gpu-architecture=compute_61', '--fmad=true',
                 '--ftz=false', '--prec-div=true', '--prec-sqrt=true')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def operands(x, gamma, beta, epsilon):
    values = [np.ascontiguousarray(a, dtype=np.float32) for a in (x, gamma, beta)]
    x, gamma, beta = values
    if (x.ndim < 2 or x.size == 0 or not 1 <= x.shape[-1] <= 16384
            or gamma.shape != (x.shape[-1],) or beta.shape != gamma.shape
            or not all(np.isfinite(a).all() for a in values)
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('LayerNorm requires finite FP32 rows, affine vectors and positive epsilon')
    rows = x.size // x.shape[-1]
    if rows > 2**31 - 1:
        raise ValueError('LayerNorm row count exceeds diagnostic limit')
    return x, gamma, beta


class CpuNorm:
    def __init__(self, directory):
        flags_file = Path('/proc/cpuinfo')
        if not flags_file.is_file() or not {'avx2', 'fma'}.issubset(flags_file.read_text().split()):
            raise RuntimeError('The ordered CPU reproduction requires Linux x86-64 AVX2 and FMA')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / 'ordered-layernorm.so'
        flags = ['-shared', '-fPIC', '-O2', '-mavx2', '-mfma',
                 '-fno-math-errno', '-ffp-contract=fast']
        compiler = os.environ.get('HDB2C_CXX', 'g++')
        subprocess.run([compiler, *flags, str(SOURCE), '-o', str(self.path)], check=True)
        self.library = C.CDLL(str(self.path.resolve()))
        self.function = self.library.hdb2c12_layernorm
        array = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        self.function.argtypes = [C.c_int]*3 + [C.c_float] + [array]*4
        self.function.restype = C.c_int
        self.metadata = {'backend': 'CPU_AVX2_ORDERED_FP32', 'source_sha256': sha(SOURCE.read_bytes()),
                         'library_sha256': sha(self.path.read_bytes()), 'flags': flags,
                         'compiler': subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]}

    def __call__(self, x, gamma, beta, epsilon):
        x, gamma, beta = operands(x, gamma, beta, epsilon)
        out = np.empty_like(x)
        code = self.function(x.size//x.shape[-1], x.shape[-1], 8, epsilon, x, gamma, beta, out)
        if code:
            raise RuntimeError('CPU LayerNorm returned '+str(code))
        return out


def cuda_source():
    """Qualify the archived scalar-lane algorithm for device compilation.

    Keep the original CPU file unchanged. Refuse source drift in the few
    syntactic substitutions; actual arithmetic remains measured, not assumed.
    """
    source = SOURCE.read_text()
    replacements = (
        ('#include <cmath>', ''), ('#include <algorithm>', ''), ('#include <cstdint>', 'typedef long long int64_t;'),
        ('static void combine(', '__device__ void combine('),
        ('static void row_moments(', '__device__ void row_moments('),
        ('extern "C" int hdb2c12_layernorm(', '__device__ int hdb2c12_layernorm('),
        ('std::min(16,vectors-chunk*16)', '((vectors-chunk*16)<16 ? (vectors-chunk*16) : 16)'),
        ('std::sqrt(var+epsilon)', 'sqrtf(var+epsilon)'),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Ordered LayerNorm source changed: '+old)
        source = source.replace(old, new, 1)
    if source.count('std::fma(') != 3:
        raise ValueError('Ordered LayerNorm FMA coverage changed')
    source = source.replace('std::fma(', 'fmaf(')
    return source + '''
extern "C" __global__ void hdb2c12_layernorm_cuda(int rows,int n,float epsilon,
    const float *x,const float *gamma,const float *beta,float *out) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<rows)hdb2c12_layernorm(1,n,8,epsilon,x+(int64_t)row*n,
        gamma,beta,out+(int64_t)row*n);
}
'''


def nvrtc_library():
    candidates = [os.environ.get('HDB2C_NVRTC_LIBRARY'), ctypes.util.find_library('nvrtc'),
                  '/opt/cuda/lib64/libnvrtc.so.12',
                  '/opt/cuda/targets/x86_64-linux/lib/libnvrtc.so.12',
                  '/usr/local/cuda/lib64/libnvrtc.so.12', 'libnvrtc.so.12']
    # Permit the standalone NVRTC wheel for compilation on a host without a GPU.
    try:
        import nvidia.cuda_nvrtc
        for folder in nvidia.cuda_nvrtc.__path__:
            candidates.append(str(Path(folder)/'lib/libnvrtc.so.12'))
    except ImportError:
        pass
    for candidate in candidates:
        if candidate:
            try:
                return C.CDLL(candidate)
            except OSError:
                pass
    raise RuntimeError('CUDA 12 NVRTC library missing; use the unified CUDA 12.9.2 image')


def compile_cuda():
    lib = nvrtc_library()
    def bind(name, args):
        fn = getattr(lib, name); fn.argtypes = args; fn.restype = C.c_int
        return fn
    version = bind('nvrtcVersion', [C.POINTER(C.c_int)]*2)
    create = bind('nvrtcCreateProgram', [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p,
                                       C.c_int, C.c_void_p, C.c_void_p])
    compile_fn = bind('nvrtcCompileProgram', [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)])
    destroy = bind('nvrtcDestroyProgram', [C.POINTER(C.c_void_p)])
    get_log_size = bind('nvrtcGetProgramLogSize', [C.c_void_p, C.POINTER(C.c_size_t)])
    get_log = bind('nvrtcGetProgramLog', [C.c_void_p, C.c_void_p])
    get_size = bind('nvrtcGetPTXSize', [C.c_void_p, C.POINTER(C.c_size_t)])
    get_ptx = bind('nvrtcGetPTX', [C.c_void_p, C.c_void_p])
    major, minor = C.c_int(), C.c_int()
    if version(C.byref(major), C.byref(minor)) or (major.value, minor.value) != (12, 9):
        raise RuntimeError('Diagnostic CUDA compiler must be NVRTC 12.9')
    source = cuda_source().encode()
    program = C.c_void_p()
    if create(C.byref(program), source, b'ordered_layernorm.cu', 0, None, None):
        raise RuntimeError('NVRTC could not create LayerNorm program')
    try:
        options = (C.c_char_p*len(NVRTC_OPTIONS))(*(s.encode() for s in NVRTC_OPTIONS))
        status = compile_fn(program, len(options), options)
        size = C.c_size_t(); get_log_size(program, C.byref(size))
        log = C.create_string_buffer(max(1, size.value)); get_log(program, log)
        if status:
            raise RuntimeError('NVRTC compilation failed: '+log.value.decode(errors='replace'))
        if get_size(program, C.byref(size)):
            raise RuntimeError('NVRTC could not size PTX')
        ptx = C.create_string_buffer(size.value)
        if get_ptx(program, ptx):
            raise RuntimeError('NVRTC could not emit PTX')
        return ptx.raw, {'backend': 'CUDA_SM61_ORDERED_FP32_DIAGNOSTIC',
                         'source_sha256': sha(SOURCE.read_bytes()), 'cuda_source_sha256': sha(source),
                         'ptx_sha256': sha(ptx.raw), 'nvrtc_version': [major.value, minor.value],
                         'options': list(NVRTC_OPTIONS), 'cpu_fallback': False,
                         'compilation_log': log.value.decode(errors='replace')}
    finally:
        destroy(C.byref(program))


class CudaNorm:
    def __init__(self):
        self.driver = C.CDLL('libcuda.so.1')
        self.context, self.module = C.c_void_p(), C.c_void_p()
        self.functions = {}
        self._bind('cuInit', [C.c_uint])(0)
        self._bind('cuDeviceGet', [C.POINTER(C.c_int), C.c_int])
        device = C.c_int(); self._call('cuDeviceGet', C.byref(device), 0)
        self._bind('cuDeviceGetAttribute', [C.POINTER(C.c_int), C.c_int, C.c_int])
        major, minor = C.c_int(), C.c_int()
        self._call('cuDeviceGetAttribute', C.byref(major), 75, device)
        self._call('cuDeviceGetAttribute', C.byref(minor), 76, device)
        if (major.value, minor.value) != (6, 1):
            raise RuntimeError('This qualification requires the physical sm61 GPU')
        name = C.create_string_buffer(256)
        self._bind('cuDeviceGetName', [C.c_void_p, C.c_int, C.c_int])
        self._call('cuDeviceGetName', name, 256, device)
        self._bind('cuDriverGetVersion', [C.POINTER(C.c_int)])
        driver_version = C.c_int(); self._call('cuDriverGetVersion', C.byref(driver_version))
        ptx, self.metadata = compile_cuda()
        self.metadata.update({'device': name.value.decode(), 'compute_capability': [6, 1],
                              'driver_version': driver_version.value})
        self._bind('cuCtxCreate_v2', [C.POINTER(C.c_void_p), C.c_uint, C.c_int])
        self._bind('cuCtxDestroy_v2', [C.c_void_p])
        self._bind('cuModuleLoadData', [C.POINTER(C.c_void_p), C.c_void_p])
        self._bind('cuModuleUnload', [C.c_void_p])
        self._bind('cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p])
        self._bind('cuMemAlloc_v2', [C.POINTER(C.c_uint64), C.c_size_t])
        self._bind('cuMemFree_v2', [C.c_uint64])
        self._bind('cuMemcpyHtoD_v2', [C.c_uint64, C.c_void_p, C.c_size_t])
        self._bind('cuMemcpyDtoH_v2', [C.c_void_p, C.c_uint64, C.c_size_t])
        self._bind('cuLaunchKernel', [C.c_void_p]+[C.c_uint]*7+[C.c_void_p]*3)
        self._bind('cuCtxSynchronize', [])
        try:
            self._call('cuCtxCreate_v2', C.byref(self.context), 0, device)
            self._call('cuModuleLoadData', C.byref(self.module), C.create_string_buffer(ptx))
            self.kernel = C.c_void_p()
            self._call('cuModuleGetFunction', C.byref(self.kernel), self.module, b'hdb2c12_layernorm_cuda')
        except BaseException:
            self.close(); raise

    def _bind(self, name, args):
        fn = getattr(self.driver, name); fn.argtypes = args; fn.restype = C.c_int
        self.functions[name] = fn
        return lambda *values: self._call(name, *values)

    def _call(self, name, *values):
        result = self.functions[name](*values)
        if result:
            raise RuntimeError(f'CUDA driver {name} failed with code {result}; no CPU fallback')

    def __call__(self, x, gamma, beta, epsilon):
        x, gamma, beta = operands(x, gamma, beta, epsilon)
        out = np.empty_like(x); buffers = []
        try:
            for a in (x, gamma, beta, out):
                pointer = C.c_uint64()
                self._call('cuMemAlloc_v2', C.byref(pointer), a.nbytes)
                buffers.append(pointer)
            for a, pointer in zip((x, gamma, beta), buffers):
                self._call('cuMemcpyHtoD_v2', pointer, C.c_void_p(a.ctypes.data), a.nbytes)
                echo = np.empty_like(a)
                self._call('cuMemcpyDtoH_v2', C.c_void_p(echo.ctypes.data), pointer, a.nbytes)
                if echo.tobytes() != a.tobytes():
                    raise RuntimeError('CUDA common input transfer changed bytes')
            rows, width, eps = C.c_int(x.size//x.shape[-1]), C.c_int(x.shape[-1]), C.c_float(epsilon)
            values = [rows, width, eps, *buffers]
            args = (C.c_void_p*len(values))(*(C.cast(C.byref(v), C.c_void_p) for v in values))
            self._call('cuLaunchKernel', self.kernel, (rows.value+31)//32, 1, 1, 32, 1, 1, 0, None, args, None)
            self._call('cuCtxSynchronize')
            self._call('cuMemcpyDtoH_v2', C.c_void_p(out.ctypes.data), buffers[-1], out.nbytes)
            return out
        finally:
            for pointer in reversed(buffers):
                self.functions['cuMemFree_v2'](pointer)

    def close(self):
        if self.module.value:
            self.functions['cuModuleUnload'](self.module); self.module = C.c_void_p()
        if self.context.value:
            self.functions['cuCtxDestroy_v2'](self.context); self.context = C.c_void_p()
