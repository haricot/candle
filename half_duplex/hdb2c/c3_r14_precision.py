"""R14 FP64 operator audit and elementwise tolerance survey.

This is evidence collection, not a replacement qualification policy. FP64
replays consume the exact observed FP32 operands and never feed ASR inference.
No tolerance is fitted or approved from the candidate's outputs.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import math
from pathlib import Path

import numpy as np

from .c3_r3 import read_tensor, STAGES as SUBSAMPLING
from .c3_r7 import encoder_audit
from .c3_r8 import actual_input_for

SIGNATURE = 'hdb2c3-r14-fp64-operator-audit-v0.7.14'
# A prospective survey, not an acceptance threshold or an automatic fitter.
RTOL_SURVEY = (0., 1e-7, 1e-6, 1e-5, 1e-4)


def arrays(reference, candidate):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or not a.size:
        raise ValueError('R14 empty tensor or shape mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('R14 refuses nonfinite operands/results')
    return a, b


def errors(reference, candidate):
    """Retain the FP64 reference; never round it to FP32 before comparison."""
    a, b = arrays(reference, candidate)
    delta = np.abs(a-b)
    norm = float(np.linalg.norm(a))
    scale = float(np.max(np.abs(a)))
    return {'shape': list(a.shape), 'elements': int(a.size),
            'max_abs': float(delta.max()), 'rmse': float(np.sqrt(np.mean(delta*delta))),
            'reference_max_abs': scale,
            'relative_l2': float(np.linalg.norm(a-b)/norm) if norm else None,
            'max_error_over_reference_max_abs': float(delta.max()/scale) if scale else None}


def tolerance_survey(reference, candidate, *, atol=1e-3):
    """Check every element, including cancellation/near-zero values.

    A global cosine or max-error/max-amplitude ratio cannot establish this
    pointwise condition. A zero reference exceeding atol admits no finite rtol.
    """
    if not math.isfinite(atol) or atol < 0:
        raise ValueError('R14 invalid absolute tolerance')
    a, b = arrays(reference, candidate)
    delta, magnitude = np.abs(a-b), np.abs(a)
    zero_failure = (magnitude == 0) & (delta > atol)
    needed = np.zeros_like(delta)
    np.divide(np.maximum(delta-atol, 0), magnitude, out=needed, where=magnitude != 0)
    rows = []
    for rtol in RTOL_SURVEY:
        allowance = atol + rtol*magnitude
        failed = delta > allowance
        first = np.argwhere(failed)
        rows.append({'rtol': rtol, 'violations': int(failed.sum()),
                     'all_elements_within_tolerance': bool(not failed.any()),
                     'max_excess': float(np.maximum(delta-allowance, 0).max()),
                     'first_violation_index': first[0].tolist() if len(first) else None})
    return {**errors(a, b), 'atol': atol, 'criterion': 'abs(candidate-reference) <= atol + rtol*abs(reference)',
            'rtol_survey': rows, 'zero_reference_violations': int(zero_failure.sum()),
            'minimum_required_rtol_diagnostic': None if zero_failure.any() else float(needed.max()),
            'policy_status': 'SURVEY_ONLY_NO_THRESHOLD_APPROVED'}


def gamma(operations, *, dtype=np.float32):
    """Classical roundoff bound n*u/(1-n*u), u=epsilon/2."""
    if not isinstance(operations, int) or operations <= 0:
        raise ValueError('R14 requires a positive operation count')
    nu = operations * float(np.finfo(dtype).eps)/2
    if nu >= 1:
        raise ValueError('R14 roundoff bound is not informative')
    return nu/(1-nu)


def dot_bound(reference64, candidate32, absolute_product_sum, products):
    """Conservative non-FMA/FMA dot bound; diagnostic, not a blanket gate.

    2*k+1 operations cover separate products, summation and bias. The absolute
    term accounts for cancellation; the relative term scales with abs(y64).
    A small bound does not establish the correctness of a composed network.
    """
    a, b = arrays(reference64, candidate32)
    products_sum, _ = arrays(absolute_product_sum, a)
    if np.any(products_sum < 0):
        raise ValueError('R14 negative absolute product sum')
    operations = 2*products+1
    g32, g64 = gamma(operations), gamma(operations, dtype=np.float64)
    # Inflate the computed FP64 absolute sum to cover its own rounding.
    upper_sum = products_sum/(1-g64)
    relative = (g32+g64)*np.abs(a)
    absolute = (g32+g64)*np.maximum(upper_sum-np.abs(a), 0)
    allowance = relative+absolute
    delta = np.abs(a-b)
    return {'scope': 'CONSERVATIVE_LOCAL_DOT_ROUNDOFF_DIAGNOSTIC',
            'products_per_output': products, 'operations_bound': operations,
            'rtol': g32+g64, 'atol_elementwise_range': [float(absolute.min()), float(absolute.max())],
            'violations': int((delta > allowance).sum()),
            'max_excess': float(np.maximum(delta-allowance, 0).max()),
            'acceptance_gate': False,
            'assumptions': 'direct dot reduction; normal IEEE arithmetic without overflow/underflow; '
                           'FP32 inputs/weights exact in FP64; FFT/Winograd transforms not certified by this bound'}


def tensor_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<f4').tobytes()).hexdigest()


def operator_family(name):
    if name in ('out', 'pre_encode'):return 'subsampling_projection'
    if name.startswith('conv_') or name == 'block0_convolution':return 'convolution'
    if name.startswith('relu_'):return 'relu'
    if name.startswith('block0_norm_'):return 'normalization'
    if name.startswith('layer_'):return 'full_encoder_layer'
    if name in ('block0_ff1', 'block0_ff2'):return 'feed_forward'
    if name == 'block0_attention':return 'attention'
    if name.startswith('block0_after_'):return 'residual'
    if name == 'block0_pos_emb':return 'relative_position_embedding'
    if name in ('post_pos_enc', 'block0_input'):return 'position_scaling'
    raise ValueError('R14 unclassified operation '+name)


def _numpy64(value):
    import torch
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or value.device.type != 'cpu':
        raise ValueError('R14 reference did not execute in CPU FP64')
    out = value.detach().contiguous().numpy()
    if not np.isfinite(out).all():
        raise ValueError('R14 nonfinite FP64 reference')
    return out


class Fp64Auditor:
    """Copies of actual checkpoint modules, with original weights widened to FP64."""

    def __init__(self, model):
        import torch
        if any(m.training for m in model.modules()):
            raise ValueError('R14 requires the original model in eval mode')
        encoder = model.encoder
        if len(encoder.layers) != 24 or encoder.self_attention_model != 'rel_pos':
            raise ValueError('R14 requires the qualified 24-block relative-position model')
        original = {name: p.dtype for name, p in model.named_parameters()}
        self.subsampling = {f'conv_{i:02}': copy.deepcopy(encoder.pre_encode.conv[i]).double().eval()
                            for i in (0, 2, 3, 5, 6)}
        self.subsampling['out'] = copy.deepcopy(encoder.pre_encode.out).double().eval()
        self.block = copy.deepcopy(encoder.layers[0]).double().eval()
        self.xscale = float(encoder.pos_enc.xscale or 1.)
        if any(p.dtype != torch.float64 for m in [self.block, *self.subsampling.values()] for p in m.parameters()):
            raise ValueError('R14 mixed precision in FP64 reference')
        if original != {name: p.dtype for name, p in model.named_parameters()}:
            raise ValueError('R14 changed original model parameter dtypes')
        classes = [*self.subsampling.values(), self.block.feed_forward1,
                   self.block.self_attn, self.block.conv, self.block.norm_out]
        self.manifest = {'signature': SIGNATURE, 'torch_version': torch.__version__,
                         'reference_dtype': 'float64', 'reference_device': 'cpu',
                         'original_model_dtype_preserved': True,
                         'scope': 'LOCAL_OPERATORS_ON_OBSERVED_FP32_INPUTS_NOT_FULL_FP64_ASR',
                         'reference_forward_sha256': {type(m).__module__+'.'+type(m).__name__:
                             hashlib.sha256(inspect.getsource(type(m).forward).encode()).hexdigest() for m in classes}}

    @staticmethod
    def _tensor(a):
        import torch
        x = np.array(a, dtype=np.float64, copy=True)
        if not x.size or not np.isfinite(x).all():
            raise ValueError('R14 invalid operator operand')
        return torch.from_numpy(x)

    def _subsampling(self, name, operand):
        import torch
        import torch.nn.functional as functional
        x = self._tensor(operand)
        if name.startswith('relu_'):
            return _numpy64(x.relu()), None
        module = self.subsampling[name]
        with torch.inference_mode():
            result = module(x)
            bias = None if module.bias is None else module.bias.abs()
            if name == 'out':
                total = functional.linear(x.abs(), module.weight.abs(), bias)
                count = module.in_features
            else:
                if module.padding_mode != 'zeros':
                    raise ValueError('R14 unsupported convolution padding mode')
                total = functional.conv2d(x.abs(), module.weight.abs(), bias, module.stride,
                                          module.padding, module.dilation, module.groups)
                count = module.weight.shape[1]*math.prod(module.kernel_size)
        return _numpy64(result), (_numpy64(total), int(count))

    def evaluate(self, reference, candidate=None, replay=None, *, include_reference=True):
        """Common operands and autonomous operands stay separate in every row."""
        import torch
        reference = Path(reference)
        rows, failures = [], []
        trace = reference/'encoder_trace' if candidate is None else Path(candidate)/'encoder_trace'
        read = lambda folder, name: read_tensor(Path(folder)/(name+'.f32le'))

        def collect(scope, name, operands, actual, calculate):
            try:
                with torch.inference_mode():
                    high, bound = calculate()
                data = {'scope': scope, 'name': name, 'family': operator_family(name), 'reference_dtype': 'float64',
                        'operand_sha256': {k: tensor_hash(v) for k, v in operands.items()},
                        'fp32_vs_fp64': tolerance_survey(high, actual)}
                if bound is not None:
                    data['roundoff_bound'] = dot_bound(high, actual, bound[0], bound[1])
                rows.append(data)
            except Exception as exc:
                failures.append({'scope': scope, 'name': name, 'error': type(exc).__name__+': '+str(exc)})

        for name in SUBSAMPLING:
            ref_input = read(reference/'subsampling'/name, 'input')
            ref_output = read(reference/'subsampling'/name, 'output')
            if include_reference:
                collect('NEMO_FP32_ON_NEMO_OPERANDS', name, {'input': ref_input}, ref_output,
                        lambda n=name, x=ref_input: self._subsampling(n, x))
            if replay is not None:
                operand = read(Path(replay)/'subsampling'/name, 'input')
                if tensor_hash(ref_input) != tensor_hash(operand):
                    raise ValueError('R14 common replay operand identity changed')
                collect('CUDA_FP32_ON_NEMO_OPERANDS', name, {'input': operand},
                        read(Path(replay)/'subsampling'/name, 'output'),
                        lambda n=name, x=operand: self._subsampling(n, x))
            if candidate is not None:
                operand = actual_input_for(name, Path(candidate))
                collect('AUTONOMOUS_FP32_ON_OWN_OPERANDS', name, {'input': operand}, read(trace, name),
                        lambda n=name, x=operand: self._subsampling(n, x))

        scope = 'NEMO_FP32_ON_NEMO_OPERANDS' if candidate is None else 'AUTONOMOUS_FP32_ON_OWN_OPERANDS'
        module_inputs = {
            'block0_norm_ff1': ('norm_feed_forward1', 'block0_input'),
            'block0_ff1': ('feed_forward1', 'block0_norm_ff1'),
            'block0_norm_att': ('norm_self_att', 'block0_after_ff1'),
            'block0_norm_conv': ('norm_conv', 'block0_after_attention'),
            'block0_norm_ff2': ('norm_feed_forward2', 'block0_after_convolution'),
            'block0_ff2': ('feed_forward2', 'block0_norm_ff2'),
            'block0_norm_out': ('norm_out', 'block0_after_ff2')}
        for name, (module_name, input_name) in module_inputs.items():
            operand = read(trace, input_name)
            module = getattr(self.block, module_name)
            collect(scope, name, {'input': operand}, read(trace, name),
                    lambda m=module, x=operand: (_numpy64(m(self._tensor(x))), None))
        norm_att = read(trace, 'block0_norm_att')
        position = read(trace, 'block0_pos_emb')
        att_mask = read(trace, 'encoder_att_mask')
        pad_mask = read(trace, 'encoder_pad_mask')
        if not np.isin(att_mask, [0, 1]).all() or not np.isin(pad_mask, [0, 1]).all():
            raise ValueError('R14 non-boolean observed mask')
        collect(scope, 'block0_attention', {'input': norm_att, 'position': position, 'mask': att_mask},
                read(trace, 'block0_attention'), lambda: (_numpy64(self.block.self_attn(
                    query=self._tensor(norm_att), key=self._tensor(norm_att), value=self._tensor(norm_att),
                    mask=torch.from_numpy(att_mask.astype(bool)), pos_emb=self._tensor(position))), None))
        norm_conv = read(trace, 'block0_norm_conv')
        collect(scope, 'block0_convolution', {'input': norm_conv, 'mask': pad_mask},
                read(trace, 'block0_convolution'), lambda: (_numpy64(self.block.conv(
                    self._tensor(norm_conv), pad_mask=torch.from_numpy(pad_mask[..., 0].astype(bool)))), None))
        residuals = {'block0_after_ff1': ('block0_input', 'block0_ff1', .5),
                     'block0_after_attention': ('block0_after_ff1', 'block0_attention', 1.),
                     'block0_after_convolution': ('block0_after_attention', 'block0_convolution', 1.),
                     'block0_after_ff2': ('block0_after_convolution', 'block0_ff2', .5)}
        for name, (left, right, factor) in residuals.items():
            a, b = read(trace, left), read(trace, right)
            collect(scope, name, {'left': a, 'right': b}, read(trace, name),
                    lambda a=a, b=b, f=factor: (np.asarray(a, dtype=np.float64)+np.asarray(b, dtype=np.float64)*f, None))
        pre = read(trace, 'pre_encode')
        collect(scope, 'post_pos_enc', {'input': pre}, read(trace, 'post_pos_enc'),
                lambda: (np.asarray(pre, dtype=np.float64)*self.xscale, None))
        return {'signature': SIGNATURE, 'reference': self.manifest,
                'execution_status': 'COMPLETE' if not failures else 'INCOMPLETE',
                'local_operator_replays': rows, 'errors': failures,
                'qualification': 'AUDIT_ONLY_TOLERANCE_REVIEW_PENDING',
                'production_injection': 'NONE', 'bitwise_policy': 'CROSS_BACKEND_DIAGNOSTIC_ONLY'}


def encoder_survey(reference, candidate, valid):
    """Survey the original stage gates without replacing them."""
    original = encoder_audit(Path(reference)/'encoder_trace', Path(candidate), valid)
    rows = []
    for stage in original['stages']:
        name = stage['name']
        a = read_tensor(Path(reference)/'encoder_trace'/(name+'.f32le'))
        b = read_tensor(Path(candidate)/'encoder_trace'/(name+'.f32le'))
        rows.append({'name': name, 'family': operator_family(name), 'legacy_numeric_pass': stage['numeric_pass'],
                     'bitwise_equal_diagnostic': stage['bitwise_equal'], **tolerance_survey(a, b)})
    return rows
