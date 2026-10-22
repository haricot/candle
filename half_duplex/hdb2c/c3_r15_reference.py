"""Reusable original CPU NeMo reference; no FP64 operator replay.

The short profile observes the final encoder and the actual production TDT.
The full profile also records encoder stages, with an unobserved-forward control.
"""
from pathlib import Path

from .audio import read_pcm
from .c3_r3 import _dump, attach_nemo_hooks, verify_nemo_stages
from .c3_nemo import save_stage
from .c3_r4 import attach_nemo_r4_hooks, verify_r4_hooks
from .c3_r5 import attach_nemo_r5_hooks
from .c3_r6 import attach_nemo_r6_hooks
from .c3_r7_nemo import capture
from .c3_r7_metrics import tensor_metrics
from .c3_r8 import write_json
from .nemo_runner import load_nemo, ensure_probe_eval


class Reference:
    def __init__(self, checkpoint):
        self.model, self.torch = load_nemo(checkpoint)
        ensure_probe_eval(self.model)
        if len(self.model.encoder.layers) != 24:
            raise ValueError('R15 requires the qualified 24-block TranscribePsy checkpoint')

    def run(self, wav: Path, output: Path, *, detailed=False):
        model, torch = self.model, self.torch
        eval_state = ensure_probe_eval(model)
        output.mkdir(parents=True)
        pcm = read_pcm(wav)
        audio = torch.from_numpy(pcm.copy()).reshape(1, -1)
        with torch.inference_mode():
            mel, mel_length = model.preprocessor(
                input_signal=audio, length=torch.tensor([len(pcm)], dtype=torch.int64))
            encoded, lengths = model.encoder(audio_signal=mel, length=mel_length)
        _dump(output, 'pcm', audio.unsqueeze(-1), 'B,T,1')
        _dump(output, 'mel', mel.transpose(1, 2), 'B,T,F')
        _dump(output, 'encoder', encoded.transpose(1, 2), 'B,T,D')
        control = None
        if detailed:
            trace = output/'encoder_trace'
            trace.mkdir()
            hooks = []
            try:
                dw, dw_seen = attach_nemo_hooks(model.encoder, trace); hooks.extend(dw)
                r4, r4_seen = attach_nemo_r4_hooks(model.encoder, trace); hooks.extend(r4)
                r5, r5_seen = attach_nemo_r5_hooks(model.encoder, trace); hooks.extend(r5)
                r6, r6_seen = attach_nemo_r6_hooks(model.encoder, trace); hooks.extend(r6)
                stages = set()
                def observe(name):
                    def hook(_module, _args, result):
                        if name in stages:
                            raise ValueError('Duplicate R15 reference stage '+name)
                        save_stage(trace, name, result)
                        stages.add(name)
                    return hook
                hooks.append(model.encoder.pre_encode.register_forward_hook(observe('pre_encode')))
                hooks.append(model.encoder.pos_enc.register_forward_hook(observe('post_pos_enc')))
                hooks.extend(layer.register_forward_hook(observe(f'layer_{i:02}'))
                             for i, layer in enumerate(model.encoder.layers))
                with torch.inference_mode():
                    observed, observed_lengths = model.encoder(audio_signal=mel, length=mel_length)
                verify_nemo_stages(dw_seen)
                verify_r4_hooks(r4_seen)
                if (len(stages) != 26 or r5_seen != {'pos_input', 'pos_output'} or
                        r6_seen != {'encoder_pad_mask', 'encoder_att_mask', 'block0_conv_after_mask'}):
                    raise ValueError('Incomplete R15 reference stage coverage')
                control = tensor_metrics(encoded.numpy(), observed.numpy())
                if not control['bitwise_equal'] or not torch.equal(lengths, observed_lengths):
                    raise ValueError('R15 observation changed the original CPU encoder')
            finally:
                for hook in hooks:
                    hook.remove()
        # Decode the unobserved encoder with the original NeMo algorithm.
        result = capture(model, encoded, lengths, output)
        result.update(mel_valid=int(mel_length[0]), encoder_valid=int(lengths[0]),
                      eval_state=eval_state, encoder_observation_control=control,
                      reference_scope='ORIGINAL_CPU_NEMO_PRODUCTION_TDT',
                      fp64_operator_audit='NOT_REQUESTED')
        write_json(output/'manifest.json', result)
        return result
