# Attributions

- FireRedTeam / Xiaohongshu, FireRedVAD, copyright 2026 Xiaohongshu
  (Kaituo Xu, Wenpeng Li, Kai Huang, Kun Liu), Apache-2.0.
  Architecture : https://github.com/FireRedTeam/FireRedVAD
  Poids Stream-VAD : https://huggingface.co/FireRedTeam/FireRedVAD
- Eric Schmidbauer, conversion ONNX FireRedVAD, MIT :
  https://github.com/eschmidbauer/fireredvad-onnx
  Révision d'actifs : `faa757dafa0058d0b231ec82e084ec910a3a46d5`.
- Daily / Pipecat, Smart Turn, BSD-2-Clause. Architecture :
  https://github.com/pipecat-ai/smart-turn ; poids v3.2 :
  https://huggingface.co/pipecat-ai/smart-turn-v3
  Révision des poids : `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`.
- Prétraitements : conventions Kaldi FBank et Transformers WhisperFeatureExtractor.
- Le moteur ASR est l'adaptation R9 existante de n0madic/parakeet-candle,
  préparée par `half_duplex/ci/extended_build.py` et compilée avec Candle.

Les adaptations Rust nouvelles suivent la licence MIT OR Apache-2.0 du projet,
avec maintien des attributions et conditions des sources concernées. Les poids
restent des actifs externes sous leurs licences respectives. Pour les portions
issues de Smart Turn, la notice suivante est conservée :

BSD 2-Clause License

Copyright (c) 2024–2025, Daily

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
