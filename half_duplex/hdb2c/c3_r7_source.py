"""Pinned-source TDT correction and observation, on the verified R6 encoder."""

MARKER = 'HDB2C3_R7_TDT_ADVANCE_AND_TRACE_V1'


def transform_model(source, helper):
    if MARKER in source:
        raise ValueError('R7 already installed')
    start = source.index('    fn decode_greedy(')
    end = source.index('    fn decode_beam(', start)
    head, source, tail = source[:start], source[start:end], source[end:]
    def once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('R7 pinned model anchor differs: '+old[:80])
        source = source.replace(old, new, 1)
    once('        let tr = self.time_ratio();\n\n        for (batch, &length)',
         '        let tr = self.time_ratio();\n        let mut hdb2c7_rows = Vec::new();\n\n        for (batch, &length)')
    once('                let step_before = step;\n', '''                let step_before = step;
                let trace_index = hdb2c7_rows.len();
                let input_token = last;
                if std::env::var_os("HDB2C7_TDT_TRACE_DIR").is_some() && b != 1 {
                    return Err(candle::Error::Msg("R7 trace requires batch 1".into()));
                }
''')
    once('                let enc_step = feature.narrow(1, step, 1)?;\n', '''                hdb2c7_tensor(trace_index,"g",&decoder_out)?;
                hdb2c7_tensor(trace_index,"h",&decoder_state.0)?;
                hdb2c7_tensor(trace_index,"c",&decoder_state.1)?;
                let enc_step = feature.narrow(1, step, 1)?;
                hdb2c7_tensor(trace_index,"enc",&enc_step)?;
''')
    once('                let joint_out = self.joint.forward(&enc_step, &decoder_out)?;\n',
         '                let joint_out = self.joint.forward(&enc_step, &decoder_out)?;\n                hdb2c7_tensor(trace_index,"logits",&joint_out)?;\n')
    once('''                let decision = (duration_logits.argmax(D::Minus1)?.to_vec0::<u32>()? as usize)
                    .min(self.durations.len() - 1);''', '''                if duration_logits.elem_count() != self.durations.len() {
                    return Err(candle::Error::Msg("R7 TDT duration head/config mismatch".into()));
                }
                let decision = duration_logits.argmax(D::Minus1)?.to_vec0::<u32>()? as usize;''')
    once('                    new_symbols += 1;\n', '')
    once('''                step += duration;
                let mut forced_advance = false;
                if duration != 0 {
                    new_symbols = 0;
                } else if let Some(max_symbols) = self.max_symbols
                    && new_symbols >= max_symbols
                {
                    step += 1;
                    forced_advance = true;
                    new_symbols = 0;
                }''', '''                let (next, count, forced_advance) = hdb2c7_advance(
                    step,pred_token,self.vocabulary.len(),duration,new_symbols,self.max_symbols);
                step=next; new_symbols=count;
                hdb2c7_rows.push(serde_json::json!({"frame":step_before,"token":pred_token,
                    "duration_index":decision,"duration":duration,"advance":step-step_before,
                    "forced_advance":forced_advance,"input_token":input_token,
                    "state_committed":pred_token!=self.vocabulary.len()}));''')
    once('''        Ok((results, next_hidden))''', '''        if let Some(root) = std::env::var_os("HDB2C7_TDT_TRACE_DIR") {
            std::fs::create_dir_all(&root).map_err(|e|candle::Error::Msg(e.to_string()))?;
            std::fs::write(std::path::PathBuf::from(root).join("tdt_trace.json"),
                serde_json::to_vec_pretty(&hdb2c7_rows).map_err(|e|candle::Error::Msg(e.to_string()))?)
                .map_err(|e|candle::Error::Msg(e.to_string()))?;
        }
        Ok((results, next_hidden))''')
    return head+source+tail+'\n'+helper
