use candle_core::backend::BackendDevice;
use candle_core::{Device, Result, Tensor};
use serde::Serialize;
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Clone, Debug, Serialize)]
pub struct LayerEventSample {
    pub name: String,
    pub category: String,
    pub ms: f64,
    pub output_shape: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MetricStats {
    pub samples_ms: Vec<f64>,
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct LayerEventSummary {
    pub name: String,
    pub category: String,
    pub output_shape: Vec<usize>,
    pub stats: MetricStats,
    pub pct_of_sum_event_medians: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct AggregateSummary {
    pub name: String,
    pub sum_of_event_medians_ms: f64,
    pub pct_of_sum_event_medians: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct LayerAuditReport {
    pub schema: &'static str,
    pub methodology: &'static str,
    pub audit_runs: usize,
    pub event_count: usize,
    pub sync_baseline: MetricStats,
    pub profiled_pass: MetricStats,
    pub estimated_sync_baseline_contribution_ms: f64,
    pub estimated_sync_baseline_pct_of_profiled_pass_median: f64,
    pub sum_event_medians_ms: f64,
    pub events: Vec<LayerEventSummary>,
    pub categories: Vec<AggregateSummary>,
    pub stages: Vec<AggregateSummary>,
    pub top_hotspots: Vec<LayerEventSummary>,
}

#[derive(Debug)]
pub struct LayerProfiler {
    device: Device,
    events: Vec<LayerEventSample>,
}

impl LayerProfiler {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
            events: Vec::new(),
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        match &self.device {
            Device::Cpu => Ok(()),
            Device::Cuda(device) => device.synchronize(),
            Device::Metal(_) => candle_core::bail!("layer profiler currently supports CPU/CUDA only"),
        }
    }

    pub fn measure_tensor<F>(&mut self, name: &str, category: &str, f: F) -> Result<Tensor>
    where
        F: FnOnce() -> Result<Tensor>,
    {
        // The caller synchronizes before the first event and every event ends
        // with a synchronization. Therefore each leaf starts at a completed
        // barrier without paying a redundant second pre-op synchronize. This
        // is still intentionally intrusive; the empty-sync baseline and
        // profiled-pass total expose the observer effect.
        let start = Instant::now();
        let out = f()?;
        self.synchronize()?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        self.events.push(LayerEventSample {
            name: name.to_string(),
            category: category.to_string(),
            ms,
            output_shape: out.dims().to_vec(),
        });
        Ok(out)
    }

    pub fn into_events(self) -> Vec<LayerEventSample> {
        self.events
    }
}

pub fn measure_sync_baseline(device: &Device, samples: usize) -> Result<MetricStats> {
    let profiler = LayerProfiler::new(device);
    let mut values = Vec::with_capacity(samples.max(1));
    for _ in 0..samples.max(1) {
        let start = Instant::now();
        profiler.synchronize()?;
        values.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(metric_stats(values))
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn metric_stats(samples_ms: Vec<f64>) -> MetricStats {
    if samples_ms.is_empty() {
        return MetricStats {
            samples_ms,
            min_ms: f64::NAN,
            max_ms: f64::NAN,
            mean_ms: f64::NAN,
            median_ms: f64::NAN,
            p90_ms: f64::NAN,
            p95_ms: f64::NAN,
        };
    }
    let mut sorted = samples_ms.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mean_ms = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
    MetricStats {
        min_ms: sorted[0],
        max_ms: sorted[sorted.len() - 1],
        mean_ms,
        median_ms: percentile_sorted(&sorted, 0.5),
        p90_ms: percentile_sorted(&sorted, 0.90),
        p95_ms: percentile_sorted(&sorted, 0.95),
        samples_ms,
    }
}

fn stage_for(name: &str) -> &'static str {
    if name.starts_with("backbone.stem") {
        "backbone.stem"
    } else if name.starts_with("backbone.stage1") {
        "backbone.stage1"
    } else if name.starts_with("backbone.stage2") {
        "backbone.stage2"
    } else if name.starts_with("backbone.stage3") {
        "backbone.stage3"
    } else if name.starts_with("backbone.stage4") {
        "backbone.stage4"
    } else if name.starts_with("head.gau") {
        "head.gau"
    } else if name.starts_with("head") {
        "head.other"
    } else {
        "other"
    }
}

pub fn summarize_layer_audit(
    runs: Vec<Vec<LayerEventSample>>,
    pass_ms: Vec<f64>,
    sync_baseline: MetricStats,
) -> Result<LayerAuditReport> {
    if runs.is_empty() {
        candle_core::bail!("layer audit requires at least one run");
    }

    let expected_names: Vec<String> = runs[0].iter().map(|event| event.name.clone()).collect();
    for (run_index, run) in runs.iter().enumerate().skip(1) {
        let names: Vec<String> = run.iter().map(|event| event.name.clone()).collect();
        if names != expected_names {
            candle_core::bail!(
                "layer audit event order mismatch at run {run_index}: expected {} events got {}",
                expected_names.len(),
                names.len()
            );
        }
    }

    let mut summaries = Vec::with_capacity(runs[0].len());
    for event_index in 0..runs[0].len() {
        let first = &runs[0][event_index];
        let samples = runs
            .iter()
            .map(|run| run[event_index].ms)
            .collect::<Vec<_>>();
        summaries.push(LayerEventSummary {
            name: first.name.clone(),
            category: first.category.clone(),
            output_shape: first.output_shape.clone(),
            stats: metric_stats(samples),
            pct_of_sum_event_medians: 0.0,
        });
    }

    let sum_event_medians_ms = summaries.iter().map(|event| event.stats.median_ms).sum::<f64>();
    if sum_event_medians_ms > 0.0 {
        for event in &mut summaries {
            event.pct_of_sum_event_medians =
                event.stats.median_ms * 100.0 / sum_event_medians_ms;
        }
    }

    let mut category_sums: BTreeMap<String, f64> = BTreeMap::new();
    let mut stage_sums: BTreeMap<String, f64> = BTreeMap::new();
    for event in &summaries {
        *category_sums.entry(event.category.clone()).or_default() += event.stats.median_ms;
        *stage_sums
            .entry(stage_for(&event.name).to_string())
            .or_default() += event.stats.median_ms;
    }

    let to_aggregate = |items: BTreeMap<String, f64>| {
        let mut out = items
            .into_iter()
            .map(|(name, sum)| AggregateSummary {
                name,
                sum_of_event_medians_ms: sum,
                pct_of_sum_event_medians: if sum_event_medians_ms > 0.0 {
                    sum * 100.0 / sum_event_medians_ms
                } else {
                    0.0
                },
            })
            .collect::<Vec<_>>();
        out.sort_by(|a, b| {
            b.sum_of_event_medians_ms
                .total_cmp(&a.sum_of_event_medians_ms)
        });
        out
    };

    let mut top_hotspots = summaries.clone();
    top_hotspots.sort_by(|a, b| b.stats.median_ms.total_cmp(&a.stats.median_ms));
    top_hotspots.truncate(20);

    let profiled_pass = metric_stats(pass_ms);
    let event_count = summaries.len();
    let estimated_sync_baseline_contribution_ms =
        sync_baseline.median_ms * (event_count as f64 + 2.0);
    let estimated_sync_baseline_pct_of_profiled_pass_median =
        if profiled_pass.median_ms > 0.0 {
            estimated_sync_baseline_contribution_ms * 100.0 / profiled_pass.median_ms
        } else {
            0.0
        };

    Ok(LayerAuditReport {
        schema: "haricot.pose.body26.cuda-layer-audit.v1",
        methodology: "device synchronized before first event and after every leaf op; previous post-op barrier is next pre-op barrier; diagnostic and intentionally intrusive",
        audit_runs: runs.len(),
        event_count,
        sync_baseline,
        profiled_pass,
        estimated_sync_baseline_contribution_ms,
        estimated_sync_baseline_pct_of_profiled_pass_median,
        sum_event_medians_ms,
        events: summaries,
        categories: to_aggregate(category_sums),
        stages: to_aggregate(stage_sums),
        top_hotspots,
    })
}
