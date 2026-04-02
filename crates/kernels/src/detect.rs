//! Detection post-processing: TopK and Non-Maximum Suppression.
//!
//! These complete the detection pipeline (YOLO, SSD, RetinaNet, etc.):
//!   backbone → neck → head → **TopK score filter → NMS** → final detections
//!
//! TopK: CPU-based sort+select (called once per inference, N is manageable).
//! IoU matrix: GPU-parallel pairwise computation.
//! NMS: GPU IoU + CPU greedy suppression.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use warp_ir::{DType, Shape};

use crate::cache::KernelCache;
use crate::device::{DeviceError, WarpDevice};
use crate::tensor::GpuTensor;

macro_rules! launch_err {
    ($e:expr) => {
        $e.map_err(|e: cudarc::driver::result::DriverError| DeviceError::Launch(e.to_string()))
    };
}

// ── IoU matrix (GPU) ────────────────────────────────────────────

const IOU_MATRIX_SRC: &str = r#"
extern "C" __global__ void warp_iou_matrix(
    float *iou_out,         // [N, N]
    const float *boxes,     // [N, 4] — (x1, y1, x2, y2)
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * N;
    if (idx >= total) return;

    unsigned int i = idx / N;
    unsigned int j = idx % N;

    float x1_i = boxes[i * 4 + 0], y1_i = boxes[i * 4 + 1];
    float x2_i = boxes[i * 4 + 2], y2_i = boxes[i * 4 + 3];
    float x1_j = boxes[j * 4 + 0], y1_j = boxes[j * 4 + 1];
    float x2_j = boxes[j * 4 + 2], y2_j = boxes[j * 4 + 3];

    float ix1 = fmaxf(x1_i, x1_j);
    float iy1 = fmaxf(y1_i, y1_j);
    float ix2 = fminf(x2_i, x2_j);
    float iy2 = fminf(y2_i, y2_j);

    float inter = fmaxf(ix2 - ix1, 0.0f) * fmaxf(iy2 - iy1, 0.0f);
    float area_i = (x2_i - x1_i) * (y2_i - y1_i);
    float area_j = (x2_j - x1_j) * (y2_j - y1_j);
    float uni = area_i + area_j - inter;

    iou_out[idx] = (uni > 0.0f) ? inter / uni : 0.0f;
}
"#;

/// TopK result.
pub struct TopKResult {
    pub values: Vec<f32>,
    pub indices: Vec<usize>,
}

/// TopK: find K largest (or smallest) elements.
///
/// Reads tensor to CPU, sorts, returns top K values and original indices.
/// For inference post-processing this is efficient — called once per step.
pub fn topk(
    device: &WarpDevice,
    input: &GpuTensor<f32>,
    k: u32,
    largest: bool,
) -> Result<TopKResult, DeviceError> {
    let data = input.to_host(device)?;
    let n = data.len();
    let k = k as usize;
    assert!(k <= n, "K ({k}) must be <= N ({n})");

    let mut indexed: Vec<(f32, usize)> = data.into_iter().zip(0..).collect();

    if largest {
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    let values: Vec<f32> = indexed[..k].iter().map(|x| x.0).collect();
    let indices: Vec<usize> = indexed[..k].iter().map(|x| x.1).collect();

    Ok(TopKResult { values, indices })
}

/// NMS result.
pub struct NmsResult {
    pub keep_indices: Vec<usize>,
    pub num_kept: usize,
}

/// Non-maximum suppression.
///
/// boxes: [N, 4] as (x1, y1, x2, y2)
/// scores: [N] confidence scores
/// iou_threshold: suppress boxes with IoU above this
/// score_threshold: ignore boxes with score below this
/// max_output: maximum number of detections to keep
pub fn nms(
    cache: &KernelCache,
    device: &WarpDevice,
    boxes: &GpuTensor<f32>,
    scores: &GpuTensor<f32>,
    iou_threshold: f32,
    score_threshold: f32,
    max_output: u32,
) -> Result<NmsResult, DeviceError> {
    let n = scores.numel;
    let scores_host = scores.to_host(device)?;
    let boxes_host = boxes.to_host(device)?;

    // Sort by score descending, filter by threshold
    let mut order: Vec<usize> = (0..n)
        .filter(|&i| scores_host[i] > score_threshold)
        .collect();
    order.sort_by(|&a, &b| scores_host[b].partial_cmp(&scores_host[a]).unwrap());

    if order.is_empty() {
        return Ok(NmsResult { keep_indices: vec![], num_kept: 0 });
    }

    let filtered_n = order.len();

    // Reorder boxes for GPU IoU
    let mut reordered: Vec<f32> = vec![0.0; filtered_n * 4];
    for (new_i, &orig_i) in order.iter().enumerate() {
        reordered[new_i * 4..new_i * 4 + 4]
            .copy_from_slice(&boxes_host[orig_i * 4..orig_i * 4 + 4]);
    }

    // GPU IoU matrix
    let gpu_boxes = GpuTensor::from_host(device, &reordered,
        Shape::from_static(&[filtered_n, 4]), DType::F32)?;
    let mut iou_matrix = GpuTensor::<f32>::zeros(device,
        Shape::from_static(&[filtered_n, filtered_n]), DType::F32)?;

    let iou_f = cache.get_or_compile(device, IOU_MATRIX_SRC, "warp_iou_matrix")?;
    let total = (filtered_n * filtered_n) as u32;
    let cfg = LaunchConfig::for_num_elems(total);

    unsafe {
        launch_err!(device.stream.launch_builder(&iou_f)
            .arg(&mut iou_matrix.data)
            .arg(&gpu_boxes.data)
            .arg(&(filtered_n as u32))
            .launch(cfg))?;
    }
    device.synchronize()?;

    let iou_host = iou_matrix.to_host(device)?;

    // Greedy suppression
    let mut suppressed = vec![false; filtered_n];
    let mut keep = Vec::new();

    for i in 0..filtered_n {
        if suppressed[i] { continue; }
        keep.push(order[i]);
        if keep.len() >= max_output as usize { break; }

        for j in (i + 1)..filtered_n {
            if !suppressed[j] && iou_host[i * filtered_n + j] > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    let num_kept = keep.len();
    Ok(NmsResult { keep_indices: keep, num_kept })
}

// ═════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Option<(WarpDevice, KernelCache)> {
        Some((WarpDevice::new(0).ok()?, KernelCache::new()))
    }

    #[test]
    fn topk_largest() {
        let (dev, _cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let data: Vec<f32> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0];
        let input = GpuTensor::from_host(&dev, &data,
            Shape::from_static(&[10]), DType::F32).unwrap();

        let result = topk(&dev, &input, 3, true).unwrap();

        println!("TopK(3, largest): values={:?}, indices={:?}", result.values, result.indices);
        assert_eq!(result.values[0], 9.0);
        assert_eq!(result.values[1], 6.0);
        assert_eq!(result.values[2], 5.0);
        assert_eq!(result.indices[0], 5);
        assert_eq!(result.indices[1], 7);
    }

    #[test]
    fn topk_smallest() {
        let (dev, _cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let data: Vec<f32> = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let input = GpuTensor::from_host(&dev, &data,
            Shape::from_static(&[8]), DType::F32).unwrap();

        let result = topk(&dev, &input, 3, false).unwrap();
        println!("TopK(3, smallest): values={:?}", result.values);
        assert_eq!(result.values[0], 1.0);
        assert_eq!(result.values[1], 1.0);
        assert_eq!(result.values[2], 2.0);
    }

    #[test]
    fn topk_large_array() {
        let (dev, _cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let n = 32000usize;
        let data: Vec<f32> = (0..n).map(|i| ((i * 7 + 13) % 1000) as f32 * 0.001).collect();

        let mut expected: Vec<f32> = data.clone();
        expected.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let input = GpuTensor::from_host(&dev, &data,
            Shape::from_static(&[n]), DType::F32).unwrap();

        let result = topk(&dev, &input, 5, true).unwrap();

        println!("TopK(5, N=32000): values={:?}", result.values);
        assert!((result.values[0] - expected[0]).abs() < 1e-5);
        assert!((result.values[4] - expected[4]).abs() < 1e-5);
    }

    #[test]
    fn nms_basic() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // Boxes with clear overlaps: IoU(0,1)≈0.68, IoU(0,3)≈0.47, IoU(2,4)≈0.68
        #[rustfmt::skip]
        let boxes_data: Vec<f32> = vec![
            0.0,  0.0,  10.0, 10.0,  // box 0: high score
            1.0,  1.0,  11.0, 11.0,  // box 1: IoU≈0.68 with 0 → suppressed
            50.0, 50.0, 60.0, 60.0,  // box 2: separate object
            0.5,  0.5,  10.5, 10.5,  // box 3: IoU≈0.82 with 0 → suppressed
            51.0, 51.0, 61.0, 61.0,  // box 4: IoU≈0.68 with 2 → suppressed
        ];
        let scores_data: Vec<f32> = vec![0.9, 0.8, 0.7, 0.6, 0.5];

        let boxes = GpuTensor::from_host(&dev, &boxes_data,
            Shape::from_static(&[5, 4]), DType::F32).unwrap();
        let scores = GpuTensor::from_host(&dev, &scores_data,
            Shape::from_static(&[5]), DType::F32).unwrap();

        let result = nms(&cache, &dev, &boxes, &scores, 0.5, 0.1, 100).unwrap();

        println!("NMS (5 boxes, IoU=0.5): kept {:?} ({} boxes)", result.keep_indices, result.num_kept);

        assert!(result.keep_indices.contains(&0), "Box 0 should be kept");
        assert!(result.keep_indices.contains(&2), "Box 2 should be kept");
        assert!(!result.keep_indices.contains(&1), "Box 1 should be suppressed");
        assert!(!result.keep_indices.contains(&3), "Box 3 should be suppressed");
        assert_eq!(result.num_kept, 2);
    }

    #[test]
    fn nms_score_threshold() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        let boxes_data: Vec<f32> = vec![
            0.0, 0.0, 10.0, 10.0,
            20.0, 20.0, 30.0, 30.0,
            40.0, 40.0, 50.0, 50.0,
        ];
        let scores_data: Vec<f32> = vec![0.9, 0.3, 0.8];

        let boxes = GpuTensor::from_host(&dev, &boxes_data,
            Shape::from_static(&[3, 4]), DType::F32).unwrap();
        let scores = GpuTensor::from_host(&dev, &scores_data,
            Shape::from_static(&[3]), DType::F32).unwrap();

        let result = nms(&cache, &dev, &boxes, &scores, 0.5, 0.5, 100).unwrap();

        println!("NMS score_threshold=0.5: kept {:?}", result.keep_indices);
        assert_eq!(result.num_kept, 2);
        assert!(result.keep_indices.contains(&0));
        assert!(result.keep_indices.contains(&2));
    }

    #[test]
    fn detection_pipeline() {
        let (dev, cache) = match setup() {
            Some(s) => s,
            None => { println!("No CUDA, skipping"); return; }
        };

        // YOLO-style: 1000 anchors → TopK(100) → generate boxes → NMS
        let n_anchors = 1000u32;

        let obj_scores: Vec<f32> = (0..n_anchors as usize)
            .map(|i| {
                if i < 20 { 0.5 + (i as f32 * 0.025) }
                else { ((i * 7 + 3) % 100) as f32 * 0.003 }
            })
            .collect();

        let input = GpuTensor::from_host(&dev, &obj_scores,
            Shape::from_static(&[n_anchors as usize]), DType::F32).unwrap();

        // TopK(100)
        let top = topk(&dev, &input, 100, true).unwrap();

        println!("\n=== Detection Pipeline ===");
        println!("  {n_anchors} anchors → TopK(100) → top score = {:.3}", top.values[0]);
        assert!(top.values[0] > 0.9);

        // Generate boxes for top candidates
        let n_top = 100;
        let boxes_data: Vec<f32> = (0..n_top).flat_map(|i| {
            let x = (i % 10) as f32 * 50.0;
            let y = (i / 10) as f32 * 50.0;
            vec![x, y, x + 30.0 + (i % 3) as f32, y + 30.0 + (i % 3) as f32]
        }).collect();

        let top_scores: Vec<f32> = top.values;
        let boxes = GpuTensor::from_host(&dev, &boxes_data,
            Shape::from_static(&[n_top, 4]), DType::F32).unwrap();
        let scores = GpuTensor::from_host(&dev, &top_scores,
            Shape::from_static(&[n_top]), DType::F32).unwrap();

        // NMS
        let nms_result = nms(&cache, &dev, &boxes, &scores, 0.45, 0.3, 50).unwrap();

        println!("  NMS (IoU=0.45, score>0.3): {n_top} → {} detections", nms_result.num_kept);
        println!("  Kept: {:?}", &nms_result.keep_indices[..nms_result.num_kept.min(10)]);

        assert!(nms_result.num_kept > 0);
        assert!(nms_result.num_kept < n_top);
        println!("  Detection pipeline complete!");
    }
}
