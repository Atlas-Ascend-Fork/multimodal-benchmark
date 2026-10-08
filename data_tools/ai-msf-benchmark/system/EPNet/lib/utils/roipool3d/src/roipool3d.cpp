#include <algorithm>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.options().device().is_cuda(), #x, " must be a CUDA tensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

void roipool3dLauncher_slow(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num, 
                           const float *xyz, const float *boxes3d, const float *pts_feature, float *pooled_features, int *pooled_empty_flag);

void roipool3dLauncher(int batch_size, int pts_num, int boxes_num, int feature_in_len, int sampled_pts_num, 
                       const float *xyz, const float *boxes3d, const float *pts_feature, float *pooled_features, int *pooled_empty_flag);

struct RoiPoolInputs {
    at::Tensor xyz;
    at::Tensor boxes3d;
    at::Tensor pts_feature;
    at::Tensor pooled_features;
    at::Tensor pooled_empty_flag;
};

using RoiPoolLauncher = void (*)(int, int, int, int, int, const float *,
                                 const float *, const float *, float *, int *);

int run_roipool3d(const RoiPoolInputs &inputs, RoiPoolLauncher launcher) {
    // params xyz: (B, N, 3)
    // params boxes3d: (B, M, 7)
    // params pts_feature: (B, N, C)
    // params pooled_features: (B, M, 512, 3+C)
    // params pooled_empty_flag: (B, M)
    CHECK_INPUT(inputs.xyz);
    CHECK_INPUT(inputs.boxes3d);
    CHECK_INPUT(inputs.pts_feature);
    CHECK_INPUT(inputs.pooled_features);
    CHECK_INPUT(inputs.pooled_empty_flag);

    int batch_size = inputs.xyz.size(0);
    int pts_num = inputs.xyz.size(1);
    int boxes_num = inputs.boxes3d.size(1);
    int feature_in_len = inputs.pts_feature.size(2);
    int sampled_pts_num = inputs.pooled_features.size(2);

    const float * xyz_data = inputs.xyz.data_ptr<float>();
    const float * boxes3d_data = inputs.boxes3d.data_ptr<float>();
    const float * pts_feature_data = inputs.pts_feature.data_ptr<float>();
    float * pooled_features_data = inputs.pooled_features.data_ptr<float>();
    int * pooled_empty_flag_data = inputs.pooled_empty_flag.data_ptr<int>();

    launcher(batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num,
             xyz_data, boxes3d_data, pts_feature_data, pooled_features_data,
             pooled_empty_flag_data);

    return 1;
}

int roipool3d_gpu_slow(at::Tensor xyz, at::Tensor boxes3d, at::Tensor pts_feature,
                       at::Tensor pooled_features, at::Tensor pooled_empty_flag) {
    return run_roipool3d(
        {xyz, boxes3d, pts_feature, pooled_features, pooled_empty_flag},
        roipool3dLauncher_slow);
}

int roipool3d_gpu(at::Tensor xyz, at::Tensor boxes3d, at::Tensor pts_feature,
                  at::Tensor pooled_features, at::Tensor pooled_empty_flag) {
    return run_roipool3d(
        {xyz, boxes3d, pts_feature, pooled_features, pooled_empty_flag},
        roipool3dLauncher);
}


int pt_in_box3d_cpu(float x, float y, float z, float cx, float bottom_y, float cz, float h, float w, float l, float angle){
    float max_dis = 10.0, x_rot, z_rot, cosa, sina, cy;
    int in_flag;
    cy = bottom_y - h / 2.0;
    if ((fabsf(x - cx) > max_dis) || (fabsf(y - cy) > h / 2.0) || (fabsf(z - cz) > max_dis)){
        return 0;
    }
    cosa = cos(angle); sina = sin(angle);
    x_rot = (x - cx) * cosa + (z - cz) * (-sina);
    z_rot = (x - cx) * sina + (z - cz) * cosa;

    in_flag = (x_rot >= -l / 2.0) & (x_rot <= l / 2.0) & (z_rot >= -w / 2.0) & (z_rot <= w / 2.0);
    return in_flag;
}

int pts_in_boxes3d_cpu(at::Tensor pts_flag, at::Tensor pts, at::Tensor boxes3d){
    // param in_flag: (M, N), 0 or 1
    // param pts: (N, 3)
    // param boxes3d: (M, 7)  [x, y, z, h, w, l, ry]

    CHECK_CONTIGUOUS(pts_flag);
    CHECK_CONTIGUOUS(pts);
    CHECK_CONTIGUOUS(boxes3d);

    long boxes_num = boxes3d.size(0);
    long pts_num = pts.size(0);

    int64_t * pts_flag_flat = pts_flag.data_ptr<int64_t>();
    float * pts_flat = pts.data_ptr<float>();
    float * boxes3d_flat = boxes3d.data_ptr<float>();

    std::fill_n(pts_flag_flat, static_cast<int64_t>(boxes_num) * pts_num, 0L);

    int i, j, cur_in_flag;
    for (i = 0; i < boxes_num; i++){
        for (j = 0; j < pts_num; j++){
            cur_in_flag = pt_in_box3d_cpu(pts_flat[j * 3], pts_flat[j * 3 + 1], pts_flat[j * 3 + 2], boxes3d_flat[i * 7],
                                          boxes3d_flat[i * 7 + 1], boxes3d_flat[i * 7 + 2], boxes3d_flat[i * 7 + 3],
                                          boxes3d_flat[i * 7 + 4], boxes3d_flat[i * 7 + 5], boxes3d_flat[i * 7 + 6]);
            pts_flag_flat[i * pts_num + j] = cur_in_flag;
        }
    }
    return 1;
}

struct RoiPoolCpuData {
    float *pts;
    float *boxes;
    float *features;
    float *pooled_pts;
    float *pooled_features;
    int64_t pts_num;
    int64_t feature_len;
    int64_t sampled_pts_num;
};

void copy_pooled_point(const RoiPoolCpuData &data, int64_t box_idx, int64_t point_idx, int64_t sample_idx) {
    int64_t output_idx = box_idx * data.sampled_pts_num * 3 + sample_idx * 3;
    for (int64_t k = 0; k < 3; ++k) data.pooled_pts[output_idx + k] = data.pts[point_idx * 3 + k];
    output_idx = box_idx * data.sampled_pts_num * data.feature_len + sample_idx * data.feature_len;
    for (int64_t k = 0; k < data.feature_len; ++k) {
        data.pooled_features[output_idx + k] = data.features[point_idx * data.feature_len + k];
    }
}

void duplicate_pooled_points(const RoiPoolCpuData &data, int64_t box_idx, int64_t count) {
    if (count <= 0) return;
    for (int64_t j = count; j < data.sampled_pts_num; ++j) {
        int64_t output_idx = box_idx * data.sampled_pts_num * 3 + j * 3;
        int64_t source_idx = box_idx * data.sampled_pts_num * 3
            + (j % std::max<int64_t>(count, 1)) * 3;
        for (int64_t k = 0; k < 3; ++k) data.pooled_pts[output_idx + k] = data.pooled_pts[source_idx + k];
        output_idx = box_idx * data.sampled_pts_num * data.feature_len + j * data.feature_len;
        source_idx = box_idx * data.sampled_pts_num * data.feature_len
            + (j % std::max<int64_t>(count, 1)) * data.feature_len;
        for (int64_t k = 0; k < data.feature_len; ++k) {
            data.pooled_features[output_idx + k] = data.pooled_features[source_idx + k];
        }
    }
}

int64_t pool_single_box(const RoiPoolCpuData &data, int64_t box_idx) {
    int64_t count = 0;
    for (int64_t j = 0; j < data.pts_num && count < data.sampled_pts_num; ++j) {
        int64_t box_offset = box_idx * 7;
        int inside = pt_in_box3d_cpu(data.pts[j * 3], data.pts[j * 3 + 1], data.pts[j * 3 + 2],
                                     data.boxes[box_offset], data.boxes[box_offset + 1],
                                     data.boxes[box_offset + 2], data.boxes[box_offset + 3],
                                     data.boxes[box_offset + 4], data.boxes[box_offset + 5],
                                     data.boxes[box_offset + 6]);
        if (inside) copy_pooled_point(data, box_idx, j, count++);
    }
    return count;
}

int roipool3d_cpu(at::Tensor pts, at::Tensor boxes3d, at::Tensor pts_feature, at::Tensor pooled_pts,
                  at::Tensor pooled_features, at::Tensor pooled_empty_flag) {
    CHECK_CONTIGUOUS(pts);
    CHECK_CONTIGUOUS(boxes3d);
    CHECK_CONTIGUOUS(pts_feature);
    CHECK_CONTIGUOUS(pooled_pts);
    CHECK_CONTIGUOUS(pooled_features);
    CHECK_CONTIGUOUS(pooled_empty_flag);

    int64_t boxes_num = boxes3d.size(0);
    RoiPoolCpuData data = {pts.data<float>(), boxes3d.data<float>(), pts_feature.data<float>(),
                           pooled_pts.data<float>(), pooled_features.data<float>(), pts.size(0),
                           pts_feature.size(1), pooled_pts.size(1)};
    long *pooled_empty = pooled_empty_flag.data<long>();
    std::fill_n(pooled_empty, boxes_num, 0L);
    for (int64_t i = 0; i < boxes_num; ++i) {
        int64_t count = pool_single_box(data, i);
        if (count == 0) {
            pooled_empty[i] = 1;
        } else if (count < data.sampled_pts_num) {
            duplicate_pooled_points(data, i, count);
        }
    }
    return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pts_in_boxes3d_cpu", &pts_in_boxes3d_cpu, "pts_in_boxes3d_cpu");
    m.def("roipool3d_cpu", &roipool3d_cpu, "roipool3d_cpu");
    m.def("forward", &roipool3d_gpu, "roipool3d forward (CUDA)");
    m.def("forward_slow", &roipool3d_gpu_slow, "roipool3d forward (CUDA)");
}

