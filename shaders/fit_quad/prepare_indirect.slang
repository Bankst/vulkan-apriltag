// Writes VkDispatchIndirectCommand structs based on ClusterTotals.
// Single workgroup, single invocation.
#include "shared.slang"

[vk::push_constant]
ConstantBuffer<FitQuadPush> pc;

[vk::constant_id(0)]
const uint32_t WG_SIZE = 256;

struct DispatchIndirectCommand {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    const ClusterTotals totals = *(ClusterTotals *)pc.totals;
    DispatchIndirectCommand *cmds =
        (DispatchIndirectCommand *)(pc.scratch +
                                   FIT_QUAD_INDIRECT_OFFSET(capacity(pc)));

    const uint32_t sort_groups = 512;
    const uint32_t groups = 256;
    const uint32_t wg_size = WG_SIZE;
    const uint32_t scan_tile = wg_size * 8;

    const uint32_t sort_units = totals.sort_unit_count;
    const uint32_t points = totals.clustered_point_count;
    const uint32_t clusters = totals.cluster_count;
    const uint32_t tile_count = (points + scan_tile - 1) / scan_tile;
    const uint32_t point_groups = (points + wg_size - 1) / wg_size;

    // sort/merge/gather: cap at sort_groups, or actual sort_units if fewer
    cmds[FIT_QUAD_INDIRECT_SORT] = { min(sort_groups, max(sort_units, 1u)), 1, 1 };
    // scan_reduce: cap at groups, or tile_count if fewer
    cmds[FIT_QUAD_INDIRECT_SCAN_REDUCE] = { min(groups, max(tile_count, 1u)), 1, 1 };
    // scan_fused: same
    cmds[FIT_QUAD_INDIRECT_SCAN_FUSED] = { min(groups, max(tile_count, 1u)), 1, 1 };
    // window_errors/filter/maxima: cap at groups, or point_groups if fewer
    cmds[FIT_QUAD_INDIRECT_WINDOW_ERRORS] = { min(groups, max(point_groups, 1u)), 1, 1 };
    cmds[FIT_QUAD_INDIRECT_FILTER] = { min(groups, max(point_groups, 1u)), 1, 1 };
    cmds[FIT_QUAD_INDIRECT_MAXIMA] = { min(groups, max(point_groups, 1u)), 1, 1 };
    // search: one WG per cluster, cap at groups
    cmds[FIT_QUAD_INDIRECT_SEARCH] = { min(groups, max(clusters, 1u)), 1, 1 };
    // units: cap at groups
    const uint32_t unit_groups = (sort_units + wg_size - 1) / wg_size;
    cmds[FIT_QUAD_INDIRECT_UNITS] = { min(groups, max(unit_groups, 1u)), 1, 1 };
}
