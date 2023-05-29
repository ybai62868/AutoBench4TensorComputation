import os
import json

from tvm import meta_schedule as ms
from tvm.target import Target

from tvm.meta_schedule import postproc
from tvm.meta_schedule import schedule_rule as M
from tvm.meta_schedule.tune import TuneConfig


def load_config():
    cur_path = os.path.dirname(__file__)
    config_path = os.path.join(cur_path, "configs")
    with open(config_path) as f:
        return json.load(f)


def sch_rules_tensor_core():
    return [
        M.MultiLevelTiling(
            structure="SSSRRSRS",
            tile_binds=["blockIdx.y", "blockIdx.x", "threadIdx.y"],
            use_tensor_core=True,
            max_innermost_factor=4,
            vector_load_lens=[1, 2, 4, 8],
            reuse_read=M.ReuseType(
                req="must",
                levels=[4],
                scope="shared.dyn",
            ),
            reuse_write=M.ReuseType(
                req="no",
                levels=[3],
                scope="shared.dyn",
            ),
        ),
        M.AutoInline(
            into_producer=True,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=False,
            require_ordered=False,
            disallow_op=None,
        ),
        M.AutoInline(
            into_producer=True,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=False,
            require_ordered=False,
            disallow_op=None,
        ),
        M.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
        M.ParallelizeVectorizeUnroll(
            max_jobs_per_core=-1,  # disable parallelize
            max_vectorize_extent=-1,  # disable vectorize
            unroll_max_steps=[0, 16, 64, 512, 1024],
            unroll_explicit=True,
        ),
    ]


def postprocs_tensor_core():
    return [
        postproc.RewriteCooperativeFetch(),
        postproc.RewriteUnboundBlock(),
        postproc.RewriteParallelVectorizeUnroll(),
        postproc.RewriteReductionBlock(),
        postproc.RewriteTensorCore(),
        postproc.VerifyGPUCode(),
    ]
