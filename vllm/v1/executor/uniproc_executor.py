# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from concurrent.futures import Future
from functools import cached_property
from multiprocessing import Lock
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class SeqLock:
    def __init__(self):
        from threading import Lock, Condition
        self._lock = Lock()
        self._cond = Condition(self._lock)
        self.next_seq = 1

    def acquire(self, seq: int):
        self._lock.acquire()
        while self.next_seq != seq:
            self._cond.wait()

    def release(self, increment: int = 1):
        self.next_seq += increment
        self._cond.notify_all()
        self._lock.release()



class AsyncOutputFuture(Future):
    def __init__(self, async_output: AsyncModelRunnerOutput, single_value: bool):
        self.async_output = async_output
        self.single_value = single_value
        super().__init__()

    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        if not super().done():
            try:
                output = self.async_output.get_output()
                self.set_result(output if self.single_value else [output])
            except Exception as e:
                self.set_exception(e)
        return super().result()


class UniProcExecutor(Executor):
    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        distributed_init_method, rank, local_rank = self._distributed_args()
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
            shared_worker_lock=Lock(),
        )

        self.driver_worker.init_worker(all_kwargs=[kwargs])
        self.driver_worker.init_device()

        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.driver_worker.elastic_ep_execute("load_model")
        else:
            self.driver_worker.load_model()
        current_platform.update_block_size_for_backend(self.vllm_config)

        from concurrent.futures import ThreadPoolExecutor
        import os
        self.baseline_mode = os.getenv("VLLM_BASELINE_CONCURRENCY") == "1"
        
        if self.baseline_mode:
            logger.info("Initializing UniProcExecutor in SYNCHRONOUS BASELINE MODE")
            self.worker_pools = [ThreadPoolExecutor(max_workers=1) for _ in range(1)]
            from threading import Lock as ThreadLock
            self.execution_lock = ThreadLock()
        else:
            logger.info("Initializing UniProcExecutor in OPTIMIZED PARALLEL OVERLAP MODE")
            self.worker_pools = [ThreadPoolExecutor(max_workers=1) for _ in range(2)]
            self.execution_lock = SeqLock()
        self._step_idx = 0

    def _distributed_args(self) -> tuple[str, int, int]:
        """Return (distributed_init_method, rank, local_rank)."""
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(":")
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        return distributed_init_method, 0, local_rank

    @cached_property
    def max_concurrent_batches(self) -> int:
        return 2 if self.scheduler_config.async_scheduling else 1

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
        skip_sample: bool = False,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            return result if single_value else [result]

        if self.baseline_mode:
            pool = self.worker_pools[0]
            step_id = self._step_idx
            
            def _run_baseline():
                import time
                t_submit = time.perf_counter()
                with self.execution_lock:
                    t_acq = time.perf_counter()
                    res = run_method(self.driver_worker, method, args, kwargs)
                    if hasattr(res, "get_output"):
                        ret = res.get_output()
                        t_out = time.perf_counter()
                        logger.info(f"[AGENT_METRIC_CONCURRENCY] step={step_id} | method={method} | lock_wait={(t_acq-t_submit)*1000:.3f}ms | run_dispatch={(t_out-t_acq)*1000:.3f}ms | total={(t_out-t_submit)*1000:.3f}ms")
                        return ret if single_value else [ret]
                    t_out = time.perf_counter()
                    return res if single_value else [res]
            return pool.submit(_run_baseline)

        pool = self.worker_pools[self._step_idx % 2]
        step_id = self._step_idx

        if method == "execute_model":
            seq = 2 * step_id - 1
            kwargs = dict(kwargs)
            kwargs["execution_lock"] = self.execution_lock
            kwargs["seq"] = seq
        elif method == "sample_tokens":
            seq = 2 * step_id
        else:
            seq = None

        def _run():
            import time
            t_submit = time.perf_counter()
            if seq is not None and method != "execute_model":
                self.execution_lock.acquire(seq)
            t_acq = time.perf_counter()
            
            try:
                res = run_method(self.driver_worker, method, args, kwargs)
            finally:
                t_dispatch = time.perf_counter()
                if seq is not None:
                    increment = 2 if skip_sample else 1
                    self.execution_lock.release(increment=increment)

            if hasattr(res, "get_output"):
                ret = res.get_output()
                t_out = time.perf_counter()
                logger.info(f"[AGENT_METRIC_CONCURRENCY] step={step_id} | method={method} | lock_wait={(t_acq-t_submit)*1000:.3f}ms | run_dispatch={(t_dispatch-t_acq)*1000:.3f}ms | tpu_wait={(t_out-t_dispatch)*1000:.3f}ms | total={(t_out-t_submit)*1000:.3f}ms")
                return ret if single_value else [ret]
            logger.info(f"[AGENT_METRIC_CONCURRENCY] step={step_id} | method={method} | lock_wait={(t_acq-t_submit)*1000:.3f}ms | run_dispatch={(t_dispatch-t_acq)*1000:.3f}ms | total={(t_dispatch-t_submit)*1000:.3f}ms")
            return res if single_value else [res]

        return pool.submit(_run)

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        self._step_idx += 1
        skip_sample = (
            (self.vllm_config.model_config.runner_type == "pooling")
            or scheduler_output.total_num_scheduled_tokens == 0
        )
        output = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
            skip_sample=skip_sample,
        )
        # In non-blocking mode, surface any exception as early as possible.
        if non_block and output.done():
            # Raise the exception in-line if the task failed.
            output.result()
        return output

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    def check_health(self) -> None:
        # UniProcExecutor will always be healthy as long as
        # it's running.
        return

    def shutdown(self) -> None:
        if worker := self.driver_worker:
            worker.shutdown()

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return True


class ExecutorWithExternalLauncher(UniProcExecutor):
    """An executor that uses external launchers to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.

    see https://github.com/vllm-project/vllm/issues/11400 for
    the motivation, and examples/features/torchrun/torchrun_example_offline.py
    for the usage example.

    The key idea: although it is tensor-parallel inference, we only
    create one worker per executor, users will launch multiple
    engines with torchrun-compatible launchers, and all these engines
    work together to process the same prompts. When scheduling is
    deterministic, all the engines will generate the same outputs,
    and they don't need to synchronize the states with each other.
    """

    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, (
            "To get deterministic execution, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )
        super()._init_executor()

    def _distributed_args(self) -> tuple[str, int, int]:
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return distributed_init_method, rank, local_rank

    def determine_available_memory(self) -> list[int]:  # in bytes
        # we need to get the min across all ranks.
        memory = super().determine_available_memory()
        from vllm.distributed.parallel_state import get_world_group

        cpu_group = get_world_group().cpu_group
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return [memory_tensor.item()]
