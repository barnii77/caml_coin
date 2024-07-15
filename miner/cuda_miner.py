import time
import json
import os
import queue
import threading
import math
import cupy as cp

from typing import List, Union, Optional, Dict, Set

from miner.miner_utils import thread_yield, sha256

_cuda_miner_code: Optional[str] = None
CUDA_MINER_FUNCTION_NAME = "mine_sha256"
AUTOTUNE_INPUTS = (
    1234,
    0x00000003FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF,
    1,
)
AUTOTUNE_PAUSE_AFTER_MEASUREMENTS = 0.1
_miner_input_pools = {}
ALIGNMENT_REQUIREMENT = 32


def copy_tensors(*args):
    return tuple(cp.copy(arg) if isinstance(arg, cp.ndarray) else arg for arg in args)


def move_tensors_to_device(*args, device_id: int):
    with cp.cuda.Device(device_id):
        return tuple(
            cp.asarray(arg) if isinstance(arg, cp.ndarray) else arg for arg in args
        )


class CudaDeviceLaunchConfig:
    def __init__(
        self,
        grid_dim: Union[tuple[int], tuple[int, int], tuple[int, int, int]],
        block_dim: Union[tuple[int], tuple[int, int], tuple[int, int, int]],
        template_params: dict[str, str] = None,
    ):
        if template_params is None:
            template_params = {}
        assert 1 <= len(grid_dim) <= 3, "grid must be 1, 2, or 3 dimensional"
        assert 1 <= len(block_dim) <= 3, "blocks must be 1, 2, or 3 dimensional"
        n_threads = math.prod(block_dim)
        n_blocks = math.prod(grid_dim)
        assert n_threads > 0, "n_threads must be positive"
        assert 1024 >= n_threads, "cannot have more than 1024 threads per block"
        assert n_blocks > 0, "n_blocks must be positive"
        self.n_blocks = n_blocks
        self.n_threads = n_threads
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.template_params = template_params

    def __repr__(self):
        return f"CudaDeviceLaunchConfig(grid_dim={self.grid_dim}, block_dim={self.block_dim}, template_params={self.template_params})"


class CudaMultiDeviceLaunchConfigAutotuneThread(threading.Thread):
    def __init__(
        self,
        kernel,
        autotune_configs: List[CudaDeviceLaunchConfig],
        kernel_arg_settings: List[tuple],
        device_id: int = 0,
        n_warmup_repeats: int = 8,
        n_measure_repeats: int = 32,
        timeout_ns: Optional[int] = None,
        check_cache: bool = True,
        save_to_cache: bool = True,
        show_progress: bool = False,
    ):
        super().__init__()
        self.best_config = None
        self.kernel = kernel
        self.autotune_configs = autotune_configs
        self.kernel_arg_settings = [
            move_tensors_to_device(*args, device_id=device_id)
            for args in kernel_arg_settings
        ]
        self.device_id = device_id
        self.n_warmup_repeats = n_warmup_repeats
        self.n_measure_repeats = n_measure_repeats
        self.timeout_ns = timeout_ns
        self.check_cache = check_cache
        self.save_to_cache = save_to_cache
        self.show_progress = show_progress

    def run(self):
        self.best_config = self.kernel.autotune(
            self.autotune_configs,
            self.kernel_arg_settings,
            self.device_id,
            self.n_warmup_repeats,
            self.n_measure_repeats,
            self.timeout_ns,
            self.check_cache,
            self.save_to_cache,
            self.show_progress,
        )


class CudaMultiDeviceLaunchConfig:
    def __init__(self, device_id_to_config: Dict[int, CudaDeviceLaunchConfig]):
        if not all(isinstance(k, int) for k in device_id_to_config.keys()):
            raise TypeError("device_id_to_config keys must be integers")
        if not all(
            isinstance(v, CudaDeviceLaunchConfig) for v in device_id_to_config.values()
        ):
            raise TypeError("device_id_to_config values must be CudaDeviceLaunchConfig")
        self.device_id_to_config = device_id_to_config

    def __repr__(self):
        return f"CudaMultiDeviceLaunchConfig(device_id_to_config={self.device_id_to_config})"

    @staticmethod
    def autotune_for_each_device(
        kernel: "KernelTemplate",
        autotune_configs: Dict[int, List["CudaDeviceLaunchConfig"]],
        kernel_arg_settings: List[tuple],
        n_warmup_repeats: int = 8,
        n_measure_repeats: int = 32,
        timeout_ns: Optional[int] = None,
        show_progress: bool = False,
    ) -> "CudaMultiDeviceLaunchConfig":
        # autotune every device config in parallel, then combine the best for each device
        best_configs = {}
        autotune_threads = [
            CudaMultiDeviceLaunchConfigAutotuneThread(
                kernel,
                autotune_configs[device_id],
                kernel_arg_settings,
                device_id=device_id,
                n_warmup_repeats=n_warmup_repeats,
                n_measure_repeats=n_measure_repeats,
                timeout_ns=timeout_ns,
                show_progress=show_progress,
            )
            for device_id in autotune_configs.keys()
        ]
        for thread in autotune_threads:
            thread.start()
        for thread in autotune_threads:
            thread.join()
            best_configs[thread.device_id] = thread.best_config
        return CudaMultiDeviceLaunchConfig(best_configs)


class KernelTemplate:
    def __init__(self, raw_kernel_code: str, _kernel_lru_cache_max_size=1):
        self.raw_kernel_code = raw_kernel_code
        self._kernel_lru_cache_keys = []
        self._kernel_lru_cache_values = []
        self._kernel_lru_cache_max_size = _kernel_lru_cache_max_size

    def __call__(
        self,
        launch_config: CudaDeviceLaunchConfig,
        args,
        device_id: int = 0,
    ):
        if launch_config in self._kernel_lru_cache_keys:
            idx = self._kernel_lru_cache_keys.index(launch_config)
            kernel = self._kernel_lru_cache_values[idx]
            self._kernel_lru_cache_keys.pop(idx)
            self._kernel_lru_cache_values.pop(idx)
            self._kernel_lru_cache_keys.append(launch_config)
            self._kernel_lru_cache_values.append(kernel)
        else:
            nvrtc_options = (
                tuple(
                    self.raw_kernel_code[
                        len("// NVRTC-OPTIONS: ") : self.raw_kernel_code.find("\n")
                    ].split()
                )
                if self.raw_kernel_code.startswith("// NVRTC-OPTIONS: ")
                else ()
            )
            kernel_code = (
                "\n".join(
                    map(
                        lambda p: f"#define {p[0]} ({p[1]})",
                        launch_config.template_params.items(),
                    )
                )
                + "\n\n"
                + self.raw_kernel_code
            )
            kernel = cp.RawModule(code=kernel_code, options=nvrtc_options).get_function(
                CUDA_MINER_FUNCTION_NAME
            )
            if len(self._kernel_lru_cache_keys) >= self._kernel_lru_cache_max_size:
                self._kernel_lru_cache_keys.pop(0)
                self._kernel_lru_cache_values.pop(0)
            self._kernel_lru_cache_keys.append(launch_config)
            self._kernel_lru_cache_values.append(kernel)

        with cp.cuda.Device(device_id):
            if math.prod(launch_config.block_dim) > 1024:
                raise ValueError(
                    "Invalid launch config: product of CudaDeviceLaunchConfig.block_dim must not exceed 1024, but got",
                    math.prod(launch_config.block_dim),
                )
            kernel(launch_config.grid_dim, launch_config.block_dim, args)

    def autotune(
        self,
        autotune_configs: List[CudaDeviceLaunchConfig],
        kernel_arg_settings: List[tuple],
        device_id: int = 0,
        n_warmup_repeats: int = 8,
        n_measure_repeats: int = 32,
        timeout_ns: Optional[int] = None,
        check_cache: bool = True,
        save_to_cache: bool = True,
        show_progress: bool = False,
    ) -> CudaDeviceLaunchConfig:
        """
        Autotunes the kernel to find the best launch configuration.

        :param autotune_configs: List of CudaDeviceLaunchConfig objects to test.
        :param kernel_arg_settings: List of argument collections to test (and average across).
        :param device_id: The device to run the autotuning on.
        :param n_warmup_repeats: Number of times to run each configuration with each argument collection before measuring.
        :param n_measure_repeats: Number of times to run each configuration with each argument collection.
        :param timeout_ns: Optional timeout in nanoseconds after which a tested kernel run will be aborted
        :param check_cache: Whether to check the cache file .autotune_cache.json for the best configuration.
        :param save_to_cache: Whether to save the best configuration to the cache file.
        :param show_progress: Whether to print progress information.
        """
        device_name = cp.cuda.runtime.getDeviceProperties(device_id)["name"]
        # check_cache = check_cache and device_name != "UNAVAILABLE"
        # save_to_cache = save_to_cache and device_name != "UNAVAILABLE"
        cache_key = sha256(
            f"DEVICE_NAME: {device_name}\nDEVICE_ID: {device_id}\nCODE:\n{self.raw_kernel_code}".encode("utf-8")
        ).hex()
        if check_cache and os.path.exists(".autotune_cache.json"):
            with open(".autotune_cache.json") as f:
                cache = json.load(f)
            if cache_key in cache:
                return CudaDeviceLaunchConfig(
                    tuple(cache[cache_key][0]),
                    tuple(cache[cache_key][1]),
                    cache[cache_key][2],
                )

        total_times = [0] * len(autotune_configs)
        error_dummy_tensor = cp.zeros((1,))
        n_kernel_runs = (n_warmup_repeats + n_measure_repeats) * len(kernel_arg_settings) * len(autotune_configs)
        n_runs_completed = 0
        for j in range(n_warmup_repeats + n_measure_repeats):
            for args in kernel_arg_settings:
                for i, config in enumerate(autotune_configs):
                    if show_progress:
                        print(
                            f"Running kernel {n_runs_completed + 1}/{n_kernel_runs} for device {device_name} with id {device_id}, "
                            f"arg {kernel_arg_settings.index(args) + 1}/{len(kernel_arg_settings)}, "
                            f"config {i + 1}/{len(autotune_configs)}",
                        )
                    start_event, end_event = cp.cuda.Event(), cp.cuda.Event()
                    start = time.perf_counter_ns()
                    start_event.record()
                    self(config, copy_tensors(*args), device_id)
                    with cp.cuda.Device(device_id):
                        while (
                            timeout_ns is None
                            or time.perf_counter_ns() - start < timeout_ns
                        ) and not cp.cuda.get_current_stream().done:
                            try:
                                error_dummy_tensor[0].item()
                            except Exception as e:
                                raise RuntimeError(
                                    f"Aborting because an autotuning kernel crashed: {e}"
                                )
                            thread_yield()
                        end_event.record()
                        cp.cuda.runtime.deviceSynchronize()
                    if j >= n_warmup_repeats:
                        # cpu_elapsed_time = time.perf_counter_ns() - start
                        total_times[i] += cp.cuda.get_elapsed_time(
                            start_event, end_event
                        )
                    n_runs_completed += 1
                    time.sleep(AUTOTUNE_PAUSE_AFTER_MEASUREMENTS)
        top_config = autotune_configs[total_times.index(min(total_times))]

        if save_to_cache:
            if os.path.exists(".autotune_cache.json"):
                with open(".autotune_cache.json", "r") as f:
                    cache = json.load(f)
            else:
                cache = {}
            cache[cache_key] = [
                list(top_config.grid_dim),
                list(top_config.block_dim),
                top_config.template_params,
            ]
            with open(".autotune_cache.json", "w") as f:
                json.dump(cache, f)

        return top_config


class FixedLaunchConfigKernel:
    def __init__(
        self,
        kernel: "KernelTemplate",
        launch_config: Union[CudaDeviceLaunchConfig, CudaMultiDeviceLaunchConfig],
    ):
        self.kernel = kernel
        self.launch_config = launch_config
        self._is_multi_device = isinstance(launch_config, CudaMultiDeviceLaunchConfig)

    def __call__(
        self,
        args_per_device: Union[Dict[int, tuple], tuple],
        device_ids: Set[int] = None,
    ):
        if self._is_multi_device:
            if device_ids is None:
                device_ids = self.launch_config.device_id_to_config.keys()
            filtered_device_configs = {
                k: v
                for k, v in self.launch_config.device_id_to_config.items()
                if k in device_ids
            }
            for device_id, config in filtered_device_configs.items():
                self.kernel(config, args_per_device[device_id], device_id)
        else:
            assert (
                device_ids is None
            ), "device_ids must be None for single-device kernel"
            device_id = (
                0
                if isinstance(args_per_device, tuple)
                else next(iter(args_per_device.keys()))
            )
            args = (
                args_per_device
                if isinstance(args_per_device, tuple)
                else args_per_device[device_id]
            )
            self.kernel(
                self.launch_config,
                args,
                device_id,
            )


def init(miner_code_path="miner_template.cu"):
    global _cuda_miner_code
    with open(miner_code_path) as f:
        _cuda_miner_code = f.read()


def get_reasonable_autotune_configs() -> List[CudaDeviceLaunchConfig]:
    possible_thread_counts = [64, 128, 256, 512, 1024]
    possible_block_counts = [4, 8, 16, 32, 64, 128, 256, 512]
    return [
        CudaDeviceLaunchConfig((block_count,), (thread_count,))
        for block_count in possible_block_counts
        for thread_count in possible_thread_counts
    ][1:]  # remove 4, 64 (too small)


def get_reasonable_multi_device_autotune_configs() -> (
    Dict[int, List[CudaDeviceLaunchConfig]]
):
    reasonable_config = get_reasonable_autotune_configs()
    return {i: reasonable_config for i in range(cp.cuda.runtime.getDeviceCount())}


def get_cuda_miner_kernel() -> "KernelTemplate":
    assert (
        _cuda_miner_code is not None
    ), "init must be called before get_cuda_miner_kernel"
    return KernelTemplate(_cuda_miner_code)


def get_autotuned_cuda_miner_kernel(
    used_devices: List[int] = None,
    n_warmup_repeats=8,
    n_measure_repeats: int = 32,
    show_progress: bool = False,
) -> "FixedLaunchConfigKernel":
    if used_devices is None:
        used_devices = list(range(cp.cuda.runtime.getDeviceCount()))
    miner_kernel = get_cuda_miner_kernel()
    mem, const_in, max_valid_hash, nonce_out, init_nonce_ = (
        _get_miner_kernel_args_tensors(*AUTOTUNE_INPUTS)
    )
    if len(used_devices) == 1:
        configs = get_reasonable_autotune_configs()
        max_batch = max(map(lambda config: config.n_threads, configs)) * max(
            map(lambda config: config.n_blocks, configs)
        )
        kernel_args = (
            const_in,
            nonce_out,
            max_batch,
            max_batch,
            max_valid_hash,
            init_nonce_,
        )
        opt_config = miner_kernel.autotune(
            configs,
            [kernel_args],
            device_id=used_devices[0],
            n_warmup_repeats=n_warmup_repeats,
            n_measure_repeats=n_measure_repeats,
            show_progress=show_progress,
        )
    else:
        configs = get_reasonable_multi_device_autotune_configs()
        max_batch = max(map(lambda config: config.n_threads, configs)) * max(
            map(lambda config: config.n_blocks, configs)
        )
        kernel_args = (
            const_in,
            nonce_out,
            max_batch,
            max_batch,
            max_valid_hash,
            init_nonce_,
        )
        opt_config = CudaMultiDeviceLaunchConfig.autotune_for_each_device(
            miner_kernel,
            configs,
            [kernel_args],
            n_warmup_repeats=n_warmup_repeats,
            n_measure_repeats=n_measure_repeats,
            show_progress=show_progress,
        )

    return FixedLaunchConfigKernel(miner_kernel, opt_config)


def _get_miner_kernel_args_tensors(
    h: int, valid_block_max_hash: int, init_nonce: int
) -> tuple["cp.ndarray", "cp.ndarray", "cp.ndarray", "cp.ndarray", "cp.ndarray"]:
    if cp.cuda.Device().id not in _miner_input_pools:
        _miner_input_pools[cp.cuda.Device().id] = queue.Queue()
    _miner_input_pool = _miner_input_pools[cp.cuda.Device().id]
    if _miner_input_pool.empty():
        # reduce allocations by making one big array
        mem = cp.empty(12 + ALIGNMENT_REQUIREMENT // 8 - 1, dtype=cp.uint64)
    else:
        mem = _miner_input_pool.get_nowait()

    # offset to make stuff 32-byte aligned
    i = (
        ALIGNMENT_REQUIREMENT - mem.data.ptr % ALIGNMENT_REQUIREMENT
    ) % ALIGNMENT_REQUIREMENT
    const_in = mem[i : i + 4]
    max_valid_hash = mem[i + 4 : i + 8]
    nonce_out = mem[i + 8 : i + 10]
    init_nonce_ = mem[i + 10 : i + 12]
    uint64_max = 0xFFFFFFFFFFFFFFFF
    const_in[0] = h & uint64_max
    const_in[1] = (h >> 64) & uint64_max
    const_in[2] = (h >> 128) & uint64_max
    const_in[3] = (h >> 192) & uint64_max
    max_valid_hash[0] = valid_block_max_hash & uint64_max
    max_valid_hash[1] = (valid_block_max_hash >> 64) & uint64_max
    max_valid_hash[2] = (valid_block_max_hash >> 128) & uint64_max
    max_valid_hash[3] = (valid_block_max_hash >> 192) & uint64_max
    nonce_out[0] = 0
    nonce_out[1] = 0
    init_nonce_[0] = init_nonce & uint64_max
    init_nonce_[1] = (init_nonce >> 64) & uint64_max
    return mem, const_in, max_valid_hash, nonce_out, init_nonce_


def _mine_single_device(
    kernel: "FixedLaunchConfigKernel",
    h: int,
    valid_block_max_hash: int,
    kill_event=None,
    init_nonce: int = 1,
) -> Optional[int]:
    assert isinstance(kernel, FixedLaunchConfigKernel)
    assert (
        not kernel._is_multi_device
    ), "_mine_single_device requires kernel to be single-device"
    # prepare kernel input
    n_batch = kernel.launch_config.n_blocks * kernel.launch_config.n_threads
    mem, const_in, max_valid_hash, nonce_out, init_nonce_ = (
        _get_miner_kernel_args_tensors(h, valid_block_max_hash, init_nonce)
    )
    args = (const_in, nonce_out, n_batch, n_batch, max_valid_hash, init_nonce_)

    # run kernel
    kernel(args)
    kill_event_value = False
    while (
        not kill_event_value and nonce_out[0].item() == 0 and nonce_out[1].item() == 0
    ):
        thread_yield()
        kill_event_value = kill_event is not None and kill_event.is_set()
    if not kill_event_value:
        cp.cuda.runtime.deviceSynchronize()
        out = nonce_out[0].item() + (nonce_out[1].item() << 64)
        _miner_input_pools[cp.cuda.Device().id].put_nowait(mem)
        return out


def _mine_multi_device(
    kernel: "FixedLaunchConfigKernel",
    h: int,
    valid_block_max_hash: int,
    kill_event=None,
    init_nonce: int = 1,
) -> Optional[int]:
    assert isinstance(kernel, FixedLaunchConfigKernel)
    assert (
        kernel._is_multi_device
    ), "_mine_single_device requires kernel to be single-device"
    assert (
        kernel.launch_config.device_id_to_config
    ), "at least one device must be specified"
    # prepare kernel input
    arg_collections = {}
    n_batches = {}
    for device_id, config in kernel.launch_config.device_id_to_config.items():
        n_batches[device_id] = config.n_blocks * config.n_threads
    nonce_step_size = sum(n_batches.values())
    for device_id in kernel.launch_config.device_id_to_config.keys():
        with cp.cuda.Device(device_id):
            mem, const_in, max_valid_hash, nonce_out, init_nonce_ = (
                _get_miner_kernel_args_tensors(h, valid_block_max_hash, init_nonce)
            )
        n_batch_device = n_batches[device_id]
        args = (
            const_in,
            nonce_out,
            nonce_step_size,
            n_batch_device,
            max_valid_hash,
            init_nonce_,
        )
        arg_collections[device_id] = args
        init_nonce += n_batch_device

    # run kernel
    kernel(arg_collections)
    kill_event_value = False  # this avoids race conditions if kill_event is cleared before return at the bottom
    while not kill_event_value and all(
        nonce_out[0].item() == 0 and nonce_out[1].item() == 0
        for _, nonce_out, _, _, _, _ in arg_collections.values()
    ):
        thread_yield()
        kill_event_value = kill_event is not None and kill_event.is_set()

    found_nonce = None
    for device_id, nonce_out in arg_collections.items():
        nonce = nonce_out[0].item() + (nonce_out[1].item() << 64)
        if nonce != 0:
            found_nonce = nonce
            break

    # make all threads on all devices stop by telling them a nonce was found
    for args in arg_collections.values():
        nonce_buffer = args[1]
        nonce_buffer[0] = 1

    if not kill_event_value:
        for device_id in kernel.launch_config.device_id_to_config.keys():
            with cp.cuda.Device(device_id):
                cp.cuda.runtime.deviceSynchronize()
            _miner_input_pools[device_id].put_nowait(mem)
        return found_nonce  # may return None (if something went wrong)


def mine(
    kernel: "FixedLaunchConfigKernel",
    h: int,
    valid_block_max_hash: int,
    kill_event=None,
    init_nonce: int = 1,  # have to expose this because if I always start at 1, I will interfere with other miners
) -> Optional[int]:
    assert isinstance(kernel, FixedLaunchConfigKernel)
    if kernel._is_multi_device:
        return _mine_multi_device(
            kernel, h, valid_block_max_hash, kill_event, init_nonce
        )
    return _mine_single_device(kernel, h, valid_block_max_hash, kill_event, init_nonce)
