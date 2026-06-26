import dataclasses
import ipaddress
import json
import logging
import multiprocessing
import os
import re
import time
from collections import Counter
from urllib.parse import quote

import requests
import sglang_router
from packaging.version import parse
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from .qwen3_5 import is_qwen35_model_path, maybe_prepare_qwen35_text_model, patch_sglang_qwen35
from slime.backends.sglang_utils.external import get_server_info
from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)


def get_base_gpu_id(args, rank):
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
        if getattr(args, "use_critic", False):
            num_critic_gpus = args.critic_num_gpus_per_node * args.critic_num_nodes
            start_index = (num_actor_gpus + num_critic_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id  # no remapping
    # CUDA_VISIBLE_DEVICES can be like "4,5,6,7"
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    # In a remapped process, valid torch device indices are 0..len(visible)-1
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    # If we're already getting local IDs, allow them
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )


def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:
    if getattr(server_args, "encoder_only", False):
        from sglang.srt.disaggregation.encode_server import launch_server_process as sglang_launch_server_process

        return sglang_launch_server_process(
            server_args,
            start_method="spawn",
            wait_for_server=True,
        )

    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")
    p = multiprocessing.Process(target=_launch_server_entry, args=(server_args,))
    p.start()

    if getattr(server_args, "node_rank", 0) != 0:
        return p

    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )

    return p


def _launch_server_entry(server_args: ServerArgs):
    try:
        patch_sglang_qwen35()
    except (ImportError, ModuleNotFoundError):
        pass
    from sglang.srt.entrypoints.http_server import launch_server

    launch_server(server_args)


def _wait_server_healthy(base_url, api_key, is_process_alive):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    with requests.Session() as session:
        while True:
            try:
                healthy = False
                for endpoint in ("/health", "/health_generate"):
                    response = session.get(f"{base_url}{endpoint}", headers=headers, timeout=5)
                    if response.status_code == 200:
                        healthy = True
                        break
                    if endpoint == "/health" and response.status_code == 404:
                        continue
                if healthy:
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)

        # Make sure the working queue is empty before offload or weight update.
        while True:
            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers, timeout=5)
                if response.status_code == 200:
                    break
                if response.status_code == 400 and "Cache flushed." in response.text:
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)


class SGLangEngine(RayActor):
    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine

    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host=None,
        disaggregation_bootstrap_port=None,
        router_ip=None,
        router_port=None,
    ):
        self.router_ip = router_ip if router_ip is not None else self.args.sglang_router_ip
        self.router_port = router_port if router_port is not None else self.args.sglang_router_port

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict, external_engine_need_check_fields = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
            sglang_overrides=self.sglang_overrides,
            num_gpus_per_engine=self.num_gpus_per_engine,
        )

        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]

        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict)

    def _init_external(self, expect_server_args, external_engine_need_check_fields):
        logger.info(f"Use external SGLang engine (rank={self.rank}, expect_server_args={expect_server_args})")

        def _sanity_check_server_args(actual_server_args, expect_server_args):
            for name in external_engine_need_check_fields:
                expect_value = expect_server_args.get(name)
                actual_value = actual_server_args.get(name)
                assert (
                    actual_value == expect_value
                ), f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"

        actual_server_args = get_server_info(f"http://{self.server_host}:{self.server_port}")
        _sanity_check_server_args(actual_server_args, expect_server_args)
        self._register_to_router(expect_server_args)

    def _init_normal(self, server_args_dict):
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        self.process = launch_server_process(ServerArgs(**server_args_dict))
        self._register_to_router(server_args_dict)

    def _register_to_router(self, server_args_dict):
        if self.worker_type == "encoder":
            return

        if self.node_rank == 0 and self.router_ip and self.router_port:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            if parse(sglang_router.__version__) <= parse("0.2.1"):
                assert self.worker_type == "regular", "pd disaggregation is not supported in old router."
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/add_worker?url={worker_url}",
                )
            else:
                payload = {
                    "url": worker_url,
                    "worker_type": self.worker_type,
                }
                if self.worker_type == "prefill":
                    bootstrap_port = server_args_dict.get("disaggregation_bootstrap_port")
                    if bootstrap_port is None:
                        raise RuntimeError(
                            f"Prefill worker {worker_url} does not have disaggregation_bootstrap_port; "
                            "cannot register it to the PD router."
                        )
                    payload["bootstrap_port"] = bootstrap_port
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json=payload,
                )
            response.raise_for_status()

    def _make_request(self, endpoint: str, payload: dict | None = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """
        if self.node_rank != 0:
            return True

        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/health_generate",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    def generation_quality_check(self, label: str = "") -> dict | None:
        if self.node_rank != 0:
            return None

        prompt = os.environ.get("ROLLOUT_GENERATION_QUALITY_GATE_PROMPT", "Answer exactly with the digit 2.")
        expected_regex = os.environ.get("ROLLOUT_GENERATION_QUALITY_GATE_EXPECTED_REGEX", r"^\s*2\s*$")
        max_tokens = int(os.environ.get("ROLLOUT_GENERATION_QUALITY_GATE_MAX_TOKENS", "16"))
        timeout = float(os.environ.get("ROLLOUT_GENERATION_QUALITY_GATE_TIMEOUT_SEC", "120"))
        max_repeat_ratio = float(os.environ.get("ROLLOUT_GENERATION_QUALITY_GATE_MAX_REPEAT_CHAR_RATIO", "0.8"))
        require_nonempty_content = str(
            os.environ.get("ROLLOUT_GENERATION_QUALITY_GATE_REQUIRE_NONEMPTY_CONTENT", "1")
        ).lower() in {"1", "true", "yes", "on"}
        extra_body = os.environ.get("ROLLOUT_GENERATION_QUALITY_GATE_EXTRA_BODY_JSON", "")

        payload = {
            "model": getattr(self.args, "model_name", None) or "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "stream": False,
        }
        if extra_body:
            try:
                extra_payload = json.loads(extra_body)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid ROLLOUT_GENERATION_QUALITY_GATE_EXTRA_BODY_JSON: {exc}") from exc
            if not isinstance(extra_payload, dict):
                raise ValueError("ROLLOUT_GENERATION_QUALITY_GATE_EXTRA_BODY_JSON must decode to an object")
            payload.update(extra_payload)

        url = f"http://{self.server_host}:{self.server_port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = getattr(self.args, "sglang_api_key", None)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if not isinstance(reasoning, str):
            reasoning = json.dumps(reasoning, ensure_ascii=False)
        finish_reason = choice.get("finish_reason")

        errors = []
        if require_nonempty_content and not content.strip():
            errors.append("empty content")
        if expected_regex and not re.search(expected_regex, content, flags=re.DOTALL):
            errors.append(f"content does not match {expected_regex!r}")

        repeat_text = "".join(ch for ch in (content + reasoning) if not ch.isspace())
        repeat_ratio = 0.0
        if repeat_text:
            repeat_ratio = max(Counter(repeat_text).values()) / len(repeat_text)
            if repeat_ratio > max_repeat_ratio:
                errors.append(f"repeat_char_ratio {repeat_ratio:.3f} exceeds {max_repeat_ratio:.3f}")

        return {
            "ok": not errors,
            "label": label,
            "engine_rank": self.rank,
            "url": url,
            "errors": errors,
            "finish_reason": finish_reason,
            "content_preview": content[:512],
            "reasoning_preview": reasoning[:512],
            "repeat_char_ratio": repeat_ratio,
            "usage": data.get("usage") or {},
        }

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        # flush cache will not return status_code 200 when there are pending requests
        for _ in range(60):
            try:
                response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
                if response.status_code == 200:
                    break
                logger.info(f"Error flushing cache: HTTP {response.status_code} {response.text!r}")
                time.sleep(1)
            except NewConnectionError as e:
                raise e
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                time.sleep(1)
                continue
        else:
            raise TimeoutError("Timeout while flushing cache.")

    def get_url(self):
        if self.node_rank != 0:
            return None
        return f"http://{self.server_host}:{self.server_port}"

    def shutdown(self):
        if self.args.rollout_external:
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        if self.worker_type != "encoder" and self.node_rank == 0:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            response = None
            if parse(sglang_router.__version__) <= parse("0.2.1"):
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/remove_worker?url=http://{self.server_host}:{self.server_port}"
                )
            elif parse(sglang_router.__version__) < parse("0.3.0"):
                worker_url = quote(worker_url, safe="")
                response = requests.delete(f"http://{self.router_ip}:{self.router_port}/workers/{worker_url}")
            else:
                try:
                    all_workers = requests.get(f"http://{self.router_ip}:{self.router_port}/workers").json()["workers"]
                    for worker in all_workers:
                        if worker["url"] == worker_url:
                            worker_id = worker["id"]
                            response = requests.delete(
                                f"http://{self.router_ip}:{self.router_port}/workers/{worker_id}"
                            )
                            break
                    else:
                        logger.warning(f"Worker {worker_url} not found in router during shutdown.")
                except Exception as e:
                    logger.warning(f"Failed to fetch workers list or remove worker: {e}")

            if response is not None:
                response.raise_for_status()
        kill_process_tree(self.process.pid)

    def get_weight_version(self):
        if self.node_rank != 0:
            return
        url = f"http://{self.server_host}:{self.server_port}/get_weight_version"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["weight_version"]

    def set_weight_version(self, new_version: str):
        """Bump the engine's recorded weight version without changing weights.

        Used by the delta-update path when a sync produced no bytes (e.g. an
        all-zero diff): we still need the engine's version to track the
        updater's, otherwise the CI version-equality check will trip.
        """
        return self._make_request("update_weight_version", {"new_version": str(new_version)})

    def release_memory_occupation(self):
        self.flush_cache()
        return self._make_request("release_memory_occupation")

    def resume_memory_occupation(self, tags: list[str] = None):
        """
        Available tags for multi-stage resume: weights, kv_cache
        """
        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    def check_weights(self, action: str):
        return self._make_request("weights_checker", {"action": action})

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str | None = None,
        weight_version: str | None = None,
        files: list[str] | None = None,
    ):
        """Reload weights from *model_path* without restarting the engine.

        Standard HF reload: ``model_path`` is the checkpoint directory.
        Delta (``load_format="delta"``): ``model_path`` is the parent of the
        per-sync version subdir and ``files`` is the basenames within it to read +
        apply. Each delta call is independent — sender owns batching, sync
        boundaries, cleanup.
        """
        payload: dict = {"model_path": model_path}
        if load_format is not None:
            payload["load_format"] = load_format
        if weight_version is not None:
            payload["weight_version"] = weight_version
        if files is not None:
            payload["files"] = files
        return self._make_request("update_weights_from_disk", payload)

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
            )
        except requests.exceptions.RequestException:
            # catch the case there the engine is just created and does not have the group.
            pass

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=False,
        weight_version: str | None = None,
        load_format: str | None = None,
        delta=None,
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        if load_format is not None:
            payload["load_format"] = load_format
        if delta is not None:
            # DeltaSpec → JSON string. Receiver reconstructs via DeltaEncoding(...) +
            # DeltaParam(**p); avoids depending on FastAPI's nested-dataclass coercion.
            import json
            from dataclasses import asdict

            payload["delta"] = json.dumps(
                {
                    "encoding": delta.encoding.value,
                    "params": [asdict(p) for p in delta.params],
                    "checksum": delta.checksum,
                }
            )
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def pause_generation(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/pause_generation", json={})
        response.raise_for_status()
        return response

    def continue_generation(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/continue_generation", json={})
        response.raise_for_status()
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.
        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """

        return self._make_request(
            "post_process_weights",
            {
                "restore_weights_before_load": restore_weights_before_load,
                "post_process_quantization": post_process_quantization,
            },
        )

    def start_profile(
        self,
        # The output directory
        output_dir: str | None = None,
        # If set, it profile as many as this number of steps.
        # If it is set, profiling is automatically stopped after this step, and
        # the caller doesn't need to run stop_profile.
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/start_profile",
            json={
                "output_dir": output_dir,
                "start_step": start_step,
                "num_steps": num_steps,
                "activities": activities,
                "profile_by_stage": profile_by_stage,
                "with_stack": with_stack,
                "record_shapes": record_shapes,
            },
        )
        response.raise_for_status()
        return response

    def stop_profile(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/stop_profile", json={})
        response.raise_for_status()
        return response

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return

        logger.info(f"Simulating crash on engine {self.server_host}:{self.server_port}...")
        self.shutdown()


def _compute_server_args(
    args,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
    sglang_overrides: dict | None = None,
    num_gpus_per_engine: int | None = None,
):
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    original_model_path = getattr(args, "rollout_model_path", None) or args.hf_checkpoint
    model_path = maybe_prepare_qwen35_text_model(
        original_model_path,
        language_only=getattr(args, "sglang_language_only", False),
    )
    server_language_only = getattr(args, "sglang_language_only", False)
    # Once Qwen3.5 has been materialized as a text-only shadow checkpoint, stop
    # forwarding language_only; new SGLang treats it as encoder disaggregation.
    if model_path != original_model_path and is_qwen35_model_path(model_path):
        server_language_only = False
    if is_qwen35_model_path(model_path) or is_qwen35_model_path(original_model_path):
        os.environ["SLIME_ENABLE_QWEN35_SGLANG_PATCH"] = "1"
        os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "slime_plugins.sglang_models"

    nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    base = _to_local_gpu_id(base)
    kwargs = {
        "model_path": model_path,
        "trust_remote_code": True,
        "random_seed": args.seed + rank,
        # memory
        "enable_memory_saver": args.offload_rollout,
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,
        # parallel
        "tp_size": _gpus_per_engine // args.sglang_pp_size,
        "dp_size": args.sglang_dp_size,
        "pp_size": args.sglang_pp_size,
        "ep_size": args.sglang_ep_size,
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
        # Always enable Prometheus metrics so the router /engine_metrics endpoint
        # is available for external scraping.
        "enable_metrics": True,
    }

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs["load_balance_method"] = "follow_bootstrap_room"
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True
    elif worker_type == "encoder":
        kwargs["encoder_only"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    server_arg_fields = dataclasses.fields(ServerArgs)
    server_arg_field_names = {attr.name for attr in server_arg_fields}
    unused_keys = set(kwargs.keys())
    for attr in server_arg_fields:
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if attr.name == "language_only":
            kwargs[attr.name] = server_language_only
            unused_keys.discard(attr.name)
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)

    # Per-server-group overrides from --sglang-config YAML.
    # Applied after base args so they take highest priority.
    if sglang_overrides:
        for key, value in sglang_overrides.items():
            normalized_key = key.replace("-", "_")
            if normalized_key != key:
                logger.warning(
                    f"sglang_overrides key '{key}' normalized to '{normalized_key}' (rank={rank}). "
                    "Please use underscore style in YAML overrides."
                )
            if normalized_key in kwargs:
                logger.info(
                    f"sglang_overrides: overriding {normalized_key}={kwargs[normalized_key]} -> {value} (rank={rank})"
                )
            kwargs[normalized_key] = value
            if normalized_key in server_arg_field_names:
                unused_keys.discard(normalized_key)
            else:
                unused_keys.add(normalized_key)

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    return kwargs, external_engine_need_check_fields


_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    "model_path",
    "trust_remote_code",
    "random_seed",
    "host",
    "port",
    "nccl_port",
    "nnodes",
    "node_rank",
    "dist_init_addr",
    "gpu_id_step",
    "base_gpu_id",
    "tp_size",
    "dp_size",
    "pp_size",
    "ep_size",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "enable_metrics",
    "mem_fraction_static",
]
