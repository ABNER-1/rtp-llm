import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.async_decoder_engine.rpc_engine import LanguageCppEngine
from rtp_llm.lora.lora_exception import LoraCountException, LoraException
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.utils.time_util import Timer


class LoraManager:
    thread_lock_ = threading.Lock()
    lora_infos_: Dict[str, str]

    engine_: BaseEngine
    lora_cpp_wrapper_: Any
    weights_loader_: ModelLoader

    def __init__(
        self, engine: BaseEngine, max_lora_model_size: int = -1, local_rank: int = 0
    ) -> None:
        self.engine_ = engine
        self.lora_infos_ = {}
        self.max_lora_model_size_ = max_lora_model_size
        self.device: str = f"cuda:{local_rank}"
        assert isinstance(self.engine_, LanguageCppEngine)
        self.lora_cpp_wrapper_ = self.engine_.rtp_llm_op_.ft_op
        assert isinstance(self.engine_.model.model_weights_loader, ModelLoader)
        self.weights_loader_ = self.engine_.model.model_weights_loader
        with Timer() as timer:
            model_lora_infos = self.engine_.model.model_config.lora_infos
            if model_lora_infos is not None and len(model_lora_infos) > 1:
                logging.info(f"model_lora_infos is {model_lora_infos}")
                self._batch_load_loras(model_lora_infos)
        logging.info(f"update lora weights time: {timer.cost_ms() / 1000 :.2f} s")

    def _check_loraInfo_size(self, lora_infos: Dict[str, str]):
        if (
            self.max_lora_model_size_ != -1
            and len(lora_infos) > self.max_lora_model_size_
        ):
            raise LoraCountException(
                f"lora_infos[{lora_infos}]'s size exceed MAX_LORA_MODEL_SIZE[{self.max_lora_model_size_}]"
            )

    def get_add_lora_map(self, lora_infos: Dict[str, str]) -> Dict[str, str]:
        with self.thread_lock_:
            self._check_loraInfo_size(lora_infos)
            add_lora_map: Dict[str, str] = {}
            for adapter_name, lora_path in lora_infos.items():
                if (
                    adapter_name not in self.lora_infos_
                    or lora_path != self.lora_infos_[adapter_name]
                ):
                    add_lora_map[adapter_name] = lora_path
            return add_lora_map

    def get_remove_lora_map(self, lora_infos: Dict[str, str]) -> Dict[str, str]:
        with self.thread_lock_:
            self._check_loraInfo_size(lora_infos)
            remove_lora_map: Dict[str, str] = {}
            for adapter_name, lora_path in self.lora_infos_.items():
                if (
                    adapter_name not in lora_infos
                    or lora_path != lora_infos[adapter_name]
                ):
                    remove_lora_map[adapter_name] = lora_path
            return remove_lora_map

    def _batch_load_loras(self, model_lora_infos: Dict[str, str]):
        """Batch load multiple LoRA adapters with parallel I/O.

        Phase 1: Serial database registration - register all adapters to the database
                 sequentially (each is fast: file discovery + config parsing).
        Phase 2: Parallel I/O preload + weight assembly - each adapter's safetensors
                 file is loaded in parallel via preload_lora_tensors(), then layer
                 weights are assembled from the in-memory cache.
        Phase 3: Serial cleanup + C++ registration - remove database entries and
                 register weights to the C++ engine sequentially.
        """
        database = self.weights_loader_._load_config.database
        num_layers = self.weights_loader_._load_config.num_layers
        weight_style = self.weights_loader_._weights_info.weight_style

        # Phase 1: Serial database registration (each is fast: ~0.01s)
        lora_configs = {}
        for adapter_name, lora_path in model_lora_infos.items():
            database.load_lora(adapter_name, lora_path)
            lora_configs[adapter_name] = database.get_lora_config(adapter_name)
            logging.info(
                f"registered adapter to database: {adapter_name}, "
                f"rank={lora_configs[adapter_name].rank}"
            )

        # Phase 2: Parallel I/O preload + weight assembly
        def _load_adapter_weights(adapter_name: str, lora_path: str):
            """Load a single adapter's weights from pre-registered database entry."""
            from rtp_llm.lora.lora_weights import LoRAWeights
            from rtp_llm.utils.model_weight import WeightStyle

            lora_config = lora_configs[adapter_name]
            lora_weights = LoRAWeights(num_layers)
            lora_weights.set_lora_rank(lora_config.rank)

            if weight_style == WeightStyle.RTP_LLM_STYLE:
                raise ValueError("load_lora_weights only support non-ft-style weight")

            # Batch I/O: one load_tensors() call instead of 640+ safe_open() calls
            tensor_cache = database.preload_lora_tensors(adapter_name, "cpu")
            logging.info(
                f"preloaded {len(tensor_cache)} tensors for adapter {adapter_name}"
            )

            for layer_id in range(num_layers):
                result = self.weights_loader_._load_layer_lora_weights(
                    adapter_name, layer_id, "cpu", tensor_cache=tensor_cache
                )
                for name, tensor in result.items():
                    lora_weights.set_layer_weight(False, layer_id, name, tensor)

            lora_weights.apply_scale(lora_config.lora_alpha / lora_config.rank)
            del tensor_cache
            return adapter_name, lora_path, lora_weights

        max_workers = min(4, len(model_lora_infos))
        loaded_results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_load_adapter_weights, k, v): k
                for k, v in model_lora_infos.items()
            }
            for future in as_completed(futures):
                adapter_name = futures[future]
                try:
                    name, path, weights = future.result()
                    loaded_results[name] = (path, weights)
                    logging.info(f"parallel loaded adapter: {name}")
                except Exception as e:
                    logging.error(f"failed to load adapter {adapter_name}: {e}")
                    raise

        # Phase 3: Serial cleanup + C++ registration
        for adapter_name in model_lora_infos:
            database.remove_lora(adapter_name)

        for adapter_name, (lora_path, weights) in loaded_results.items():
            self.lora_infos_[adapter_name] = lora_path
            self.lora_cpp_wrapper_.add_lora(
                adapter_name, weights.lora_a_weights, weights.lora_b_weights
            )
            logging.info(f"registered adapter to C++ engine: {adapter_name}")

    def add_lora(self, adapter_name: str, lora_path: str) -> Optional[LoraException]:
        with self.thread_lock_:
            assert adapter_name not in self.lora_infos_.keys()
            self.lora_infos_[adapter_name] = lora_path
            weights = self.weights_loader_.load_lora_weights(
                adapter_name, lora_path, "cpu"
            )
            self.lora_cpp_wrapper_.add_lora(
                adapter_name, weights.lora_a_weights, weights.lora_b_weights
            )

    def remove_lora(self, adapter_name: str) -> Optional[LoraException]:
        with self.thread_lock_:
            assert adapter_name in self.lora_infos_.keys()
            del self.lora_infos_[adapter_name]
            self.lora_cpp_wrapper_.remove_lora(adapter_name)
