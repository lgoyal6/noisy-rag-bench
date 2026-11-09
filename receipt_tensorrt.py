"""Bounded TensorRT export feasibility, decision parity and paired GPU timing.

Called with the already loaded model/processor. Parser failures are retained as
conversion evidence; successful builds still require numerical and task parity.
"""
import hashlib
import json
import statistics
import time


def compare_export(onnx_path, wrapper, processor, samples, output_dir, policy):
    import torch
    from receipt_model_study import encode_sample, _prediction_record
    record = {"onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
              "status": "not_started", "logit_atol": .05,
              "task_decision_rule": "Same field values, source boxes, currency and candidate counts on 16 frozen development receipts",
              "speedup_claim": False, "samples": []}
    started = time.perf_counter()
    try:
        import tensorrt as trt
    except ImportError:
        record.update(status="blocked_tensorrt_python_unavailable", engine_built=False)
        (output_dir / "tensorrt-study.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        record.update(status="unsupported_onnx_operators", engine_built=False,
                      parser_errors=[str(parser.get_error(i)) for i in range(parser.num_errors)])
        (output_dir / "tensorrt-study.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    shapes = {"input_ids": (1, 512), "bbox": (1, 512, 4),
              "attention_mask": (1, 512), "pixel_values": (1, 3, 224, 224)}
    for i in range(network.num_inputs):
        name = network.get_input(i).name
        if name not in shapes:
            raise ValueError("unexpected TensorRT input " + name)
        shape = shapes[name]
        profile.set_shape(name, shape, shape, shape)
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        record.update(status="engine_build_failed", engine_built=False)
        (output_dir / "tensorrt-study.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    engine_path = output_dir / "layoutlmv3.engine"
    engine_path.write_bytes(bytes(plan))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    record.update(engine_built=True, engine_sha256=hashlib.sha256(engine_path.read_bytes()).hexdigest(),
                  build_seconds=time.perf_counter() - started, tensorrt_version=trt.__version__)

    def execute(inputs):
        buffers = {}
        for name, value in inputs.items():
            value = value.contiguous()
            expected = torch.from_numpy(__import__("numpy").empty((), dtype=trt.nptype(engine.get_tensor_dtype(name)))).dtype
            if value.dtype != expected:
                value = value.to(expected)
            buffers[name] = value
            context.set_input_shape(name, tuple(value.shape))
            context.set_tensor_address(name, value.data_ptr())
        output_name = "logits"
        dtype = torch.from_numpy(__import__("numpy").empty((), dtype=trt.nptype(engine.get_tensor_dtype(output_name)))).dtype
        output = torch.empty(tuple(context.get_tensor_shape(output_name)), device="cuda", dtype=dtype)
        context.set_tensor_address(output_name, output.data_ptr())
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT enqueue failed")
        return output

    def time_call(call):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        result = call()
        b.record()
        b.synchronize()
        return result, a.elapsed_time(b)

    torch_times, trt_times = [], []
    with torch.inference_mode():
        for sample in samples:
            if not sample.input_supported or not sample.ocr_words:
                continue
            enc, word_ids = encode_sample(processor, sample)
            inputs = {key: enc[key].cuda() for key in shapes}
            for _ in range(3):
                wrapper(**inputs)
                execute(inputs)
            reference = wrapper(**inputs)
            candidate = execute(inputs)
            torch.cuda.synchronize()
            error = float((reference.float() - candidate.float()).abs().max())
            a = _prediction_record(sample, reference, word_ids, sample.boxes, source="native")
            b = _prediction_record(sample, candidate, word_ids, sample.boxes, source="tensorrt")
            keys = ["field", "value", "currency", "box", "unique_candidate_count"]
            task_same = [{k: field.get(k) for k in keys} for field in a["fields"]] == [{k: field.get(k) for k in keys} for field in b["fields"]]
            from receipt_calibration import probability, supported
            decisions_a = [f["field"] in policy["autoaccept_fields"] and supported(f) and probability(policy["isotonic"], f["confidence"]) >= policy["threshold"] for f in a["fields"]]
            decisions_b = [f["field"] in policy["autoaccept_fields"] and supported(f) and probability(policy["isotonic"], f["confidence"]) >= policy["threshold"] for f in b["fields"]]
            task_same = task_same and decisions_a == decisions_b
            record["samples"].append({"document_id": sample.document_id, "max_logit_abs_error": error,
                                      "task_decisions_equal": task_same})
            for rep in range(10):
                for arm in (["native", "trt"] if rep % 2 == 0 else ["trt", "native"]):
                    _, ms = time_call((lambda: wrapper(**inputs)) if arm == "native" else (lambda: execute(inputs)))
                    (torch_times if arm == "native" else trt_times).append(ms)
            if len(record["samples"]) == 16:
                break
    passed = len(record["samples"]) == 16 and all(s["max_logit_abs_error"] <= .05 and s["task_decisions_equal"] for s in record["samples"])
    record.update(status="parity_verified" if passed else "failed_parity", parity_pass=passed,
                  native_median_gpu_ms=statistics.median(torch_times) if torch_times else None,
                  tensorrt_median_gpu_ms=statistics.median(trt_times) if trt_times else None,
                  timing_scope="paired/interleaved CUDA events, same 512-token inputs, model only; OCR/encoding/service and conversion build cost excluded and separately reported",
                  native_gpu_ms=torch_times, tensorrt_gpu_ms=trt_times)
    (output_dir / "tensorrt-study.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
