from manga_translator.utils.metrics import record_api_request, flush, record_stage_duration, set_queue_depth

def test_api_request():
    record_api_request("deepl", "ok")
    flush()

def test_stage_duration():
    record_stage_duration(3.2, "ocr")
    flush()

def test_set_queue_depth():
    set_queue_depth(3)
    flush()