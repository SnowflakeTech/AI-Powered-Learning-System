"""
sat_ai_core/api_throttler.py
-----------------------------------
Bộ điều tiết (throttler) và cơ chế retry nâng cao cho các lệnh gọi OpenAI API.
Tránh lỗi HTTP 429 ("Too Many Requests") hoặc lỗi mạng tạm thời.

✅ Điểm nổi bật:
- Giới hạn tốc độ theo model hoặc toàn cục (per-model throttling)
- Tự động retry với backoff theo cấp số nhân + jitter
- Tôn trọng header Retry-After của OpenAI (nếu có)
- Phân biệt lỗi tạm thời (retry được) và lỗi vĩnh viễn (ngừng retry)
- Thread-safe, không làm nghẽn luồng khác
- Logging rõ ràng, có thể tích hợp vào hệ thống giám sát
"""

import time
import random
import logging
from threading import Lock
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError

# ==============================
# ⚙️ Cấu hình logging
# ==============================
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ==============================
# 🧩 Lớp Exception tùy biến
# ==============================
class ThrottlerError(Exception):
    """Báo lỗi khi hết lượt retry hoặc API liên tục thất bại."""

    def __init__(self, message: str, last_exception: Optional[BaseException], attempts: int):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


# ==============================
# 🚀 Lớp ApiThrottler (bản cải tiến)
# ==============================
class ApiThrottler:
    def __init__(
        self,
        min_interval: float = 2.0,
        max_retries: int = 5,
        max_wait: float = 30.0,
        per_model: bool = True,
    ):
        """
        Tham số:
            min_interval: Khoảng cách tối thiểu giữa 2 lần gọi API (giây)
            max_retries: Số lần retry tối đa
            max_wait: Thời gian chờ tối đa giữa các lần retry
            per_model: Giới hạn riêng theo từng model (True) hoặc toàn cục (False)
        """
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.max_wait = max_wait
        self.per_model = per_model

        self._lock = Lock()
        self._last_call: Dict[str, float] = {}

    # ------------------------------
    # 🔧 Xử lý thời gian an toàn
    # ------------------------------
    def _now(self) -> float:
        return time.monotonic()

    def _key(self, model: str) -> str:
        return model if self.per_model else "__global__"

    # ------------------------------
    # ⏳ Chờ slot an toàn (thread-safe)
    # ------------------------------
    def _wait_for_slot(self, key: str):
        with self._lock:
            now = self._now()
            last = self._last_call.get(key, 0.0)
            elapsed = now - last
            if elapsed < self.min_interval:
                wait = self.min_interval - elapsed
                logger.debug(f"⏳ Chờ {wait:.2f}s để tránh vượt giới hạn API ({key})")
                self._lock.release()
                try:
                    time.sleep(wait)
                finally:
                    self._lock.acquire()
            self._last_call[key] = self._now()

    # ------------------------------
    # 🧠 Tính toán thời gian backoff
    # ------------------------------
    def _compute_backoff(self, attempt: int, retry_after: Optional[float]) -> float:
        if retry_after is not None:
            return min(self.max_wait, max(0.0, retry_after))
        return min(self.max_wait, 2 ** attempt + random.uniform(0.5, 2.0))

    # ------------------------------
    # 📥 Hàm chính: gọi API an toàn
    # ------------------------------
    def safe_openai_chat(
        self,
        client: OpenAI,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4o-mini",
        **kwargs,
    ):
        """
        Gọi API OpenAI với throttling + retry tự động.
        Trả về response nếu thành công, ném ThrottlerError nếu thất bại sau N lần.
        """
        key = self._key(model)
        last_exc: Optional[BaseException] = None

        for attempt in range(1, self.max_retries + 1):
            self._wait_for_slot(key)

            try:
                response = client.chat.completions.create(model=model, messages=messages, **kwargs)
                return response

            # ----- Xử lý lỗi giới hạn -----
            except RateLimitError as e:
                retry_after = self._get_retry_after(e)
                wait_time = self._compute_backoff(attempt, retry_after)
                logger.warning(f"⚠️ Rate limit (HTTP 429). Chờ {wait_time:.1f}s trước khi retry ({attempt}/{self.max_retries})")
                time.sleep(wait_time)
                last_exc = e

            # ----- Lỗi timeout -----
            except APITimeoutError as e:
                wait_time = self._compute_backoff(attempt, None)
                logger.warning(f"⏱️ Timeout API. Chờ {wait_time:.1f}s rồi retry ({attempt}/{self.max_retries})")
                time.sleep(wait_time)
                last_exc = e

            # ----- Lỗi máy chủ (5xx) -----
            except APIError as e:
                status = getattr(e, "status_code", None)
                if status and 500 <= status < 600:
                    wait_time = self._compute_backoff(attempt, None)
                    logger.warning(f"💥 Lỗi máy chủ ({status}). Chờ {wait_time:.1f}s rồi retry ({attempt}/{self.max_retries})")
                    time.sleep(wait_time)
                    last_exc = e
                else:
                    logger.error(f"🚫 Lỗi API không thể retry ({status}): {e}")
                    raise

            # ----- Các lỗi khác -----
            except Exception as e:
                logger.error(f"🚨 Lỗi không xác định khi gọi OpenAI: {e}")
                last_exc = e
                break

        # Nếu hết lượt retry
        raise ThrottlerError("❌ Hết lượt retry — API thất bại.", last_exc, self.max_retries)

    # ------------------------------
    # 🔍 Hàm phụ lấy Retry-After
    # ------------------------------
    def _get_retry_after(self, exc: Exception) -> Optional[float]:
        try:
            headers = getattr(exc, "response", None)
            if headers and hasattr(headers, "headers"):
                val = headers.headers.get("Retry-After")
                if val:
                    return float(val)
        except Exception:
            pass
        return None
