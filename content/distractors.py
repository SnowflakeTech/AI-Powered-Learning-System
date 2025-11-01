from __future__ import annotations
import random
from typing import List, Union, Tuple


def ensure_unique_distractors(
    correct_value: Union[int, float, str],
    candidates: List[Union[int, float, str]],
    max_distractors: int = 3,
) -> List[str]:
    """
    Chọn ra tối đa 3 distractor duy nhất khác đáp án đúng.
    Padding các giá trị tương tự để đảm bảo đủ số lượng.
    """
    seen = {str(correct_value)}
    uniq: List[str] = []

    for value in candidates:
        s = str(value)
        if s not in seen:
            uniq.append(s)
            seen.add(s)
        if len(uniq) >= max_distractors:
            break

    # Bổ sung nếu thiếu (các number lân cận hoặc phương án text cố định)
    if isinstance(correct_value, (int, float)):
        base = int(correct_value)
        k = 1
        while len(uniq) < max_distractors:
            for cand in (base + k, base - k):
                s = str(cand)
                if s not in seen:
                    uniq.append(s)
                    seen.add(s)
                    if len(uniq) >= max_distractors:
                        break
            k += 1
    else:
        filler = random.choice(["Không rõ", "Thông tin thiếu", "Không phải đáp án"])
        while len(uniq) < max_distractors:
            uniq.append(filler)

    return uniq


def make_option_set(
    correct: str,
    distractors: List[str],
) -> Tuple[List[dict], str]:
    """
    Tạo danh sách phương án dạng:
    [
       {id: "A", text: "..."},
       ...
    ]
    và trả về answer_key (A/B/C/D)
    """
    letters = ["A", "B", "C", "D"]
    pool = [correct] + distractors[:3]
    random.shuffle(pool)

    options = [
        {"id": letters[i], "text": pool[i]}
        for i in range(4)
    ]
    answer_key = next(o["id"] for o in options if o["text"] == correct)
    return options, answer_key
