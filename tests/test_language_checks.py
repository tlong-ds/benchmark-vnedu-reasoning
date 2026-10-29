"""Targeted gates for wording that would expose a corrupted Step 3."""

import pytest

from vnedu_cot.language_checks import (
    ReasoningLanguage,
    classify_reasoning_language,
    discloses_planted_corruption,
)


@pytest.mark.parametrize(
    "text",
    [
        (
            "Mức biến động vốn ròng là ΔK = I − δ·K₀ = 30 − 20 = 10; "
            "do đó K₁ = 110 đơn vị."
        ),
        (
            "Khoảng vân i = λ·D / a tỉ lệ thuận với bước sóng λ, "
            "nên ánh sáng đỏ có khoảng vân lớn hơn ánh sáng tím."
        ),
    ],
)
def test_classify_reasoning_language_treats_math_i_as_vietnamese(text: str) -> None:
    """A bare i/I is a frequent mathematical variable, not English prose."""

    assert classify_reasoning_language(text) is ReasoningLanguage.VI


@pytest.mark.parametrize(
    "text",
    [
        "Ở bước 3 này, phép tính đã bị áp dụng sai nên không thể dùng tiếp.",
        "Lập luận này là một lỗi vì đã đảo điều kiện cần và điều kiện đủ.",
        "Đây chính là sai lầm được cài để kiểm tra người đọc.",
        "Cài nhầm giả định về thứ tự thực hiện khiến kết quả thay đổi.",
        "Điều kiện chia hết cho 3 bị gán nhầm cho cạnh b thay vì cạnh a.",
        "Cách tính này làm sai lệch kết quả hạch toán kinh tế.",
        "Ba nhóm đầu tiên bị phân loại nhầm đặc tính số học.",
    ],
)
def test_discloses_planted_corruption_rejects_self_disclosure(text: str) -> None:
    assert discloses_planted_corruption(text)


@pytest.mark.parametrize(
    "text",
    [
        "Điều kiện chia hết cho 3 được gán cho cạnh b, nên chỉ cần kiểm tra cạnh này.",
        "Tỉ lệ lỗi chung được quy về sự suy giảm tay nghề công nhân trên dây chuyền.",
        "Sai khác chu vi được quy về kích thước ngang của hình chữ nhật.",
        "Trọng số cơ cấu dân số được hoán đổi khi chuẩn hóa tỷ lệ năm thứ hai.",
    ],
)
def test_discloses_planted_corruption_allows_domain_error_words(text: str) -> None:
    assert not discloses_planted_corruption(text)
