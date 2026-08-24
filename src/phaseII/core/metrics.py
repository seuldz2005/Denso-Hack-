"""
metrics.py -- C-index, IBS, calibration, và điểm RUL suy ra để so RMSE cũ.

Dùng lifelines cho C-index (concordance_index_censored có sẵn, không tự
viết lại công thức cặp đôi), scikit-survival cho IBS. Cài đặt:
    pip install lifelines scikit-survival --break-system-packages

KHÔNG import các thư viện này ở đầu module để tránh việc chỉ cần build
model/loss cũng phải cài scikit-survival -- import cục bộ trong hàm.
"""

import numpy as np


def hazard_to_survival(h: np.ndarray) -> np.ndarray:
    """
    h: (T,) chuỗi hazard của MỘT engine (đã cắt đúng phần thật, không
    padding). Trả về S(t) = prod_{s<=t} (1 - h(s)).
    """
    return np.cumprod(1.0 - h)


def median_survival_time(h: np.ndarray) -> float:
    """
    Điểm S(t) cắt qua 0.5 -- dùng để suy ra một con số "RUL" có thể so
    sánh (không phải tối ưu trực tiếp) với RMSE 15.7 của pipeline cũ.
    Nếu S(t) không bao giờ xuống dưới 0.5 trong toàn bộ sequence quan sát
    được, trả về len(h) như một giá trị "ít nhất còn sống đến đây"
    (right-censored ước lượng, không phải giá trị chính xác).
    """
    s = hazard_to_survival(h)
    below = np.where(s <= 0.5)[0]
    if len(below) == 0:
        return float(len(h))
    return float(below[0])


def concordance_index(event_times: np.ndarray, predicted_risk: np.ndarray,
                       event_observed: np.ndarray) -> float:
    """
    event_times: thời điểm cuối cùng quan sát được của mỗi engine (bin xảy
                 ra sự kiện, hoặc bin cuối nếu censored)
    predicted_risk: điểm rủi ro mô hình cho ra -- dùng NEGATIVE của median
                 survival time (thời gian sống dự đoán càng ngắn thì rủi ro
                 càng cao), hoặc trực tiếp dùng h(T) tại bin cuối quan sát.
    event_observed: 1 nếu có sự kiện thật, 0 nếu censored

    lifelines quy ước: risk càng cao => event xảy ra càng sớm. Nếu bạn
    truyền predicted_risk = median_survival_time (số càng lớn = càng an
    toàn), cần đảo dấu trước khi gọi hàm này.
    """
    from lifelines.utils import concordance_index as _ci
    return _ci(event_times, predicted_risk, event_observed)


def integrated_brier_score(train_event_times, train_event_observed,
                            test_event_times, test_event_observed,
                            survival_curves: np.ndarray, time_grid: np.ndarray) -> float:
    """
    survival_curves: (n_test, n_time_grid) -- S(t) dự đoán cho từng engine
                      tại các mốc thời gian trong time_grid.

    scikit-survival cần structured array cho event/time -- xem ví dụ dùng
    trong train.py. Đây là hàm mỏng bọc lại, không tự implement Brier
    score, tránh sai công thức IPCW.
    """
    from sksurv.metrics import integrated_brier_score as _ibs
    from sksurv.util import Surv

    y_train = Surv.from_arrays(train_event_observed.astype(bool), train_event_times)
    y_test = Surv.from_arrays(test_event_observed.astype(bool), test_event_times)
    return _ibs(y_train, y_test, survival_curves, time_grid)

def cumulative_hazard(h: np.ndarray) -> float:
    """
    -log(S(T)) = tổng -log(1-h(t)) qua toàn bộ chuỗi quan sát được.
    Không bao giờ bão hòa như median_survival_time -- càng nhiều bin có
    h cao (dù chưa vượt 0.5), giá trị này càng tăng, phản ánh đúng "tổng
    lượng rủi ro tích lũy" thay vì chỉ hỏi "đã vượt ngưỡng 0.5 chưa".
    Đây là thước đo rủi ro chuẩn trong survival analysis (tương đương
    linear predictor trong Cox model), phù hợp làm predicted_risk cho
    C-index hơn hẳn median survival time.
    """
    eps = 1e-7
    h = np.clip(h, eps, 1 - eps)
    return float(-np.sum(np.log(1 - h)))
