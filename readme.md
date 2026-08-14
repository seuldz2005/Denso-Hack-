
# Table of Contents

1.  [Diagram](#org7fd77f4)
2.  [Reality Facts](#org8126fd7)
    1.  [The data is not that beautiful &hellip;](#orgfff59d1)
    2.  [The data isn&rsquo;t avaiable](#orgcbb7ccf)
3.  [Brain Storm Section](#org495969d)
    1.  [Custom loss function](#org9888441)
        1.  [Discrete-time Hazard](#orgf014f3b)
        2.  [Monotonicity Regularizer (Ham rang buoc don dieu)](#orga15c7a2)
    2.  [Autoencoder](#org95bea48)
4.  [How to evaluate our model](#org20a229f)
    1.  [Labeling + LSTM](#org7e8cc7c)
        1.  [Concordance Index (C-index)](#orgff3e2e5)
        2.  [Time-dependent Brier Score / Integrated Brier Score (IBS)](#orge81c0b1)
        3.  [Calibration plot](#org69a604a)
5.  [Phase Description](#orgea8fa87)
6.  [Suggestion](#org7885a49)
    1.  [Problem statement](#org3a67d22)
    2.  [Demo](#org0c2e9f7)



<a id="org7fd77f4"></a>

# Diagram

![img](my-diagram.png)


<a id="org8126fd7"></a>

# Reality Facts


<a id="orgfff59d1"></a>

## The data is not that beautiful &hellip;

-   In reality, we can&rsquo;t just let the machine broken down just to get the data but
    always fix it after some degree of degradation signals
-   Therefore, we need to consider that our data will always contain mostly **health phase**


<a id="orgcbb7ccf"></a>

## The data isn&rsquo;t avaiable

-   One more thing is that if we have a new machine and get it into work. We can&rsquo;t
    just have all of it data immediately &hellip;
-   That&rsquo;s why we should consider this aspect


<a id="org495969d"></a>

# Brain Storm Section


<a id="org9888441"></a>

## Custom loss function


<a id="orgf014f3b"></a>

### Discrete-time Hazard

-   is basically **Probability**
-   has different loss function :

$$\text{Loss}_t = -[y_t \log(\hat{y}_t) + (1 - y_t) \log(1 - \hat{y}_t)]$$


<a id="orga15c7a2"></a>

### Monotonicity Regularizer (Ham rang buoc don dieu)

-   basically to make sure the percantage of the engine which will broken down can
    decrease &hellip; (ban chat chung phai tuyen tinh tang len)


<a id="org95bea48"></a>

## Autoencoder

-   its purpose is to find anomaly in health phase
-   so it can also help us to revaluate the degradation point instead of using
    130 - fixed point like my previous plan, also we can use this as a transition
    point to use phase II


<a id="org20a229f"></a>

# How to evaluate our model

-   Include C-index, IBS, Calibration and it is completely different from RMSE


<a id="org7e8cc7c"></a>

## Labeling + LSTM


<a id="orgff3e2e5"></a>

### Concordance Index (C-index)

-   đây là chỉ số quan trọng nhất, tương đương AUC nhưng cho dữ liệu
    survival/censored. Nó đo: trong mọi cặp máy (i, j) mà bạn biết chắc máy nào
    &ldquo;hỏng trước&rdquo; (kể cả khi một trong hai bị censor, miễn còn so sánh được), mô
    hình có xếp hạng đúng thứ tự rủi ro không? C-index = 0.5 là đoán ngẫu nhiên,
    1.0 là hoàn hảo. Đây là chỉ số bạn nên báo cáo làm &ldquo;con số chính&rdquo; thay thế vai
    trò của RMSE trước đây, vì nó đánh giá đúng bản chất bài toán bạn đang giải
    (xếp hạng rủi ro tương đối), không đòi hỏi biết chính xác RUL của từng máy
    censored.


<a id="orge81c0b1"></a>

### Time-dependent Brier Score / Integrated Brier Score (IBS)

-   đây là thứ gần với &ldquo;MSE cho xác suất&rdquo;: tại mỗi mốc thời gian t, so sánh xác
    suất dự đoán với outcome thực tế (đã xảy ra sự kiện hay chưa), có điều chỉnh
    trọng số cho dữ liệu censored (IPCW). IBS lấy tích phân Brier score qua toàn
    bộ khoảng thời gian quan sát thành một con số duy nhất — cho bạn biết mô hình
    có hiệu chỉnh tốt không (calibration), bổ sung cho C-index vốn chỉ đo thứ hạng
    chứ không đo độ chính xác xác suất tuyệt đối.


<a id="org69a604a"></a>

### Calibration plot

-   vẽ xác suất dự đoán &ldquo;sẽ cần bảo trì trong X cycle tới&rdquo; so với tần suất thực tế
    quan sát được trong nhóm đó. Đây là biểu đồ trực quan rất tốt để demo — dễ
    giải thích cho ban giám khảo hơn nhiều so với con số C-index trừu tượng.


<a id="orgea8fa87"></a>

# Phase Description

Ta sẽ có 2 phase, khi data vẫn còn ít, ta sẽ sử dụng autoencoder nhằm phát
hiện bất thường, vì là thời điểm healthy phase nên việc sử dụng chúng chỉ nhằm
mục tiêu đảm bảo trường hợp xấu nhất (rất khó xảy ra). Khi đạt đến 1 thời điểm
nhất định, ta sẽ sử dụng pipeline : MFPCA + GMM + Youden index -> label, và sử
dụng LSTM cùng với custom loss function (Discrete-time Hazard , Label,


<a id="org7885a49"></a>

# Suggestion


<a id="org3a67d22"></a>

## Problem statement

-   Tham dự triển lãm Intelligent Asia Hanoi 2026 và các phiên talkshow chuyên đề
    về chuyển đổi số trong sản xuất, chúng tôi nhận thấy Việt Nam đang bước vào
    giai đoạn chuyển dịch mạnh mẽ hướng tới Smart Manufacturing, được thúc đẩy bởi
    loạt chính sách hỗ trợ từ Nhà nước dành cho doanh nghiệp và các cơ sở nghiên
    cứu. Trong bối cảnh đó, việc đầu tư máy móc nhập khẩu — vốn có chi phí lớn và
    thời gian khấu hao dài — đang được khuyến khích như một bước đi tất yếu để
    nâng cao năng lực sản xuất. Tuy nhiên, đi kèm với làn sóng đầu tư này là một
    khoảng trống đáng lo ngại: phần lớn các nhà cung cấp cấp 2, cấp 3 trong chuỗi
    cung ứng — nơi vận hành song song nhiều máy móc cùng loại nhưng lại có ngân
    sách và nhân lực kỹ thuật hạn chế — không có khả năng tiếp cận các công cụ
    đánh giá độ bền/dự đoán tuổi thọ (RUL) đi kèm, vốn thường bị khóa trong hệ
    sinh thái độc quyền và chi phí cao của chính hãng sản xuất máy. Tình trạng này
    càng trở nên cấp thiết khi nhiều khu công nghiệp hiện vẫn phụ thuộc phần lớn
    vào giám sát thủ công bằng con người — vừa tốn kém nhân lực, vừa thiếu nhất
    quán và không thể phát hiện sớm các dấu hiệu suy thoái tiềm ẩn. Nhu cầu về một
    giải pháp đánh giá sức khỏe/tuổi thọ máy móc — độc lập với hãng sản xuất, phù
    hợp với năng lực dữ liệu thực tế và chi phí vận hành của nhóm doanh nghiệp này
    — do đó không còn là câu hỏi &ldquo;có cần không&rdquo;, mà chỉ còn là vấn đề thời gian
    trước khi nó trở thành một yêu cầu bắt buộc trong chuỗi cung ứng công nghiệp
    4.0. Xuất phát từ thực trạng đó, chúng tôi đề xuất&hellip;


<a id="org0c2e9f7"></a>

## Demo

-   At the overlap range of 2 main phase, we can check if the anomaly are provided
    the same between these 2 phase.
-   

Team information — gắn với năng lực đã chứng minhKhông chỉ tên/trường — nêu rõ
vì sao đội bạn phù hợp với chính bài toán này: bạn đã có RMSE 15.7 trên C-MAPSS,
đã tham dự Intelligent Asia Hanoi, hiểu văn hóa TPM. Đây là slide tạo niềm tin
ngay từ đầu, không phải thủ tục.
2
Problem statement and background — rút gọn thành 1 slide cốt lõiDùng đoạn dẫn
dắt đã viết (Intelligent Asia → chính sách Smart Manufacturing → khoảng trống
OEM khóa vendor → Tier 2/3 thiếu công cụ), nhưng rút gọn còn 3-4 câu trên slide,
phần diễn giải đầy đủ để vào speaker note nếu template có chỗ đó. Thêm 1 số liệu
thịnh (ví dụ chi phí IIoT trung bình cho SME) để tăng sức nặng.
3
Market analysis — so sánh trực tiếp với giải pháp hiện cóĐặt sản phẩm bạn bên
cạnh Augury/SKF/Factory AI trong một bảng so sánh ngắn: họ đắt + khóa vendor,
bạn rẻ + sensor-agnostic + nhẹ cho Tier 2/3. Đây là slide dễ gây ấn tượng nhất
vì cho thấy bạn hiểu thị trường thật, không chỉ hiểu thuật toán.
4
Technical design — trục chiều là sơ đồ pipelineĐây là phần nặng nhất — dùng
chính sơ đồ pipeline vừa hoàn thiện làm hero visual của toàn bộ deck. Một slide
cho toàn cảnh, sau đó tách riêng 1-2 slide zoom vào Phase I (autoencoder) và
Phase II (MFPCA/hazard) với giải thích ngắn mỗi phần làm gì, tại sao cần.
5
Objectives and expected impact — gắn với mốc cụ thểNêu rõ cụ thể sẽ làm gì nếu
được chọn vào top 10/top 5: ví dụ áp dụng pipeline lên dữ liệu thật từ Factory
Tour, đo C-index/IBS cụ thể, xây demo dashboard. Tránh chung chung kiểu &ldquo;cải
thiện hiệu suất sản xuất&rdquo; — càng đo đếm được càng tốt.
6
Relevance to competition theme — nói rõ không ngầm địnhMột câu hoặc hai câu kết
nối trực tiếp với tên chủ đề &ldquo;Predictive & Knowledge AI&rdquo; — tránh để giám khảo tự
suy ra, nêu thẳng.

