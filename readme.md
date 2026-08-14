
# Table of Contents

1.  [Diagram](#orge356558)
2.  [Reality Facts](#org130829a)
    1.  [The data is not that beautiful &hellip;](#orgbf765ea)
    2.  [The data isn&rsquo;t avaiable](#org1b78042)
    3.  [The engine won&rsquo;t work in sync](#org58ce1cd)
3.  [Brain Storm Section](#org0825ae2)
    1.  [Custom loss function](#org2e41b5d)
        1.  [Discrete-time Hazard](#org00b78d6)
        2.  [Monotonicity Regularizer (Ham rang buoc don dieu)](#org6c1945d)
    2.  [Autoencoder](#org9febb64)
4.  [How to evaluate our model](#org20c7494)
    1.  [Labeling + LSTM](#orge1359dd)
        1.  [Concordance Index (C-index)](#org1552c1b)
        2.  [Time-dependent Brier Score / Integrated Brier Score (IBS)](#orgf2c569f)
        3.  [Calibration plot](#orgb520c22)
5.  [Phase Description](#orgb34c958)
6.  [Suggestion](#orgda590a4)
    1.  [Problem statement](#org57ec494)
    2.  [Demo](#org0c40576)
    3.  [Preview](#orgeaf3407)
    4.  [Asignment](#org4947e9e)
7.  [Potential Question](#orga099ae8)
    1.  [Why for Tier 2/3](#org3ace2b9)
    2.  [What is the reason for Phase II?](#org151fd46)
    3.  [Why not using RMSE for validate check?](#org57fe78f)
    4.  [Do we need to start all the engine at the same time?](#orge90ae48)



<a id="orge356558"></a>

# Diagram

![img](my-diagram.png)


<a id="org130829a"></a>

# Reality Facts


<a id="orgbf765ea"></a>

## The data is not that beautiful &hellip;

-   In reality, we can&rsquo;t just let the machine broken down just to get the data but
    always fix it after some degree of degradation signals
-   Therefore, we need to consider that our data will always contain mostly **health phase**


<a id="org1b78042"></a>

## The data isn&rsquo;t avaiable

-   One more thing is that if we have a new machine and get it into work. We can&rsquo;t
    just have all of it data immediately &hellip;
-   That&rsquo;s why we should consider this aspect


<a id="org58ce1cd"></a>

## The engine won&rsquo;t work in sync

-   We should consider that all the engines should work in different term of its


<a id="org0825ae2"></a>

# Brain Storm Section


<a id="org2e41b5d"></a>

## Custom loss function


<a id="org00b78d6"></a>

### Discrete-time Hazard

-   is basically **Probability**
-   has different loss function :

$$\text{Loss}_t = -[y_t \log(\hat{y}_t) + (1 - y_t) \log(1 - \hat{y}_t)]$$


<a id="org6c1945d"></a>

### Monotonicity Regularizer (Ham rang buoc don dieu)

-   basically to make sure the percantage of the engine which will broken down can
    decrease &hellip; (ban chat chung phai tuyen tinh tang len)


<a id="org9febb64"></a>

## Autoencoder

-   its purpose is to find anomaly in health phase
-   so it can also help us to revaluate the degradation point instead of using
    130 - fixed point like my previous plan, also we can use this as a transition
    point to use phase II


<a id="org20c7494"></a>

# How to evaluate our model

-   Include C-index, IBS, Calibration and it is completely different from RMSE


<a id="orge1359dd"></a>

## Labeling + LSTM


<a id="org1552c1b"></a>

### Concordance Index (C-index)

-   đây là chỉ số quan trọng nhất, tương đương AUC nhưng cho dữ liệu
    survival/censored. Nó đo: trong mọi cặp máy (i, j) mà bạn biết chắc máy nào
    &ldquo;hỏng trước&rdquo; (kể cả khi một trong hai bị censor, miễn còn so sánh được), mô
    hình có xếp hạng đúng thứ tự rủi ro không? C-index = 0.5 là đoán ngẫu nhiên,
    1.0 là hoàn hảo. Đây là chỉ số bạn nên báo cáo làm &ldquo;con số chính&rdquo; thay thế vai
    trò của RMSE trước đây, vì nó đánh giá đúng bản chất bài toán bạn đang giải
    (xếp hạng rủi ro tương đối), không đòi hỏi biết chính xác RUL của từng máy
    censored.


<a id="orgf2c569f"></a>

### Time-dependent Brier Score / Integrated Brier Score (IBS)

-   đây là thứ gần với &ldquo;MSE cho xác suất&rdquo;: tại mỗi mốc thời gian t, so sánh xác
    suất dự đoán với outcome thực tế (đã xảy ra sự kiện hay chưa), có điều chỉnh
    trọng số cho dữ liệu censored (IPCW). IBS lấy tích phân Brier score qua toàn
    bộ khoảng thời gian quan sát thành một con số duy nhất — cho bạn biết mô hình
    có hiệu chỉnh tốt không (calibration), bổ sung cho C-index vốn chỉ đo thứ hạng
    chứ không đo độ chính xác xác suất tuyệt đối.


<a id="orgb520c22"></a>

### Calibration plot

-   vẽ xác suất dự đoán &ldquo;sẽ cần bảo trì trong X cycle tới&rdquo; so với tần suất thực tế
    quan sát được trong nhóm đó. Đây là biểu đồ trực quan rất tốt để demo — dễ
    giải thích cho ban giám khảo hơn nhiều so với con số C-index trừu tượng.


<a id="orgb34c958"></a>

# Phase Description

Ta sẽ có 2 phase, khi data vẫn còn ít, ta sẽ sử dụng autoencoder nhằm phát
hiện bất thường, vì là thời điểm healthy phase nên việc sử dụng chúng chỉ nhằm
mục tiêu đảm bảo trường hợp xấu nhất (rất khó xảy ra). Khi đạt đến 1 thời điểm
nhất định, ta sẽ sử dụng pipeline : MFPCA + GMM + Youden index -> label, và sử
dụng LSTM cùng với custom loss function (Discrete-time Hazard , Label,


<a id="orgda590a4"></a>

# Suggestion


<a id="org57ec494"></a>

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


<a id="org0c40576"></a>

## Demo

-   At the overlap range of 2 main phase, we can check if the anomaly are provided
    the same between these 2 phase.


<a id="orgeaf3407"></a>

## Preview

-   Team information — gắn với năng lực đã chứng minh Không chỉ tên/trường — nêu
    rõ vì sao đội bạn phù hợp với chính bài toán này: bạn đã có RMSE 15.7 trên
    C-MAPSS, đã tham dự Intelligent Asia Hanoi, hiểu văn hóa TPM. Đây là slide tạo
    niềm tin ngay từ đầu, không phải thủ tục.
-   Problem statement and background — rút gọn thành 1 slide cốt lõi. Dùng đoạn
    dẫn dắt đã viết (Intelligent Asia → chính sách Smart Manufacturing → khoảng
    trống OEM khóa vendor → Tier 2/3 thiếu công cụ), nhưng rút gọn còn 3-4 câu
    trên slide, phần diễn giải đầy đủ để vào speaker note nếu template có chỗ đó.
    Thêm 1 số liệu thịnh (ví dụ chi phí IIoT trung bình cho SME) để tăng sức nặng.
-   Market analysis — so sánh trực tiếp với giải pháp hiện có. Đặt sản phẩm bạn
    bên cạnh Augury/SKF/Factory AI trong một bảng so sánh ngắn: họ đắt + khóa
    vendor, bạn rẻ + sensor-agnostic + nhẹ cho [Tier 2/3](#org3ace2b9). Đây là slide dễ gây ấn
    tượng nhất vì cho thấy bạn hiểu thị trường thật, không chỉ hiểu thuật toán.
-   Technical design — trục chiều là sơ đồ pipelineĐây là phần nặng nhất — dùng
    chính sơ đồ pipeline vừa hoàn thiện làm hero visual của toàn bộ deck. Một
    slide cho toàn cảnh, sau đó tách riêng 1-2 slide zoom vào Phase I
    (autoencoder) và Phase II (MFPCA/hazard) với giải thích ngắn mỗi phần làm gì,
    tại sao cần.
-   Objectives and expected impact — gắn với mốc cụ thểNêu rõ cụ thể sẽ làm gì nếu
    được chọn vào top 10/top 5: ví dụ áp dụng pipeline lên dữ liệu thật từ Factory
    Tour, đo C-index/IBS cụ thể, xây demo dashboard. Tránh chung chung kiểu &ldquo;cải
    thiện hiệu suất sản xuất&rdquo; — càng đo đếm được càng tốt.
-   Relevance to competition theme — nói rõ không ngầm địnhMột câu hoặc hai câu
    kết nối trực tiếp với tên chủ đề &ldquo;Predictive & Knowledge AI&rdquo; — tránh để giám
    khảo tự suy ra, nêu thẳng.


<a id="org4947e9e"></a>

## Asignment

-   Bạn — Phase 2 (MFPCA/GMM/hazard-LSTM), như đã định.
-   Người 2 — Phase 1 (autoencoder), như đã định.
-   Người 3 — thay vì &ldquo;quản lý transition + draft pitch&rdquo;, nên là chủ trì phần đánh
    giá & tích hợp xuyên suốt cả hai phase: xây bộ harness đo lường (Spearman
    correlation cho AE, C-index/IBS/calibration cho Phase 2, multi-seed
    evaluation), và cùng bạn với người 2 thống nhất tiêu chí PhaseCheck (dựa trên
    số liệu thực nghiệm hai người kia cung cấp, không tự quyết một mình). Đây là
    việc rất dễ bị bỏ quên vì cả hai người làm model đều sẽ mải tối ưu phần riêng
    của mình — nếu không ai chuyên trách, phần &ldquo;hai module có thực sự ăn khớp với
    nhau không&rdquo; dễ chỉ được kiểm tra vội vào phút cuối.


<a id="orga099ae8"></a>

# Potential Question


<a id="org3ace2b9"></a>

## Why for Tier 2/3


<a id="org151fd46"></a>

## What is the reason for Phase II?

-   2 different for 2 phase :
-   AE will answer for the question about anomaly and when does it start?
-   While phase II will answer for the question : which engine need to maintain
    first (priority)

-   Một điểm tôi khuyên bạn thành thật thừa nhận thay vì né tránh — vì nó thực ra
    làm lập luận của bạn mạnh hơn: đúng là có trường hợp Phase II không cần thiết
    — nếu máy đó rẻ, hậu quả hỏng thấp, chi phí kiểm tra thủ công thấp, thì cứ có
    cảnh báo từ AE là đủ, không cần đầu tư thêm phân tích sâu. Việc bạn chủ động
    nói ra ranh giới này (&ldquo;Phase II chỉ đáng đầu tư cho máy/dây chuyền có giá trị
    cao hoặc hậu quả hỏng nghiêm trọng — đúng kiểu máy nhập khẩu đắt tiền mà đề
    xuất ban đầu của bạn đang nhắm tới&rdquo;) sẽ cho ban giám khảo thấy bạn hiểu rõ
    trade-off chi phí-lợi ích của chính hệ thống mình đề xuất, không chỉ cố bán
    &ldquo;càng nhiều mô hình càng tốt&rdquo; — đây chính xác là điều phân biệt một đề xuất kỹ
    thuật trưởng thành với một đề xuất chỉ đang khoe số lượng thuật toán.


<a id="org57fe78f"></a>

## Why not using RMSE for validate check?


<a id="orge90ae48"></a>

## Do we need to start all the engine at the same time?

