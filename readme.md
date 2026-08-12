
# Table of Contents

1.  [Diagram](#orgce836ec)
2.  [Brain Storm Section](#org0c9a17b)
    1.  [Custom loss function](#orgf512fb8)
        1.  [Discrete-time Hazard](#org1eb89c2)
        2.  [Monotonicity Regularizer (Ham rang buoc don dieu)](#orgd8eb119)
    2.  [Autoencoder](#org5fba7ca)
3.  [How to evaluate our model](#org17af508)
    1.  [Concordance Index (C-index)](#org4ab248d)
    2.  [Time-dependent Brier Score / Integrated Brier Score (IBS)](#org4678f81)
    3.  [Calibration plot](#org3fc5ffb)
4.  [Reality Facts](#orgb72bcb3)
    1.  [The data is not that beautiful &hellip;](#orgf2c3ba9)
    2.  [The data isn&rsquo;t avaiable](#orgc934b5b)
5.  [Phase Description](#orgba3ce68)
6.  [Suggest Demo](#orgefc7310)



<a id="orgce836ec"></a>

# Diagram

    flowchart TD
        %% Đây là comment trong Mermaid (sẽ không hiển thị)
    
        Start(Bắt đầu) --> Input[/Nhập ma trận/]
        Input --> Check{Kiểm tra định dạng}
    
        Check -->|Hợp lệ| Process[Chạy thuật toán Jacobi]
        Check -->|Lỗi| Error[Báo lỗi Dữ liệu]
    
        Process -.-> DB[(Lưu kết quả ngầm)]
        Process ==> Output[/Xuất mảng NumPy/]
       
        Error --> End((Kết thúc))
        Output --> End


<a id="org0c9a17b"></a>

# Brain Storm Section


<a id="orgf512fb8"></a>

## Custom loss function


<a id="org1eb89c2"></a>

### Discrete-time Hazard

-   is basically **Probability**
-   has different loss function :
    $$\text{Loss}_{t} = -[y_{t}log(\hat{y_{t}})+(1-y_{t})log(1-\hat{y_{t}})] $$
-   **answer : how is the result of guessing which unit will break first**


<a id="orgd8eb119"></a>

### Monotonicity Regularizer (Ham rang buoc don dieu)

-   basically to make sure the percantage of the engine which will broken down can
    decrease &hellip; (ban chat chung phai tuyen tinh tang len)


<a id="org5fba7ca"></a>

## Autoencoder

-   its purpose is to find anomaly in health phase
-   so it can also help us to revaluate the degradation point instead of using
    130 - fixed point like my previous plan, also we can use this as a transition
    point to use phase II


<a id="org17af508"></a>

# How to evaluate our model

-   Include C-index, IBS, Calibration and it is completely different from RMSE


<a id="org4ab248d"></a>

## Concordance Index (C-index)

-   đây là chỉ số quan trọng nhất, tương đương AUC nhưng cho dữ liệu
    survival/censored. Nó đo: trong mọi cặp máy (i, j) mà bạn biết chắc máy nào
    &ldquo;hỏng trước&rdquo; (kể cả khi một trong hai bị censor, miễn còn so sánh được), mô
    hình có xếp hạng đúng thứ tự rủi ro không? C-index = 0.5 là đoán ngẫu nhiên,
    1.0 là hoàn hảo. Đây là chỉ số bạn nên báo cáo làm &ldquo;con số chính&rdquo; thay thế vai
    trò của RMSE trước đây, vì nó đánh giá đúng bản chất bài toán bạn đang giải
    (xếp hạng rủi ro tương đối), không đòi hỏi biết chính xác RUL của từng máy
    censored.


<a id="org4678f81"></a>

## Time-dependent Brier Score / Integrated Brier Score (IBS)

-   đây là thứ gần với &ldquo;MSE cho xác suất&rdquo;: tại mỗi mốc thời gian t, so sánh xác
    suất dự đoán với outcome thực tế (đã xảy ra sự kiện hay chưa), có điều chỉnh
    trọng số cho dữ liệu censored (IPCW). IBS lấy tích phân Brier score qua toàn
    bộ khoảng thời gian quan sát thành một con số duy nhất — cho bạn biết mô hình
    có hiệu chỉnh tốt không (calibration), bổ sung cho C-index vốn chỉ đo thứ hạng
    chứ không đo độ chính xác xác suất tuyệt đối.


<a id="org3fc5ffb"></a>

## Calibration plot

-   vẽ xác suất dự đoán &ldquo;sẽ cần bảo trì trong X cycle tới&rdquo; so với tần suất thực tế
    quan sát được trong nhóm đó. Đây là biểu đồ trực quan rất tốt để demo — dễ
    giải thích cho ban giám khảo hơn nhiều so với con số C-index trừu tượng.


<a id="orgb72bcb3"></a>

# Reality Facts


<a id="orgf2c3ba9"></a>

## The data is not that beautiful &hellip;

-   In reality, we can&rsquo;t just let the machine broken down just to get the data but
    always fix it after some degree of degradation signals
-   Therefore, we need to consider that our data will always contain mostly **health phase**


<a id="orgc934b5b"></a>

## The data isn&rsquo;t avaiable

-   One more thing is that if we have a new machine and get it into work. We can&rsquo;t
    just have all of it data immediately &hellip;
-   That&rsquo;s why we should consider this aspect


<a id="orgba3ce68"></a>

# Phase Description

Ta sẽ có 2 phase, khi data vẫn còn ít, ta sẽ sử dụng autoencoder nhằm phát
hiện bất thường, vì là thời điểm healthy phase nên việc sử dụng chúng chỉ nhằm
mục tiêu đảm bảo trường hợp xấu nhất (rất khó xảy ra). Khi đạt đến 1 thời điểm
nhất định, ta sẽ sử dụng pipeline : MFPCA + GMM + Youden index -> label, và sử
dụng LSTM cùng với custom loss function (Discrete-time Hazard , Label,


<a id="orgefc7310"></a>

# Suggest Demo

-   At the overlap range of 2 main phase, we can check if the anomaly are provided
    the same between these 2 phase.
-   

