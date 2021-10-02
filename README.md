# Cài đặt các modules cần thiết
`pip install -r requirements.txt`

# Thiết lập tham số

Tất cả tham số dùng cho việc training và testing mô hình đều được đặt trong một file `.yaml` và thường được đặt trong thư mục `config`.

Cấu trúc một file `.yaml` tương đối giống cấu trúc giữa các khối lệnh trong Python - khối lệnh con nằm lùi vào một khoảng so với khối lệnh cha - tương tự, trường dữ liệu con nằm lùi vào một khoảng so với cha của nó.

Ý nghĩa các tham số (chi tiết tham khảo trong file `config/default.yaml`):

- dataset:
>- train: Đường dẫn đến file chứa dữ liệu huấn luyện
>- test: Đường dẫn đến file chứa dữ liệu kiểm thử
- model: Chứa các cài đặt liên quan đến model, các tham số có thể khác nhau tùy thuộc từng model. Một số tham số thường được sử dụng:
>- module: Tên của model, trong mã nguồn nên có 1 `dict` ánh xạ từ tên model sang class tương ứng.
>- data_config: Các cài đặt liên quan đến dữ liệu, ví dụ như số lượng trạm, kích thước cửa sổ dữ liệu input, hệ số chuẩn hóa,...
>- Các modules con.
- resource: Các tham số cài đặt tài nguyên cấp phát cho quá trình huấn luyện và kiểm thử.
>- num_workers: Cài đặt số luồng chạy cho DataLoader.
>- gpus: Danh sách GPUs được dùng.
- training: Các cài đặt riêng cho quá trình huấn luyện.
>- batch_size
>- max_epochs: Việc huấn luyện sẽ dừng lại sau khi đạt đủ số epochs, kể cả chưa đạt các điều kiện dừng khác.
>- loss: Cài đặt loss function
- testing: Các cài đặt riêng cho quá trình kiểm thử.

# Training
`python train.py`

Tham số dòng lệnh:

- --config: Đường dẫn đến file `.yaml` chứa các tham số.

Kết quả huấn luyện được lưu trong thư mục `lightning_logs/version_x`:

- File `hparams.yaml` chứa các thiết lập tham số của mô hình.
- File `events.out.tfevents*` chứa logs.
- Các file trong thư mục con `checkpoints` chứa tham số học được của mô hình.

# Testing

`python test.py`

Tham số dòng lệnh:

- --checkpoint: Đường dẫn đến thư mục chứa 2 file `checkpoint.ckpt` và `hparams.yaml`.