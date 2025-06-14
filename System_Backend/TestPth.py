import torch
from DualEfficientNetB0 import DualEfficientNetB0
from preprocessing import RadarPreprocessor


def getResult():
    # 1. 加载模型
    model_path = r"DualEfficientNetB0_best_model.pth"
    model = DualEfficientNetB0(num_classes=20)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 2. 类别索引
    id2label = {
        0: "交替腿举-不标准", 1: "交替腿举-标准",
        2: "仰卧起坐-不标准", 3: "仰卧起坐-标准",
        4: "俄罗斯转体-不标准", 5: "俄罗斯转体-标准",
        6: "俯卧撑-不标准", 7: "俯卧撑-标准",
        8: "前弓步-不标准", 9: "前弓步-标准",
        10: "开合跳-不标准", 11: "开合跳-标准",
        12: "深蹲-不标准", 13: "深蹲-标准",
        14: "登山者-不标准", 15: "登山者-标准",
        16: "高抬腿-不标准", 17: "高抬腿-标准",
        18: "鸟狗式-不标准", 19: "鸟狗式-标准"
    }

    # 3. 处理一条数据
    pre = RadarPreprocessor()
    mat_path = r"..\MATLAB code\adcSampleAll.mat"
    rtm, dtm = pre.preprocess_pipeline(mat_path)  # [H, W, 3] np.ndarray

    # 4. 转成模型输入格式
    rtm_tensor = torch.tensor(rtm).permute(2, 0, 1).unsqueeze(0).float() / 255.
    dtm_tensor = torch.tensor(dtm).permute(2, 0, 1).unsqueeze(0).float() / 255.
    with torch.no_grad():
        logits = model(rtm_tensor, dtm_tensor)  # [1, 20]
        pred = logits.argmax(dim=1).item()
        print(f"预测结果类别: {pred} -> {id2label[pred]}")
        return pred
