import timm
import torch
import os
from tqdm import tqdm

def download_models():
    # 设置模型保存路径到当前项目的 models 目录
    models_dir = './timm_models'
    os.makedirs(models_dir, exist_ok=True)
    
    # 需要下载的模型列表
    model_names = [
        'vit_base_patch16_224',
        'vit_small_patch16_224',
        'vit_large_patch16_224',
        # 可以添加其他需要的模型
    ]
    
    print("开始下载预训练模型...")
    for model_name in tqdm(model_names, desc="下载进度"):
        try:
            print(f"\n正在下载 {model_name}...")
            # 创建模型
            model = timm.create_model(model_name, pretrained=True)
            
            # 保存模型到指定目录
            save_path = os.path.join(models_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), save_path)
            
            print(f"{model_name} 下载并保存成功！保存位置: {save_path}")
        except Exception as e:
            print(f"{model_name} 下载失败: {str(e)}")
    
    print("\n下载完成！模型文件保存在:", models_dir)
    print("已下载的模型可以离线使用。")


if __name__ == "__main__":
    # 如果需要使用代理，取消下面两行的注释并设置代理地址
    # os.environ['http_proxy'] = 'http://your-proxy:port'
    # os.environ['https_proxy'] = 'http://your-proxy:port'
    
    download_models()