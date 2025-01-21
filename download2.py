from transformers import DistilBertModel, DistilBertTokenizer
import os
import argparse

def download_model(model_name, save_dir):
    """
    下载预训练模型到指定目录
    
    Args:
        model_name: 模型名称
        save_dir: 保存目录
    """
    print(f"开始下载模型 {model_name}...")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 下载模型
        model = DistilBertModel.from_pretrained(model_name)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # 保存到本地
        model_path = os.path.join(save_dir, model_name)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        print(f"模型已成功下载并保存到 {model_path}")
        
    except Exception as e:
        print(f"下载失败: {str(e)}")
        
def main():
    parser = argparse.ArgumentParser(description='下载预训练模型')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                       help='要下载的模型名称')
    parser.add_argument('--save_dir', type=str, default='pretrained_models',
                       help='模型保存目录')
    
    args = parser.parse_args()
    
    download_model(args.model_name, args.save_dir)

if __name__ == '__main__':
    main()