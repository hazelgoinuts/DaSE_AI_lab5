from transformers import RobertaTokenizer, RobertaModel

def download_model():
    print("下载RoBERTa模型和tokenizer...")
    try:
        # 使用镜像站点
        tokenizer = RobertaTokenizer.from_pretrained(
            'roberta-base',
            mirror='https://hf-mirror.com'  # 使用镜像站点
        )
        model = RobertaModel.from_pretrained(
            'roberta-base',
            mirror='https://hf-mirror.com'  # 使用镜像站点
        )
        
        # 保存到本地
        save_path = './models/roberta-base'
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print(f"下载成功！模型保存在: {save_path}")
        
    except Exception as e:
        print(f"下载出错: {str(e)}")
        print("请检查网络连接或代理设置")
        
        print("\n尝试以下解决方案：")
        print("1. 使用镜像站点：https://hf-mirror.com")
        print("2. 使用代理")
        print("3. 清理缓存目录：rm -rf ~/.cache/huggingface")

if __name__ == "__main__":
    # 如果需要使用代理，取消注释下面的代码
    # import os
    # os.environ['http_proxy'] = 'http://your-proxy:port'
    # os.environ['https_proxy'] = 'http://your-proxy:port'
    
    download_model()